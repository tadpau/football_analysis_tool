"""Motion-aware ball tracker — replaces naive linear interpolation.

Why the simpler pandas-interpolate approach was not enough:
  * pandas fills a gap linearly between the last real detection and the NEXT
    real detection. If that next detection is a false positive (a white sock,
    a corner-flag shadow, a stray line on the ad board), the gap is filled
    with a straight line toward the wrong spot — the green triangle slides
    across the pitch to meet it.
  * The outlier filter only drops detections that fall inside a player's
    bbox. A false ball in open grass gets through.

The fix: track the ball as a point with *velocity*. Each frame we maintain:
    pos       — last accepted (x, y) centre
    vel       — (dx, dy) per-frame velocity, EMA-smoothed
    last_wh   — last known bbox width/height (ball size rarely changes)
    gap       — frames since the last accepted detection

On each new frame:

  1. Predict  pred = pos + vel.
  2. If there IS a detection and it lies within TRAJECTORY_GATE_PX of `pred`
     (OR we've been lost for a while and any detection is welcome), accept it:
     update velocity, zero the gap, keep the detection's bbox.
  3. If there's a detection but it's OFF the predicted trajectory, REJECT it
     (it's a false positive — a sock, a white line, a duplicate on a player).
     Extrapolate instead.
  4. If no detection (or we just rejected one), and gap < MAX_EXTRAP_FRAMES,
     emit the predicted position as an interpolated frame and keep going.
  5. If gap exceeds MAX_EXTRAP_FRAMES, the ball is genuinely lost — drop state
     and wait for the next real detection to re-initialise.

Player-proximity gate (added later)
-----------------------------------
Real broadcasts are full of out-of-pitch ball detections: warm-up balls behind
the goal, decoration balls in the corners, training balls on the touchline.
With nothing tying the ball to "where humans are", the tracker would happily
latch onto these the moment we lost the real ball (e.g. when a player blocks
it from camera view).

A ball is real iff it's near a person, with one exception: a ball mid-flight
during a long pass is *supposed* to be far from everyone for a few frames.
We resolve that by only requiring proximity when we have NO trajectory
evidence — bootstrap (first sighting, or after we gave up) and "lost long
enough" recovery. Mid-flight (state alive, small gap) keeps the pure
trajectory gate so legitimate long passes still pass.

Velocity-conditional trajectory gate (Idea A)
---------------------------------------------
A held / dribbled ball moves at walking pace (≤ a few px/frame). It does not
teleport 200 px between frames — if a "ball" detection appears that far away
when the prior ball was stationary, it's almost certainly a false positive on
someone's sock or sleeve. So the gate width depends on the prior speed:

  * speed < CARRIER_SPEED_THRESHOLD  → tight gate (TIGHT_GATE_PX, ~50 px)
  * speed ≥ CARRIER_SPEED_THRESHOLD  → loose gate (TRAJECTORY_GATE_PX, ~200 px)
  * (carrier-anchored — see below)   → loose gate (must allow throw release)

Carrier anchoring (Idea B)
--------------------------
When a ball detection is accepted close to a person AND the ball is moving
slowly, we record THAT person's track id as the "carrier". On subsequent
frames where the detector loses the ball, instead of velocity-extrapolating
into stale space, we PIN the ball to the carrier's bbox (using the in-bbox
offset captured at acceptance). That handles:

  * Throw-in setup: ball held above head for several seconds with no detection.
  * Slow dribble / carry: detector occasionally drops the small ball at feet.
  * Carrier-occlusion: player's body blocks the ball from camera, but we
    know they have it.

Carrier anchoring gets a longer extrap budget than velocity extrap (30 vs 10
frames) because "player walking with ball" is a much more stable extrapolation
than "ballistic ball trajectory". When a real detection re-appears, it's
evaluated against the loose gate (so a throw or kick that suddenly moves the
ball clear of the carrier still gets accepted), and on acceptance the carrier
status is recomputed — if the ball is now moving fast or far from any person,
the carrier is dropped and we go back to ballistic mode.

Outputs the same schema as `interpolate_ball_positions` so main.py + the
annotator don't need to know which method was used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable


# Tuning — these are for broadcast football at ~24 fps / 1920×1080.
# A hard pass tops out around 30 m/s; at 105 m pitch length viewed across
# ~1700 px, that's ~140 px/frame in the long direction. 200 px is a generous
# gate that still catches spurious detections on the wrong side of the pitch.
TRAJECTORY_GATE_PX = 200.0

# After this many consecutive gap frames, we stop extrapolating. Beyond ~0.4 s
# the velocity extrapolation is no longer reliable — ball may have been
# kicked / stopped / changed direction.
MAX_EXTRAP_FRAMES = 10

# EMA weight on velocity updates. Higher = more responsive to direction
# changes, lower = smoother but slower to adapt.
VELOCITY_EMA = 0.5

# Clip extreme velocity estimates (e.g. first-frame noise from a big bbox).
MAX_SPEED_PX_PER_FRAME = 220.0

# Player-proximity gate (in pixels, measured from ball centre to nearest edge
# of any player/GK/ref bbox). 120 px ≈ a player's outstretched arm + a couple
# of strides at this clip's scale. Used at bootstrap and during "lost-long-
# enough" recovery to prevent latching onto out-of-bounds false positives
# (warm-up balls, corner-flag shadows, ad-board patterns). NOT used while a
# trajectory is alive — a long pass is supposed to be far from everyone for
# a few frames.
PLAYER_PROXIMITY_PX = 120.0

# Velocity-conditional gate (Idea A). Gate width depends on whether the ball
# was last seen moving (mid-flight) vs stationary (held / dribbled).
TIGHT_GATE_PX = 50.0
CARRIER_SPEED_THRESHOLD = 20.0   # px/frame; below = walking-pace = carried

# Carrier anchoring (Idea B). When the ball is held / dribbled, we pin its
# position to the carrying player's bbox during detection gaps instead of
# velocity-extrapolating. Carrier-anchored extrap gets a much longer budget
# than ballistic extrap because "player carrying ball" is a stable state.
MAX_CARRIER_FRAMES = 30          # ≈ 1.25 s @ 24 fps; cap so a carrier walking
                                 # away from the ball doesn't pin forever.

# After being genuinely lost (gap > MAX_EXTRAP_FRAMES with no carrier, or
# > MAX_CARRIER_FRAMES with carrier), any next real detection is accepted —
# we have no trajectory anymore. (Modified by player-proximity gate: now
# requires the detection to be near a person.)


@dataclass
class _State:
    x: float
    y: float
    vx: float
    vy: float
    w: float           # last known bbox width
    h: float           # last known bbox height
    gap: int           # frames since last real detection
    confidence: float  # last real detection's confidence
    # Carrier-anchoring fields. carrier_id is None when the ball is in flight
    # or sitting alone on the pitch. Set to (cls_name, track_id) when the
    # last accepted detection was near a person and slow.
    carrier_id: tuple | None = None
    # Ball position relative to the carrier's bbox top-left, in pixels. Used
    # to keep the ball at the same spot inside the bbox as the carrier moves
    # (above head for throw-in, at feet for dribble, etc.).
    carrier_offset: tuple[float, float] | None = None


def _center(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _wh(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return x2 - x1, y2 - y1


def _clip_speed(vx: float, vy: float, max_speed: float) -> tuple[float, float]:
    speed = math.hypot(vx, vy)
    if speed <= max_speed or speed == 0.0:
        return vx, vy
    k = max_speed / speed
    return vx * k, vy * k


def _bbox_dist(point, bbox) -> float:
    """Euclidean distance from a point to the nearest edge of an axis-aligned
    bbox. Returns 0 if the point is inside the bbox."""
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _nearest_person(
    point, persons,
) -> tuple[float, Hashable | None, tuple | None]:
    """Find nearest person to a point. Returns (dist, person_id, bbox).
    `persons` is a list of (id, bbox) tuples. Returns (inf, None, None)
    if persons is empty."""
    if not persons:
        return float("inf"), None, None
    best_d = float("inf")
    best_id: Hashable | None = None
    best_bbox: tuple | None = None
    for pid, bbox in persons:
        d = _bbox_dist(point, bbox)
        if d < best_d:
            best_d = d
            best_id = pid
            best_bbox = bbox
            if best_d == 0.0:
                break
    return best_d, best_id, best_bbox


def _find_person_bbox(persons, target_id) -> tuple | None:
    """Return the bbox of `target_id` in this frame's person list, or None."""
    for pid, bbox in persons:
        if pid == target_id:
            return bbox
    return None


def track_ball_with_motion(
    ball_positions: list[dict],
    persons_per_frame: list[list[tuple[Hashable, tuple]]] | None = None,
    trajectory_gate_px: float = TRAJECTORY_GATE_PX,
    tight_gate_px: float = TIGHT_GATE_PX,
    max_extrap_frames: int = MAX_EXTRAP_FRAMES,
    max_carrier_frames: int = MAX_CARRIER_FRAMES,
    velocity_ema: float = VELOCITY_EMA,
    max_speed_px_per_frame: float = MAX_SPEED_PX_PER_FRAME,
    player_proximity_px: float = PLAYER_PROXIMITY_PX,
    carrier_speed_threshold: float = CARRIER_SPEED_THRESHOLD,
) -> tuple[list[dict], dict[str, int]]:
    """Walk ball detections frame-by-frame with a position + velocity model.

    Args:
        ball_positions: per-frame dict, either ``{}`` or ``{1: {"bbox": ...}}``.
        persons_per_frame: optional per-frame list of (person_id, bbox)
            tuples for any humans on the field. ``person_id`` should be
            unique across all classes — ``(class_name, track_id)`` works
            well. Enables the player-proximity gate AND carrier anchoring.
            Pass None or omit to disable both (legacy behaviour).

    Returns:
        (out, stats) where ``out`` has the same schema with gaps filled and
        off-trajectory detections rejected. ``stats`` reports per-bucket
        counts: accepted / rejected / extrapolated / carrier_anchored /
        rejected_far_from_person / lost.
    """
    state: _State | None = None
    out: list[dict] = []
    stats = {
        "accepted": 0, "rejected": 0, "extrapolated": 0, "lost": 0,
        "rejected_far_from_person": 0, "carrier_anchored": 0,
    }

    proximity_enabled = persons_per_frame is not None

    for fi, fp in enumerate(ball_positions):
        det = fp.get(1)
        det_bbox = det.get("bbox") if det else None
        det_conf = det.get("confidence", 0.0) if det else 0.0

        persons_now = (
            persons_per_frame[fi]
            if proximity_enabled and fi < len(persons_per_frame)
            else []
        )

        # Compute proximity info for the current detection (if any). Used
        # for bootstrap, lost-recovery, AND for re-evaluating carrier status
        # on every accepted detection.
        near_person = True
        nearest_id: Hashable | None = None
        nearest_bbox: tuple | None = None
        if proximity_enabled and det_bbox is not None:
            d_person, nearest_id, nearest_bbox = _nearest_person(
                _center(det_bbox), persons_now,
            )
            near_person = d_person <= player_proximity_px

        # ----- no state yet: wait for first real detection ---------------
        if state is None:
            if det_bbox is not None and near_person:
                dx, dy = _center(det_bbox)
                dw, dh = _wh(det_bbox)
                new_state = _State(
                    x=dx, y=dy, vx=0.0, vy=0.0,
                    w=dw, h=dh, gap=0, confidence=det_conf,
                )
                # First detection — speed is unknown (no prior frame). Treat
                # as "carried" if near a person, since most match restarts
                # (kickoff, throw-in) feature a player holding/standing-with
                # the ball. Carrier is dropped immediately if next frame's
                # speed proves it's actually mid-flight.
                if proximity_enabled and nearest_id is not None and nearest_bbox is not None:
                    new_state.carrier_id = nearest_id
                    new_state.carrier_offset = (
                        dx - nearest_bbox[0],
                        dy - nearest_bbox[1],
                    )
                state = new_state
                out.append({1: {
                    "bbox": list(det_bbox),
                    "confidence": det_conf,
                    "interpolated": False,
                }})
                stats["accepted"] += 1
            else:
                if det_bbox is not None and not near_person:
                    stats["rejected_far_from_person"] += 1
                out.append({})
            continue

        # ----- we have state: predict, then decide ------------------------
        pred_x = state.x + state.vx
        pred_y = state.y + state.vy

        # Velocity-conditional gate (Idea A). When carrier-anchored, force
        # the loose gate so a thrown / kicked ball that suddenly moves clear
        # of the carrier still gets accepted. When free + slow, use the tight
        # gate to refuse "ball teleported 200 px" false positives.
        prior_speed = math.hypot(state.vx, state.vy)
        if state.carrier_id is not None:
            current_gate = trajectory_gate_px
        elif prior_speed < carrier_speed_threshold:
            current_gate = tight_gate_px
        else:
            current_gate = trajectory_gate_px

        if det_bbox is not None:
            dx, dy = _center(det_bbox)
            dist = math.hypot(dx - pred_x, dy - pred_y)
            # Acceptance paths:
            #   (a) on-trajectory     — within the velocity-conditional gate
            #   (b) lost-recovery     — we have no trajectory anymore; require
            #                           proximity to a person (kills false
            #                           positives in empty regions)
            on_trajectory = dist <= current_gate
            # Carrier-anchored extra rule: a release event (throw, kick) ALWAYS
            # has the ball near the releaser. If we're carrier-anchored and a
            # "ball" appears far from any person, it's almost certainly a
            # false positive on a sock / jersey patch elsewhere — refuse.
            # Without this, the loose carrier gate (200 px) would happily
            # latch onto teleports.
            if state.carrier_id is not None and not near_person:
                on_trajectory = False

            extrap_budget = (
                max_carrier_frames if state.carrier_id is not None
                else max_extrap_frames
            )
            lost_recovery = state.gap >= extrap_budget and near_person
            accept = on_trajectory or lost_recovery

            if not accept and not near_person:
                # Whether we hit the lost-recovery branch or the carrier
                # extra rule, the rejection reason is the same: too far
                # from anyone to be a real ball.
                if state.gap >= extrap_budget or state.carrier_id is not None:
                    stats["rejected_far_from_person"] += 1

            if accept:
                # Update velocity (EMA). Divide by gap+1 because the last
                # accepted detection may have been several frames back.
                new_vx = (dx - state.x) / (state.gap + 1)
                new_vy = (dy - state.y) / (state.gap + 1)
                new_vx = velocity_ema * new_vx + (1 - velocity_ema) * state.vx
                new_vy = velocity_ema * new_vy + (1 - velocity_ema) * state.vy
                new_vx, new_vy = _clip_speed(new_vx, new_vy, max_speed_px_per_frame)
                state.x, state.y = dx, dy
                state.vx, state.vy = new_vx, new_vy
                state.w, state.h = _wh(det_bbox)
                state.gap = 0
                state.confidence = det_conf

                # Re-evaluate carrier on every acceptance. Ball must be both
                # near a person AND moving slowly to be considered carried.
                # Anything else (mid-flight, alone in midfield) clears the
                # carrier so we go back to ballistic mode.
                new_speed = math.hypot(new_vx, new_vy)
                if (proximity_enabled and near_person
                        and new_speed < carrier_speed_threshold
                        and nearest_id is not None and nearest_bbox is not None):
                    state.carrier_id = nearest_id
                    state.carrier_offset = (
                        dx - nearest_bbox[0],
                        dy - nearest_bbox[1],
                    )
                else:
                    state.carrier_id = None
                    state.carrier_offset = None

                out.append({1: {
                    "bbox": list(det_bbox),
                    "confidence": det_conf,
                    "interpolated": False,
                }})
                stats["accepted"] += 1
                continue
            # Else: detection failed both gates — drop through to extrap.
            stats["rejected"] += 1

        # ----- no usable detection this frame ---------------------------
        # Prefer carrier-anchored extrap: if we know who has the ball and
        # they're still in this frame, pin the ball to their bbox using
        # the stored offset. Otherwise fall back to ballistic extrap.
        carrier_bbox = None
        if state.carrier_id is not None and proximity_enabled:
            carrier_bbox = _find_person_bbox(persons_now, state.carrier_id)

        if carrier_bbox is not None and state.gap < max_carrier_frames:
            ox, oy = state.carrier_offset
            new_x = carrier_bbox[0] + ox
            new_y = carrier_bbox[1] + oy
            # Velocity now reflects how the carrier is moving (frame-to-frame
            # delta of pinned ball position). Useful for the next acceptance.
            state.vx = new_x - state.x
            state.vy = new_y - state.y
            state.x, state.y = new_x, new_y
            state.gap += 1
            bbox = [
                state.x - state.w * 0.5,
                state.y - state.h * 0.5,
                state.x + state.w * 0.5,
                state.y + state.h * 0.5,
            ]
            out.append({1: {
                "bbox": bbox,
                # Carrier-anchored guess is more reliable than pure ballistic
                # extrap, so keep its confidence higher.
                "confidence": state.confidence * 0.85,
                "interpolated": True,
            }})
            stats["carrier_anchored"] += 1
        elif state.gap < max_extrap_frames:
            state.x = pred_x
            state.y = pred_y
            state.gap += 1
            # Decay velocity slightly each extrap frame so a stale velocity
            # doesn't keep the phantom ball cruising across the pitch.
            state.vx *= 0.9
            state.vy *= 0.9
            bbox = [
                state.x - state.w * 0.5,
                state.y - state.h * 0.5,
                state.x + state.w * 0.5,
                state.y + state.h * 0.5,
            ]
            out.append({1: {
                "bbox": bbox,
                "confidence": state.confidence * 0.7,
                "interpolated": True,
            }})
            stats["extrapolated"] += 1
        else:
            # Given up — neither carrier nor velocity extrap is usable any
            # more. Wait for a fresh detection to re-init state.
            state = None
            out.append({})
            stats["lost"] += 1

    return out, stats
