"""Ball-possession assignment.

Rules:
  * A frame has an owner only if (a) the ball was *actually detected* in that
    frame (interpolated frames don't count — we don't know where the ball really
    is) AND (b) some player's nearest foot is within `MAX_BALL_DIST_PX`.
  * A ball "in the air" or in open space -> no owner that frame, even if a
    player happens to be within the distance budget. We approximate this by
    requiring the ball to be at or below the player's bbox bottom (i.e., near
    the feet, not floating above their head).
  * A short smoothing pass suppresses 1-frame owner flips between two players
    in a tight contest. It NEVER invents an owner for a frame that had none.

Foot approximation: bbox bottom-left and bottom-right corners. Cheaper than
running pose estimation; accurate enough at broadcast-camera scale.
"""
from __future__ import annotations

from collections import Counter

from src.utils.bbox_utils import get_center, measure_distance

MAX_BALL_DIST_PX = 60   # foot-to-ball distance cap; tune per resolution
MAX_BALL_ABOVE_FEET_PX = 40   # ball this far above a player's feet -> "in the air"


def assign_ball_to_player(players: dict, ball_bbox) -> int | None:
    """Return track_id of the player with the ball, or None.

    Returns None if (a) nobody is within MAX_BALL_DIST_PX, or (b) the ball
    center is well above the closest player's feet (ball in the air).
    """
    ball_center = get_center(ball_bbox)
    bx, by = ball_center

    best_id: int | None = None
    best_dist = MAX_BALL_DIST_PX

    for pid, info in players.items():
        x1, _y1, x2, y2 = info["bbox"]
        # Reject: ball is well above this player's feet (ball is airborne /
        # at chest or head height — not a foot-level dribble).
        if (y2 - by) > MAX_BALL_ABOVE_FEET_PX:
            continue
        left_foot = (x1, y2)
        right_foot = (x2, y2)
        d = min(
            measure_distance(ball_center, left_foot),
            measure_distance(ball_center, right_foot),
        )
        if d < best_dist:
            best_dist = d
            best_id = pid
    return best_id


def smooth_owner_series(
    raw: list[int | None], window: int = 9, min_consensus: int = 4
) -> list[int | None]:
    """Reduce owner flips WITHOUT inventing owners for None frames.

    For each frame:
      * if raw[i] is None    -> output None (never carry forward a stale owner)
      * if raw[i] is X       -> output X only if:
          - at least ``min_consensus`` frames in a ±window/2 neighbourhood also
            voted X, AND
          - X is also the *plurality* winner in that window (no other owner
            has strictly more votes).

    Window size was doubled (5→9) and min_consensus raised (2→4) after the red
    ball-owner triangle was flickering frame-to-frame when two players contest
    possession. With these settings, X only gets painted if they genuinely held
    the ball for ~⅓ of a ~9-frame window (≈0.4 s at 24 fps) and dominated any
    rival in the window.
    """
    out: list[int | None] = []
    half = window // 2
    for i, current in enumerate(raw):
        if current is None:
            out.append(None)
            continue
        lo, hi = max(0, i - half), min(len(raw), i + half + 1)
        votes = Counter(v for v in raw[lo:hi] if v is not None)
        own_votes = votes.get(current, 0)
        plurality = max(votes.values()) if votes else 0
        if own_votes >= min_consensus and own_votes >= plurality:
            out.append(current)
        else:
            out.append(None)
    return out


def compute_rolling_team_share(
    tracks: list[dict],
    per_frame_owner: list[int | None],
    window_frames: int,
) -> list[dict[int, float]]:
    """Per-frame possession ratios over a trailing window.

    Returns ``out[i] = {1: share1, 2: share2}`` where each share is the
    fraction of *owned* frames in ``[i-window_frames+1, i]`` that belonged
    to that team (un-owned / interpolated frames are excluded from the
    denominator). If the window contains no owned frames, both shares are 0.

    O(n) total via a sliding counter — safe for hour-long matches.
    """
    n = len(per_frame_owner)
    out: list[dict[int, float]] = []
    if window_frames < 1:
        window_frames = 1

    # Resolve per-frame owning team once (None if no owner this frame).
    owning_team: list[int | None] = []
    for ft, owner in zip(tracks, per_frame_owner):
        if owner is None:
            owning_team.append(None)
            continue
        owning_team.append(ft.get("player", {}).get(owner, {}).get("team"))

    counts = {1: 0, 2: 0}
    for i in range(n):
        # Add the new frame entering the right edge of the window.
        t = owning_team[i]
        if t in counts:
            counts[t] += 1
        # Drop the frame leaving the left edge.
        drop_i = i - window_frames
        if drop_i >= 0:
            t_drop = owning_team[drop_i]
            if t_drop in counts:
                counts[t_drop] -= 1

        total = counts[1] + counts[2]
        if total == 0:
            out.append({1: 0.0, 2: 0.0})
        else:
            out.append({1: counts[1] / total, 2: counts[2] / total})

    return out


def compute_team_possession(
    tracks: list[dict],
    smooth: bool = True,
) -> tuple[list[int | None], dict[int, float]]:
    """Per-frame owning player + team possession ratios.

    Only frames whose ball was *actually detected* (not interpolated) can
    produce an owner. This is the user-facing rule: the red triangle should
    only ever appear when we can confirm the ball is at someone's feet.
    """
    raw_owner: list[int | None] = []
    for frame_tracks in tracks:
        ball = frame_tracks.get("ball", {}).get(1)
        players = frame_tracks.get("player", {})
        if not ball or ball.get("interpolated") or not players:
            raw_owner.append(None)
            continue
        raw_owner.append(assign_ball_to_player(players, ball["bbox"]))

    per_frame_owner = smooth_owner_series(raw_owner) if smooth else raw_owner

    counts = {1: 0, 2: 0}
    for ft, owner in zip(tracks, per_frame_owner):
        if owner is None:
            continue
        team = ft.get("player", {}).get(owner, {}).get("team")
        if team in counts:
            counts[team] += 1

    total = counts[1] + counts[2]
    if total == 0:
        return per_frame_owner, {1: 0.0, 2: 0.0}
    return per_frame_owner, {1: counts[1] / total, 2: counts[2] / total}
