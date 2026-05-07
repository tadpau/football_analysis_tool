"""Find frames where the ball is *rising into the stands* and the detector lost it.

Round 5 dataset miner. The remaining failure mode you identified visually:
the ball goes high in the air, crosses the far touchline horizon, and the
background flips from green grass to stands / buildings / sky. The detector
trained on grass-background balls fails on that distribution shift, so the
ball goes missing for a sustained run, then snaps back when the ball
descends below the horizon again.

We *cannot* detect "ball is over the stands" directly without re-running
inference, but the geometric signature is unambiguous from the stub alone:

  1. The ball was detected and *rising* (image-y decreasing) for at least
     RISING_MIN_FRAMES consecutive frames.
  2. Right after that streak, the ball detection vanishes (no entry in
     ``ft["ball"]``) for at least LOSS_MIN_FRAMES.
  3. The last seen ball position was already in the *upper* portion of the
     frame — i.e. close to the visual horizon where stands begin.

Frames matching that pattern almost always show "ball above the horizon,
detector lost it" — exactly the failure we want to mine. We score each
candidate and emit:

  * the most informative frame from each loss episode (one frame per
    episode — no point labelling 30 near-identical missed frames);
  * a CSV manifest with frame index, timestamp, last seen y, gap length;
  * JPEG cutouts ready for Label Studio.

The output dir is a sibling of frames_to_label/<clip>/ — call it
``frames_to_label/<clip>_high_ball/`` so it doesn't collide with previous
mining rounds. Run dedupe_frames.py on it before labelling, same as before.

Usage::

    python scripts/find_high_ball_failures.py stubs/VFA_KFM_pass1.pkl \
        --video video_clips/VFA_KFM.mp4 \
        --out-dir frames_to_label/VFA_KFM_high_ball \
        --top 80
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# Heuristic thresholds — tuned from the VFA_KFM stub. The hard discriminator
# between a *true high ball* and a *dribble/forward-pass* is whether a player
# is near the ball at the moment of loss: a kicker stays on the ground when
# they boot the ball skyward, so the ball ends up >> a player width away from
# anyone. A dribbled ball stays at the kicker's feet, so a player is always
# within tens of pixels.
RISING_MIN_FRAMES = 3        # need at least this many consecutive rising obs.
RISING_MIN_DY_PX = 6.0       # per-frame upward speed (image px) — slow lobs OK
LOSS_MIN_FRAMES = 4          # the ball must vanish for this long
HORIZON_FRAC = 0.45          # last-seen ball y must be in upper 45% of frame
MIN_PLAYER_DIST_PX = 80.0    # last-seen ball must be ≥ this far from every
                             # player bbox edge — kills the dribble false-flag
MAX_GAP_TO_FLAG = 60         # don't flag once we've waited >2 s @ 30 fps
EPISODE_MIN_GAP = 30         # frames; min spacing between flagged episodes


def _ball_centers(tracks: list[dict]) -> list[tuple[float, float] | None]:
    """Per-frame ball center (cx, cy) — None if no ball detection."""
    out: list[tuple[float, float] | None] = []
    for ft in tracks:
        ball = ft.get("ball", {}).get(1)
        if not ball:
            out.append(None)
            continue
        x1, y1, x2, y2 = ball["bbox"]
        out.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return out


def _nearest_player_dist(ft: dict, bx: float, by: float) -> float:
    """Pixel distance from (bx, by) to the nearest player/GK/ref bbox edge.

    Returns 0 if (bx, by) is inside any bbox, ``inf`` if no people in frame.
    Used to gate out dribbles: if the ball's last seen position is right next
    to a player, the "rising" was almost certainly the ball moving with that
    player rather than a real lob into the air.
    """
    best = float("inf")
    for cls in ("player", "goalkeeper", "referee"):
        for info in ft.get(cls, {}).values():
            x1, y1, x2, y2 = info["bbox"]
            dx = max(x1 - bx, 0.0, bx - x2)
            dy = max(y1 - by, 0.0, by - y2)
            d = (dx * dx + dy * dy) ** 0.5
            if d < best:
                best = d
    return best


def find_episodes(
    tracks: list[dict],
    frame_height: int,
    rising_min_frames: int = RISING_MIN_FRAMES,
    rising_min_dy: float = RISING_MIN_DY_PX,
    loss_min_frames: int = LOSS_MIN_FRAMES,
    horizon_frac: float = HORIZON_FRAC,
    min_player_dist: float = MIN_PLAYER_DIST_PX,
    max_gap_to_flag: int = MAX_GAP_TO_FLAG,
) -> list[dict]:
    """Walk the timeline, return one entry per high-ball-loss episode.

    Each entry contains the *recommended frame to label* (the one a few
    frames into the loss, where the ball is most likely cleanly above the
    horizon), plus diagnostic metadata so the CSV makes sense.
    """
    centers = _ball_centers(tracks)
    n = len(centers)
    horizon_px = horizon_frac * frame_height

    episodes: list[dict] = []
    rising_streak = 0          # length of current rising streak (frames)
    rising_apex_y: float | None = None  # highest (smallest y) seen during streak
    last_y: float | None = None
    in_loss = False
    loss_start: int | None = None
    pre_loss_y: float | None = None
    pre_loss_x: float | None = None
    pre_loss_idx: int | None = None
    pre_loss_was_rising = False

    for i in range(n):
        c = centers[i]
        if c is not None:
            cy = c[1]
            # Is this an upward step?
            if last_y is not None and (last_y - cy) >= rising_min_dy:
                rising_streak += 1
                rising_apex_y = cy if rising_apex_y is None else min(rising_apex_y, cy)
            else:
                # Reset only if we see a clear downward / sideways step.
                # Slight noise around a hovering ball shouldn't reset.
                if last_y is not None and (cy - last_y) > rising_min_dy:
                    rising_streak = 0
                    rising_apex_y = None

            # If we were inside a loss and the ball came back, finalise nothing
            # extra here — the episode was already recorded at loss-start. Just
            # update bookkeeping so future detections work.
            if in_loss:
                in_loss = False
                loss_start = None

            pre_loss_y = cy
            # Stash x too — needed for the player-proximity gate when the
            # ball later disappears.
            pre_loss_x = c[0]
            pre_loss_idx = i
            pre_loss_was_rising = rising_streak >= rising_min_frames
            last_y = cy
        else:
            # Ball is missing this frame.
            if not in_loss:
                in_loss = True
                loss_start = i
            # Detect a "high-ball loss" the moment the loss has lasted long
            # enough AND the precursor was a rising streak ending high in
            # frame.
            assert loss_start is not None
            gap = i - loss_start + 1
            if (
                gap == loss_min_frames
                and pre_loss_was_rising
                and pre_loss_y is not None
                and pre_loss_x is not None
                and pre_loss_y < horizon_px
                and pre_loss_idx is not None
                and (i - pre_loss_idx) <= max_gap_to_flag
            ):
                # Player-proximity gate — kills dribbles. A genuine high
                # ball is far from any player at the moment of loss because
                # the kicker stayed on the ground.
                player_dist = _nearest_player_dist(
                    tracks[pre_loss_idx], pre_loss_x, pre_loss_y
                )
                if player_dist < min_player_dist:
                    # Suppress this candidate but keep walking; future
                    # episodes are still possible.
                    pre_loss_was_rising = False
                    continue

                # Pick the labelling frame: the midpoint of the loss so far.
                # The ball is most likely cleanly above the horizon there
                # rather than at the very edge of the loss boundary.
                label_frame = loss_start + max(1, loss_min_frames // 2)
                # Score: higher when ball was further above horizon AND
                # rising fast. Recency-weighted; longer gaps slightly lower
                # (the ball might have already left the frame).
                height_score = max(0.0, (horizon_px - pre_loss_y) / horizon_px)
                rise_score = min(1.0, rising_streak / 6.0)
                gap_to_loss = i - pre_loss_idx
                recency = math.exp(-gap_to_loss / 20.0)
                score = (height_score * 1.5 + rise_score) * recency
                episodes.append({
                    "frame": label_frame,
                    "loss_start": loss_start,
                    "pre_loss_idx": pre_loss_idx,
                    "pre_loss_y": pre_loss_y,
                    "rising_streak": rising_streak,
                    "rising_apex_y": rising_apex_y,
                    "nearest_player_dist": player_dist,
                    "score": score,
                })
                # Don't reset rising_streak — if the ball reappears and goes
                # up again, that's a new episode. But we don't want to flag
                # the same episode twice, so once flagged at gap == min, we
                # mark pre_loss_was_rising = False to suppress duplicates.
                pre_loss_was_rising = False

    # Episodes are produced in time order. Apply a min-spacing dedupe so two
    # back-to-back losses at almost the same moment don't both get labelled.
    episodes.sort(key=lambda d: d["frame"])
    deduped: list[dict] = []
    for ep in episodes:
        if deduped and ep["frame"] - deduped[-1]["frame"] < EPISODE_MIN_GAP:
            # Keep whichever has the higher score.
            if ep["score"] > deduped[-1]["score"]:
                deduped[-1] = ep
            continue
        deduped.append(ep)
    return deduped


def dump_frames(
    video_path: Path, frames: list[int], out_dir: Path, clip_stem: str
) -> dict[str, dict]:
    """Read the requested frames from the source video, save as JPEG."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    manifest: dict[str, dict] = {}
    target = sorted(set(frames))
    target_idx = 0
    cur = 0
    while target_idx < len(target):
        want = target[target_idx]
        if want - cur > 30:
            cap.set(cv2.CAP_PROP_POS_FRAMES, want)
            cur = want
        while cur < want:
            ok, _ = cap.read()
            if not ok:
                break
            cur += 1
        ok, frame = cap.read()
        if not ok:
            break
        cur += 1
        fname = f"{clip_stem}_hi_f{want:06d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame)
        manifest[fname] = {
            "clip": clip_stem,
            "frame_idx": want,
            "timestamp_s": want / fps,
            "kind": "high_ball_loss",
        }
        target_idx += 1
    cap.release()
    return manifest


def _load_stub(path: Path) -> list[dict]:
    """Handle both stub formats:
      * bare ``list[dict]`` — output of legacy ``Tracker.get_object_tracks``;
      * ``{"tracks": [...], "camera_movement": [...]}`` — the new
        ``--analysis-stub`` written by ``main.run_streaming``.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "tracks" in obj:
        return obj["tracks"]
    if isinstance(obj, list):
        return obj
    raise RuntimeError(
        f"Unrecognised stub format in {path}: type={type(obj).__name__}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stub", type=Path,
                   help="Tracker stub (.pkl) or analysis-stub (.pkl) to read")
    p.add_argument("--video", type=Path, default=None,
                   help="Source clip — required to dump JPEG cutouts")
    p.add_argument("--out-dir", type=Path,
                   default=Path("frames_to_label/high_ball"),
                   help="Where to write CSV / JSON / JPEGs")
    p.add_argument("--top", type=int, default=80,
                   help="Cap on number of episodes to dump as JPEGs")
    p.add_argument("--frame-height", type=int, default=1080,
                   help="Source video height in pixels (used for the horizon "
                        "threshold — defaults to 1080p)")
    p.add_argument("--horizon-frac", type=float, default=HORIZON_FRAC,
                   help=f"Last-seen ball y must be < this fraction of "
                        f"frame_height to count as 'over the horizon'. "
                        f"Default {HORIZON_FRAC}.")
    p.add_argument("--rising-min-frames", type=int, default=RISING_MIN_FRAMES)
    p.add_argument("--rising-min-dy", type=float, default=RISING_MIN_DY_PX)
    p.add_argument("--loss-min-frames", type=int, default=LOSS_MIN_FRAMES)
    p.add_argument(
        "--min-player-dist", type=float, default=MIN_PLAYER_DIST_PX,
        help=f"Minimum px distance from the last-seen ball to any player "
             f"bbox edge. Below this, the loss is treated as a dribble and "
             f"suppressed. Default {MIN_PLAYER_DIST_PX}.",
    )
    p.add_argument("--frame-offset", type=int, default=0,
                   help="Add to every reported frame index — use when stub was "
                        "made with --start-frame N so output indices match the "
                        "ORIGINAL video.")
    args = p.parse_args()

    tracks = _load_stub(args.stub)
    print(f"Loaded {len(tracks)} frames from {args.stub}")

    episodes = find_episodes(
        tracks,
        frame_height=args.frame_height,
        rising_min_frames=args.rising_min_frames,
        rising_min_dy=args.rising_min_dy,
        loss_min_frames=args.loss_min_frames,
        horizon_frac=args.horizon_frac,
        min_player_dist=args.min_player_dist,
    )
    print(f"Found {len(episodes)} high-ball loss episodes "
          f"({len(episodes) / max(1, len(tracks)) * 1000:.2f} per 1000 frames)")
    if not episodes:
        print("No episodes flagged — either the model is healthy on high "
              "balls, or thresholds need loosening (try --horizon-frac 0.65).")
        return

    # Sort by score descending for top-K selection, but write CSV in time order.
    episodes_by_score = sorted(episodes, key=lambda d: d["score"], reverse=True)
    selected = sorted(episodes_by_score[: args.top], key=lambda d: d["frame"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clip_stem = args.video.stem if args.video else args.stub.stem
    off = args.frame_offset

    fps = 25.0
    if args.video is not None:
        import cv2
        cap = cv2.VideoCapture(str(args.video))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            cap.release()

    csv_path = args.out_dir / f"{clip_stem}_high_ball.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "timestamp_s", "score",
            "loss_start", "pre_loss_idx", "pre_loss_y",
            "rising_streak", "rising_apex_y", "nearest_player_dist",
        ])
        for ep in selected:
            global_fi = ep["frame"] + off
            w.writerow([
                global_fi,
                f"{global_fi / fps:.2f}",
                f"{ep['score']:.3f}",
                ep["loss_start"] + off,
                ep["pre_loss_idx"] + off,
                f"{ep['pre_loss_y']:.1f}",
                ep["rising_streak"],
                f"{ep['rising_apex_y']:.1f}" if ep['rising_apex_y'] else "",
                f"{ep['nearest_player_dist']:.1f}",
            ])
    print(f"  wrote {csv_path}")

    print("\nTop 10 episodes (by score):")
    for ep in episodes_by_score[:10]:
        print(
            f"  f={ep['frame'] + off:>6}  score={ep['score']:5.2f}  "
            f"last-seen-y={ep['pre_loss_y']:6.1f}  "
            f"rising-streak={ep['rising_streak']}f  "
            f"nearest-player={ep['nearest_player_dist']:5.1f}px"
        )

    if args.video is None:
        print("\n--video not supplied; skipping JPEG dump.")
        return

    manifest = dump_frames(
        args.video, [ep["frame"] + off for ep in selected],
        args.out_dir, clip_stem,
    )
    manifest_path = args.out_dir / f"{clip_stem}_high_ball_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {len(manifest)} JPEGs and {manifest_path}")


if __name__ == "__main__":
    main()
