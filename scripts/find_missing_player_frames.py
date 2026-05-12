"""Mine frames where the detector likely DROPPED players.

Crowded scenes — corners, set pieces, scrums in the box — are where
YOLO struggles most. The signature is that the visible-person count
for one frame is noticeably LOWER than the typical count in the
surrounding seconds: the camera didn't pan away, the play didn't
suddenly empty the screen, the model just lost a few players.

We surface those frames so the operator can label every visible
player (including the ones the model missed) and feed that as
targeted training data for the next round. This is the missing
counterpart to the high-ball miner — that one targets ball failures,
this one targets people failures.

Signal
------
For every frame, compute ``n_visible = |player| + |goalkeeper| +
|referee|`` after the basic stitcher has had its say. Compute the
ROLLING MEDIAN of that count over ``WINDOW_FRAMES`` (~2.5 s @ 30 fps,
adjustable). Any frame where ``n_visible`` is at least
``DROP_THRESHOLD`` people below the rolling median is flagged. The
score is the size of the drop — bigger drops mean the model lost
more people in that single frame.

Skips
-----
Frames where the rolling median is already low (camera off the pitch,
half-time, replays) get skipped. We only want frames where the
context says "many people are normally here, and right now they
aren't all detected".

Output
------
Same shape as the other miners:
  * CSV per-flagged-frame
  * JPEG cutouts of the top-K
  * JSON manifest

Usage::

    python scripts/find_missing_player_frames.py stubs/<clip>_pass1.pkl \\
        --video video_clips/<clip>.mp4 \\
        --out-dir frames_to_label/<clip>_missing_players \\
        --top 80
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# Defaults — match the high-ball miner's spacing conventions so output
# directories interleave cleanly.
WINDOW_FRAMES_DEFAULT = 75          # 2.5 s @ 30 fps
DROP_THRESHOLD_DEFAULT = 3          # this many fewer than rolling median = flag
MIN_CONTEXT_DEFAULT = 8             # rolling median needs >= this to be a "crowded" segment
EPISODE_MIN_GAP_DEFAULT = 30        # min frames between consecutive flagged episodes


def _per_frame_people_counts(tracks: list[dict]) -> list[int]:
    """Total people-detection count per frame (player + GK + ref)."""
    return [
        sum(len(ft.get(c, {})) for c in ("player", "goalkeeper", "referee"))
        for ft in tracks
    ]


def find_density_drops(
    tracks: list[dict],
    window_frames: int = WINDOW_FRAMES_DEFAULT,
    drop_threshold: int = DROP_THRESHOLD_DEFAULT,
    min_context: int = MIN_CONTEXT_DEFAULT,
    episode_min_gap: int = EPISODE_MIN_GAP_DEFAULT,
) -> list[dict]:
    """Return one entry per flagged "the model lost people here" episode.

    Episodes within ``episode_min_gap`` of each other collapse to the
    highest-scoring one — labelling 5 near-identical consecutive frames
    of the same scrum wastes effort.
    """
    counts = _per_frame_people_counts(tracks)
    n = len(counts)
    half = window_frames // 2
    flagged: list[dict] = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        nearby = counts[lo:hi]
        if not nearby:
            continue
        median = statistics.median(nearby)
        if median < min_context:
            # Rolling median is low — probably off-pitch / replay / half-time.
            # Not a "the model lost people in a crowded scene" frame.
            continue
        drop = median - counts[i]
        if drop >= drop_threshold:
            flagged.append({
                "frame": i,
                "score": float(drop),
                "n_visible": counts[i],
                "rolling_median": float(median),
            })

    # Greedy min-spacing collapse — keep the highest-score frame per
    # cluster. Same idiom as scripts/find_high_ball_failures.py.
    flagged.sort(key=lambda d: d["frame"])
    deduped: list[dict] = []
    for ep in flagged:
        if deduped and ep["frame"] - deduped[-1]["frame"] < episode_min_gap:
            if ep["score"] > deduped[-1]["score"]:
                deduped[-1] = ep
            continue
        deduped.append(ep)
    return deduped


def _load_stub(path: Path) -> list[dict]:
    """Accept both raw stub (legacy ``Tracker.get_object_tracks`` pickle)
    and the new ``--analysis-stub`` dict shape."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "tracks" in obj:
        return obj["tracks"]
    if isinstance(obj, list):
        return obj
    raise RuntimeError(
        f"Unrecognised stub format in {path}: type={type(obj).__name__}"
    )


def dump_frames(
    video_path: Path, frames: list[int], out_dir: Path, clip_stem: str,
) -> dict[str, dict]:
    """Extract each frame and save as JPEG. Mirrors the helper in
    ``find_high_ball_failures.py`` — same seek-or-decode logic, same
    file-name convention."""
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
        fname = f"{clip_stem}_miss_f{want:06d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame)
        manifest[fname] = {
            "clip": clip_stem,
            "frame_idx": want,
            "timestamp_s": want / fps,
            "kind": "density_drop",
        }
        target_idx += 1
    cap.release()
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stub", type=Path,
                   help="Tracker stub (.pkl) or analysis-stub (.pkl) to read")
    p.add_argument("--video", type=Path, default=None,
                   help="Source clip — required to dump JPEG cutouts")
    p.add_argument("--out-dir", type=Path,
                   default=Path("frames_to_label/missing_players"),
                   help="Where to write CSV / JSON / JPEGs")
    p.add_argument("--top", type=int, default=80,
                   help="Cap on episodes to dump as JPEGs")
    p.add_argument("--window", type=int, default=WINDOW_FRAMES_DEFAULT,
                   help=f"Rolling-median window in frames "
                        f"(default {WINDOW_FRAMES_DEFAULT} ≈ 2.5 s)")
    p.add_argument("--drop", type=int, default=DROP_THRESHOLD_DEFAULT,
                   help=f"Min drop vs rolling median to flag "
                        f"(default {DROP_THRESHOLD_DEFAULT}). Higher = "
                        f"only flag bigger losses.")
    p.add_argument("--min-context", type=int, default=MIN_CONTEXT_DEFAULT,
                   help=f"Rolling median must be ≥ this to consider the "
                        f"segment crowded (default {MIN_CONTEXT_DEFAULT}). "
                        f"Lower for tight cameras where the whole team "
                        f"is rarely on screen.")
    p.add_argument("--episode-gap", type=int, default=EPISODE_MIN_GAP_DEFAULT,
                   help=f"Min frame gap between flagged episodes "
                        f"(default {EPISODE_MIN_GAP_DEFAULT})")
    p.add_argument("--frame-offset", type=int, default=0,
                   help="Add to every reported frame index — use when stub "
                        "was made with --start-frame N so output indices "
                        "match the ORIGINAL video.")
    args = p.parse_args()

    tracks = _load_stub(args.stub)
    print(f"Loaded {len(tracks)} frames from {args.stub}")

    episodes = find_density_drops(
        tracks,
        window_frames=args.window,
        drop_threshold=args.drop,
        min_context=args.min_context,
        episode_min_gap=args.episode_gap,
    )
    print(f"Found {len(episodes)} density-drop episodes "
          f"({len(episodes) / max(1, len(tracks)) * 1000:.2f} per 1000 frames)")
    if not episodes:
        print("No drops flagged — model density looks consistent. Try "
              "lowering --drop or --min-context.")
        return

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

    csv_path = args.out_dir / f"{clip_stem}_missing_players.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "timestamp_s", "score",
            "n_visible", "rolling_median",
        ])
        for ep in selected:
            gfi = ep["frame"] + off
            w.writerow([
                gfi, f"{gfi / fps:.2f}",
                f"{ep['score']:.1f}",
                ep["n_visible"],
                f"{ep['rolling_median']:.1f}",
            ])
    print(f"  wrote {csv_path}")

    print("\nTop 10 episodes (by drop size):")
    for ep in episodes_by_score[:10]:
        print(
            f"  f={ep['frame'] + off:>6}  drop={ep['score']:4.1f}  "
            f"n_visible={ep['n_visible']:>2}  median={ep['rolling_median']:5.1f}"
        )

    if args.video is None:
        print("\n--video not supplied; skipping JPEG dump.")
        return

    manifest = dump_frames(
        args.video, [ep["frame"] + off for ep in selected],
        args.out_dir, clip_stem,
    )
    manifest_path = args.out_dir / f"{clip_stem}_missing_players_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {len(manifest)} JPEGs and {manifest_path}")


if __name__ == "__main__":
    main()
