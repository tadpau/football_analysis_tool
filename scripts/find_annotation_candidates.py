"""Find frames worth annotating to maximally improve the detector.

Where the model is *currently wrong* is where new labels pay back the most.
This script reads a tracker stub (the pickled output of `Tracker.get_object_tracks`)
and scores each frame on six independent failure-mode signals, then emits:

  * a CSV ranking every frame by total score (descending);
  * the top-N as JPEG cutouts in an output dir (so they're ready for Label
    Studio drag-and-drop);
  * a JSON manifest mapping frame index → (clip, timestamp_s, reasons[]).

Failure-mode signals (all derived from raw stub data — no model re-run needed):

  1. **Ball gap** — runs of ≥ MIN_BALL_GAP frames where the detector returned
     no ball at all. The ball is the single hardest class for the model and
     long gaps are exactly the frames where re-training would help.

  2. **Ball-in-player** — the ball detection center lies inside a player bbox.
     Either the model is firing on a white shoe / glove, or it's a true
     foot-on-ball moment that the Phase-5a outlier filter would (correctly)
     drop. Both are valuable — white-shoe FPs need negative examples; real
     foot-on-ball moments need positive examples so the filter doesn't have
     to drop them.

  3. **Low-confidence detection** — any detection with confidence below
     LOW_CONF. Frames packed with hesitant detections are exactly the visual
     conditions the model finds hardest (motion blur, far end of the pitch,
     unusual pose).

  4. **Short-lived track** — a track ID that exists for fewer than
     SHORT_TRACK_FRAMES total frames is overwhelmingly a false positive
     (ad-hoc occlusion blob, ref shadow, kit detail). Logging the frame
     where the short track first appeared lets the annotator mark it as
     background.

  5. **Detection-count jump** — frames where the active count of (player +
     GK + ref) changes by ≥ COUNT_JUMP from the previous frame. Either we
     just had a sudden missed-detection burst, or someone genuinely walked
     in/out of frame at a clip boundary. Both worth eyeballing.

  6. **New track-ID birth** — a frame where ByteTrack first assigned a new
     track ID, weighted by clip-position. New IDs late in a clip almost
     always = an ID swap during a crossing, the highest-leverage frames for
     improving tracking robustness via better detector outputs.

The scores are summed (with separate weights per signal) and the top-N are
returned. Each signal contributes uncorrelated information, so the union
covers different failure modes rather than piling up the same kind of frame.

Usage::

    python scripts/find_annotation_candidates.py stubs/clip5.pkl \
        --video video_clips/clip_5.mp4 \
        --out-dir frames_to_label/clip_5_candidates \
        --top 60

If ``--video`` is omitted, no JPEGs are written — only the CSV/JSON manifests.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# Signal thresholds — tuned to surface ~50–100 frames per 2-minute clip.
MIN_BALL_GAP = 5            # frames — runs shorter than this are normal
LOW_CONF = 0.40             # detection confidence floor
SHORT_TRACK_FRAMES = 8      # track shorter than this = almost certain FP
COUNT_JUMP = 4              # sudden detection-count change threshold

# Per-signal score weight. Weights are deliberately not normalised — the goal
# is to make sure a single "ball-in-player" frame outranks a single
# "low-confidence detection" frame, since the former is much more diagnostic.
W_BALL_GAP = 3.0
W_BALL_IN_PLAYER = 5.0
W_LOW_CONF = 0.5            # per low-conf detection in a frame
W_SHORT_TRACK = 2.0         # at the frame the short track first appeared
W_COUNT_JUMP = 2.0
W_NEW_ID = 1.0


# ---------------------------------------------------------------- per-signal
def _score_ball_gaps(tracks: list[dict]) -> dict[int, list[str]]:
    """Mark every frame inside a run-of-no-ball ≥ MIN_BALL_GAP."""
    flagged: dict[int, list[str]] = defaultdict(list)
    run_start: int | None = None
    for i, ft in enumerate(tracks):
        has_ball = bool(ft.get("ball"))
        if not has_ball:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= MIN_BALL_GAP:
                for k in range(run_start, i):
                    flagged[k].append(f"ball_gap_run={i - run_start}")
            run_start = None
    if run_start is not None and len(tracks) - run_start >= MIN_BALL_GAP:
        for k in range(run_start, len(tracks)):
            flagged[k].append(f"ball_gap_run={len(tracks) - run_start}")
    return flagged


def _score_ball_in_player(tracks: list[dict]) -> dict[int, list[str]]:
    flagged: dict[int, list[str]] = defaultdict(list)
    for i, ft in enumerate(tracks):
        ball = ft.get("ball", {}).get(1)
        if not ball:
            continue
        bx1, by1, bx2, by2 = ball["bbox"]
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        for info in ft.get("player", {}).values():
            px1, py1, px2, py2 = info["bbox"]
            if px1 <= bcx <= px2 and py1 <= bcy <= py2:
                flagged[i].append("ball_in_player")
                break
    return flagged


def _score_low_conf(tracks: list[dict]) -> dict[int, list[str]]:
    flagged: dict[int, list[str]] = defaultdict(list)
    for i, ft in enumerate(tracks):
        n_low = 0
        for cls in ("player", "goalkeeper", "referee", "ball"):
            for info in ft.get(cls, {}).values():
                if info.get("confidence", 1.0) < LOW_CONF:
                    n_low += 1
        if n_low > 0:
            flagged[i].append(f"low_conf_n={n_low}")
    return flagged


def _score_short_tracks(tracks: list[dict]) -> dict[int, list[str]]:
    """Find short-lived track ids; flag the frame they first appeared."""
    spans: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, ft in enumerate(tracks):
        for cls in ("player", "goalkeeper", "referee"):
            for tid in ft.get(cls, {}).keys():
                spans[(cls, tid)].append(i)

    flagged: dict[int, list[str]] = defaultdict(list)
    for (cls, tid), frames in spans.items():
        if len(frames) < SHORT_TRACK_FRAMES:
            flagged[frames[0]].append(f"short_track_{cls}{tid}_n={len(frames)}")
    return flagged


def _score_count_jumps(tracks: list[dict]) -> dict[int, list[str]]:
    flagged: dict[int, list[str]] = defaultdict(list)
    prev = None
    for i, ft in enumerate(tracks):
        cur = sum(len(ft.get(c, {})) for c in ("player", "goalkeeper", "referee"))
        if prev is not None and abs(cur - prev) >= COUNT_JUMP:
            flagged[i].append(f"count_jump={cur - prev:+d}")
        prev = cur
    return flagged


def _score_new_ids(tracks: list[dict]) -> dict[int, list[str]]:
    """Frame each new player track-id was first seen. Weighted by clip
    position — IDs born in the first 10% of the clip are mostly bootstrap and
    not interesting; later births tend to be ID swaps during crossings."""
    flagged: dict[int, list[str]] = defaultdict(list)
    seen: set[int] = set()
    n = max(1, len(tracks))
    for i, ft in enumerate(tracks):
        for tid in ft.get("player", {}).keys():
            if tid in seen:
                continue
            seen.add(tid)
            if i / n < 0.10:
                continue   # bootstrap region, ignore
            flagged[i].append(f"new_id_player{tid}")
    return flagged


# ---------------------------------------------------------------- aggregation
def aggregate_scores(tracks: list[dict]) -> list[dict]:
    """Compute per-frame total score and reasons. Returns a list sorted by
    descending score, with frames that scored 0 omitted (most frames)."""
    sigs = [
        (_score_ball_gaps(tracks),       W_BALL_GAP),
        (_score_ball_in_player(tracks),  W_BALL_IN_PLAYER),
        (_score_low_conf(tracks),        W_LOW_CONF),
        (_score_short_tracks(tracks),    W_SHORT_TRACK),
        (_score_count_jumps(tracks),     W_COUNT_JUMP),
        (_score_new_ids(tracks),         W_NEW_ID),
    ]

    by_frame: dict[int, dict] = {}
    for flagged, weight in sigs:
        for fi, reasons in flagged.items():
            slot = by_frame.setdefault(fi, {"frame": fi, "score": 0.0, "reasons": []})
            slot["score"] += weight * len(reasons)
            slot["reasons"].extend(reasons)

    out = sorted(by_frame.values(), key=lambda d: d["score"], reverse=True)
    return out


# ---------------------------------------------------------------- frame dump
def dump_frames(
    video_path: Path, frames: list[int], out_dir: Path, clip_stem: str
) -> dict[str, dict]:
    """Extract each requested frame index and save as JPEG. Returns a
    manifest dict suitable for JSON dump."""
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
        # Seek if we're far away — VideoCapture.set is slow but cheaper than
        # decoding hundreds of frames we're going to drop.
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
        fname = f"{clip_stem}_f{want:06d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame)
        manifest[fname] = {
            "clip": clip_stem,
            "frame_idx": want,
            "timestamp_s": want / fps,
        }
        target_idx += 1
    cap.release()
    return manifest


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stub", type=Path, help="Tracker stub (pickle) to inspect")
    p.add_argument("--video", type=Path, default=None,
                   help="Source clip — required to dump JPEG cutouts")
    p.add_argument("--out-dir", type=Path,
                   default=Path("frames_to_label/candidates"),
                   help="Where to write CSV / JSON / JPEGs")
    p.add_argument("--top", type=int, default=60,
                   help="Number of top-scoring frames to export as JPEGs")
    p.add_argument("--fps", type=float, default=25.0,
                   help="Fallback frame-rate for timestamp column when no "
                        "video is supplied")
    p.add_argument("--frame-offset", type=int, default=0,
                   help="Add this offset to every reported frame index when "
                        "writing CSV/JPEGs. Use this when the stub was "
                        "generated from a windowed run (--start-frame N): "
                        "pass --frame-offset N so the indices match the "
                        "ORIGINAL video, not the stub's local 0-based index.")
    args = p.parse_args()

    # Stub dispatch — handle both formats:
    #   * bare ``list[dict]`` — legacy Tracker.get_object_tracks output;
    #   * ``{"tracks": [...], "camera_movement": [...]}`` — the new
    #     ``--analysis-stub`` written by ``main.run_streaming``.
    with open(args.stub, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "tracks" in obj:
        tracks = obj["tracks"]
    elif isinstance(obj, list):
        tracks = obj
    else:
        raise SystemExit(
            f"Unrecognised stub format in {args.stub}: "
            f"type={type(obj).__name__}"
        )
    print(f"Loaded {len(tracks)} frames from {args.stub}")

    ranked = aggregate_scores(tracks)
    print(f"Flagged {len(ranked)} frames with at least one signal "
          f"({len(ranked) / max(1, len(tracks)) * 100:.1f}%)")
    if not ranked:
        print("No frames flagged — model looks healthy on this clip "
              "(or thresholds need loosening).")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clip_stem = args.video.stem if args.video else args.stub.stem

    # --- CSV (full ranked list, easy to filter in a spreadsheet) ----------
    csv_path = args.out_dir / f"{clip_stem}_candidates.csv"
    fps = args.fps
    if args.video is not None:
        import cv2
        cap = cv2.VideoCapture(str(args.video))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            cap.release()
    off = args.frame_offset
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "timestamp_s", "score", "reasons"])
        for row in ranked:
            global_fi = row["frame"] + off
            w.writerow([
                global_fi,
                f"{global_fi / fps:.2f}",
                f"{row['score']:.2f}",
                "; ".join(row["reasons"]),
            ])
    print(f"  wrote {csv_path}")

    # --- top-N JPEG cutouts (only if --video given) -----------------------
    top = ranked[: args.top]
    print("\nTop 10 candidate frames (global indices):")
    for row in top[:10]:
        print(f"  f={row['frame'] + off:>6}  score={row['score']:5.1f}  "
              f"reasons={row['reasons']}")

    if args.video is None:
        print("\n--video not supplied; skipping JPEG dump.")
        return

    # Add the offset before seeking — so we read the right frames from the
    # ORIGINAL video, not the stub-local position.
    manifest = dump_frames(
        args.video, [r["frame"] + off for r in top], args.out_dir, clip_stem
    )
    manifest_path = args.out_dir / f"{clip_stem}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {len(manifest)} JPEGs and {manifest_path}")


if __name__ == "__main__":
    main()
