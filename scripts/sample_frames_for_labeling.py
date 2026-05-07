"""Sample diverse frames from all clips in video_clips/ for Label Studio.

We want ~150 frames to annotate. Naive approach (grab every Nth frame from
each clip) wastes budget on near-duplicate frames when the camera is still.
Better strategy:

  1. Walk each clip with a stride of STRIDE_S seconds (default 2 s).
  2. For each candidate frame, compute a cheap perceptual hash (downscale to
     8x8 grayscale and threshold at the mean — classic dHash-ish).
  3. Keep the frame only if its hash differs from the previous *kept* frame's
     hash by ≥ HAMMING_MIN bits. This drops still-camera repeats.
  4. Budget is spread roughly evenly across clips: each clip contributes
     `PER_CLIP_MAX` frames at most, so one shaky clip can't hog the budget.

Output layout:
    frames_to_label/
      clip_1_f000120.jpg
      clip_1_f000300.jpg
      ...
      manifest.json     # maps filename -> {clip, frame_idx} for traceback

Usage:
    python scripts/sample_frames_for_labeling.py
    python scripts/sample_frames_for_labeling.py --per-clip 20 --stride-s 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


HAMMING_MIN = 10          # min bit-diff between consecutive kept frames
PHASH_SIZE = 8            # 8×8 → 64-bit hash


def _phash(frame: np.ndarray) -> int:
    """Cheap 64-bit perceptual hash — resize + threshold at mean."""
    small = cv2.resize(frame, (PHASH_SIZE, PHASH_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bits = (gray > gray.mean()).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sample_clip(
    clip_path: Path, stride_frames: int, max_kept: int, hamming_min: int
) -> list[tuple[int, np.ndarray]]:
    """Return a list of (frame_idx, frame) diverse samples from one clip."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"  [warn] could not open {clip_path}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kept: list[tuple[int, np.ndarray]] = []
    last_hash: int | None = None

    idx = 0
    while idx < total and len(kept) < max_kept:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        h = _phash(frame)
        if last_hash is None or _hamming(h, last_hash) >= hamming_min:
            kept.append((idx, frame))
            last_hash = h
        idx += stride_frames

    cap.release()
    return kept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir", type=Path,
        default=REPO_ROOT / "video_clips",
        help="Folder of .mp4 clips (default: video_clips/).",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "frames_to_label",
        help="Where to dump sampled JPEGs + manifest.json.",
    )
    p.add_argument("--per-clip", type=int, default=18,
                   help="Max frames kept per clip (default 18 → ~180 total for 10 clips).")
    p.add_argument("--stride-s", type=float, default=2.0,
                   help="Seconds between candidate frames (default 2.0).")
    p.add_argument("--hamming-min", type=int, default=HAMMING_MIN,
                   help="Min perceptual-hash bit difference to keep a frame (default 10).")
    args = p.parse_args()

    clips = sorted(args.input_dir.glob("*.mp4"))
    if not clips:
        sys.exit(f"No .mp4 files in {args.input_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    total_kept = 0

    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        stride_frames = max(1, int(round(args.stride_s * fps)))

        print(f"Sampling {clip.name} (stride={stride_frames} frames, budget={args.per_clip}) ...")
        kept = sample_clip(clip, stride_frames, args.per_clip, args.hamming_min)
        print(f"  kept {len(kept)} diverse frames")

        for frame_idx, frame in kept:
            name = f"{clip.stem}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(args.out_dir / name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest[name] = {"clip": clip.name, "frame_idx": frame_idx}
            total_kept += 1

    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {total_kept} frames saved to {args.out_dir}")
    print(f"Manifest written: {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
