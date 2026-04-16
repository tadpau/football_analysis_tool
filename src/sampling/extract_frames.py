"""Sample frames from source clips for annotation.

Strategy: evenly spaced sampling per clip — this avoids near-duplicate frames that
add labeling cost without teaching the model anything new. For 10 clips × ~120s at
~24 fps, one frame every ~3 seconds gives ~400 frames total, which is a sensible
starting budget for fine-tuning on top of a Roboflow-pretrained model.

Run:
    python -m src.sampling.extract_frames --every 72 --out data/raw_frames
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from src.utils.video_utils import iter_video_frames, get_video_info


def extract_from_clip(video_path: Path, out_dir: Path, every_n_frames: int) -> int:
    """Sample every Nth frame from one clip. Returns number of frames written."""
    info = get_video_info(video_path)
    stem = video_path.stem.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    expected = max(1, info["frame_count"] // every_n_frames)
    pbar = tqdm(total=expected, desc=stem[:30], unit="frame")
    for idx, frame in iter_video_frames(video_path):
        if idx % every_n_frames != 0:
            continue
        out_path = out_dir / f"{stem}_f{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written += 1
        pbar.update(1)
    pbar.close()
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("video_clips"),
        help="Directory containing source .mp4 clips",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw_frames"),
        help="Directory to write sampled JPEGs into",
    )
    p.add_argument(
        "--every",
        type=int,
        default=72,
        help="Sample every Nth frame (72 ≈ one frame per 3s at 24fps)",
    )
    args = p.parse_args()

    clips = sorted(args.clips_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"No .mp4 files found in {args.clips_dir}")

    total = 0
    for clip in clips:
        total += extract_from_clip(clip, args.out, args.every)
    print(f"\nDone. Wrote {total} frames to {args.out}")


if __name__ == "__main__":
    main()
