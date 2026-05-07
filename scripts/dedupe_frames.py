"""Dedupe near-duplicate JPEGs in a frames-to-label directory.

After running the annotation miner across multiple windowed stubs (and/or the
diversity sampler), it's common to end up with two frames that differ by 1–2
real frames — the miner flagged frame 3 712 for "ball-in-player" and the
sampler picked 3 720 for "diverse coverage", but visually they're the same
scene with the camera barely moved. Labelling both wastes time without adding
training signal (they'd be near-identical examples to the model).

This script keeps the *higher-priority* frame from each near-duplicate cluster
and moves the rest into a ``_dropped/`` sibling directory (not deleted, so you
can recover anything if a heuristic mis-fired).

Priority rule
-------------
We don't have explicit priorities, but we can derive a useful proxy:

  * Frames produced by the miner are named with their original global frame
    index, e.g. ``VFA_KFM_f001310.jpg``.
  * Frames produced by the sampler use the same naming convention.

Within a near-duplicate cluster we keep the frame whose name comes first
alphabetically (lowest frame index) — that's the one a human is most likely
to have already opened up in Label Studio if they were going through the dir
in order, and it's a stable rule that produces deterministic results.

Comparison metric
-----------------
Cheap 64-bit perceptual hash (resize 8×8 → grayscale → threshold at mean →
flatten to 64 bits). Two frames are considered duplicates when their Hamming
distance is ≤ HAMMING_MAX. Default 6 — empirically tight enough that frames
showing the same play but a clear cut to a different camera angle survive,
loose enough to catch the 1–2-frame-apart cases.

Usage::

    python scripts/dedupe_frames.py frames_to_label/VFA_KFM
    python scripts/dedupe_frames.py frames_to_label/VFA_KFM --hamming 8 --dry-run
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


PHASH_SIZE = 8
HAMMING_MAX_DEFAULT = 6


def phash(img_path: Path) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read {img_path}")
    small = cv2.resize(img, (PHASH_SIZE, PHASH_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bits = (gray > gray.mean()).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path, help="Directory of JPEGs to dedupe")
    p.add_argument("--hamming", type=int, default=HAMMING_MAX_DEFAULT,
                   help="Max Hamming distance (in 64 bits) for two frames to "
                        "be considered duplicates. Lower = stricter.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be moved without touching files.")
    args = p.parse_args()

    img_dir: Path = args.dir
    if not img_dir.is_dir():
        raise SystemExit(f"{img_dir} is not a directory")

    jpegs = sorted(img_dir.glob("*.jpg"))
    if not jpegs:
        raise SystemExit(f"No .jpg files in {img_dir}")
    print(f"Hashing {len(jpegs)} JPEGs ...")

    hashes: list[tuple[Path, int]] = [(p, phash(p)) for p in jpegs]

    # Greedy clustering: walk in name order; for each frame, compare to every
    # already-kept frame. If it matches any, mark as duplicate of the older
    # one. O(n²) — fine for a few hundred frames; not for 100k.
    kept: list[tuple[Path, int]] = []
    drops: list[tuple[Path, Path]] = []   # (duplicate, kept_original)
    for path, h in hashes:
        match = next(
            ((kp, kh) for kp, kh in kept if hamming(kh, h) <= args.hamming),
            None,
        )
        if match is None:
            kept.append((path, h))
        else:
            drops.append((path, match[0]))

    print(f"  kept     {len(kept):4d}")
    print(f"  dropped  {len(drops):4d}  "
          f"({len(drops) / len(jpegs) * 100:.1f}% of input)")

    if not drops:
        return

    # Show the first few clusters so the user can sanity-check the threshold.
    print("\nFirst 8 dropped frames (and which one they collapse into):")
    for dup, original in drops[:8]:
        print(f"  drop  {dup.name}  ->  keep  {original.name}")

    if args.dry_run:
        print("\n--dry-run: no files moved.")
        return

    dropped_dir = img_dir / "_dropped"
    dropped_dir.mkdir(exist_ok=True)
    for dup, _ in drops:
        shutil.move(str(dup), str(dropped_dir / dup.name))
    print(f"\nMoved {len(drops)} duplicates to {dropped_dir}")


if __name__ == "__main__":
    main()
