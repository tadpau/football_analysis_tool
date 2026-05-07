"""Split own_labels into train/val, zip for Colab upload.

Takes `datasets/own_labels/{images,labels}/` (flat folders produced by
`export_label_studio.py`) and reshapes them into the YOLO canonical layout:

    own_labels_split/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      classes.txt

Then zips the directory as `own_labels_split.zip` for one-click upload to
Colab.

Split strategy: deterministic by filename hash so re-runs give the same
train/val assignment (important if you retrain later and want a stable val
metric to compare against).

Usage:
    python scripts/prepare_own_labels_split.py
    python scripts/prepare_own_labels_split.py --val-frac 0.15
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_val(stem: str, val_frac: float) -> bool:
    """Deterministic val/train split by hashing the filename stem."""
    h = int(hashlib.md5(stem.encode("utf-8")).hexdigest(), 16)
    return (h % 1_000_000) / 1_000_000 < val_frac


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path,
                   default=REPO_ROOT / "datasets" / "own_labels",
                   help="Flat own_labels folder (from export_label_studio.py).")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "datasets" / "own_labels_split",
                   help="Where to build the split layout.")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Fraction of frames used for validation (default 0.2).")
    p.add_argument("--zip", dest="make_zip", action="store_true",
                   default=True, help="Also produce own_labels_split.zip.")
    p.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = p.parse_args()

    img_src = args.in_dir / "images"
    lbl_src = args.in_dir / "labels"
    if not img_src.is_dir() or not lbl_src.is_dir():
        sys.exit(f"Expected {img_src} and {lbl_src} to exist. Run export first.")

    # Clean output
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    n_train = n_val = n_skipped = 0
    for img in sorted(img_src.glob("*.jpg")):
        lbl = lbl_src / f"{img.stem}.txt"
        if not lbl.exists():
            n_skipped += 1
            continue
        bucket = "val" if _is_val(img.stem, args.val_frac) else "train"
        shutil.copy2(img, args.out_dir / "images" / bucket / img.name)
        shutil.copy2(lbl, args.out_dir / "labels" / bucket / lbl.name)
        if bucket == "val":
            n_val += 1
        else:
            n_train += 1

    # Copy classes.txt if present
    classes_src = args.in_dir / "classes.txt"
    if classes_src.exists():
        shutil.copy2(classes_src, args.out_dir / "classes.txt")

    print(f"Split: {n_train} train | {n_val} val | {n_skipped} skipped (no label)")

    if args.make_zip:
        zip_path = args.out_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in args.out_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(args.out_dir))
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"Zipped -> {zip_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
