"""Merge multiple Label Studio YOLO-export zips into one training archive.

The retraining notebook expects a SINGLE zip of own labels. With the labels
spread across multiple Label Studio projects (one per mining round), each
retraining is at risk of forgetting whichever rounds didn't get uploaded —
exactly the catastrophic-forgetting bug we hit on best_v4.pt.

This script flattens N input zips into one canonical layout::

    own_labels_master.zip
    ├── classes.txt
    ├── images/                 (all .jpg side by side, no train/val split)
    │   ├── round3_VFA_KFM_f001310.jpg
    │   ├── round3_VFA_KFM_f001445.jpg
    │   ├── round5_hi_f000046.jpg
    │   └── ...
    └── labels/                 (matching .txt YOLO label files)
        ├── round3_VFA_KFM_f001310.txt
        └── ...

Cell 4 of ``notebooks/train_yolo_colab.ipynb`` already handles this flat
layout — it does a deterministic MD5-hash 80/20 split — so no notebook
changes are required.

Filename collisions: if two input zips share an image basename (rare, but
possible across rounds), the later input wins UNLESS ``--prefix`` is given.
With ``--prefix``, every file is renamed ``<zip_stem>_<original_name>``,
making collisions impossible.

Usage::

    # combine three rounds into one master zip
    python scripts/merge_yolo_zips.py \
        models/own_labels_round3.zip \
        models/own_labels_round4.zip \
        models/own_labels_round5.zip \
        --out models/own_labels_master.zip --prefix

After running, upload ``own_labels_master.zip`` to Colab and bump
``THIS_VERSION`` in the notebook. The growing dataset is the whole point.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _find_image_label_pairs(root: Path) -> list[tuple[Path, Path | None]]:
    """Find every image in ``root`` and its matching .txt label.

    Walks recursively because Label Studio's exports vary — sometimes images
    sit at top level, sometimes inside ``images/``, sometimes inside
    ``images/train/``. Labels mirror that layout. We pair by stem.
    """
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    pairs: list[tuple[Path, Path | None]] = []
    for img in images:
        # Find the label by stem, anywhere in the extracted tree. Prefer a
        # match in a 'labels' directory if multiple .txt with the same stem
        # exist (shouldn't happen, but be safe).
        candidates = list(root.rglob(f"{img.stem}.txt"))
        if not candidates:
            pairs.append((img, None))
            continue
        candidates.sort(
            key=lambda p: 0 if "label" in p.parent.name.lower() else 1
        )
        pairs.append((img, candidates[0]))
    return pairs


def merge(zip_paths: list[Path], out_zip: Path, prefix: bool) -> None:
    """Extract every input zip, deduplicate, write a single output zip."""
    if not zip_paths:
        raise SystemExit("Need at least one input zip.")
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as merged_td:
        merged = Path(merged_td)
        merged_imgs = merged / "images"
        merged_lbls = merged / "labels"
        merged_imgs.mkdir()
        merged_lbls.mkdir()

        seen_stems: set[str] = set()
        per_zip_kept: list[tuple[str, int]] = []
        per_zip_no_label: list[tuple[str, int]] = []
        classes_bytes: bytes | None = None

        for zp in zip_paths:
            if not zp.exists():
                raise SystemExit(f"Zip not found: {zp}")
            with tempfile.TemporaryDirectory() as ex_td:
                ex = Path(ex_td)
                with zipfile.ZipFile(zp) as z:
                    z.extractall(ex)
                # Cache classes.txt content while the temp dir is alive — by
                # the time we write the output zip, the inner temp dir will
                # be gone so a stored Path would dangle.
                if classes_bytes is None:
                    for c in ex.rglob("classes.txt"):
                        classes_bytes = c.read_bytes()
                        break

                kept = 0
                missing_label = 0
                for img, lbl in _find_image_label_pairs(ex):
                    stem = img.stem
                    if prefix:
                        new_stem = f"{zp.stem}_{stem}"
                    elif stem in seen_stems:
                        # Collision — overwrite with later zip's version.
                        # This matches "newest-round wins" semantics.
                        new_stem = stem
                    else:
                        new_stem = stem
                    seen_stems.add(new_stem)

                    new_img = merged_imgs / f"{new_stem}{img.suffix.lower()}"
                    shutil.copy2(img, new_img)
                    if lbl is None:
                        missing_label += 1
                        # Still keep the image — but a missing label means
                        # YOLO will treat the image as "all-background".
                        # That's almost never what we want for fine-tuning,
                        # so warn loudly.
                        continue
                    new_lbl = merged_lbls / f"{new_stem}.txt"
                    shutil.copy2(lbl, new_lbl)
                    kept += 1

                per_zip_kept.append((zp.name, kept))
                if missing_label:
                    per_zip_no_label.append((zp.name, missing_label))

        # Sanity: classes.txt is required by Cell 6's assert.
        if classes_bytes is not None:
            (merged / "classes.txt").write_bytes(classes_bytes)
        else:
            print(
                "WARNING: no classes.txt found in any input zip. "
                "The training notebook will likely fail Cell 6's class-order "
                "assert. Verify your Label Studio export settings."
            )

        # Write the output zip with a flat layout.
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            if (merged / "classes.txt").exists():
                z.write(merged / "classes.txt", arcname="classes.txt")
            for f in sorted(merged_imgs.iterdir()):
                z.write(f, arcname=f"images/{f.name}")
            for f in sorted(merged_lbls.iterdir()):
                z.write(f, arcname=f"labels/{f.name}")

        # Report.
        print("\nMerge summary")
        for name, n in per_zip_kept:
            print(f"  {name:40s}  {n:4d} labeled image(s)")
        if per_zip_no_label:
            print("\n  (the following images had no matching label .txt — "
                  "labels NOT propagated:)")
            for name, n in per_zip_no_label:
                print(f"  {name:40s}  {n:4d} unlabeled image(s)")
        total_labeled = sum(n for _, n in per_zip_kept)
        print(
            f"\n  total labeled in merged zip:   {total_labeled}"
            f"\n  unique stems after merge:      {len(seen_stems)}"
        )
        print(f"  wrote -> {out_zip}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("zips", type=Path, nargs="+",
                   help="One or more YOLO-export zips to merge")
    p.add_argument("--out", type=Path,
                   default=Path("models/own_labels_master.zip"),
                   help="Output path for the merged zip")
    p.add_argument(
        "--prefix", action="store_true",
        help="Prefix every filename with the source zip stem. Use this when "
             "two input zips might share basenames — guarantees no "
             "overwrites. Without --prefix, later zips overwrite earlier "
             "ones on stem collision.",
    )
    args = p.parse_args()
    merge(args.zips, args.out, args.prefix)


if __name__ == "__main__":
    main()
