"""Drop a sparsely-labeled class from a YOLO-format dataset zip.

When a class has too few labeled instances for reliable training (rule of
thumb: under ~20 frames), training on it produces a confused detector —
either it never predicts the class, or it hallucinates false positives
because the model never saw enough variety. Dropping the class is cleaner
than training a degraded one.

This script takes an input zip in the layout ``merge_yolo_zips.py`` produces:

    classes.txt
    images/<name>.jpg
    labels/<name>.txt           (YOLO format: <class_id> <cx> <cy> <w> <h>)

…and writes a new zip with:

  * the requested class removed from ``classes.txt``;
  * every line in every label file referring to that class deleted;
  * ALL remaining class IDs renumbered to stay contiguous (YOLO requires
    class IDs in [0, nc-1] with no gaps).

Empty label files (where every bbox referred to the dropped class) are
written as empty files — the corresponding image becomes a "background"
training sample, which is fine for YOLO.

Usage::

    python scripts/drop_class_from_zip.py models/own_labels_master.zip \
        --drop penalty_spot_left \
        --out models/own_labels_master_trim.zip
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def drop_class(in_zip: Path, drop_name: str, out_zip: Path) -> None:
    if not in_zip.exists():
        raise SystemExit(f"Input zip not found: {in_zip}")
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(in_zip) as z:
            z.extractall(td)

        # Find classes.txt (top-level, sometimes inside a subdir).
        candidates = list(td.rglob("classes.txt"))
        if not candidates:
            raise SystemExit(f"No classes.txt in {in_zip}")
        classes_path = candidates[0]
        classes = classes_path.read_text().strip().splitlines()
        if drop_name not in classes:
            raise SystemExit(
                f"Class '{drop_name}' not found in {classes_path}. "
                f"Available: {classes}"
            )
        drop_id = classes.index(drop_name)
        new_classes = [c for c in classes if c != drop_name]

        # Build the renumber map: every old ID > drop_id shifts down by 1.
        # Old drop_id maps to None (lines get filtered out).
        remap: dict[int, int | None] = {}
        for i in range(len(classes)):
            if i == drop_id:
                remap[i] = None
            elif i < drop_id:
                remap[i] = i
            else:
                remap[i] = i - 1
        print(f"Dropping class '{drop_name}' (was ID {drop_id})")
        print(f"  remap: {remap}")
        print(f"  new classes ({len(new_classes)}): {new_classes}")

        # Rewrite every label file.
        labels_root = None
        for cand in (td / "labels", *td.rglob("labels")):
            if cand.is_dir():
                labels_root = cand
                break
        if labels_root is None:
            raise SystemExit(f"No labels/ directory in {in_zip}")

        n_files = 0
        n_lines_kept = 0
        n_lines_dropped = 0
        for lbl in labels_root.rglob("*.txt"):
            text = lbl.read_text().strip()
            if not text:
                continue
            new_lines: list[str] = []
            for line in text.splitlines():
                parts = line.split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                new_cls = remap[old_cls]
                if new_cls is None:
                    n_lines_dropped += 1
                    continue
                new_lines.append(" ".join([str(new_cls)] + parts[1:]))
                n_lines_kept += 1
            # Write back — empty file is fine (background-only image).
            lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            n_files += 1

        # Rewrite classes.txt
        classes_path.write_text("\n".join(new_classes) + "\n")
        print(f"  rewrote {n_files} label files: kept {n_lines_kept} bbox "
              f"lines, dropped {n_lines_dropped}")

        # Re-zip with the same flat layout merge_yolo_zips.py produces.
        # Walk relative to the temp dir root so the archive is portable.
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            # classes.txt at the top
            z.write(classes_path, arcname="classes.txt")
            # images and labels
            for sub in ("images", "labels"):
                src = td / sub
                if not src.exists():
                    # Fall back to wherever they lived in the input zip
                    found = list(td.rglob(sub))
                    if found:
                        src = found[0]
                    else:
                        continue
                for f in sorted(src.iterdir()):
                    if f.is_file():
                        z.write(f, arcname=f"{sub}/{f.name}")
        print(f"  wrote -> {out_zip}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("zip", type=Path, help="Input YOLO-format zip to filter")
    p.add_argument("--drop", required=True,
                   help="Class name to remove (e.g. penalty_spot_left)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output zip path")
    args = p.parse_args()
    drop_class(args.zip, args.drop, args.out)


if __name__ == "__main__":
    main()
