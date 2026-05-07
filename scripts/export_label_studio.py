"""Pull annotations from Label Studio API → build a clean YOLO dataset.

Works around the Windows export bug where LS writes label files with URL-
encoded Windows paths in their names (`?d=frames_to_label%5C...`), which the
Windows filesystem rejects.

What it does:
  1. GET /api/projects/<id>/export?exportType=JSON  — plain JSON, no file-
     naming tricks. Always works on Windows.
  2. For each LS task, figure out which source frame it refers to by matching
     against `frames_to_label/` on disk (LS mangles the filename, we recover
     it from the basename).
  3. Convert each bbox annotation into a YOLO `class_id cx cy w h` line,
     normalised to the actual image dimensions.
  4. Copy the image + write the .txt label to `datasets/own_labels/{images,labels}/`.
  5. Write `classes.txt` so main.py / Colab training knows the class order.

Setup — get these values once from Label Studio:
  * API token:  Account & Settings -> Access Token  (or /user/account)
  * Project ID: open your project; the number in the URL is the ID.

Usage:
    python scripts/export_label_studio.py --token XXXXXX --project-id 1
    # defaults: --base-url http://localhost:8080
    #           --frames-dir frames_to_label
    #           --out-dir    datasets/own_labels

If you prefer not to pass the token on the command line, export it first:
    $env:LS_TOKEN = "XXXXXX"
    python scripts/export_label_studio.py --project-id 1
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import requests


def _imread_unicode(path: Path):
    """cv2.imread equivalent that works with Unicode paths on Windows.

    cv2.imread on Windows uses the ANSI Win32 API and silently returns None
    when the path has non-ASCII characters (common with Lithuanian /
    Polish / Cyrillic filenames from screen recordings). Python's open() uses
    the Unicode Win32 API and has no such issue, so we read bytes ourselves
    and decode with cv2.imdecode.
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _ascii_safe(name: str) -> str:
    """Strip non-ASCII chars + whitespace from a filename stem for YOLO safety."""
    import re
    # Replace non-alphanumeric (keeping _ and -) with _
    out = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "frame"


def _imwrite_unicode(path: Path, img, quality: int = 92) -> bool:
    """cv2.imwrite equivalent that works with Unicode output paths."""
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


def _maybe_exchange_refresh_for_access(token: str, base_url: str) -> str:
    """If `token` is a JWT refresh token, POST it to /api/token/refresh/ to get
    an access token. Otherwise return `token` unchanged.

    LS's Personal Access Token setup gives you a refresh token, but API calls
    need an access token. The UI sometimes shows the refresh token as 'Access
    Token' which trips everyone up.
    """
    # JWTs have three dot-separated base64-url segments.
    parts = token.split(".")
    if len(parts) != 3:
        return token     # not a JWT — legacy token, use as-is
    try:
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)   # pad for decode
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return token
    if payload.get("token_type") != "refresh":
        return token     # already an access token

    refresh_url = f"{base_url.rstrip('/')}/api/token/refresh/"
    print(f"  token is a refresh token; exchanging at {refresh_url} ...")
    r = requests.post(refresh_url, json={"refresh": token}, timeout=30)
    if r.status_code != 200:
        sys.exit(
            f"Refresh-token exchange failed ({r.status_code}): {r.text}\n"
            "Tip: in Label Studio -> Organization -> enable Legacy API Tokens,\n"
            "then copy the resulting legacy token (no expiry, simpler auth)."
        )
    access = r.json().get("access")
    if not access:
        sys.exit(f"Refresh-token response had no 'access' field: {r.text}")
    print("  got access token; continuing ...")
    return access

REPO_ROOT = Path(__file__).resolve().parents[1]

# Class taxonomy. Two presets supported — pick via --classes when you call the
# script, or rely on auto-detection (default) which looks at what label names
# appear in the export and picks the matching preset.
#
#   "v3"          : the 4-class Roboflow-compatible order used through best_v3.pt
#   "v_landmarks" : the 8-class extension with Tier-1 pitch landmarks, in
#                   alphabetical YOLO order (matches ClassMap.custom_landmarks).
CLASS_PRESETS: dict[str, list[str]] = {
    "v3": ["ball", "goalkeeper", "player", "referee"],
    "v_landmarks": [
        "ball", "center_spot", "corner_flag", "goalkeeper",
        "penalty_spot_left", "penalty_spot_right", "player", "referee",
    ],
}
DEFAULT_CLASSES = CLASS_PRESETS["v3"]


def _find_source_image(
    frames_dir: Path,
    ls_filename: str,
    _basename_index: dict[str, Path] | None = None,
) -> Path | None:
    """Given the (possibly URL-encoded, possibly prefixed) filename LS stored,
    locate the original frame anywhere under ``frames_dir``.

    Walks subdirectories — round-N candidates are conventionally stored in
    ``frames_to_label/<clip>_round_N/`` rather than at the top level, so a
    flat ``frames_dir / basename`` lookup misses every round's subfolder.

    For speed, ``main`` builds an index of basename → Path once and passes
    it as ``_basename_index``; without it we fall back to ``rglob``.
    """
    clean = unquote(ls_filename)
    if "?" in clean:
        clean = clean.split("?", 1)[-1]
    if "=" in clean:
        clean = clean.split("=", 1)[-1]
    clean = clean.replace("\\", "/").split("/")[-1]
    stem_variants = [clean]
    if " - Copy" in clean:
        stem_variants.append(clean.replace(" - Copy", ""))

    for name in stem_variants:
        # 1. Top-level (legacy convention).
        p = frames_dir / name
        if p.exists():
            return p
        # 2. Indexed lookup (pre-built; instant).
        if _basename_index is not None and name in _basename_index:
            return _basename_index[name]
        # 3. Recursive fallback — slow if frames_dir has thousands of files
        #    and no index was built. Use only as a last resort.
        if _basename_index is None:
            matches = list(frames_dir.rglob(name))
            if matches:
                return matches[0]
    return None


def _convert_task(
    task: dict, frames_dir: Path, out_img_dir: Path, out_lbl_dir: Path,
    class_to_idx: dict[str, int],
    basename_index: dict[str, Path] | None = None,
) -> tuple[bool, str]:
    """Convert one LS task to image + YOLO .txt. Returns (ok, reason)."""
    data = task.get("data", {})
    ls_filename = data.get("image") or data.get("file") or ""
    if not ls_filename:
        return False, "no image field in task"

    src = _find_source_image(frames_dir, ls_filename, basename_index)
    if src is None:
        return False, f"source frame not found for {ls_filename!r}"

    img = _imread_unicode(src)
    if img is None:
        return False, f"cv2 failed to read {src}"
    H, W = img.shape[:2]

    annotations = task.get("annotations", [])
    if not annotations:
        return False, "no annotations"
    # Label Studio can have multiple annotation passes; take the most recent
    # *non-empty* one.
    last = None
    for a in annotations:
        if a.get("result"):
            last = a
    if last is None:
        return False, "all annotations empty"

    lines: list[str] = []
    for r in last["result"]:
        if r.get("type") != "rectanglelabels":
            continue
        val = r.get("value", {})
        labels = val.get("rectanglelabels", [])
        if not labels:
            continue
        cls = labels[0]
        if cls not in class_to_idx:
            # Unknown label — skip, don't error out.
            continue

        # LS stores bbox as percentages of ORIGINAL_WIDTH/HEIGHT on the result,
        # not of the raw image. Use those if present; else fall back to image.
        ow = r.get("original_width", W)
        oh = r.get("original_height", H)
        x_pct = val["x"];  y_pct = val["y"]
        w_pct = val["width"];  h_pct = val["height"]

        # x/y in pixels, top-left of bbox
        x_px = x_pct / 100.0 * ow
        y_px = y_pct / 100.0 * oh
        w_px = w_pct / 100.0 * ow
        h_px = h_pct / 100.0 * oh

        # YOLO: (cx, cy, w, h) normalised by ORIGINAL image size (should match
        # the image on disk — ow=W, oh=H — but we normalise by whichever LS
        # gave us so the math is self-consistent).
        cx = (x_px + w_px / 2.0) / ow
        cy = (y_px + h_px / 2.0) / oh
        wn = w_px / ow
        hn = h_px / oh

        # Clamp to [0, 1] — LS occasionally overshoots by 0.1%.
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        wn = min(max(wn, 0.0), 1.0)
        hn = min(max(hn, 0.0), 1.0)

        lines.append(f"{class_to_idx[cls]} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

    if not lines:
        return False, "no rectangle labels found"

    # Normalise stem to ASCII-safe — YOLO training toolchain is happier that
    # way, and the manifest keeps the source-video traceback if you ever need it.
    stem = _ascii_safe(src.stem)
    shutil.copy2(src, out_img_dir / f"{stem}.jpg")
    (out_lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8080",
                   help="Label Studio base URL (default http://localhost:8080)")
    p.add_argument("--token", default=os.environ.get("LS_TOKEN"),
                   help="LS API token (or set $env:LS_TOKEN).")
    p.add_argument("--project-id", type=int, required=True,
                   help="LS project ID (number in project URL).")
    p.add_argument("--frames-dir", type=Path,
                   default=REPO_ROOT / "frames_to_label",
                   help="Folder with the original .jpg frames you imported into LS.")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "datasets" / "own_labels",
                   help="Where to build the YOLO-format dataset.")
    p.add_argument(
        "--classes", choices=("auto", *CLASS_PRESETS.keys()), default="auto",
        help=("Class taxonomy to write into classes.txt. "
              "'auto' (default) inspects the export and picks 'v_landmarks' "
              "if any landmark labels are present, else 'v3' (4-class). "
              "Pass an explicit preset to override. Both presets follow "
              "alphabetical YOLO order to match the training notebook's "
              "Cell-6 assertion."),
    )
    args = p.parse_args()

    if not args.token:
        sys.exit("Missing --token (or set $env:LS_TOKEN).")

    token = _maybe_exchange_refresh_for_access(args.token, args.base_url)

    url = f"{args.base_url.rstrip('/')}/api/projects/{args.project_id}/export"
    print(f"GET {url}?exportType=JSON ...")

    # Label Studio has two auth schemes depending on version/token type:
    #   - Legacy Token  -> "Authorization: Token <token>"
    #   - Personal Access Token (JWT, newer default) -> "Authorization: Bearer <token>"
    # Try "Token" first (matches the UI's own docs for Legacy Token), then
    # fall back to "Bearer" before giving up.
    last_err: Exception | None = None
    tasks = None
    for scheme in ("Token", "Bearer"):
        try:
            r = requests.get(
                url,
                params={"exportType": "JSON"},
                headers={"Authorization": f"{scheme} {token}"},
                timeout=120,
            )
            if r.status_code == 401:
                print(f"  auth scheme '{scheme}' rejected (401) — trying next ...")
                continue
            r.raise_for_status()
            tasks = r.json()
            print(f"  auth scheme '{scheme}' accepted, received {len(tasks)} tasks")
            break
        except requests.exceptions.HTTPError as e:
            last_err = e
            if r.status_code != 401:
                raise
    if tasks is None:
        sys.exit(
            "Both 'Token' and 'Bearer' auth schemes returned 401.\n"
            "Check in Label Studio:\n"
            "  * Account & Settings -> Access Token: copy the CURRENT token.\n"
            "  * If you see both 'Legacy Token' and 'Personal Access Token' options,\n"
            "    prefer Legacy Token — it doesn't expire.\n"
            f"Last error: {last_err}"
        )

    out_img_dir = args.out_dir / "images"
    out_lbl_dir = args.out_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Build a basename → Path index over the entire frames_dir so each task
    # lookup is O(1) and resilient to round-N subfolders. Cheap (~10 ms even
    # for thousands of files) compared to per-task rglob.
    basename_index: dict[str, Path] = {}
    if args.frames_dir.exists():
        for p in args.frames_dir.rglob("*.jpg"):
            # First match wins — if you ever have duplicate basenames across
            # round folders, the dedupe step should have caught them, but
            # we won't silently overwrite a different copy.
            basename_index.setdefault(p.name, p)
        for p in args.frames_dir.rglob("*.jpeg"):
            basename_index.setdefault(p.name, p)
        for p in args.frames_dir.rglob("*.png"):
            basename_index.setdefault(p.name, p)
    print(f"Indexed {len(basename_index)} source frames under {args.frames_dir}")

    # Pick the class taxonomy. In 'auto' mode, scan all annotations once for
    # any landmark label name; if found, switch to the 8-class preset.
    if args.classes == "auto":
        landmark_names = set(CLASS_PRESETS["v_landmarks"]) - set(CLASS_PRESETS["v3"])
        seen_landmark = False
        for t in tasks:
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    for lbl in r.get("value", {}).get("rectanglelabels", []):
                        if lbl in landmark_names:
                            seen_landmark = True
                            break
                    if seen_landmark:
                        break
                if seen_landmark:
                    break
            if seen_landmark:
                break
        chosen_preset = "v_landmarks" if seen_landmark else "v3"
        print(f"  auto-detected class preset: {chosen_preset}")
    else:
        chosen_preset = args.classes
    classes = CLASS_PRESETS[chosen_preset]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    ok = 0
    skipped: list[str] = []
    for t in tasks:
        success, reason = _convert_task(
            t, args.frames_dir, out_img_dir, out_lbl_dir,
            class_to_idx=class_to_idx, basename_index=basename_index,
        )
        if success:
            ok += 1
        else:
            skipped.append(reason)

    (args.out_dir / "classes.txt").write_text(
        "\n".join(classes) + "\n", encoding="utf-8"
    )

    print(f"\nDone. {ok} labeled frames exported to {args.out_dir}")
    print(f"  images: {out_img_dir}")
    print(f"  labels: {out_lbl_dir}")
    if skipped:
        print(f"\nSkipped {len(skipped)} tasks. First few reasons:")
        for r in skipped[:5]:
            print(f"  - {r}")


if __name__ == "__main__":
    main()
