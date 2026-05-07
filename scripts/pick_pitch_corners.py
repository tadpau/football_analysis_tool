"""Interactive pitch-corner picker.

Pops up frame 0 of a clip in an OpenCV window. Click 4 reference points in
this order:

    1. TL — top-left   pitch landmark (e.g. far-touchline / halfway intersection)
    2. TR — top-right  pitch landmark (e.g. far-touchline / penalty-area edge)
    3. BR — bottom-right landmark (near-touchline / penalty-area edge)
    4. BL — bottom-left  landmark (near-touchline / halfway intersection)

Then enter the matching real-world (m_x, m_y) coordinates of those 4 points
on a canonical 105×68 pitch (origin = bottom-left, +x along the long axis).

Result is saved to `calibrations/<clip_stem>.json`:

    {
      "image_corners": [[px,py], [px,py], [px,py], [px,py]],
      "world_corners": [[mx,my], [mx,my], [mx,my], [mx,my]]
    }

Usage:
    python scripts/pick_pitch_corners.py --input video_clips/clip_7.mp4
    python scripts/pick_pitch_corners.py --input video_clips/clip_7.mp4 \\
                                         --frame 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import read_video_frames     # noqa: E402


CORNER_LABELS = ["TL", "TR", "BR", "BL"]


def _pick_image_points(frame, window_title: str) -> list[list[float]]:
    points: list[list[float]] = []
    work = frame.copy()

    def on_click(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([float(x), float(y)])
            cv2.circle(work, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(
                work, CORNER_LABELS[len(points) - 1], (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            cv2.imshow(window_title, work)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, work)
    cv2.setMouseCallback(window_title, on_click)

    print("Click 4 points in order: TL, TR, BR, BL. Press 'q' to abort, "
          "'u' to undo last, ENTER to confirm.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit("Aborted.")
        if key == ord("u") and points:
            points.pop()
            work = frame.copy()
            for i, (px, py) in enumerate(points):
                cv2.circle(work, (int(px), int(py)), 6, (0, 255, 255), -1)
                cv2.putText(work, CORNER_LABELS[i], (int(px) + 8, int(py) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window_title, work)
        if key in (10, 13) and len(points) == 4:
            break
    cv2.destroyAllWindows()
    return points


def _ask_world_points() -> list[list[float]]:
    print("\nEnter the matching world (metric) coordinates on a 105×68 pitch.")
    print("Origin = bottom-left of the pitch, +x along the long axis.\n")
    pts: list[list[float]] = []
    for label in CORNER_LABELS:
        while True:
            raw = input(f"  {label} world (m_x m_y): ").strip()
            try:
                a, b = raw.replace(",", " ").split()
                pts.append([float(a), float(b)])
                break
            except ValueError:
                print("  -> please enter two numbers, e.g. '52.5 68'")
    return pts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0,
                   help="Frame index to use as calibration reference (default 0)")
    p.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "calibrations",
        help="Where to write <clip_stem>.json",
    )
    args = p.parse_args()

    frames = read_video_frames(args.input, max_frames=args.frame + 1,
                               start_frame=args.frame)
    if not frames:
        sys.exit(f"No frames read from {args.input}")
    frame = frames[-1]

    image_corners = _pick_image_points(frame, f"Pick 4 corners — {args.input.name}")
    world_corners = _ask_world_points()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.input.stem}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_corners": image_corners,
                "world_corners": world_corners,
                "reference_frame": args.frame,
            },
            f, indent=2,
        )
    print(f"\nSaved calibration -> {out}")


if __name__ == "__main__":
    main()
