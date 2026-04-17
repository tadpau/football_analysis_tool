"""Pipeline orchestrator.

Phase 4 wiring (detection + tracking + overlay).
Later phases (team colors, ball interpolation, possession, camera, pitch, speed)
are commented below and will be enabled as they land.

Run (COCO mode — no training needed, works today):
    python main.py --input "video_clips/<clip>.mp4" --output output_videos/out.mp4

Run (custom weights, after Colab training):
    python main.py --input "video_clips/<clip>.mp4" \
                   --output output_videos/out.mp4 \
                   --model models/best.pt --mode custom
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import read_video_frames, save_video, get_video_info
from src.trackers import Tracker, draw_annotations


def run(
    input_path: Path,
    output_path: Path,
    model_path: str,
    mode: str,
    stub_path: Path | None,
    read_from_stub: bool,
    max_frames: int | None,
    start_frame: int,
    imgsz: int,
) -> None:
    info = get_video_info(input_path)
    print(
        f"Loaded {input_path.name}: {info['frame_count']} frames "
        f"@ {info['fps']:.1f} fps ({info['width']}x{info['height']})"
    )

    print(f"Reading frames (max_frames={max_frames}, start_frame={start_frame}) ...")
    frames = read_video_frames(input_path, max_frames=max_frames, start_frame=start_frame)
    print(f"  loaded {len(frames)} frames")

    # --- Phase 4: detection + tracking -------------------------------------
    tracker = Tracker(model_path=model_path, mode=mode, imgsz=imgsz)
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=read_from_stub,
        stub_path=str(stub_path) if stub_path else None,
    )

    # --- Phase 6 (pending): camera movement --------------------------------
    # cam_motion = CameraMovementEstimator(frames[0]).get_camera_movement(frames, ...)

    # --- Phase 7 (pending): pitch homography -------------------------------
    # ViewTransformer().add_transformed_positions_to_tracks(tracks)

    # --- Phase 5 (pending): ball interpolation + possession ----------------
    # tracks = interpolate_ball(tracks); possession = PossessionAssigner().assign(tracks)

    # --- Phase 4b (pending): team color assignment -------------------------
    # TeamAssigner().assign(frames, tracks)

    # --- Phase 7b (pending): speed/distance in meters ----------------------
    # SpeedAndDistanceEstimator(frame_rate=info["fps"]).add(tracks)

    # --- Render --------------------------------------------------------------
    print("Drawing overlays ...")
    annotated = draw_annotations(frames, tracks)

    print(f"Writing {output_path} ...")
    save_video(annotated, output_path, fps=info["fps"])
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("output_videos/out.mp4"))
    p.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Path to weights, OR a name ultralytics can auto-download (yolov8m.pt)",
    )
    p.add_argument("--mode", choices=["coco", "custom"], default="coco")
    p.add_argument(
        "--stub",
        type=Path,
        default=None,
        help="Cache tracker output here; reuse with --use-stub next run",
    )
    p.add_argument("--use-stub", action="store_true")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames (dev affordance — CPU inference is slow)",
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference size. Lower = faster, worse at small objects (ball).",
    )
    args = p.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        mode=args.mode,
        stub_path=args.stub,
        read_from_stub=args.use_stub,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
