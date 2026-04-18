"""Pipeline orchestrator.

Phases wired:
  4   detection + ByteTrack
  4b  team assignment (shirt color)
  5   ball interpolation + possession assignment

Phases pending (stubs raise NotImplementedError):
  6   camera movement compensation
  7   pitch homography + speed/distance in meters

Run:
    python main.py --input "video_clips/<clip>.mp4" \\
                   --output output_videos/out.mp4 \\
                   --model models/best.pt --mode custom \\
                   --max-frames 300 --imgsz 1280
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import read_video_frames, save_video, get_video_info
from src.trackers import Tracker, draw_annotations
from src.team_assigner import TeamAssigner
from src.ball import interpolate_ball_positions
from src.possession import compute_team_possession


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

    # --- Phase 5: ball interpolation --------------------------------------
    ball_series = [ft["ball"] for ft in tracks]
    ball_series = interpolate_ball_positions(ball_series)
    for ft, bf in zip(tracks, ball_series):
        ft["ball"] = bf

    # --- Phase 4b: team color assignment ----------------------------------
    team_assigner = TeamAssigner()
    team_assigner.assign_all(frames, tracks)
    team_colors = {k: v.tolist() for k, v in team_assigner.team_colors.items()}
    if not team_colors:
        print("  ⚠  Team bootstrap never fired (no frame had enough players).")

    # --- Phase 5b: possession --------------------------------------------
    per_frame_owner, team_share = compute_team_possession(tracks)
    print(
        f"Possession: team1 {team_share.get(1, 0)*100:.1f}% | "
        f"team2 {team_share.get(2, 0)*100:.1f}%"
    )

    # --- Phase 6 (pending): camera motion compensation --------------------
    # cam_motion = CameraMovementEstimator(frames[0]).get_camera_movement(frames, ...)

    # --- Phase 7 (pending): pitch homography + world-coord metrics --------
    # ViewTransformer().add_transformed_positions_to_tracks(tracks)
    # SpeedAndDistanceEstimator(frame_rate=info["fps"]).add(tracks)

    # --- Render --------------------------------------------------------------
    print("Drawing overlays ...")
    annotated = draw_annotations(
        frames,
        tracks,
        per_frame_owner=per_frame_owner,
        team_share=team_share,
        team_colors=team_colors,
    )

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
        default="models/best.pt",
        help="Path to weights, or a name ultralytics can auto-download (yolov8m.pt)",
    )
    p.add_argument("--mode", choices=["coco", "custom"], default="custom")
    p.add_argument("--stub", type=Path, default=None)
    p.add_argument("--use-stub", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=1280)
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
