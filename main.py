"""Pipeline orchestrator.

Phases wired:
  4   detection + ByteTrack
  4b  team assignment (shirt color)
  5a  ball outlier filter (drop white-shoe / jersey false positives)
  5b  ball interpolation across remaining gaps
  5c  ball possession (per-frame owner + per-team %)
  6   camera-motion compensation (Lucas-Kanade optical flow)
  7a  pitch homography (image px → metric pitch coords)
  7b  per-player distance + speed (m, km/h)

Phase 7 is gated on a per-clip calibration JSON produced by
`scripts/pick_pitch_corners.py`. If absent, phases 7a/7b are skipped and the
overlay shows only ball/possession + camera-shift HUD.

Run:
    python main.py --input "video_clips/clip_7.mp4" \\
                   --output output_videos/clip_7_full.mp4 \\
                   --model models/best.pt --mode custom \\
                   --max-frames 300 --imgsz 1280 \\
                   --calibration calibrations/clip_7.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from src.utils import (
    read_video_frames,
    save_video,
    get_video_info,
    iter_video_chunks,
    iter_video_frames,
    StreamingVideoWriter,
)
from src.trackers import (
    Tracker, draw_annotations, draw_one_frame, stitch_tracks_team_aware,
)
from src.team_assigner import TeamAssigner
from src.ball import (
    interpolate_ball_positions,
    filter_ball_outliers,
    track_ball_with_motion,
)
from src.possession import compute_team_possession, compute_rolling_team_share
from src.camera_movement import CameraMovementEstimator
from src.pitch import (
    ViewTransformer,
    LandmarkCalibrator,
    apply_auto_calibration_to_tracks,
)
from src.speed_distance import SpeedAndDistanceEstimator
from src.team_stats import (
    compute_dominant_teams,
    compute_team_distances,
    compute_team_heatmaps,
    save_team_heatmap_png,
)


# Default chunk size for the streaming pipeline. 600 frames at 1080p ≈ 3.6 GB
# of resident frame data per chunk — fits well under 16 GB while keeping
# detection batches large enough that YOLO startup overhead is amortised.
STREAM_CHUNK_FRAMES = 600

# Rolling possession HUD window — 30 s of game time, converted to frames at
# the clip's native fps in the render loop. 30 s is short enough to react to
# changes of phase (a long defensive period followed by a counter shows up
# in the HUD before the moment is over) and long enough that one missed
# detection doesn't yank the percentages.
ROLLING_HUD_SECONDS = 30.0


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
    calibration_path: Path | None,
    cam_stub_path: Path | None,
    heatmap_dir: Path | None,
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
    # frame_rate is passed through to ByteTrack + the stitcher so the default
    # "3 s buffer" translates to a sensible number of frames for this clip.
    tracker = Tracker(
        model_path=model_path, mode=mode, imgsz=imgsz,
        frame_rate=info["fps"],
    )
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=read_from_stub,
        stub_path=str(stub_path) if stub_path else None,
    )

    # --- Phase 5a: drop ball misclassifications inside a player bbox ------
    dropped = filter_ball_outliers(tracks)
    print(f"  filtered out {dropped} suspect ball detections")

    # --- Phase 5b: motion-aware ball tracking -----------------------------
    # Velocity-based extrapolation + trajectory gating + player-proximity
    # gate + carrier anchoring. The carrier anchor pins the ball to a
    # specific tracked person while they're holding/dribbling, which fixes
    # the throw-in / occluded-by-carrier cases (ball stays on the player
    # instead of drifting into stale ballistic predictions).
    ball_series = [ft["ball"] for ft in tracks]
    persons_per_frame = [
        [((cls, tid), info["bbox"])
         for cls in ("player", "goalkeeper", "referee")
         for tid, info in ft.get(cls, {}).items()]
        for ft in tracks
    ]
    ball_series, ball_stats = track_ball_with_motion(
        ball_series,
        persons_per_frame=persons_per_frame,
    )
    for ft, bf in zip(tracks, ball_series):
        ft["ball"] = bf
    print(
        f"  ball: {ball_stats['accepted']} accepted, "
        f"{ball_stats['rejected']} off-trajectory rejected, "
        f"{ball_stats['rejected_far_from_person']} far-from-person rejected, "
        f"{ball_stats['extrapolated']} extrapolated, "
        f"{ball_stats['carrier_anchored']} carrier-anchored, "
        f"{ball_stats['lost']} gave up"
    )

    # --- Phase 4b: team color assignment ----------------------------------
    team_assigner = TeamAssigner()
    team_assigner.assign_all(frames, tracks)
    team_colors = {k: v.tolist() for k, v in team_assigner.team_colors.items()}
    if not team_colors:
        print("  Team bootstrap never fired (no frame had enough players).")

    # --- Phase 4c: team-aware second-pass stitcher ------------------------
    # The baseline stitcher inside Tracker (gap≤150f, dist≤200px) is tuned
    # conservatively because without any team signal, merging across teams
    # would be worse than leaving a fragment unstitched. Now that team
    # classification is stable per-detection, we run a second pass with
    # much wider reach (300f / 350px) but HARD-GATED on team agreement —
    # never merges across teams, never tries to stitch a ref-like
    # unassigned track. This catches the ID flicker that happens during
    # crossings and sharp turns.
    if team_colors:
        stats = stitch_tracks_team_aware(tracks)
        print(
            f"  team-aware stitcher: {stats['merges']} extra merges "
            f"({stats['skipped_team_mismatch']} refused for team mismatch, "
            f"{stats['skipped_no_team']} skipped — unassigned tracks)"
        )

    # --- Phase 5c: possession --------------------------------------------
    per_frame_owner, team_share = compute_team_possession(tracks)
    print(
        f"Possession: team1 {team_share.get(1, 0)*100:.1f}% | "
        f"team2 {team_share.get(2, 0)*100:.1f}%"
    )

    # --- Phase 6: camera motion compensation ------------------------------
    print("Estimating camera motion ...")
    cam_est = CameraMovementEstimator(frames[0])
    camera_movement = cam_est.get_camera_movement(
        frames,
        read_from_stub=read_from_stub,
        stub_path=str(cam_stub_path) if cam_stub_path else None,
    )
    cam_est.add_adjusted_positions_to_tracks(tracks, camera_movement)
    total_dx = sum(d[0] for d in camera_movement)
    total_dy = sum(d[1] for d in camera_movement)
    print(f"  cumulative camera drift over clip: dx={total_dx:+.1f} px  dy={total_dy:+.1f} px")

    # --- Phase 7: homography + speed/distance -----------------------------
    if calibration_path and calibration_path.exists():
        with open(calibration_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        view = ViewTransformer(
            image_corners=calib["image_corners"],
            world_corners=calib["world_corners"],
        )
        view.add_transformed_positions_to_tracks(tracks)

        sd = SpeedAndDistanceEstimator(frame_rate=info["fps"])
        sd.add_speed_and_distance_to_tracks(tracks)

        # --- Phase 8: team-level aggregates ------------------------------
        # Distance + heatmaps grouped by dominant-team-per-track. These are
        # the metrics that survive ID flicker (a swap within a team doesn't
        # change the team total) so they're the most trustworthy stats we
        # can show on academy footage.
        dom_teams = compute_dominant_teams(tracks)
        team_km = compute_team_distances(sd.summary, dom_teams)
        print(
            f"Team distance:  team1 {team_km.get(1, 0):5.2f} km  "
            f"|  team2 {team_km.get(2, 0):5.2f} km"
        )

        # Per-team top contributors — easier to read than one global top-5
        # that often skews toward whichever team had longer-lived tracks.
        if sd.summary:
            for team_id in (1, 2):
                team_players = [
                    (tid, s) for tid, s in sd.summary.items()
                    if dom_teams.get(tid) == team_id
                ]
                team_players.sort(
                    key=lambda kv: kv[1]["total_distance_m"], reverse=True,
                )
                if not team_players:
                    continue
                print(f"  Team {team_id} top distance covered:")
                for tid, s in team_players[:5]:
                    print(
                        f"    player {tid:>3}:  {s['total_distance_m']:6.1f} m   "
                        f"longest sprint: {s['longest_sprint_m']:5.1f} m"
                    )

        if heatmap_dir is not None:
            heatmaps = compute_team_heatmaps(tracks, dom_teams)
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            stem = output_path.stem
            for team_id, hm in heatmaps.items():
                color = team_colors.get(team_id)
                out_png = heatmap_dir / f"{stem}_team{team_id}_heatmap.png"
                save_team_heatmap_png(
                    hm,
                    out_png,
                    title=f"Team {team_id} — occupancy heatmap",
                    team_color_bgr=tuple(color) if color else None,
                )
                print(f"  wrote {out_png}")
    else:
        if calibration_path:
            print(f"  no calibration at {calibration_path} — skipping homography/speed.")
        else:
            print("  no --calibration provided — skipping homography/speed.")

    # --- Render --------------------------------------------------------------
    # Rolling possession share replaces the static aggregate in the HUD —
    # the static number was constant across the clip, which made the HUD
    # look frozen. The rolling number reacts to phases of play.
    print("Drawing overlays ...")
    window_frames = max(1, int(round(ROLLING_HUD_SECONDS * info["fps"])))
    rolling_share = compute_rolling_team_share(
        tracks, per_frame_owner, window_frames=window_frames,
    )
    annotated = draw_annotations(
        frames,
        tracks,
        per_frame_owner=per_frame_owner,
        team_share=rolling_share,
        team_colors=team_colors,
        camera_movement=camera_movement,
        hud_label=f"last {int(ROLLING_HUD_SECONDS)}s",
    )

    print(f"Writing {output_path} ...")
    save_video(annotated, output_path, fps=info["fps"])
    print("Done.")


def run_streaming(
    input_path: Path,
    output_path: Path,
    model_path: str,
    mode: str,
    max_frames: int | None,
    start_frame: int,
    imgsz: int,
    calibration_path: Path | None,
    heatmap_dir: Path | None,
    chunk_size: int = STREAM_CHUNK_FRAMES,
    analysis_stub_path: Path | None = None,
    use_analysis_stub: bool = False,
    auto_calibrate: bool = False,
) -> None:
    """Memory-bounded two-pass pipeline for full-length clips.

    Pass 1 (analyse): stream frames in chunks of ``chunk_size``; per chunk run
        detection+tracking, accumulate ByteTrack state, run optical-flow camera
        estimation statefully, and stash a kit-colour sample on each player
        detection. After the last chunk, finalise the tracker (stitcher) and
        run all the track-only post-processing (ball, possession, team-aware
        stitch, homography, speed/distance, heatmaps).

    Pass 2 (render): re-stream frames with :func:`iter_video_frames` and write
        the annotated MP4 with :class:`StreamingVideoWriter`. Frames are never
        held more than two at a time (raw + annotated copy).

    Memory: bounded by ``chunk_size`` × frame-size during pass 1 and by a
    single frame during pass 2. The track list itself is small (~few MB even
    for an hour-long clip).
    """
    info = get_video_info(input_path)
    print(
        f"Loaded {input_path.name}: {info['frame_count']} frames "
        f"@ {info['fps']:.1f} fps ({info['width']}x{info['height']})"
    )
    print(
        f"Streaming pipeline: chunk_size={chunk_size}, "
        f"start_frame={start_frame}, max_frames={max_frames}"
    )

    # ---- Pass 1: detect + track + cam + colour samples (chunked) ----------
    # If a cached analysis stub is available and requested, skip pass 1 entirely.
    # The stub holds (tracks-with-color-samples, camera_movement) — everything
    # downstream re-derives from those, so calibration / heatmap / HUD tweaks
    # become near-instant iteration loops instead of multi-hour reruns.
    tracks: list[dict] = []
    camera_movement: list[list[float]] = []
    team_assigner = TeamAssigner()

    if (
        use_analysis_stub
        and analysis_stub_path is not None
        and analysis_stub_path.exists()
    ):
        with open(analysis_stub_path, "rb") as f:
            cache = pickle.load(f)
        tracks = cache["tracks"]
        camera_movement = cache["camera_movement"]
        print(
            f"Loaded analysis stub from {analysis_stub_path} "
            f"({len(tracks)} frames) — skipping pass 1."
        )
    else:
        tracker = Tracker(
            model_path=model_path, mode=mode, imgsz=imgsz,
            frame_rate=info["fps"],
        )
        cam_est: CameraMovementEstimator | None = None
        n_seen = 0

        for chunk in iter_video_chunks(
            input_path,
            chunk_size=chunk_size,
            start_frame=start_frame,
            max_frames=max_frames,
        ):
            # Lazy first-frame init for cam_est (it needs a frame to size the mask).
            if cam_est is None:
                cam_est = CameraMovementEstimator(chunk[0])

            chunk_tracks = tracker.feed_chunk(chunk)
            chunk_cam = cam_est.feed_chunk(chunk)

            # Colour sampling — frame-bound, must happen before frames are dropped.
            for frame, ft in zip(chunk, chunk_tracks):
                team_assigner.attach_color_samples(frame, ft)

            tracks.extend(chunk_tracks)
            camera_movement.extend(chunk_cam)
            n_seen += len(chunk)
            print(f"  pass 1: processed {n_seen} frames")

            # Drop chunk reference so Python frees the frames before next iter.
            del chunk, chunk_tracks, chunk_cam

        if not tracks:
            print("No frames read — aborting.")
            return

        # Stitching needs the full timeline.
        tracker.finalize_tracks(tracks)

        # Cache pass-1 results so subsequent runs (e.g. tweaking calibration,
        # HUD, heatmap params) skip the expensive YOLO + optical-flow stage.
        if analysis_stub_path is not None:
            analysis_stub_path.parent.mkdir(parents=True, exist_ok=True)
            with open(analysis_stub_path, "wb") as f:
                pickle.dump(
                    {"tracks": tracks, "camera_movement": camera_movement},
                    f,
                )
            print(f"Cached pass-1 analysis -> {analysis_stub_path}")

    # ---- Track-only post-processing ----------------------------------------
    dropped = filter_ball_outliers(tracks)
    print(f"  filtered out {dropped} suspect ball detections")

    ball_series = [ft["ball"] for ft in tracks]
    persons_per_frame = [
        [((cls, tid), info["bbox"])
         for cls in ("player", "goalkeeper", "referee")
         for tid, info in ft.get(cls, {}).items()]
        for ft in tracks
    ]
    ball_series, ball_stats = track_ball_with_motion(
        ball_series, persons_per_frame=persons_per_frame,
    )
    for ft, bf in zip(tracks, ball_series):
        ft["ball"] = bf
    print(
        f"  ball: {ball_stats['accepted']} accepted, "
        f"{ball_stats['rejected']} off-trajectory rejected, "
        f"{ball_stats['rejected_far_from_person']} far-from-person rejected, "
        f"{ball_stats['extrapolated']} extrapolated, "
        f"{ball_stats['carrier_anchored']} carrier-anchored, "
        f"{ball_stats['lost']} gave up"
    )

    # Team assignment from cached samples — no frames needed.
    team_assigner.assign_all_from_samples(tracks)
    team_colors = {k: v.tolist() for k, v in team_assigner.team_colors.items()}
    if not team_colors:
        print("  Team bootstrap never fired (no frame had enough players).")

    if team_colors:
        stats = stitch_tracks_team_aware(tracks)
        print(
            f"  team-aware stitcher: {stats['merges']} extra merges "
            f"({stats['skipped_team_mismatch']} refused for team mismatch, "
            f"{stats['skipped_no_team']} skipped — unassigned tracks)"
        )

    per_frame_owner, team_share = compute_team_possession(tracks)
    print(
        f"Possession: team1 {team_share.get(1, 0)*100:.1f}% | "
        f"team2 {team_share.get(2, 0)*100:.1f}%"
    )

    # add_adjusted_positions_to_tracks is a @staticmethod, so call on the
    # class directly — avoids depending on a `cam_est` instance that doesn't
    # exist on the stub-loaded path.
    CameraMovementEstimator.add_adjusted_positions_to_tracks(
        tracks, camera_movement
    )
    total_dx = sum(d[0] for d in camera_movement)
    total_dy = sum(d[1] for d in camera_movement)
    print(
        f"  cumulative camera drift over clip: "
        f"dx={total_dx:+.1f} px  dy={total_dy:+.1f} px"
    )

    # --- Pitch homography ---------------------------------------------------
    # Two paths: per-frame auto-calibration from detected landmarks (preferred
    # when the model knows them), or the legacy static --calibration JSON.
    # Auto-cal is robust to camera pan/zoom and obviates the manual picker.
    homography_done = False
    if auto_calibrate:
        # If a manual calibration JSON is also provided, use it as a
        # bootstrap H for corner-flag disambiguation. The manual H doesn't
        # have to be perfect — it's only used to *identify* which detected
        # corner is which world corner, then per-frame H is fit fresh from
        # the matched landmarks. Side-of-pitch cameras need this; the
        # naive image-quadrant heuristic produces broken matches.
        bootstrap_H = None
        if calibration_path and calibration_path.exists():
            with open(calibration_path, "r", encoding="utf-8") as f:
                calib = json.load(f)
            bootstrap_H = cv2.getPerspectiveTransform(
                np.asarray(calib["image_corners"], dtype=np.float32),
                np.asarray(calib["world_corners"], dtype=np.float32),
            ).astype(np.float64)
            print(f"  bootstrap H loaded from {calibration_path} for corner disambiguation.")

        cal = LandmarkCalibrator(
            frame_width=info["width"],
            frame_height=info["height"],
            bootstrap_H=bootstrap_H,
        )
        cal_stats = apply_auto_calibration_to_tracks(tracks, cal)
        print(
            f"Auto-calibrate: fitted {cal_stats['frames_fit']}/"
            f"{cal_stats['frames_seen']} frames "
            f"({cal_stats['fit_rate']*100:.1f}%), "
            f"{cal_stats['frames_used_fallback']} used recent fallback H, "
            f"{cal_stats['frames_with_no_h']} had no H at all"
        )
        if cal_stats["fit_rate"] < 0.05 and calibration_path and calibration_path.exists():
            # Almost no landmarks visible — degrade gracefully to static cal.
            print(
                "  fewer than 5% of frames could be auto-calibrated; "
                "falling back to manual --calibration."
            )
        elif cal_stats["fit_rate"] > 0:
            homography_done = True

    if not homography_done and calibration_path and calibration_path.exists():
        with open(calibration_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        view = ViewTransformer(
            image_corners=calib["image_corners"],
            world_corners=calib["world_corners"],
        )
        view.add_transformed_positions_to_tracks(tracks)
        homography_done = True

    if homography_done:
        sd = SpeedAndDistanceEstimator(frame_rate=info["fps"])
        sd.add_speed_and_distance_to_tracks(tracks)

        dom_teams = compute_dominant_teams(tracks)
        team_km = compute_team_distances(sd.summary, dom_teams)
        print(
            f"Team distance:  team1 {team_km.get(1, 0):5.2f} km  "
            f"|  team2 {team_km.get(2, 0):5.2f} km"
        )
        if sd.summary:
            for team_id in (1, 2):
                team_players = [
                    (tid, s) for tid, s in sd.summary.items()
                    if dom_teams.get(tid) == team_id
                ]
                team_players.sort(
                    key=lambda kv: kv[1]["total_distance_m"], reverse=True,
                )
                if not team_players:
                    continue
                print(f"  Team {team_id} top distance covered:")
                for tid, s in team_players[:5]:
                    print(
                        f"    player {tid:>3}:  {s['total_distance_m']:6.1f} m   "
                        f"longest sprint: {s['longest_sprint_m']:5.1f} m"
                    )

        if heatmap_dir is not None:
            heatmaps = compute_team_heatmaps(tracks, dom_teams)
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            stem = output_path.stem
            for team_id, hm in heatmaps.items():
                color = team_colors.get(team_id)
                out_png = heatmap_dir / f"{stem}_team{team_id}_heatmap.png"
                save_team_heatmap_png(
                    hm,
                    out_png,
                    title=f"Team {team_id} — occupancy heatmap",
                    team_color_bgr=tuple(color) if color else None,
                )
                print(f"  wrote {out_png}")
    else:
        if auto_calibrate:
            print("  --auto-calibrate produced no usable H and no --calibration "
                  "fallback was provided — skipping homography/speed.")
        elif calibration_path:
            print(f"  no calibration at {calibration_path} — skipping homography/speed.")
        else:
            print("  no --calibration or --auto-calibrate — skipping homography/speed.")

    # ---- Pass 2: render annotated frames straight to disk ------------------
    # Rolling possession share — see ROLLING_HUD_SECONDS notes. Computed once,
    # then indexed per-frame in the render loop. O(n) memory, tiny dicts.
    window_frames = max(1, int(round(ROLLING_HUD_SECONDS * info["fps"])))
    rolling_share = compute_rolling_team_share(
        tracks, per_frame_owner, window_frames=window_frames,
    )
    hud_label = f"last {int(ROLLING_HUD_SECONDS)}s"

    print(f"Rendering streamed output to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with StreamingVideoWriter(output_path, fps=info["fps"]) as writer:
        for local_idx, (_global_idx, frame) in enumerate(
            iter_video_frames(input_path, start_frame=start_frame, max_frames=max_frames)
        ):
            if local_idx >= len(tracks):
                break
            owner_id = (
                per_frame_owner[local_idx]
                if per_frame_owner is not None and local_idx < len(per_frame_owner)
                else None
            )
            cam_shift = None
            if local_idx < len(camera_movement):
                cam_shift = (
                    camera_movement[local_idx][0],
                    camera_movement[local_idx][1],
                )
            canvas = draw_one_frame(
                frame,
                tracks[local_idx],
                owner_id=owner_id,
                team_share=rolling_share[local_idx]
                    if local_idx < len(rolling_share) else None,
                team_colors=team_colors,
                camera_shift=cam_shift,
                hud_label=hud_label,
            )
            writer.write(canvas)
            n_written += 1
            if n_written % 500 == 0:
                print(f"  pass 2: wrote {n_written}/{len(tracks)} frames")
    print(f"Done. Wrote {n_written} annotated frames.")


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
    p.add_argument(
        "--mode", choices=["coco", "custom", "custom_landmarks"], default="custom",
        help="Detector class mapping. 'custom' = 4-class model "
             "(ball/GK/player/ref) for v3/v4 weights. 'custom_landmarks' = "
             "8-class extension that also predicts pitch landmarks "
             "(center_spot/corner_flag/penalty_spot_left/right) — for v5+ "
             "weights trained with the extended taxonomy.",
    )
    p.add_argument("--stub", type=Path, default=None,
                   help="Pickle cache for tracker output.")
    p.add_argument("--cam-stub", type=Path, default=None,
                   help="Pickle cache for camera movement (Phase 6).")
    p.add_argument("--use-stub", action="store_true",
                   help="Read both --stub and --cam-stub if they exist.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="YOLO inference resolution. Default 640 (~4x faster than 1280 "
             "on CPU; ~3-5%% mAP loss on small objects like the ball). Bump "
             "to 1280 for the final render of a clip you care about.",
    )
    p.add_argument(
        "--calibration", type=Path, default=None,
        help="Per-clip pitch calibration JSON (from scripts/pick_pitch_corners.py).",
    )
    p.add_argument(
        "--heatmap-dir", type=Path, default=None,
        help="If set, write per-team heatmap PNGs into this directory. "
             "Requires --calibration (heatmaps need world coords).",
    )
    p.add_argument(
        "--stream", action="store_true",
        help="Use the memory-bounded streaming pipeline (two-pass: analyse "
             "in chunks, then render straight to disk). Required for any "
             "clip longer than ~2 minutes at 1080p.",
    )
    p.add_argument(
        "--chunk-size", type=int, default=STREAM_CHUNK_FRAMES,
        help=f"Streaming chunk size (frames). Default {STREAM_CHUNK_FRAMES}. "
             "Smaller = less RAM, slightly more YOLO startup overhead.",
    )
    p.add_argument(
        "--analysis-stub", type=Path, default=None,
        help="Path to cache pass-1 outputs (tracks + camera_movement). After "
             "the first run completes, re-running with the same path + "
             "--use-analysis-stub skips detection + optical flow entirely "
             "and goes straight to post-processing + render — minutes "
             "instead of hours. Use it while iterating on calibration / "
             "HUD / heatmaps without changing the model or input clip.",
    )
    p.add_argument(
        "--use-analysis-stub", action="store_true",
        help="Load --analysis-stub if it exists. Without this flag the stub "
             "is overwritten (a fresh pass 1 always runs).",
    )
    p.add_argument(
        "--auto-calibrate", action="store_true",
        help="Compute the pitch homography per-frame from detected pitch "
             "landmarks (centre spot, corner flags, penalty spots). Requires "
             "a model trained with the extended class taxonomy "
             "(--mode custom_landmarks). When set, --calibration is used "
             "only as a fallback for frames where < 4 landmarks are visible. "
             "Eliminates the need to maintain per-clip calibration JSONs.",
    )
    args = p.parse_args()

    if args.stream:
        if args.stub or args.cam_stub or args.use_stub:
            print(
                "Note: --stub / --cam-stub / --use-stub are ignored in --stream "
                "mode; streaming runs detection + camera estimation in one pass."
            )
        run_streaming(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            mode=args.mode,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            imgsz=args.imgsz,
            calibration_path=args.calibration,
            heatmap_dir=args.heatmap_dir,
            chunk_size=args.chunk_size,
            analysis_stub_path=args.analysis_stub,
            use_analysis_stub=args.use_analysis_stub,
            auto_calibrate=args.auto_calibrate,
        )
        return

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
        calibration_path=args.calibration,
        cam_stub_path=args.cam_stub,
        heatmap_dir=args.heatmap_dir,
    )


if __name__ == "__main__":
    main()
