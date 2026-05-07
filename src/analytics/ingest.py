"""Stub → DB ingest pipeline.

Takes the pickled output of ``main.run_streaming``'s ``--analysis-stub``
(post-stitch tracks + camera_movement) and writes one fully-populated
match into the analytics DB.

The stub is cached BEFORE the bulk of post-processing runs, so this
module re-applies the rest of the chain to enrich the tracks before
writing:

  1. Ball outlier filter + motion tracker
  2. Team assignment from cached colour samples
  3. Team-aware second-pass stitcher
  4. Per-frame possession owner
  5. Camera-shift compensation
  6. (If calibration available) ViewTransformer → world coords
  7. (If calibration available) Speed + distance estimator

After that, every detection in ``tracks`` carries the full set of
fields the schema expects, and we batch-insert in the order the FK
constraints require: matches → frames → frame_player_positions /
frame_ball_positions.

Used by the ``scripts/ingest_match.py`` CLI. Importable as a function
so the future PyQt app can call it directly when the operator drops a
new clip into the workflow.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from src.ball import filter_ball_outliers, track_ball_with_motion
from src.camera_movement import CameraMovementEstimator
from src.pitch import ViewTransformer
from src.possession import compute_team_possession
from src.speed_distance import SpeedAndDistanceEstimator
from src.team_assigner import TeamAssigner
from src.trackers import stitch_tracks_team_aware


# ============================================================================
# Stub enrichment — runs the post-processing chain that main.py runs after
# loading the stub. Returns enriched tracks ready for DB insertion.
# ============================================================================
def enrich_stub(
    stub_path: Path,
    calibration_path: Path | None,
    fps: float,
) -> dict:
    """Apply post-stitcher pipeline steps to a cached analysis stub.

    Returns a dict with:
        tracks            — per-frame list of class→{tid: info} dicts
        camera_movement   — per-frame [dx, dy] from the optical-flow estimator
        team_colors       — {1: [B,G,R], 2: [B,G,R]} or {} if assigner failed
        per_frame_owner   — list of owning track_ids (or None) per frame
        team_share        — {1: float, 2: float} aggregate possession ratio
        has_homography    — bool, True if calibration was applied
    """
    with open(stub_path, "rb") as f:
        cache = pickle.load(f)
    tracks = cache["tracks"]
    camera_movement = cache["camera_movement"]

    # 1. Ball outlier filter + motion-aware tracking. ``track_ball_with_motion``
    #    needs the per-frame list of all people bboxes for its proximity gate.
    filter_ball_outliers(tracks)
    ball_series = [ft["ball"] for ft in tracks]
    persons_per_frame = [
        [
            ((cls, tid), info["bbox"])
            for cls in ("player", "goalkeeper", "referee")
            for tid, info in ft.get(cls, {}).items()
        ]
        for ft in tracks
    ]
    ball_series, _stats = track_ball_with_motion(
        ball_series, persons_per_frame=persons_per_frame,
    )
    for ft, bf in zip(tracks, ball_series):
        ft["ball"] = bf

    # 2. Team assignment from cached colour samples.
    team_assigner = TeamAssigner()
    team_assigner.assign_all_from_samples(tracks)
    team_colors = {k: v.tolist() for k, v in team_assigner.team_colors.items()}

    # 3. Team-aware second-pass stitcher (only useful once teams are assigned).
    if team_colors:
        stitch_tracks_team_aware(tracks)

    # 4. Possession (per-frame owner + aggregate share).
    per_frame_owner, team_share = compute_team_possession(tracks)

    # 5. Camera-shift adjusted foot positions — needed by the ViewTransformer.
    CameraMovementEstimator.add_adjusted_positions_to_tracks(
        tracks, camera_movement,
    )

    # 6 & 7. Homography + speed/distance, if calibration is available.
    has_homography = False
    if calibration_path is not None and Path(calibration_path).exists():
        with open(calibration_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        view = ViewTransformer(
            image_corners=calib["image_corners"],
            world_corners=calib["world_corners"],
        )
        view.add_transformed_positions_to_tracks(tracks)
        SpeedAndDistanceEstimator(frame_rate=fps).add_speed_and_distance_to_tracks(tracks)
        has_homography = True

    return {
        "tracks": tracks,
        "camera_movement": camera_movement,
        "team_colors": team_colors,
        "per_frame_owner": per_frame_owner,
        "team_share": team_share,
        "has_homography": has_homography,
    }


# ============================================================================
# DB writers
# ============================================================================
def insert_match(
    con: sqlite3.Connection,
    *,
    season_id: int,
    home_team_id: int,
    away_team_id: int,
    match_date: str,
    video_path: str,
    analysis_stub_path: str,
    calibration_path: str | None,
    model_version: str,
    fps: float,
    frame_width: int,
    frame_height: int,
    n_frames_analysed: int,
    notes: str | None = None,
) -> int:
    """Insert the matches row, return its new id. Wrapped in a savepoint
    so the caller can rollback if frame insertion fails downstream."""
    cur = con.execute(
        """
        INSERT INTO matches (
            season_id, home_team_id, away_team_id, match_date,
            video_path, analysis_stub_path, calibration_path,
            model_version, fps, frame_width, frame_height,
            n_frames_analysed, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            season_id, home_team_id, away_team_id, match_date,
            video_path, analysis_stub_path, calibration_path,
            model_version, fps, frame_width, frame_height,
            n_frames_analysed, notes,
        ),
    )
    return int(cur.lastrowid)


def insert_frames_and_positions(
    con: sqlite3.Connection,
    *,
    match_id: int,
    tracks: list[dict],
    per_frame_owner: list[int | None],
    fps: float,
) -> dict[str, int]:
    """Walk tracks once, write frames + frame_player_positions + frame_ball_positions.

    Batches inserts inside a single transaction. Returns counts so the
    CLI can print a summary line.
    """
    n_player_rows = 0
    n_ball_rows = 0

    # Cache prepared statements via executemany for speed — at a 90-min
    # match scale we're inserting millions of rows.
    frame_rows: list[tuple] = []
    player_rows: list[tuple] = []
    ball_rows: list[tuple] = []

    ms_per_frame = 1000.0 / fps if fps > 0 else 0.0

    for fi, ft in enumerate(tracks):
        ts_ms = int(round(fi * ms_per_frame))
        frame_rows.append((match_id, fi, ts_ms))

        # The frame_id will only be available after the INSERT; we'll
        # backfill in a second pass. For now, key by (match_id, frame_number).

    cur = con.cursor()
    cur.executemany(
        "INSERT INTO frames (match_id, frame_number, timestamp_ms) "
        "VALUES (?, ?, ?)",
        frame_rows,
    )

    # Pull back the new frame_ids in (match_id, frame_number) order so we
    # can map them when inserting positions.
    frame_id_lookup: dict[int, int] = {
        row["frame_number"]: row["id"]
        for row in cur.execute(
            "SELECT id, frame_number FROM frames WHERE match_id = ?",
            (match_id,),
        )
    }

    # Now build position rows using the resolved frame_ids.
    for fi, ft in enumerate(tracks):
        frame_id = frame_id_lookup[fi]

        # People classes
        for cls in ("player", "goalkeeper", "referee"):
            for tid, info in ft.get(cls, {}).items():
                bbox = info.get("bbox")
                if bbox is None:
                    continue
                x1, y1, x2, y2 = (float(c) for c in bbox)
                foot_x = (x1 + x2) / 2.0
                foot_y = y2
                pos_t = info.get("position_transformed")  # (mx, my) | None
                if pos_t is not None and pos_t[0] is not None:
                    fx_w, fy_w = float(pos_t[0]), float(pos_t[1])
                else:
                    fx_w, fy_w = None, None
                player_rows.append((
                    frame_id, int(tid), cls,
                    x1, y1, x2, y2,
                    info.get("confidence"),
                    foot_x, foot_y, fx_w, fy_w,
                    info.get("team"),
                    info.get("speed_kmh"),
                ))
                n_player_rows += 1

        # Ball — at most one entry per frame (track_id always 1).
        ball = ft.get("ball", {}).get(1)
        if ball:
            bbox = ball.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = (float(c) for c in bbox)
                cx_image = (x1 + x2) / 2.0
                cy_image = (y1 + y2) / 2.0
                # Ball position_transformed comes through the same
                # ViewTransformer pass as players. May be None.
                pos_t = ball.get("position_transformed")
                if pos_t is not None and pos_t[0] is not None:
                    x_w, y_w = float(pos_t[0]), float(pos_t[1])
                else:
                    x_w, y_w = None, None
                owner_tid = per_frame_owner[fi] if fi < len(per_frame_owner) else None
                ball_rows.append((
                    frame_id, x1, y1, x2, y2,
                    cx_image, cy_image, x_w, y_w,
                    ball.get("confidence"),
                    1 if ball.get("interpolated") else 0,
                    1 if ball.get("extrapolated") else 0,
                    1 if ball.get("carrier_anchored") else 0,
                    owner_tid,
                ))
                n_ball_rows += 1

    cur.executemany(
        """INSERT INTO frame_player_positions
        (frame_id, track_id, cls,
         bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence,
         foot_x_image, foot_y_image, foot_x_world, foot_y_world,
         team, speed_kmh)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        player_rows,
    )
    cur.executemany(
        """INSERT INTO frame_ball_positions
        (frame_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
         cx_image, cy_image, x_world, y_world, confidence,
         interpolated, extrapolated, carrier_anchored, owner_track_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ball_rows,
    )
    return {
        "frames": len(frame_rows),
        "player_positions": n_player_rows,
        "ball_positions": n_ball_rows,
    }


def get_or_create_club(con: sqlite3.Connection, name: str) -> int:
    row = con.execute("SELECT id FROM clubs WHERE name = ?", (name,)).fetchone()
    if row:
        return int(row["id"])
    cur = con.execute("INSERT INTO clubs (name) VALUES (?)", (name,))
    return int(cur.lastrowid)


def get_or_create_season(
    con: sqlite3.Connection, name: str,
    start_date: str | None = None, end_date: str | None = None,
) -> int:
    row = con.execute(
        "SELECT id FROM seasons WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return int(row["id"])
    cur = con.execute(
        "INSERT INTO seasons (name, start_date, end_date) VALUES (?, ?, ?)",
        (name, start_date, end_date),
    )
    return int(cur.lastrowid)


def get_or_create_team(
    con: sqlite3.Connection, club_id: int, name: str, age_group: str | None = None,
) -> int:
    row = con.execute(
        "SELECT id FROM teams WHERE club_id = ? AND name = ?",
        (club_id, name),
    ).fetchone()
    if row:
        return int(row["id"])
    cur = con.execute(
        "INSERT INTO teams (club_id, name, age_group) VALUES (?, ?, ?)",
        (club_id, name, age_group),
    )
    return int(cur.lastrowid)
