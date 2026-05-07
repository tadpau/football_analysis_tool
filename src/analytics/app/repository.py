"""DB read/write helpers for the desktop app.

Centralises every SQL query the UI needs so we keep raw SQL out of the
widget code. Each function takes (or holds) an open ``sqlite3.Connection``
and returns plain dicts / typed dataclasses — the UI doesn't see
``sqlite3.Row`` objects.

Phase 2 only needs READ helpers. Track→player mapping (Phase 2c) and
event INSERT helpers (Phase 2d) will land in this same module.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchSummary:
    """Headline info shown in the match selector list."""
    id: int
    match_date: str
    home_team: str
    away_team: str
    n_frames: int
    fps: float
    frame_width: int
    frame_height: int
    video_path: str
    has_calibration: bool
    model_version: str
    season: str


@dataclass(frozen=True)
class FramePlayerPos:
    """One row from frame_player_positions, denormalised for rendering."""
    track_id: int
    cls: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    foot_x_image: float
    foot_y_image: float
    team: int | None
    speed_kmh: float | None


@dataclass(frozen=True)
class FrameBallPos:
    """One row from frame_ball_positions."""
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    cx_image: float
    cy_image: float
    interpolated: bool
    owner_track_id: int | None


def list_matches(con: sqlite3.Connection) -> list[MatchSummary]:
    """All ingested matches, newest match_date first."""
    rows = con.execute(
        """
        SELECT m.id, m.match_date, m.n_frames_analysed, m.fps,
               m.frame_width, m.frame_height, m.video_path,
               m.calibration_path, m.model_version,
               ht.name AS home_team, at.name AS away_team,
               s.name  AS season
        FROM matches m
        JOIN teams ht  ON ht.id = m.home_team_id
        JOIN teams at  ON at.id = m.away_team_id
        JOIN seasons s ON s.id  = m.season_id
        ORDER BY m.match_date DESC, m.id DESC
        """
    ).fetchall()
    return [
        MatchSummary(
            id=r["id"],
            match_date=r["match_date"],
            home_team=r["home_team"],
            away_team=r["away_team"],
            n_frames=r["n_frames_analysed"],
            fps=r["fps"],
            frame_width=r["frame_width"],
            frame_height=r["frame_height"],
            video_path=r["video_path"],
            has_calibration=bool(r["calibration_path"]),
            model_version=r["model_version"],
            season=r["season"],
        )
        for r in rows
    ]


def get_match(con: sqlite3.Connection, match_id: int) -> MatchSummary | None:
    """Look up a single match by id. Returns None if it doesn't exist."""
    matches = [m for m in list_matches(con) if m.id == match_id]
    return matches[0] if matches else None


def get_frame_state(
    con: sqlite3.Connection, match_id: int, frame_number: int,
) -> tuple[list[FramePlayerPos], FrameBallPos | None]:
    """Return the player + ball positions for one frame.

    The Tagger video widget calls this on every frame change. SQLite
    handles ~10k of these queries per second easily, so no caching
    needed at v1 fidelity.
    """
    frame_id_row = con.execute(
        "SELECT id FROM frames WHERE match_id = ? AND frame_number = ?",
        (match_id, frame_number),
    ).fetchone()
    if frame_id_row is None:
        return [], None
    frame_id = frame_id_row["id"]

    player_rows = con.execute(
        """
        SELECT track_id, cls, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
               foot_x_image, foot_y_image, team, speed_kmh
        FROM frame_player_positions
        WHERE frame_id = ?
        """,
        (frame_id,),
    ).fetchall()
    players = [
        FramePlayerPos(
            track_id=r["track_id"],
            cls=r["cls"],
            bbox_x1=r["bbox_x1"], bbox_y1=r["bbox_y1"],
            bbox_x2=r["bbox_x2"], bbox_y2=r["bbox_y2"],
            foot_x_image=r["foot_x_image"], foot_y_image=r["foot_y_image"],
            team=r["team"],
            speed_kmh=r["speed_kmh"],
        )
        for r in player_rows
    ]

    ball_row = con.execute(
        """
        SELECT bbox_x1, bbox_y1, bbox_x2, bbox_y2, cx_image, cy_image,
               interpolated, owner_track_id
        FROM frame_ball_positions WHERE frame_id = ?
        """,
        (frame_id,),
    ).fetchone()
    ball = (
        FrameBallPos(
            bbox_x1=ball_row["bbox_x1"], bbox_y1=ball_row["bbox_y1"],
            bbox_x2=ball_row["bbox_x2"], bbox_y2=ball_row["bbox_y2"],
            cx_image=ball_row["cx_image"], cy_image=ball_row["cy_image"],
            interpolated=bool(ball_row["interpolated"]),
            owner_track_id=ball_row["owner_track_id"],
        )
        if ball_row is not None
        else None
    )
    return players, ball


def hit_test(
    players: list[FramePlayerPos], cx: float, cy: float,
) -> FramePlayerPos | None:
    """Find which player bbox contains the click point.

    If the click hits multiple overlapping bboxes (e.g. crowded set
    piece), pick the one with the SMALLEST area — typically the player
    nearest the camera, which matches what the operator probably aimed
    at. Returns None if no bbox contains the click.
    """
    candidates = [
        p for p in players
        if p.bbox_x1 <= cx <= p.bbox_x2 and p.bbox_y1 <= cy <= p.bbox_y2
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda p: (p.bbox_x2 - p.bbox_x1) * (p.bbox_y2 - p.bbox_y1))
