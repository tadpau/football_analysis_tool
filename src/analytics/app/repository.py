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


# ---------------------------------------------------------------------------
# Team + roster CRUD (Phase 2c — track→player mapping)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TeamInfo:
    """Resolved home/away team for a match — what the sidebar buttons show."""
    id: int
    name: str
    is_home: bool


@dataclass(frozen=True)
class TrackMapping:
    """One row of match_track_to_player joined with player + team."""
    track_id: int
    player_id: int
    player_name: str
    kit_number: int | None
    team_side: int          # CV-detected side (1 or 2)
    team_id: int            # roster team
    team_name: str
    is_home: bool


def get_match_teams(con: sqlite3.Connection, match_id: int) -> tuple[TeamInfo, TeamInfo]:
    """Return ``(home, away)`` team info for the match.

    The ``is_home`` flag drives which "Assign to ..." button label the
    sidebar shows — operator never has to remember which team_side maps
    to which side of the schema.
    """
    row = con.execute(
        """
        SELECT m.home_team_id, m.away_team_id,
               ht.name AS home_name, at.name AS away_name
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE m.id = ?
        """,
        (match_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No match with id={match_id}")
    return (
        TeamInfo(id=row["home_team_id"], name=row["home_name"], is_home=True),
        TeamInfo(id=row["away_team_id"], name=row["away_name"], is_home=False),
    )


def dominant_team_side(
    con: sqlite3.Connection, match_id: int, track_id: int,
) -> int | None:
    """Return the team_side (1 or 2) the track was MOST OFTEN classified
    as during the match, or None if every position lacks a team.

    Used to suggest home-vs-away to the operator — kit colour usually
    makes this obvious in the rendered overlay, so we just preview the
    CV's best guess. Operator confirms by clicking Home or Away.
    """
    row = con.execute(
        """
        SELECT team, COUNT(*) AS n
        FROM frame_player_positions p
        JOIN frames f ON f.id = p.frame_id
        WHERE f.match_id = ? AND p.track_id = ? AND p.team IS NOT NULL
        GROUP BY team
        ORDER BY n DESC
        LIMIT 1
        """,
        (match_id, track_id),
    ).fetchone()
    if row is None:
        return None
    return int(row["team"])


def list_track_mappings(
    con: sqlite3.Connection, match_id: int,
) -> list[TrackMapping]:
    """All track→player mappings already recorded for this match."""
    rows = con.execute(
        """
        SELECT mtp.track_id, mtp.player_id, mtp.team_side,
               mtp.kit_number_in_match,
               COALESCE(p.first_name || ' ' || p.last_name,
                        p.first_name, p.last_name,
                        '#' || COALESCE(mtp.kit_number_in_match,
                                        p.default_kit_number, 'X'))
                   AS player_name,
               t.id AS team_id, t.name AS team_name,
               (t.id = m.home_team_id) AS is_home
        FROM match_track_to_player mtp
        JOIN players p ON p.id = mtp.player_id
        JOIN teams   t ON t.id = p.team_id
        JOIN matches m ON m.id = mtp.match_id
        WHERE mtp.match_id = ?
        ORDER BY mtp.track_id
        """,
        (match_id,),
    ).fetchall()
    return [
        TrackMapping(
            track_id=r["track_id"],
            player_id=r["player_id"],
            player_name=r["player_name"],
            kit_number=r["kit_number_in_match"],
            team_side=r["team_side"],
            team_id=r["team_id"],
            team_name=r["team_name"],
            is_home=bool(r["is_home"]),
        )
        for r in rows
    ]


@dataclass(frozen=True)
class TrackLabel:
    """Overlay-rendering payload for a mapped track.

    ``expected_team_side`` is the CV-detected team the track had when
    the operator assigned it. Used by the renderer to suppress the
    name when the same track_id later resurfaces with the OPPOSITE
    team's colour — that's ByteTrack reusing a freed ID for a new
    physical player on the other team, and showing the original name
    on it would actively mislead the operator.
    """
    text: str
    expected_team_side: int


def get_track_labels(
    con: sqlite3.Connection, match_id: int,
) -> dict[int, TrackLabel]:
    """Compact ``{track_id: TrackLabel}`` dict the video widget renders.

    The renderer should ONLY display the label when the current
    frame's ``team`` matches ``expected_team_side`` (or when the
    frame's team is None — too ambiguous to override).
    """
    out: dict[int, TrackLabel] = {}
    for m in list_track_mappings(con, match_id):
        kit = m.kit_number if m.kit_number is not None else "?"
        # Truncate name to keep label short — overlays sit close together.
        name = m.player_name.strip()
        if len(name) > 12:
            name = name[:11] + "…"
        out[m.track_id] = TrackLabel(
            text=f"{kit} {name}".strip(),
            expected_team_side=m.team_side,
        )
    return out


def get_or_create_player(
    con: sqlite3.Connection,
    *,
    team_id: int,
    kit_number: int | None,
    name: str | None,
) -> int:
    """Return the ``players.id`` for (team, kit). Creates one if missing.

    The (team_id, default_kit_number) UNIQUE constraint means kit 10 on
    a given team exists at most once in the roster — repeated assigns
    of "track 47, kit 10, home" all link to the same player row.

    ``name`` is only used when CREATING a new player. Existing rows
    keep whatever name they already have; renaming is a separate
    operation (UI doesn't expose that yet — direct SQL for now).
    """
    if kit_number is not None:
        row = con.execute(
            "SELECT id FROM players "
            "WHERE team_id = ? AND default_kit_number = ?",
            (team_id, kit_number),
        ).fetchone()
        if row is not None:
            return int(row["id"])

    # Split a free-text "Petras Petrauskas" name into first / last on the
    # first space. Single-token names go to first_name.
    first, last = None, None
    if name:
        n = name.strip()
        if " " in n:
            first, last = n.split(" ", 1)
        else:
            first = n

    cur = con.execute(
        "INSERT INTO players (team_id, default_kit_number, first_name, last_name) "
        "VALUES (?, ?, ?, ?)",
        (team_id, kit_number, first, last),
    )
    return int(cur.lastrowid)


def assign_track_to_player(
    con: sqlite3.Connection,
    *,
    match_id: int,
    track_id: int,
    player_id: int,
    team_side: int,
    kit_number_in_match: int | None,
) -> None:
    """Insert (or replace) the ``match_track_to_player`` row.

    Replacing on conflict means the operator can re-assign a track
    (mistake, kit changed mid-match, …) just by repeating the action —
    the latest assignment wins.
    """
    con.execute(
        """
        INSERT INTO match_track_to_player
            (match_id, track_id, player_id, team_side, kit_number_in_match)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(match_id, track_id) DO UPDATE SET
            player_id = excluded.player_id,
            team_side = excluded.team_side,
            kit_number_in_match = excluded.kit_number_in_match
        """,
        (match_id, track_id, player_id, team_side, kit_number_in_match),
    )
    con.commit()


def unassign_track(
    con: sqlite3.Connection, match_id: int, track_id: int,
) -> None:
    """Drop a track mapping. The events table keeps any prior tags;
    they're tied to track_ids, not player_ids."""
    con.execute(
        "DELETE FROM match_track_to_player WHERE match_id = ? AND track_id = ?",
        (match_id, track_id),
    )
    con.commit()


@dataclass(frozen=True)
class RosterEntry:
    """One row of the roster panel — a player + how many tracks in the
    current match are already linked to them. The track count is the
    operator's progress indicator: a goalkeeper might end up with 1-2
    tracks across a match, an outfield player closer to 5-10."""
    player_id: int
    kit_number: int | None
    name: str
    track_count: int


def list_roster_with_track_counts(
    con: sqlite3.Connection, match_id: int, team_id: int,
) -> list[RosterEntry]:
    """Players on ``team_id`` plus a track-count for the given match.

    Players are sorted by kit number (kits without a number sink to the
    bottom). The roster panel renders this list as clickable rows.
    """
    rows = con.execute(
        """
        SELECT p.id, p.default_kit_number,
               COALESCE(p.first_name || ' ' || p.last_name,
                        p.first_name, p.last_name,
                        '#' || COALESCE(p.default_kit_number, p.id))
                   AS player_name,
               (SELECT COUNT(*) FROM match_track_to_player mtp
                WHERE mtp.match_id = ? AND mtp.player_id = p.id) AS track_count
        FROM players p
        WHERE p.team_id = ?
        ORDER BY
            CASE WHEN p.default_kit_number IS NULL THEN 1 ELSE 0 END,
            p.default_kit_number,
            p.id
        """,
        (match_id, team_id),
    ).fetchall()
    return [
        RosterEntry(
            player_id=r["id"],
            kit_number=r["default_kit_number"],
            name=r["player_name"].strip(),
            track_count=r["track_count"],
        )
        for r in rows
    ]


