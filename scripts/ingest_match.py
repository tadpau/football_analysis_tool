"""Ingest one match into the analytics database.

Takes the analysis stub produced by ``main.py --stream --analysis-stub``
plus minimal match metadata (date, teams, video path) and populates the
DB with frames, player/ball positions, and the match row itself.

The operator's track→player mapping and event tagging happen LATER, in
the desktop event-tagger app. Ingest is the offline pre-processing
step that turns a stub into a queryable match.

Identity (club / season / team) can be referenced by name; if the name
doesn't exist, ``--create-missing`` will create it. Otherwise ingest
will refuse rather than silently inserting typo'd entities.

Usage::

    # First time — create club + season + teams as needed
    python scripts/ingest_match.py \\
        --stub stubs/VFA_KFM_v6_pass1.pkl \\
        --video video_clips/VFA_KFM.mp4 \\
        --calibration calibrations/VFA_KFM.json \\
        --model-version v6 \\
        --club "Vilnius FA" --season "2025-2026 U17" \\
        --home-team "VFA U17 A" --away-team "Hanner U17" \\
        --match-date 2026-04-12 \\
        --create-missing
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analytics.db.connection import open_db  # noqa: E402
from src.analytics.ingest import (  # noqa: E402
    enrich_stub,
    insert_match,
    insert_frames_and_positions,
    get_or_create_club,
    get_or_create_season,
    get_or_create_team,
)


DEFAULT_DB_PATH = REPO_ROOT / "data" / "analytics.db"


def _video_metadata(video_path: Path) -> dict:
    """Pull fps + dimensions from the source video. Cheap — opens, reads
    metadata, closes."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def _resolve_or_create(
    con: sqlite3.Connection, args: argparse.Namespace,
) -> tuple[int, int, int]:
    """Look up (or create-if-missing) club, season, home_team, away_team.
    Returns (season_id, home_team_id, away_team_id)."""
    if args.create_missing:
        club_id = get_or_create_club(con, args.club)
        season_id = get_or_create_season(con, args.season)
        home_team_id = get_or_create_team(
            con, club_id, args.home_team, age_group=args.age_group,
        )
        # Away team — we don't know what club they're from; use a placeholder
        # club "Opponents" so the FK is satisfied and the operator can
        # rename later in the UI. This avoids inventing a sibling club for
        # every away team an academy plays.
        opp_club_id = get_or_create_club(con, "Opponents")
        away_team_id = get_or_create_team(
            con, opp_club_id, args.away_team, age_group=args.age_group,
        )
        return season_id, home_team_id, away_team_id

    # Strict mode — names must already exist.
    def _lookup(table: str, where: str, params: tuple) -> int:
        row = con.execute(
            f"SELECT id FROM {table} WHERE {where}", params
        ).fetchone()
        if row is None:
            raise SystemExit(
                f"No {table} matching {params}. "
                f"Add --create-missing or use the manage script."
            )
        return int(row["id"])

    club_id = _lookup("clubs", "name = ?", (args.club,))
    season_id = _lookup("seasons", "name = ?", (args.season,))
    home_team_id = _lookup(
        "teams", "club_id = ? AND name = ?", (club_id, args.home_team),
    )
    away_team_id = _lookup("teams", "name = ?", (args.away_team,))
    return season_id, home_team_id, away_team_id


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument("--stub", type=Path, required=True,
                   help="Analysis stub from main.py --analysis-stub")
    p.add_argument("--video", type=Path, required=True,
                   help="Source video — used for fps + dimensions")
    p.add_argument("--calibration", type=Path, default=None,
                   help="Per-clip calibration JSON. Without it, world "
                        "coords + speed/distance are NULL in the DB.")
    p.add_argument("--model-version", required=True,
                   help='e.g. "v6" — recorded in matches.model_version')
    p.add_argument("--club", required=True,
                   help="Home club name (the academy)")
    p.add_argument("--season", required=True,
                   help='Season name, e.g. "2025-2026 U17"')
    p.add_argument("--home-team", required=True)
    p.add_argument("--away-team", required=True)
    p.add_argument("--match-date", required=True,
                   help="ISO date, YYYY-MM-DD")
    p.add_argument("--age-group", default=None,
                   help='e.g. "U17" — auto-applied to created teams')
    p.add_argument("--notes", default=None,
                   help="Free-text notes attached to the match row")
    p.add_argument("--create-missing", action="store_true",
                   help="Create club/season/team rows if they don't "
                        "already exist. Without this flag, ingest "
                        "errors on unknown names.")
    args = p.parse_args()

    if not args.stub.exists():
        raise SystemExit(f"Stub not found: {args.stub}")
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    print(f"Reading video metadata from {args.video} ...")
    vmeta = _video_metadata(args.video)
    print(f"  {vmeta['fps']:.1f} fps, "
          f"{vmeta['frame_width']}x{vmeta['frame_height']}")

    print(f"Enriching stub {args.stub} (post-processing pipeline) ...")
    enriched = enrich_stub(args.stub, args.calibration, vmeta["fps"])
    n_frames = len(enriched["tracks"])
    print(f"  {n_frames} frames; "
          f"team_colors={'yes' if enriched['team_colors'] else 'no'}, "
          f"homography={'yes' if enriched['has_homography'] else 'no'}")

    con = open_db(args.db)
    try:
        season_id, home_team_id, away_team_id = _resolve_or_create(con, args)
        match_id = insert_match(
            con,
            season_id=season_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            match_date=args.match_date,
            video_path=str(args.video),
            analysis_stub_path=str(args.stub),
            calibration_path=str(args.calibration) if args.calibration else None,
            model_version=args.model_version,
            fps=vmeta["fps"],
            frame_width=vmeta["frame_width"],
            frame_height=vmeta["frame_height"],
            n_frames_analysed=n_frames,
            notes=args.notes,
        )
        print(f"Inserted match id={match_id}")
        counts = insert_frames_and_positions(
            con,
            match_id=match_id,
            tracks=enriched["tracks"],
            per_frame_owner=enriched["per_frame_owner"],
            fps=vmeta["fps"],
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    print(
        f"  frames inserted        : {counts['frames']}"
        f"\n  player positions       : {counts['player_positions']}"
        f"\n  ball positions         : {counts['ball_positions']}"
    )
    print(f"\nDone. Tag events next via the desktop app (Phase 2 — coming).")


if __name__ == "__main__":
    main()
