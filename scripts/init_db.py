"""Create a fresh empty analytics database.

Reads the schema from ``src/analytics/db/schema.sql`` and applies it
to a new SQLite file at ``--db``. Refuses to overwrite an existing DB
unless ``--force`` is given.

Usage::

    # Create a new DB at the default location
    python scripts/init_db.py

    # Create at a custom location
    python scripts/init_db.py --db data/my_academy.db

    # Recreate (DELETES existing match data)
    python scripts/init_db.py --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analytics.db.connection import init_db  # noqa: E402


DEFAULT_DB_PATH = REPO_ROOT / "data" / "analytics.db"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH,
        help=f"Path to write the SQLite DB. Default: {DEFAULT_DB_PATH}",
    )
    p.add_argument(
        "--force", action="store_true",
        help="If --db already exists, delete and recreate. WARNING: "
             "destroys all match data in that DB.",
    )
    args = p.parse_args()

    con = init_db(args.db, force=args.force)
    cur = con.cursor()

    # Tiny summary so the operator sees the DB took.
    tables = sorted(
        r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    )
    n_event_types = cur.execute("SELECT COUNT(*) FROM event_types").fetchone()[0]
    version = cur.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()[0]
    con.close()

    print(f"Created {args.db}")
    print(f"  schema version : {version}")
    print(f"  tables ({len(tables)}): {', '.join(tables)}")
    print(f"  event types    : {n_event_types} seeded (run "
          f"`sqlite3 {args.db} 'SELECT * FROM event_types'` to inspect)")


if __name__ == "__main__":
    main()
