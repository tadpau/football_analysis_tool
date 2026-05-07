"""SQLite connection helpers for the analytics layer.

Centralises three things every caller would otherwise re-implement:

  * ``open_db(path)`` — opens a connection with the right pragmas
    (``foreign_keys = ON`` is OFF by default in SQLite, which would
    break our cascade deletes silently).
  * ``init_db(path, force=False)`` — creates a fresh DB by executing
    ``schema.sql``. Refuses to overwrite an existing file unless
    ``force=True``.
  * ``get_schema_version(con)`` — returns the integer version recorded
    in the ``schema_meta`` table. Future migrations key off this.

Use ``open_db`` from app code (event tagger, ingest, reports);
``init_db`` lives in :mod:`scripts.init_db`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Schema lives next to this file. Bundling it as a package data file
# (vs reading from a hardcoded repo path) means the analytics module
# can be imported from a PyInstaller bundle later without surprises.
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

EXPECTED_SCHEMA_VERSION = 1


class SchemaError(RuntimeError):
    """Raised when the DB on disk is at an unexpected schema version."""


def open_db(path: str | Path, *, check_version: bool = True) -> sqlite3.Connection:
    """Open a SQLite connection with the project's standard pragmas.

    Args:
        path: filesystem path to the .db file. Must already exist —
            use :func:`init_db` to create one.
        check_version: if True (default), refuses to open a DB whose
            ``schema_meta.version`` doesn't match
            :data:`EXPECTED_SCHEMA_VERSION`. Set False only in
            migration scripts that intentionally read older schemas.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"DB not found: {p}. Run `python scripts/init_db.py "
            f"--db {p}` first."
        )
    con = sqlite3.connect(str(p))
    # FK constraints OFF by default in SQLite — turn them on so our
    # ON DELETE CASCADE rules actually fire.
    con.execute("PRAGMA foreign_keys = ON")
    # WAL mode allows the event tagger to read while ingest writes.
    # Costs nothing for single-process use, helps if we ever split.
    con.execute("PRAGMA journal_mode = WAL")
    # Return rows as dict-like sqlite3.Row instead of tuples — much
    # nicer to use in app code.
    con.row_factory = sqlite3.Row

    if check_version:
        version = get_schema_version(con)
        if version != EXPECTED_SCHEMA_VERSION:
            raise SchemaError(
                f"DB at {p} is schema version {version}, app expects "
                f"{EXPECTED_SCHEMA_VERSION}. Run a migration or recreate."
            )
    return con


def init_db(path: str | Path, *, force: bool = False) -> sqlite3.Connection:
    """Create a new DB at ``path`` and apply ``schema.sql``.

    If ``path`` exists and ``force`` is False, raises FileExistsError —
    we never silently clobber an existing analytics DB. With
    ``force=True``, the existing file is deleted first.
    """
    p = Path(path)
    if p.exists():
        if not force:
            raise FileExistsError(
                f"{p} already exists. Pass force=True to overwrite "
                f"(this DELETES all match data)."
            )
        p.unlink()
    p.parent.mkdir(parents=True, exist_ok=True)

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    con = sqlite3.connect(str(p))
    con.executescript(schema)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.commit()
    con.row_factory = sqlite3.Row
    return con


def get_schema_version(con: sqlite3.Connection) -> int | None:
    """Return the integer ``schema_meta.version`` row, or None if absent."""
    cur = con.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    )
    row = cur.fetchone()
    if row is None:
        return None
    return int(row[0])
