"""Entry point: ``python -m src.analytics.app [--db PATH]``.

Parses CLI args, creates the QApplication, opens the main window.
Kept thin so unit tests can exercise the window classes without going
through ``main()``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "analytics.db"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH,
        help=f"Path to analytics SQLite DB. Default: {DEFAULT_DB_PATH}",
    )
    p.add_argument(
        "--match-id", type=int, default=None,
        help="Open this match directly in the tagger view, skipping "
             "the selector. Useful for dev workflows.",
    )
    args = p.parse_args()

    if not args.db.exists():
        sys.exit(
            f"DB not found: {args.db}\n"
            f"Run `python scripts/init_db.py --db {args.db}` first."
        )

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("Football Analytics")
    win = MainWindow(db_path=args.db, initial_match_id=args.match_id)
    win.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
