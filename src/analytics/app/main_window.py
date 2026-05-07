"""Top-level window — hosts the two app views and switches between them.

Two screens, one window:

  1. **Match selector** (default) — table of all ingested matches.
     Double-click a row to open the tagger.
  2. **Tagger** — video player on the left + (Phase 2c+) sidebar with
     track→player mapping and event tagging on the right. A toolbar
     button takes you back to the selector.

Implementation note: keep the two screens inside a ``QStackedWidget``
rather than spawning separate top-level windows. Single-window apps
feel more like a coherent product than a Tkinter-style explosion of
floating dialogs, and PyInstaller bundling is simpler.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..db.connection import open_db
from .repository import MatchSummary, get_match, list_matches
from .video_widget import VideoWidget


# ---------------------------------------------------------------------------
# Match selector — first thing the user sees.
# ---------------------------------------------------------------------------
class MatchSelectorWidget(QWidget):
    """Table of ingested matches. Double-click → open in tagger."""

    match_selected = pyqtSignal(int)   # match_id

    def __init__(self, matches: list[MatchSummary], parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        header = QLabel("<h2>Open a match</h2>")
        header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(header)

        if not matches:
            # Empty DB — give the operator a copy-pasteable command rather
            # than a vague hint. The scripts/ingest_match.py CLI requires
            # several flags and a usage error from omitting them is the
            # most likely first failure mode for a new user.
            empty_title = QLabel("<h3>No matches ingested yet</h3>")
            empty_title.setStyleSheet("color: #ccc;")
            layout.addWidget(empty_title)

            empty_hint = QLabel(
                "Run the ingest CLI to add a match, then reopen this app."
                "<br><br>"
                "Required flags: <code>--stub --video --model-version "
                "--club --season --home-team --away-team --match-date</code>"
                "<br>"
                "Use <code>--create-missing</code> on first run to auto-"
                "create the club / season / team rows."
            )
            empty_hint.setWordWrap(True)
            empty_hint.setStyleSheet("color: #888;")
            layout.addWidget(empty_hint)

            example = QLabel(
                "<pre style='background:#1a1a1a; color:#ddd; padding:10px; "
                "border-radius:4px;'>python scripts/ingest_match.py \\\n"
                "    --stub stubs/&lt;clip&gt;_pass1.pkl \\\n"
                "    --video video_clips/&lt;clip&gt;.mp4 \\\n"
                "    --calibration calibrations/&lt;clip&gt;.json \\\n"
                "    --model-version v6 \\\n"
                "    --club \"Your Club\" --season \"2025-2026 U17\" \\\n"
                "    --home-team \"Your U17\" --away-team \"Opponent\" \\\n"
                "    --match-date 2026-04-12 --age-group U17 \\\n"
                "    --create-missing</pre>"
            )
            example.setTextFormat(Qt.TextFormat.RichText)
            example.setWordWrap(True)
            layout.addWidget(example)
            layout.addStretch(1)
            return

        self._table = QTableWidget(len(matches), 7)
        self._table.setHorizontalHeaderLabels(
            ["Date", "Home", "Away", "Season", "Frames", "Model", "Cal"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        # Stretch to fill horizontally; first column gets a sensible
        # fixed-ish width via ResizeMode.ResizeToContents.
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        h.setStretchLastSection(True)

        for row, m in enumerate(matches):
            cells = [
                m.match_date,
                m.home_team,
                m.away_team,
                m.season,
                f"{m.n_frames:,}",
                m.model_version,
                "✓" if m.has_calibration else "—",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                # Stash the match_id on every cell so any double-click resolves.
                item.setData(Qt.ItemDataRole.UserRole, m.id)
                self._table.setItem(row, col, item)

        self._table.cellDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table, stretch=1)

        hint = QLabel("Double-click a match to open the tagger.")
        hint.setStyleSheet("color: #888;")
        layout.addWidget(hint)

    def _on_double_click(self, row: int, _col: int) -> None:
        item = self._table.item(row, 0)
        if item is None:
            return
        match_id = int(item.data(Qt.ItemDataRole.UserRole))
        self.match_selected.emit(match_id)


# ---------------------------------------------------------------------------
# Tagger — video on the left, sidebar (placeholders for now) on the right.
# ---------------------------------------------------------------------------
class TaggerWidget(QWidget):
    """Per-match tagging view. v1 = video player only; Phase 2c/2d add
    the track-mapping panel and event-tagging panel on the right."""

    back_requested = pyqtSignal()

    def __init__(
        self,
        connection,
        match: MatchSummary,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._con = connection
        self._match = match

        video_path = match.video_path
        if not Path(video_path).exists():
            # Defensive — show an error inline rather than crashing.
            err = QLabel(
                f"<h3>Video not found</h3>"
                f"<p>The match references "
                f"<code>{video_path}</code>, which doesn't exist on this "
                f"machine. Move the file there or re-ingest with the "
                f"correct path.</p>"
            )
            err.setWordWrap(True)
            err.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout = QVBoxLayout(self)
            layout.addWidget(err)
            return

        self._video = VideoWidget(
            connection=connection,
            match_id=match.id,
            video_path=video_path,
            fps=match.fps,
            frame_width=match.frame_width,
            frame_height=match.frame_height,
            n_frames=match.n_frames,
        )
        self._video.player_clicked.connect(self._on_player_clicked)

        # Right sidebar — placeholders for Phase 2c/2d panels. We lay out
        # the structure now so the window proportions don't change later
        # when the real panels land.
        self._sidebar = QWidget()
        self._sidebar.setFixedWidth(320)
        self._sidebar.setStyleSheet("background-color: #1f1f1f; color: #ccc;")
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.addWidget(QLabel(
            f"<b>{match.home_team}</b> vs <b>{match.away_team}</b>"
            f"<br>{match.match_date} &middot; {match.season}"
        ))
        self._selection_label = QLabel("Selected: <i>none</i>")
        self._selection_label.setWordWrap(True)
        sidebar_layout.addWidget(self._selection_label)
        sidebar_layout.addStretch(1)
        sidebar_layout.addWidget(QLabel(
            "<i>Track mapping and hotkey panel arrive in the next "
            "iteration. For now: scrub through the video, click a "
            "player to confirm hit-testing works.</i>"
        ))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._video, stretch=1)
        layout.addWidget(self._sidebar)

    def _on_player_clicked(self, track_id: int) -> None:
        self._selection_label.setText(
            f"Selected: <b>track {track_id}</b><br>"
            f"<small>(Phase 2c will let you map this to a roster "
            f"player by typing a kit number.)</small>"
        )


# ---------------------------------------------------------------------------
# MainWindow — switches between the two screens.
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(
        self,
        db_path: Path,
        initial_match_id: int | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Football Analytics — Event Tagger")
        self.resize(1400, 820)

        # Single shared DB connection for the lifetime of the window.
        # SQLite handles multiple cursors on one connection fine for
        # our access pattern.
        self._con = open_db(db_path)
        self._db_path = db_path

        # ---- Toolbar ----
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._back_action = QAction("← Matches", self)
        self._back_action.setShortcut(QKeySequence("Esc"))
        self._back_action.triggered.connect(self._show_selector)
        self._back_action.setEnabled(False)  # nothing to go back to initially
        toolbar.addAction(self._back_action)

        # ---- View stack ----
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # ---- Status bar ----
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"DB: {db_path}")

        # Build the selector view immediately; tagger view is constructed
        # lazily when a match is opened.
        self._build_selector()
        if initial_match_id is not None:
            self._open_match(initial_match_id)

    # ------------------------------------------------------------ views
    def _build_selector(self) -> None:
        matches = list_matches(self._con)
        sel = MatchSelectorWidget(matches)
        sel.match_selected.connect(self._open_match)
        # Replace any existing selector at index 0.
        if self._stack.count() == 0:
            self._stack.addWidget(sel)
        else:
            old = self._stack.widget(0)
            self._stack.removeWidget(old)
            old.deleteLater()
            self._stack.insertWidget(0, sel)
        self._stack.setCurrentIndex(0)

    def _open_match(self, match_id: int) -> None:
        match = get_match(self._con, match_id)
        if match is None:
            QMessageBox.warning(
                self, "Match not found",
                f"No match with id={match_id} in this DB.",
            )
            return
        tagger = TaggerWidget(self._con, match)
        # Tagger goes at index 1; replace any previous one.
        while self._stack.count() > 1:
            old = self._stack.widget(1)
            self._stack.removeWidget(old)
            old.deleteLater()
        self._stack.addWidget(tagger)
        self._stack.setCurrentIndex(1)
        self._back_action.setEnabled(True)
        self.statusBar().showMessage(
            f"DB: {self._db_path}    Match {match.id}: "
            f"{match.home_team} vs {match.away_team}    "
            f"{match.n_frames:,} frames @ {match.fps:.0f} fps"
        )

    def _show_selector(self) -> None:
        self._stack.setCurrentIndex(0)
        self._back_action.setEnabled(False)
        self.statusBar().showMessage(f"DB: {self._db_path}")

    # ---------------------------------------------------------- close
    def closeEvent(self, event) -> None:  # noqa: D401
        if self._con is not None:
            self._con.close()
        super().closeEvent(event)
