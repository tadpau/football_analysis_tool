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
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..db.connection import open_db
from .event_panel import EventPanel
from .reports_widget import ReportsWidget
from .repository import MatchSummary, ensure_event_types, get_match, list_matches
from .track_mapping_panel import TrackMappingPanel
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

        self._tracks_panel = TrackMappingPanel(connection, match)
        self._events_panel = EventPanel(connection, match)
        # Seed the events panel with whichever team is currently selected
        # for tagging — used to resolve "kit 7" → track_id during event save.
        self._events_panel.set_tagging_team_side(
            1 if self._tracks_panel._tagging_team.is_home else 2  # noqa: SLF001
        )

        # Push existing roster mappings into the video so labels appear
        # the moment the tagger view opens (rather than only after the
        # next mapping change).
        self._video.set_track_labels(self._tracks_panel.get_track_labels())

        # ---- Cross-widget signal wiring ----
        # Click on video → both panels see the selection. Players panel
        # uses it for assignment; events panel uses it as the "primary
        # actor" for the next hotkey press.
        self._video.player_clicked.connect(self._on_video_player_clicked)
        # Frame change → events panel needs to know which frame_id to
        # attach future events to.
        self._video.frame_changed.connect(self._events_panel.set_current_frame)
        # Players panel changed mappings → repush labels into the video
        # overlay AND tell the events panel about any new tagging-team flip.
        self._tracks_panel.mappings_changed.connect(self._on_mappings_changed)
        # Players panel team-radio flips → events panel needs the new side
        # for its kit→track resolver.
        self._tracks_panel._home_radio.toggled.connect(  # noqa: SLF001
            self._on_tagging_team_changed
        )
        # Players-panel list click → highlight that track on the video.
        self._tracks_panel.track_chosen_in_list.connect(self._video.select_track)
        # Events panel: clicking a recent event → seek video to that frame.
        self._events_panel.event_clicked.connect(self._video.show_frame)

        # Tabbed sidebar — Players first (setup), Events second (tagging).
        self._sidebar_tabs = QTabWidget()
        self._sidebar_tabs.setFixedWidth(380)
        self._sidebar_tabs.addTab(self._tracks_panel, "Players")
        self._sidebar_tabs.addTab(self._events_panel, "Events")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._video, stretch=1)
        layout.addWidget(self._sidebar_tabs)

    def _on_mappings_changed(self) -> None:
        self._video.set_track_labels(self._tracks_panel.get_track_labels())

    def _on_video_player_clicked(self, track_id: int) -> None:
        # Forward to both panels. Players panel does the
        # set-selected-then-show-roster flow; events panel records who
        # the actor of the next hotkey will be.
        self._tracks_panel.set_selected_track(track_id)
        labels = self._tracks_panel.get_track_labels()
        label_obj = labels.get(track_id)
        label_text = label_obj.text if label_obj else f"track {track_id}"
        self._events_panel.set_selected_track(track_id, label_text)

    def _on_tagging_team_changed(self) -> None:
        # Players panel manages the radio internally; we just read the
        # current state and propagate the side to the events panel.
        side = 1 if self._tracks_panel._home_radio.isChecked() else 2  # noqa: SLF001
        self._events_panel.set_tagging_team_side(side)


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
        # Backfill any event_types added after the DB was first created
        # (Lost ball / Won ball were added in a later version). Safe to
        # run on every startup — INSERT OR IGNORE per row.
        ensure_event_types(self._con)
        self._db_path = db_path

        # Currently-open match + its lazy reports widget. Both stay
        # None until ``_open_match`` runs and get reset on ``_show_selector``.
        self._current_match: MatchSummary | None = None
        self._current_reports: ReportsWidget | None = None

        # ---- Toolbar ----
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # No keyboard shortcut on this — Esc is reserved for the
        # event-tagger panel's "cancel current event flow" action.
        # Going back to the match selector is a deliberate, low-frequency
        # operation, so a toolbar click is fine.
        self._back_action = QAction("← Matches", self)
        self._back_action.triggered.connect(self._show_selector)
        self._back_action.setEnabled(False)  # nothing to go back to initially
        toolbar.addAction(self._back_action)

        # Two view-switch actions: toggle between Tagger and Reports for
        # the currently-open match. Only enabled once a match is open.
        self._tagger_action = QAction("🎯 Tagger", self)
        self._tagger_action.triggered.connect(self._show_tagger_view)
        self._tagger_action.setEnabled(False)
        toolbar.addAction(self._tagger_action)
        self._reports_action = QAction("📊 Reports", self)
        self._reports_action.triggered.connect(self._show_reports_view)
        self._reports_action.setEnabled(False)
        toolbar.addAction(self._reports_action)

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

        # Tear down any previously-open match widgets to free memory
        # (each VideoWidget owns a cv2.VideoCapture). Selector stays at
        # stack index 0; we always rebuild Tagger at 1 and Reports at 2.
        while self._stack.count() > 1:
            old = self._stack.widget(1)
            self._stack.removeWidget(old)
            old.deleteLater()

        self._current_tagger = TaggerWidget(self._con, match)
        self._stack.addWidget(self._current_tagger)
        # Reports is built lazily on first switch — saves a second of
        # query work and an avoidable repaint for operators who just
        # want to tag.
        self._current_reports: ReportsWidget | None = None
        self._current_match: MatchSummary = match

        self._stack.setCurrentIndex(1)
        self._back_action.setEnabled(True)
        self._tagger_action.setEnabled(True)
        self._reports_action.setEnabled(True)
        self.statusBar().showMessage(
            f"DB: {self._db_path}    Match {match.id}: "
            f"{match.home_team} vs {match.away_team}    "
            f"{match.n_frames:,} frames @ {match.fps:.0f} fps"
        )

    def _show_tagger_view(self) -> None:
        if self._stack.count() > 1:
            self._stack.setCurrentIndex(1)

    def _show_reports_view(self) -> None:
        # Lazy-build the reports widget the first time it's requested.
        # When the operator switches back to it later, refresh the
        # queries so any newly-tagged events show up.
        if self._current_match is None:
            return
        if self._current_reports is None:
            self._current_reports = ReportsWidget(self._con, self._current_match)
            self._stack.addWidget(self._current_reports)
        else:
            self._current_reports.refresh()
        self._stack.setCurrentIndex(self._stack.indexOf(self._current_reports))

    def _show_selector(self) -> None:
        self._stack.setCurrentIndex(0)
        self._back_action.setEnabled(False)
        self._tagger_action.setEnabled(False)
        self._reports_action.setEnabled(False)
        self._current_match = None
        self._current_reports = None
        self.statusBar().showMessage(f"DB: {self._db_path}")

    # ---------------------------------------------------------- close
    def closeEvent(self, event) -> None:  # noqa: D401
        if self._con is not None:
            self._con.close()
        super().closeEvent(event)
