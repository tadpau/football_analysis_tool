"""Sidebar widget — assign roster players to CV track IDs.

Driven by the operator's workflow described in Phase 2c:
  1. Click a player on the video → that track becomes the "selected" one.
  2. Type kit number (and optionally a name).
  3. Press H (Home) or A (Away) — the assignment goes into
     ``match_track_to_player`` and the video overlay relabels that
     track to ``"<kit> <name>"`` immediately.
  4. When the player disappears from the camera and reappears as a new
     track ID, repeat — multiple track IDs map to the same roster player.

Existing roster matches are detected automatically: if you assign
"kit 10, home" twice, the second click reuses the same player row,
so per-match aggregates accumulate cleanly. Renaming an existing
roster entry is intentionally NOT exposed in the panel (avoid the
operator clobbering names mid-tagging by mistake).
"""
from __future__ import annotations

import sqlite3

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .repository import (
    MatchSummary,
    TeamInfo,
    TrackMapping,
    assign_track_to_player,
    dominant_team_side,
    get_match_teams,
    get_or_create_player,
    list_track_mappings,
    unassign_track,
)


class TrackMappingPanel(QWidget):
    """Track→player mapping UI plus list of all current mappings."""

    # Emitted whenever the assignments change so the video widget can
    # repaint with new labels.
    mappings_changed = pyqtSignal()
    # Emitted when the operator clicks a row in the mappings list — the
    # video widget jumps to a frame containing that track and highlights
    # it. (Phase 2c v1: just highlight on the current frame.)
    track_chosen_in_list = pyqtSignal(int)

    def __init__(
        self,
        connection: sqlite3.Connection,
        match: MatchSummary,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._con = connection
        self._match = match
        self._home, self._away = get_match_teams(connection, match.id)
        self._selected_track: int | None = None
        self._selected_team_side: int | None = None

        self.setFixedWidth(360)
        self.setStyleSheet(
            "QWidget { background-color: #1f1f1f; color: #ddd; } "
            "QGroupBox { border: 1px solid #333; border-radius: 4px; "
            "  margin-top: 14px; padding-top: 10px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; "
            "  padding: 0 4px; color: #999; }"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # ---- Match header ----
        header = QLabel(
            f"<b>{self._home.name}</b> &nbsp;vs&nbsp; "
            f"<b>{self._away.name}</b><br>"
            f"<span style='color:#888;'>{match.match_date} · {match.season}</span>"
        )
        header.setTextFormat(Qt.TextFormat.RichText)
        outer.addWidget(header)

        # ---- Selected track group ----
        sel_group = QGroupBox("Selected track")
        sel_layout = QVBoxLayout(sel_group)

        self._selected_label = QLabel("<i>Click a player on the video.</i>")
        self._selected_label.setTextFormat(Qt.TextFormat.RichText)
        sel_layout.addWidget(self._selected_label)

        # Kit + name inputs
        kit_row = QHBoxLayout()
        kit_row.addWidget(QLabel("Kit:"))
        self._kit_input = QSpinBox()
        self._kit_input.setRange(0, 99)
        self._kit_input.setSpecialValueText(" ")  # 0 → blank, treated as None
        self._kit_input.setValue(0)
        self._kit_input.setFixedWidth(60)
        kit_row.addWidget(self._kit_input)
        kit_row.addWidget(QLabel("Name:"))
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("(optional)")
        kit_row.addWidget(self._name_input, stretch=1)
        sel_layout.addLayout(kit_row)

        # Action buttons
        btn_row = QHBoxLayout()
        self._assign_home_btn = QPushButton(f"Home  ({self._home.name})")
        self._assign_home_btn.setToolTip("Assign as a home-team player (shortcut: H)")
        self._assign_home_btn.clicked.connect(self._on_assign_home)
        self._assign_away_btn = QPushButton(f"Away  ({self._away.name})")
        self._assign_away_btn.setToolTip("Assign as an away-team player (shortcut: A)")
        self._assign_away_btn.clicked.connect(self._on_assign_away)
        btn_row.addWidget(self._assign_home_btn)
        btn_row.addWidget(self._assign_away_btn)
        sel_layout.addLayout(btn_row)

        unassign_row = QHBoxLayout()
        self._unassign_btn = QPushButton("Unassign track")
        self._unassign_btn.setToolTip("Remove the mapping (shortcut: U)")
        self._unassign_btn.clicked.connect(self._on_unassign)
        unassign_row.addStretch(1)
        unassign_row.addWidget(self._unassign_btn)
        sel_layout.addLayout(unassign_row)

        outer.addWidget(sel_group)

        # ---- Mappings list ----
        list_group = QGroupBox("Mapped tracks")
        list_layout = QVBoxLayout(list_group)
        self._mappings_list = QListWidget()
        self._mappings_list.itemClicked.connect(self._on_list_click)
        self._mappings_list.setStyleSheet(
            "QListWidget { background-color: #141414; border: 1px solid #2a2a2a; }"
            "QListWidget::item:selected { background-color: #2a4d6a; }"
        )
        list_layout.addWidget(self._mappings_list)
        self._summary_label = QLabel("0 mapped")
        self._summary_label.setStyleSheet("color: #888;")
        list_layout.addWidget(self._summary_label)
        outer.addWidget(list_group, stretch=1)

        # ---- Disable assign buttons until a track is selected ----
        self._set_buttons_enabled(False)
        self._refresh_list()

        # ---- Keyboard shortcuts on the panel ----
        # H / A / U work whenever the panel has focus. The ``shortcut``
        # is on the panel's parent so it survives focus changing to
        # the kit input — we'd hate to type "h" into the name field
        # and have it trigger Home assignment.
        QShortcut(QKeySequence("Ctrl+H"), self, activated=self._on_assign_home)
        QShortcut(QKeySequence("Ctrl+A"), self, activated=self._on_assign_away)
        QShortcut(QKeySequence("Ctrl+U"), self, activated=self._on_unassign)

    # ------------------------------------------------------------ public
    def set_selected_track(self, track_id: int) -> None:
        """Called by the main window when the operator clicks the video.

        Looks up the track's CV-detected team side as a hint, and shows
        the existing mapping (if any) so the operator can see whether
        this track is already known.
        """
        self._selected_track = track_id
        self._selected_team_side = dominant_team_side(
            self._con, self._match.id, track_id,
        )

        # Pre-fill kit + suggested side if we already have a mapping.
        existing = next(
            (m for m in list_track_mappings(self._con, self._match.id)
             if m.track_id == track_id),
            None,
        )
        existing_summary = ""
        if existing is not None:
            existing_summary = (
                f"<br>Currently mapped to "
                f"<b>{existing.player_name}</b> "
                f"(kit {existing.kit_number}, "
                f"{'home' if existing.is_home else 'away'})"
            )
            if existing.kit_number is not None:
                self._kit_input.setValue(existing.kit_number)
            self._name_input.setText(existing.player_name)
        else:
            # Don't auto-clear inputs — operator may want to reuse the
            # last typed value across consecutive same-team assignments.
            pass

        side_str = (
            f"team {self._selected_team_side}"
            if self._selected_team_side is not None else "unclassified"
        )
        self._selected_label.setText(
            f"<b>track {track_id}</b> "
            f"<span style='color:#888;'>(CV-detected side: {side_str})</span>"
            f"{existing_summary}"
        )
        self._set_buttons_enabled(True)
        # Auto-focus kit field so the operator can type immediately.
        self._kit_input.setFocus()
        self._kit_input.selectAll()

    def get_track_labels(self) -> dict[int, str]:
        """For VideoWidget overlay rendering."""
        out: dict[int, str] = {}
        for m in list_track_mappings(self._con, self._match.id):
            kit = m.kit_number if m.kit_number is not None else "?"
            name = m.player_name.strip()
            if len(name) > 12:
                name = name[:11] + "…"
            out[m.track_id] = f"{kit} {name}" if name else f"{kit}"
        return out

    # ---------------------------------------------------------- handlers
    def _on_assign_home(self) -> None:
        self._assign(self._home)

    def _on_assign_away(self) -> None:
        self._assign(self._away)

    def _assign(self, team: TeamInfo) -> None:
        if self._selected_track is None:
            return
        kit_value = self._kit_input.value()
        kit = kit_value if kit_value > 0 else None
        name = self._name_input.text().strip() or None

        player_id = get_or_create_player(
            self._con,
            team_id=team.id,
            kit_number=kit,
            name=name,
        )
        side = self._selected_team_side or (1 if team.is_home else 2)
        assign_track_to_player(
            self._con,
            match_id=self._match.id,
            track_id=self._selected_track,
            player_id=player_id,
            team_side=side,
            kit_number_in_match=kit,
        )
        self._refresh_list()
        self.mappings_changed.emit()

    def _on_unassign(self) -> None:
        if self._selected_track is None:
            return
        unassign_track(self._con, self._match.id, self._selected_track)
        self._refresh_list()
        self.mappings_changed.emit()

    def _on_list_click(self, item: QListWidgetItem) -> None:
        track_id = int(item.data(Qt.ItemDataRole.UserRole))
        self.track_chosen_in_list.emit(track_id)

    # ---------------------------------------------------------- internals
    def _refresh_list(self) -> None:
        mappings = list_track_mappings(self._con, self._match.id)
        self._mappings_list.clear()
        for m in mappings:
            kit = m.kit_number if m.kit_number is not None else "?"
            side = "home" if m.is_home else "away"
            text = f"track {m.track_id:>4} → {kit} {m.player_name}  ({side})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, m.track_id)
            self._mappings_list.addItem(item)
        self._summary_label.setText(f"{len(mappings)} mapped")

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._assign_home_btn.setEnabled(enabled)
        self._assign_away_btn.setEnabled(enabled)
        self._unassign_btn.setEnabled(enabled)
        self._kit_input.setEnabled(enabled)
        self._name_input.setEnabled(enabled)
