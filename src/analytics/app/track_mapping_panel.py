"""Sidebar widget — single-team roster + click-to-assign workflow.

Refined Phase 2c UX (per operator feedback):

  * Operator picks ONCE which team they're tagging (their own team).
    The opposite team's tracks keep their CV-detected colour and are
    never shown in this panel — they don't get player IDs assigned.

  * Operator pre-builds the 11-player roster for their team using the
    "+ Add player" form (kit number + optional name).

  * Then during playback, the workflow becomes:
        1. Click a player on the video → that track is "selected".
        2. Click a roster row → track binds to that roster player.
        3. Track count next to each roster row goes up so you can see
           progress.

  * When the same physical player reappears with a new ByteTrack ID
    after occlusion, the operator just clicks the player on screen
    and clicks the same roster row again — both track IDs link to the
    same players.id, so per-player aggregates accumulate cleanly.

The panel intentionally does NOT expose deletion (rebuild the DB or
edit SQL if the roster needs surgery). Renaming a roster entry is
also out of scope for v1.
"""
from __future__ import annotations

import sqlite3

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .repository import (
    MatchSummary,
    RosterEntry,
    TeamInfo,
    TrackMapping,
    assign_track_to_player,
    dominant_team_side,
    get_match_teams,
    get_or_create_player,
    list_roster_with_track_counts,
    list_track_mappings,
    unassign_track,
)


class TrackMappingPanel(QWidget):
    """Single-team roster panel + click-to-assign for the selected track."""

    mappings_changed = pyqtSignal()
    track_chosen_in_list = pyqtSignal(int)   # for jumping/highlighting

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

        # Tagging team — defaults to home (most common case for an
        # academy analysing their own match), but operator can flip.
        self._tagging_team: TeamInfo = self._home

        # Currently selected CV track (set by main window on video click).
        self._selected_track: int | None = None

        self.setFixedWidth(380)
        self.setStyleSheet(
            "QWidget { background-color: #1f1f1f; color: #ddd; } "
            "QGroupBox { border: 1px solid #333; border-radius: 4px; "
            "  margin-top: 14px; padding-top: 10px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; "
            "  padding: 0 4px; color: #999; } "
            "QListWidget { background-color: #141414; border: 1px solid #2a2a2a; } "
            "QListWidget::item { padding: 6px; } "
            "QListWidget::item:hover { background-color: #2a3a4a; } "
            "QListWidget::item:selected { background-color: #2a4d6a; }"
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

        # ---- Tagging-team picker ----
        team_group = QGroupBox("Tagging team")
        team_layout = QHBoxLayout(team_group)
        self._home_radio = QRadioButton(f"{self._home.name} (home)")
        self._away_radio = QRadioButton(f"{self._away.name} (away)")
        self._home_radio.setChecked(True)
        self._team_radio_group = QButtonGroup(self)
        self._team_radio_group.addButton(self._home_radio)
        self._team_radio_group.addButton(self._away_radio)
        self._home_radio.toggled.connect(self._on_team_radio_changed)
        self._away_radio.toggled.connect(self._on_team_radio_changed)
        team_layout.addWidget(self._home_radio)
        team_layout.addWidget(self._away_radio)
        outer.addWidget(team_group)

        # ---- Roster list ----
        roster_group = QGroupBox("Roster")
        roster_layout = QVBoxLayout(roster_group)

        self._roster_list = QListWidget()
        self._roster_list.itemClicked.connect(self._on_roster_click)
        roster_layout.addWidget(self._roster_list)

        # Inline "+ Add player" form — kit + optional name + Add button.
        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Kit:"))
        self._kit_input = QSpinBox()
        self._kit_input.setRange(0, 99)
        self._kit_input.setSpecialValueText(" ")
        self._kit_input.setFixedWidth(56)
        add_row.addWidget(self._kit_input)
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Name (optional)")
        self._name_input.returnPressed.connect(self._on_add_player)
        add_row.addWidget(self._name_input, stretch=1)
        self._add_btn = QPushButton("Add")
        self._add_btn.clicked.connect(self._on_add_player)
        self._add_btn.setToolTip("Add to roster (Enter on the name field also adds)")
        add_row.addWidget(self._add_btn)
        roster_layout.addLayout(add_row)

        outer.addWidget(roster_group, stretch=1)

        # ---- Selected track section ----
        sel_group = QGroupBox("Selected track")
        sel_layout = QVBoxLayout(sel_group)
        self._selected_label = QLabel(
            "<i>Click a player on the video, then click a roster "
            "row above to assign.</i>"
        )
        self._selected_label.setTextFormat(Qt.TextFormat.RichText)
        self._selected_label.setWordWrap(True)
        sel_layout.addWidget(self._selected_label)

        unassign_row = QHBoxLayout()
        self._unassign_btn = QPushButton("Unassign track")
        self._unassign_btn.setToolTip("Remove the mapping (Ctrl+U)")
        self._unassign_btn.clicked.connect(self._on_unassign)
        self._unassign_btn.setEnabled(False)
        unassign_row.addStretch(1)
        unassign_row.addWidget(self._unassign_btn)
        sel_layout.addLayout(unassign_row)

        outer.addWidget(sel_group)

        # Keyboard shortcuts (Ctrl-modified so they don't fire while
        # typing in the kit / name fields).
        QShortcut(QKeySequence("Ctrl+U"), self, activated=self._on_unassign)

        self._refresh_roster()

    # ------------------------------------------------------------ public
    def set_selected_track(self, track_id: int) -> None:
        """Called by the main window when the operator clicks the video."""
        self._selected_track = track_id
        cv_side = dominant_team_side(self._con, self._match.id, track_id)

        # Look up existing mapping (if any) so the operator can see what's
        # already on this track.
        existing = next(
            (m for m in list_track_mappings(self._con, self._match.id)
             if m.track_id == track_id),
            None,
        )
        msg = (
            f"<b>track {track_id}</b> "
            f"<span style='color:#888;'>(CV side: "
            f"{cv_side if cv_side is not None else '—'})</span>"
        )
        if existing is not None:
            kit = existing.kit_number if existing.kit_number is not None else "?"
            tag = "home" if existing.is_home else "away"
            msg += (
                f"<br>Currently mapped → <b>{kit} {existing.player_name}</b>"
                f" <span style='color:#888;'>({tag})</span>"
            )
            self._unassign_btn.setEnabled(True)
        else:
            msg += "<br><i>Click a roster row to assign.</i>"
            self._unassign_btn.setEnabled(False)
        self._selected_label.setText(msg)

    def get_track_labels(self) -> dict[int, str]:
        """For VideoWidget overlay rendering — only labels tracks that
        have been mapped to a roster player. Untagged tracks (and any
        tracks on the OTHER team) keep showing their raw track_id."""
        out: dict[int, str] = {}
        for m in list_track_mappings(self._con, self._match.id):
            kit = m.kit_number if m.kit_number is not None else "?"
            name = m.player_name.strip()
            if len(name) > 12:
                name = name[:11] + "…"
            out[m.track_id] = f"{kit} {name}".strip()
        return out

    # ---------------------------------------------------------- handlers
    def _on_team_radio_changed(self, _checked: bool) -> None:
        # Both buttons emit toggled; act only on the now-selected one.
        if self._home_radio.isChecked():
            self._tagging_team = self._home
        else:
            self._tagging_team = self._away
        self._refresh_roster()

    def _on_add_player(self) -> None:
        kit_value = self._kit_input.value()
        kit = kit_value if kit_value > 0 else None
        name = self._name_input.text().strip() or None
        if kit is None and not name:
            # Empty form — silently ignore. Operator probably hit Enter
            # by accident.
            return
        get_or_create_player(
            self._con,
            team_id=self._tagging_team.id,
            kit_number=kit,
            name=name,
        )
        # Reset the inputs so the operator can keep adding without
        # cursor-managing manually. Focus stays on the kit input.
        self._kit_input.setValue(0)
        self._name_input.clear()
        self._kit_input.setFocus()
        self._refresh_roster()

    def _on_roster_click(self, item: QListWidgetItem) -> None:
        if self._selected_track is None:
            # Operator clicked a roster row without a selected track —
            # gentle reminder rather than a silent no-op.
            self._selected_label.setText(
                "<i>Click a player on the video first, then click a "
                "roster row to assign.</i>"
            )
            return
        player_id = int(item.data(Qt.ItemDataRole.UserRole))
        cv_side = dominant_team_side(
            self._con, self._match.id, self._selected_track,
        )
        # CV side may be None for tracks that never carried a team
        # classification (refs, ambiguous fragments). Fall back to a
        # team-side that matches the tagging team's home/away role —
        # arbitrary but stable; reports filter via the player_id chain.
        side = cv_side or (1 if self._tagging_team.is_home else 2)

        # Get the kit number from the roster entry the operator just clicked,
        # so the kit_number_in_match reflects what's stored on the roster.
        roster = list_roster_with_track_counts(
            self._con, self._match.id, self._tagging_team.id,
        )
        roster_entry = next((r for r in roster if r.player_id == player_id), None)
        kit_in_match = roster_entry.kit_number if roster_entry else None

        assign_track_to_player(
            self._con,
            match_id=self._match.id,
            track_id=self._selected_track,
            player_id=player_id,
            team_side=side,
            kit_number_in_match=kit_in_match,
        )
        # Refresh visuals immediately.
        self._refresh_roster()
        self.mappings_changed.emit()
        # Keep the same track selected in case the operator wants to
        # reconsider the assignment, but show the freshly-stored state.
        self.set_selected_track(self._selected_track)

    def _on_unassign(self) -> None:
        if self._selected_track is None:
            return
        unassign_track(self._con, self._match.id, self._selected_track)
        self._refresh_roster()
        self.mappings_changed.emit()
        self.set_selected_track(self._selected_track)

    # ---------------------------------------------------------- internals
    def _refresh_roster(self) -> None:
        roster = list_roster_with_track_counts(
            self._con, self._match.id, self._tagging_team.id,
        )
        self._roster_list.clear()
        for r in roster:
            kit_str = f"#{r.kit_number}" if r.kit_number is not None else "  —"
            name = r.name if r.name else "(no name)"
            count_str = (
                f" — {r.track_count} track{'s' if r.track_count != 1 else ''}"
                if r.track_count > 0 else ""
            )
            text = f"{kit_str:>4}  {name}{count_str}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, r.player_id)
            # Soft-grey hint for players with zero tracks yet.
            if r.track_count == 0:
                item.setForeground(Qt.GlobalColor.lightGray)
            self._roster_list.addItem(item)
        # Footer hint — useful for empty-state.
        if not roster:
            empty = QListWidgetItem(
                "(roster is empty — type kit + name below, then Add)"
            )
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            empty.setForeground(Qt.GlobalColor.gray)
            self._roster_list.addItem(empty)
