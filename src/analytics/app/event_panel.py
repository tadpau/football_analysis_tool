"""Sidebar tab for hotkey event tagging (Phase 2d).

Workflow:

  1. Operator clicks a player on the video → primary track captured.
  2. Operator presses a hotkey (e.g. ``p`` for pass).
  3. An inline form appears with the fields that event_type needs:
        * For ``has_secondary`` events (pass, cross, tackle, foul,
          goal): "Receiver kit?" / "Tackled player kit?" / etc.
        * For ``has_success`` events (pass, shot, dribble, tackle,
          save): two buttons "Success" / "Failed", or 1/2 keys.
  4. Operator types kit and/or hits 1/2, then Enter to save.
  5. Event row written to DB; the recent-events list refreshes; form
     hides; panel waits for the next hotkey.

Hotkeys are bound to the EventPanel widget with
``ShortcutContext.WidgetWithChildrenShortcut`` so they don't fire
when the operator is typing in the *Players* tab's kit / name inputs.
The video widget itself doesn't take focus, so hotkey routing is
seamless during normal "scrub-and-tag" flow.

Recent events list is the "what did I just record" view. Click a
row to (Phase 2d.1) jump the video to that frame; for now it's
read-only with an Undo-last button at the bottom.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

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
    EventRow,
    EventType,
    MatchSummary,
    dominant_team_side,
    find_track_for_kit,
    get_or_create_frame_id,
    insert_event,
    list_event_types,
    list_recent_events,
    soft_delete_event,
)


# ---------------------------------------------------------------------------
# Inline form for "new event in progress"
# ---------------------------------------------------------------------------
class NewEventForm(QFrame):
    """Inline form shown when a hotkey fires; collects secondary +
    success then either ``Save``s or ``Cancel``s."""

    saved = pyqtSignal(dict)        # {"secondary_kit": int|None, "success": int|None}
    cancelled = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet(
            "NewEventForm { background-color: #232b34; border: 1px solid #3a4a5a; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self._title = QLabel("")
        self._title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self._title)

        # Secondary-player input row — hidden when the event_type doesn't
        # need one (throw-ins, corners, offsides).
        self._secondary_row = QHBoxLayout()
        self._secondary_label = QLabel("Receiver kit:")
        self._secondary_input = QSpinBox()
        self._secondary_input.setRange(0, 99)
        self._secondary_input.setSpecialValueText(" ")
        self._secondary_input.setFixedWidth(60)
        self._secondary_row.addWidget(self._secondary_label)
        self._secondary_row.addWidget(self._secondary_input)
        self._secondary_row.addStretch(1)
        self._secondary_widget = QWidget()
        self._secondary_widget.setLayout(self._secondary_row)
        layout.addWidget(self._secondary_widget)

        # Success / fail row — shown only for has_success events.
        success_row = QHBoxLayout()
        self._success_label = QLabel("Outcome:")
        self._success_btn = QPushButton("Success (1)")
        self._fail_btn = QPushButton("Failed (2)")
        self._success_btn.setCheckable(True)
        self._fail_btn.setCheckable(True)
        self._success_btn.clicked.connect(lambda: self._set_success(1))
        self._fail_btn.clicked.connect(lambda: self._set_success(0))
        success_row.addWidget(self._success_label)
        success_row.addWidget(self._success_btn)
        success_row.addWidget(self._fail_btn)
        success_row.addStretch(1)
        self._success_widget = QWidget()
        self._success_widget.setLayout(success_row)
        layout.addWidget(self._success_widget)

        # Action buttons.
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel (Esc)")
        self._cancel_btn.clicked.connect(self.cancelled.emit)
        self._save_btn = QPushButton("Save (Enter)")
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setDefault(True)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._save_btn)
        layout.addLayout(btn_row)

        self._success_value: int | None = None
        self._has_success = False
        self._has_secondary = False

        # Single-keystroke shortcuts for outcome (1/2). Local to the form.
        QShortcut(QKeySequence("1"), self,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
                  activated=lambda: self._set_success(1))
        QShortcut(QKeySequence("2"), self,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
                  activated=lambda: self._set_success(0))
        QShortcut(QKeySequence("Esc"), self,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
                  activated=self.cancelled.emit)
        QShortcut(QKeySequence("Return"), self,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
                  activated=self._on_save)

    def configure_for(self, event_type: EventType, primary_label: str) -> None:
        """Reset the form for a new event of the given type."""
        self._has_secondary = event_type.has_secondary
        self._has_success = event_type.has_success
        self._success_value = None
        self._success_btn.setChecked(False)
        self._fail_btn.setChecked(False)
        self._secondary_input.setValue(0)
        self._secondary_widget.setVisible(self._has_secondary)
        self._success_widget.setVisible(self._has_success)
        self._title.setText(
            f"<span style='color:#9ad;'>New {event_type.label}</span> "
            f"&nbsp;by&nbsp; {primary_label}"
        )
        # Auto-focus the most useful field.
        if self._has_secondary:
            self._secondary_input.setFocus()
            self._secondary_input.selectAll()
        else:
            self._save_btn.setFocus()

    def _set_success(self, value: int) -> None:
        self._success_value = value
        self._success_btn.setChecked(value == 1)
        self._fail_btn.setChecked(value == 0)

    def _on_save(self) -> None:
        # If the event needs a success flag and the operator hasn't picked
        # one, refuse to save — prompt by colouring the buttons.
        if self._has_success and self._success_value is None:
            self._success_btn.setStyleSheet("background-color: #6a3a3a;")
            self._fail_btn.setStyleSheet("background-color: #6a3a3a;")
            return
        # Reset the warning highlight.
        self._success_btn.setStyleSheet("")
        self._fail_btn.setStyleSheet("")
        kit_value = self._secondary_input.value()
        secondary_kit = kit_value if (self._has_secondary and kit_value > 0) else None
        self.saved.emit({
            "secondary_kit": secondary_kit,
            "success": self._success_value,
        })


# ---------------------------------------------------------------------------
# EventPanel — the second sidebar tab.
# ---------------------------------------------------------------------------
class EventPanel(QWidget):
    """Hotkey reference + inline new-event form + recent-events list."""

    event_logged = pyqtSignal()
    event_clicked = pyqtSignal(int)   # frame_number — for video seek

    def __init__(
        self,
        connection: sqlite3.Connection,
        match: MatchSummary,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._con = connection
        self._match = match
        self._event_types = list_event_types(connection)
        self._by_code = {et.code: et for et in self._event_types}
        self._by_hotkey = {et.hotkey.lower(): et for et in self._event_types if et.hotkey}

        # Per-tag operator state, refreshed by the main window.
        self._selected_track: int | None = None
        self._selected_label: str = "(no player selected)"
        self._current_frame: int = 0
        self._tagging_team_side: int | None = None  # filled by main window
        self._pending_event: EventType | None = None

        self.setFixedWidth(380)
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

        # ---- Selected-track strip ----
        self._selected_label_widget = QLabel(
            "<i>Click a player on the video, then press a hotkey.</i>"
        )
        self._selected_label_widget.setTextFormat(Qt.TextFormat.RichText)
        self._selected_label_widget.setWordWrap(True)
        outer.addWidget(self._selected_label_widget)

        # ---- Inline new-event form (hidden by default) ----
        self._form = NewEventForm()
        self._form.saved.connect(self._on_form_saved)
        self._form.cancelled.connect(self._on_form_cancelled)
        self._form.setVisible(False)
        outer.addWidget(self._form)

        # ---- Hotkey reference ----
        ref_group = QGroupBox("Hotkeys")
        ref_layout = QVBoxLayout(ref_group)
        for et in self._event_types:
            if not et.hotkey:
                continue
            line = f"<b>{et.hotkey.upper()}</b>  &nbsp; {et.label}"
            extras = []
            if et.has_success:
                extras.append("1=success / 2=fail")
            if et.has_secondary:
                extras.append("type kit for 2nd player")
            if extras:
                line += (
                    f" &nbsp; <span style='color:#888;'>({'; '.join(extras)})"
                    f"</span>"
                )
            lbl = QLabel(line)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            ref_layout.addWidget(lbl)
        outer.addWidget(ref_group)

        # ---- Recent events list ----
        recent_group = QGroupBox("Recent events")
        recent_layout = QVBoxLayout(recent_group)
        self._recent_list = QListWidget()
        self._recent_list.itemClicked.connect(self._on_recent_click)
        self._recent_list.setStyleSheet(
            "QListWidget { background-color: #141414; border: 1px solid #2a2a2a; }"
            "QListWidget::item { padding: 4px; }"
            "QListWidget::item:selected { background-color: #2a4d6a; }"
        )
        recent_layout.addWidget(self._recent_list)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self._undo_btn = QPushButton("Undo last (Ctrl+Z)")
        self._undo_btn.clicked.connect(self._on_undo)
        action_row.addWidget(self._undo_btn)
        recent_layout.addLayout(action_row)

        outer.addWidget(recent_group, stretch=1)

        # ---- Hotkeys ----
        # Bind every event_type's hotkey at the panel level. Context is
        # WidgetWithChildrenShortcut so they don't fire while the operator
        # is typing in the Players tab's kit/name inputs.
        for et in self._event_types:
            if not et.hotkey:
                continue
            sc = QShortcut(QKeySequence(et.hotkey), self,
                           context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
            # Capture event_type via default-arg trick (avoid late-binding bug).
            sc.activated.connect(lambda _et=et: self._on_hotkey(_et))
        QShortcut(QKeySequence("Ctrl+Z"), self,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
                  activated=self._on_undo)

        self._refresh_recent()

    # ------------------------------------------------------------ public
    def set_selected_track(self, track_id: int, label: str) -> None:
        """Called by the main window on every video click."""
        self._selected_track = track_id
        self._selected_label = label
        self._selected_label_widget.setText(
            f"Selected: <b>{label}</b> "
            f"<span style='color:#888;'>(track {track_id})</span>"
        )

    def set_current_frame(self, frame_number: int) -> None:
        """Called by the video widget on every frame change."""
        self._current_frame = frame_number

    def set_tagging_team_side(self, team_side: int | None) -> None:
        """Called by the main window when the tagging-team radio flips.

        Used to resolve "kit 7 secondary" → track_id on the same team.
        """
        self._tagging_team_side = team_side

    # ---------------------------------------------------------- handlers
    def _on_hotkey(self, event_type: EventType) -> None:
        # Need a primary actor for any event that's about a player doing
        # something. Throw-ins / corners are arguably set-pieces tied to
        # a frame rather than a player, but for v1 we still require the
        # operator to click someone (the player taking the throw-in).
        if self._selected_track is None:
            self._selected_label_widget.setText(
                "<span style='color:#cc6;'>Click a player on the "
                "video first, then press a hotkey.</span>"
            )
            return
        self._pending_event = event_type
        self._form.configure_for(event_type, self._selected_label)
        self._form.setVisible(True)

    def _on_form_saved(self, payload: dict) -> None:
        if self._pending_event is None or self._selected_track is None:
            self._form.setVisible(False)
            return

        # Resolve frame_id for the current video frame.
        frame_id = get_or_create_frame_id(
            self._con, self._match.id, self._current_frame,
        )
        if frame_id is None:
            # Operator scrubbed past n_frames_analysed somehow; refuse.
            self._selected_label_widget.setText(
                "<span style='color:#c66;'>Cannot save: this frame "
                "wasn't ingested.</span>"
            )
            self._form.setVisible(False)
            self._pending_event = None
            return

        # Resolve secondary kit → track_id on the same tagging team.
        secondary_track_id: int | None = None
        notes: str | None = None
        secondary_kit = payload.get("secondary_kit")
        if secondary_kit is not None and self._tagging_team_side is not None:
            secondary_track_id = find_track_for_kit(
                self._con,
                match_id=self._match.id,
                team_side=self._tagging_team_side,
                kit_number=secondary_kit,
            )
            if secondary_track_id is None:
                # Kit not yet mapped — preserve the intent in notes so
                # the operator can hand-fix later.
                notes = f"secondary kit {secondary_kit} (not yet mapped)"

        timestamp_ms = int(round(self._current_frame * 1000.0 / self._match.fps))
        insert_event(
            self._con,
            match_id=self._match.id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            event_type=self._pending_event.code,
            primary_track_id=self._selected_track,
            secondary_track_id=secondary_track_id,
            success=payload.get("success"),
            notes=notes,
        )

        self._form.setVisible(False)
        self._pending_event = None
        self._refresh_recent()
        self.event_logged.emit()

    def _on_form_cancelled(self) -> None:
        self._form.setVisible(False)
        self._pending_event = None

    def _on_undo(self) -> None:
        rows = list_recent_events(self._con, self._match.id, limit=1)
        if not rows:
            return
        soft_delete_event(self._con, rows[0].id)
        self._refresh_recent()
        self.event_logged.emit()

    def _on_recent_click(self, item: QListWidgetItem) -> None:
        frame = int(item.data(Qt.ItemDataRole.UserRole))
        self.event_clicked.emit(frame)

    # ---------------------------------------------------------- internals
    def _refresh_recent(self) -> None:
        rows = list_recent_events(self._con, self._match.id, limit=30)
        self._recent_list.clear()
        for r in rows:
            ts_s = r.timestamp_ms / 1000.0
            mm, ss = divmod(int(ts_s), 60)
            primary = (
                f"{r.primary_kit or '?'} {r.primary_player_name}"
                if r.primary_player_name else f"track {r.primary_track_id}"
            )
            text = f"{mm:02d}:{ss:02d}  {r.event_label:<10s}  {primary}"
            if r.secondary_track_id is not None or r.secondary_player_name:
                secondary = (
                    f"{r.secondary_kit or '?'} {r.secondary_player_name}"
                    if r.secondary_player_name else f"track {r.secondary_track_id}"
                )
                text += f"  → {secondary}"
            if r.success is not None:
                text += f"  ({'✓' if r.success else '✗'})"
            if r.notes:
                text += f"  [{r.notes}]"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, r.frame_number)
            self._recent_list.addItem(item)
        self._undo_btn.setEnabled(bool(rows))
