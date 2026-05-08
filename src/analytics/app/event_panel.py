"""Sidebar tab for hotkey event tagging — mouse-first workflow.

Operator never types kit numbers during tagging. The receiver (or
fouled player, assister, tackled opponent, etc.) is captured by
clicking them on the video, exactly the same way the primary actor
was captured. Outcomes (success / fail) are still single-keypresses
(``1`` / ``2``) for events that have a success/fail dimension.

State machine
-------------
::

    IDLE
      └─ click player on video       →  primary set, stay IDLE
      └─ press event hotkey
            ├─ event has_secondary   →  AWAITING_SECONDARY
            │     └─ click receiver
            │          ├─ event has_success → AWAITING_OUTCOME
            │          └─ else              → save & back to IDLE
            ├─ event has_success     →  AWAITING_OUTCOME
            └─ neither               →  save & back to IDLE

    AWAITING_OUTCOME
      └─ press 1 / 2                 →  save & back to IDLE
      └─ press Esc                   →  cancel & back to IDLE

After a save, the primary actor stays selected — common case is
multiple events from the same player in quick succession (a dribble
followed by a shot, etc.). The operator can simply press the next
hotkey without re-clicking.
"""
from __future__ import annotations

import sqlite3
from enum import Enum

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .repository import (
    EventType,
    MatchSummary,
    get_or_create_frame_id,
    insert_event,
    list_event_types,
    list_recent_events,
    soft_delete_event,
)


# ---------------------------------------------------------------------------
# State machine for the in-progress event.
# ---------------------------------------------------------------------------
class _State(Enum):
    IDLE = "idle"
    AWAITING_SECONDARY = "awaiting_secondary"
    AWAITING_OUTCOME = "awaiting_outcome"


# Banner colours per state — visual cue so the operator never guesses
# which key the panel is listening for next.
_BANNER_COLORS = {
    _State.IDLE:               "#1a3a4a",   # quiet teal
    _State.AWAITING_SECONDARY: "#5a3a1a",   # amber — "click the next player"
    _State.AWAITING_OUTCOME:   "#1a5a3a",   # green — "press 1 or 2"
}


class EventPanel(QWidget):
    """Mouse-first event tagging — second sidebar tab."""

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

        # ---- state ----
        self._state: _State = _State.IDLE
        self._primary_track: int | None = None
        self._primary_label: str = ""
        self._secondary_track: int | None = None
        self._secondary_label: str = ""
        self._pending_event: EventType | None = None
        self._current_frame: int = 0
        self._tagging_team_side: int | None = None

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

        # ---- Status banner (state-driven) ----
        self._banner = QFrame()
        self._banner.setFrameStyle(QFrame.Shape.StyledPanel)
        self._banner.setMinimumHeight(72)
        banner_layout = QVBoxLayout(self._banner)
        banner_layout.setContentsMargins(12, 10, 12, 10)
        self._banner_title = QLabel()
        self._banner_title.setTextFormat(Qt.TextFormat.RichText)
        self._banner_title.setStyleSheet("font-size: 14px;")
        self._banner_title.setWordWrap(True)
        self._banner_subtitle = QLabel()
        self._banner_subtitle.setTextFormat(Qt.TextFormat.RichText)
        self._banner_subtitle.setStyleSheet("color: #bbb; font-size: 12px;")
        self._banner_subtitle.setWordWrap(True)
        banner_layout.addWidget(self._banner_title)
        banner_layout.addWidget(self._banner_subtitle)
        outer.addWidget(self._banner)

        # ---- Hotkey reference ----
        ref_group = QGroupBox("Hotkeys")
        ref_layout = QVBoxLayout(ref_group)
        for et in self._event_types:
            if not et.hotkey:
                continue
            extras = []
            if et.has_secondary:
                extras.append("→ click 2nd player")
            if et.has_success:
                extras.append("then 1=success / 2=fail")
            extras_html = (
                f" <span style='color:#888;'>({'; '.join(extras)})</span>"
                if extras else ""
            )
            lbl = QLabel(
                f"<b>{et.hotkey.upper()}</b> &nbsp; {et.label}{extras_html}"
            )
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

        # ---- Hotkey wiring ----
        # Event hotkeys: P/S/C/D/T/F/I/K/G/V/O — each starts a new event.
        for et in self._event_types:
            if not et.hotkey:
                continue
            sc = QShortcut(QKeySequence(et.hotkey), self,
                           context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
            # Default-arg trick to avoid the late-binding closure bug.
            sc.activated.connect(lambda _et=et: self._on_event_hotkey(_et))

        # Outcome hotkeys: 1=success, 2=fail. Only meaningful in
        # AWAITING_OUTCOME state; the handler bails out otherwise.
        QShortcut(QKeySequence("1"), self,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
                  activated=lambda: self._on_outcome(1))
        QShortcut(QKeySequence("2"), self,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
                  activated=lambda: self._on_outcome(0))
        # Esc cancels an in-progress event.
        QShortcut(QKeySequence("Esc"), self,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
                  activated=self._on_cancel)
        # Undo last event.
        QShortcut(QKeySequence("Ctrl+Z"), self,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
                  activated=self._on_undo)

        self._refresh_recent()
        self._update_banner()

    # ============================================================ public
    def set_selected_track(self, track_id: int, label: str) -> None:
        """Called by the main window on every video click.

        Behaviour depends on state:
          * IDLE — sets primary actor for the next event.
          * AWAITING_SECONDARY — captures the receiver / fouled player /
            assist provider, then advances the state machine.
          * AWAITING_OUTCOME — silently ignored. The operator probably
            mis-clicked while we were waiting for 1/2; pressing 1/2
            commits the event with the existing primary + secondary.
        """
        if self._state == _State.IDLE:
            self._primary_track = track_id
            self._primary_label = label
            self._update_banner()
            return

        if self._state == _State.AWAITING_SECONDARY:
            # Self-pass guard — clicking the same player you just hot-keyed
            # for is almost certainly a mis-click. Refuse it and prompt.
            if track_id == self._primary_track:
                self._banner_subtitle.setText(
                    "Same player can't be both actor and receiver — "
                    "click someone else, or press Esc to cancel."
                )
                return
            self._secondary_track = track_id
            self._secondary_label = label
            assert self._pending_event is not None
            if self._pending_event.has_success:
                self._state = _State.AWAITING_OUTCOME
                self._update_banner()
            else:
                # has_secondary but not has_success → e.g. foul, goal.
                # The receiver click is the commit signal.
                self._save_event(success=None)
            return

        # AWAITING_OUTCOME — ignore stray clicks.

    def set_current_frame(self, frame_number: int) -> None:
        """Called by the video widget on every frame change."""
        self._current_frame = frame_number

    def set_tagging_team_side(self, team_side: int | None) -> None:
        """Called by the main window when the tagging-team radio flips.
        Stored only for diagnostic / future-use; not needed for the
        click-driven flow because secondary track_id is captured
        directly from the video click."""
        self._tagging_team_side = team_side

    # ============================================================ handlers
    def _on_event_hotkey(self, event_type: EventType) -> None:
        # No primary yet — refuse and prompt.
        if self._primary_track is None:
            self._banner_title.setText(
                "<span style='color:#cc6;'>Click a player first</span>"
            )
            self._banner_subtitle.setText(
                "Then press the event hotkey."
            )
            return
        # Already mid-event — ignore (operator probably double-pressed).
        if self._state != _State.IDLE:
            return

        self._pending_event = event_type
        self._secondary_track = None
        self._secondary_label = ""

        if event_type.has_secondary:
            self._state = _State.AWAITING_SECONDARY
        elif event_type.has_success:
            self._state = _State.AWAITING_OUTCOME
        else:
            # No secondary, no success → throw-in, corner, offside.
            # Hotkey alone is the commit.
            self._save_event(success=None)
            return
        self._update_banner()

    def _on_outcome(self, success_value: int) -> None:
        if self._state != _State.AWAITING_OUTCOME:
            return  # 1/2 pressed outside an outcome wait — ignore
        self._save_event(success=success_value)

    def _on_cancel(self) -> None:
        if self._state == _State.IDLE:
            return  # nothing to cancel
        self._pending_event = None
        self._secondary_track = None
        self._secondary_label = ""
        self._state = _State.IDLE
        self._update_banner()

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

    # ============================================================ internals
    def _save_event(self, *, success: int | None) -> None:
        if self._pending_event is None or self._primary_track is None:
            self._on_cancel()
            return

        frame_id = get_or_create_frame_id(
            self._con, self._match.id, self._current_frame,
        )
        if frame_id is None:
            # Operator scrubbed past n_frames_analysed — refuse and tell them.
            self._banner_title.setText(
                "<span style='color:#c66;'>Cannot save</span>"
            )
            self._banner_subtitle.setText(
                "This frame wasn't ingested. Move to a frame within the clip."
            )
            self._on_cancel()
            return

        timestamp_ms = int(round(self._current_frame * 1000.0 / self._match.fps))
        insert_event(
            self._con,
            match_id=self._match.id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            event_type=self._pending_event.code,
            primary_track_id=self._primary_track,
            secondary_track_id=self._secondary_track,
            success=success,
            notes=None,
        )

        self._pending_event = None
        self._secondary_track = None
        self._secondary_label = ""
        self._state = _State.IDLE
        self._refresh_recent()
        self.event_logged.emit()
        self._update_banner()
        # Primary stays selected — operator usually tags multiple events
        # for the same player in a row.

    def _update_banner(self) -> None:
        bg = _BANNER_COLORS[self._state]
        # Reuse the QFrame's stylesheet so we get a nice solid colour
        # regardless of theme.
        self._banner.setStyleSheet(
            f"QFrame {{ background-color: {bg}; "
            f"  border: 1px solid #555; border-radius: 4px; }}"
        )

        if self._state == _State.IDLE:
            if self._primary_track is None:
                self._banner_title.setText(
                    "<i>Click a player on the video to start.</i>"
                )
                self._banner_subtitle.setText("")
            else:
                self._banner_title.setText(
                    f"Selected: <b>{self._primary_label}</b> "
                    f"<span style='color:#888;'>(track "
                    f"{self._primary_track})</span>"
                )
                self._banner_subtitle.setText(
                    "Press an event hotkey (P/S/C/D/T/F/I/K/G/V/O), "
                    "or click a different player."
                )
            return

        assert self._pending_event is not None
        if self._state == _State.AWAITING_SECONDARY:
            self._banner_title.setText(
                f"<b>{self._pending_event.label}</b> by "
                f"<b>{self._primary_label}</b>"
            )
            secondary_role = self._secondary_role_label(self._pending_event)
            self._banner_subtitle.setText(
                f"Click the <b>{secondary_role}</b> on the video "
                f"&nbsp;·&nbsp; <span style='color:#aaa;'>Esc to cancel</span>"
            )
            return

        # AWAITING_OUTCOME
        head = (
            f"<b>{self._pending_event.label}</b>: "
            f"<b>{self._primary_label}</b>"
        )
        if self._secondary_track is not None:
            head += f" → <b>{self._secondary_label}</b>"
        self._banner_title.setText(head)
        s_options = self._success_option_labels(self._pending_event)
        self._banner_subtitle.setText(
            f"Press <b>1</b> = {s_options[0]} &nbsp;·&nbsp; "
            f"<b>2</b> = {s_options[1]} &nbsp;·&nbsp; "
            f"<span style='color:#aaa;'>Esc to cancel</span>"
        )

    @staticmethod
    def _secondary_role_label(et: EventType) -> str:
        # Surface a more specific noun than "secondary player" so the
        # operator knows what they're being asked for.
        return {
            "pass":   "receiver",
            "cross":  "receiver",
            "tackle": "opponent",
            "foul":   "fouled player",
            "goal":   "assist provider",
        }.get(et.code, "second player")

    @staticmethod
    def _success_option_labels(et: EventType) -> tuple[str, str]:
        # Custom labels for the two outcome options so 1/2 are
        # event-aware.
        return {
            "pass":    ("completed",  "incomplete"),
            "shot":    ("on target",  "off target"),
            "cross":   ("connected",  "missed"),
            "dribble": ("succeeded",  "lost ball"),
            "tackle":  ("won ball",   "missed"),
            "save":    ("saved",      "let in"),
        }.get(et.code, ("success", "fail"))

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
