"""Match-level reports / dashboard view (Phase 3).

Sits as a third entry in ``MainWindow``'s view stack — operator opens
a match in the tagger, then clicks "Reports" in the toolbar to switch
to this view for the same match. ``← Tagger`` toolbar button goes
back.

Sections
--------

  1. **Match header** — teams, date, season, frame count
  2. **Match summary** — event counts + accuracy %s in a compact grid
  3. **Player table** — CV metrics (distance, top speed, minutes) +
     event counts (passes, shots, goals, tackles, fouls) per roster
     player, sorted by kit number
  4. **Refresh button** — re-runs the queries so reports stay live
     while the operator's still tagging in the other tab

Heatmaps and pass-network visualisations land in Phase 3.1.
"""
from __future__ import annotations

import sqlite3

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .heatmap_canvas import HeatmapCanvas
from .repository import (
    MatchStats,
    MatchSummary,
    PlayerStats,
    TeamInfo,
    compute_match_stats,
    compute_player_stats,
    derive_team_side,
    get_event_locations,
    get_match_teams,
    get_player_event_locations,
    get_player_world_positions,
    get_team_world_positions,
)


# ---------------------------------------------------------------------------
# Helpers for compact stat rendering
# ---------------------------------------------------------------------------
def _pct(num: int, denom: int) -> str:
    if denom <= 0:
        return "—"
    return f"{100.0 * num / denom:.0f}%"


def _ratio(num: int, denom: int) -> str:
    if denom <= 0:
        return "—"
    return f"{num}/{denom} ({_pct(num, denom)})"


# ---------------------------------------------------------------------------
# A small stat card — label on top, value on bottom, fixed-height.
# ---------------------------------------------------------------------------
class _StatCard(QWidget):
    def __init__(self, title: str, value: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(120)
        self.setStyleSheet(
            "_StatCard { background-color: #232b34; border: 1px solid #3a4a5a; "
            "             border-radius: 4px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        self._title = QLabel(title)
        self._title.setStyleSheet(
            "color: #88aabb; font-size: 11px; background: transparent;"
        )
        self._value = QLabel(value)
        vf = QFont()
        vf.setPointSize(16)
        vf.setBold(True)
        self._value.setFont(vf)
        self._value.setStyleSheet("color: #fff; background: transparent;")
        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, value: str) -> None:
        self._value.setText(value)


# ---------------------------------------------------------------------------
# Reports view
# ---------------------------------------------------------------------------
class ReportsWidget(QWidget):
    """Match dashboard — summary + per-player table for one match."""

    back_requested = pyqtSignal()

    # Column headers for the player table — rendering is handled by
    # ``_render_row`` which knows each column position explicitly. The
    # name column is the only string-aligned one; everything else is
    # right-aligned as a numeric cell.
    _PLAYER_HEADERS: list[str] = [
        "#", "Player", "Dist (m)", "Top km/h", "Min",
        "Passes", "Shots", "Goals", "Tackles", "Fouls",
    ]
    _NAME_COLUMN = 1

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
        self._team_for_player_table: TeamInfo = self._home
        # Drill-down state: when set, the heatmap shows only this
        # player's positions + events. None = full team view.
        self._filtered_player_id: int | None = None
        self._filtered_player_label: str = ""

        self.setStyleSheet(
            "QWidget { background-color: #1f1f1f; color: #ddd; } "
            "QGroupBox { border: 1px solid #333; border-radius: 4px; "
            "  margin-top: 14px; padding-top: 10px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; "
            "  padding: 0 4px; color: #999; } "
            "QTableWidget { background-color: #141414; gridline-color: #2a2a2a; "
            "  alternate-background-color: #181818; }"
            "QHeaderView::section { background-color: #2a2a2a; color: #ccc; "
            "  border: 0; padding: 6px; font-weight: bold; }"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        # ---- Header ----
        header_row = QHBoxLayout()
        title = QLabel(
            f"<h2 style='margin:0;'>{self._home.name} "
            f"<span style='color:#888;'>vs</span> {self._away.name}</h2>"
            f"<span style='color:#888;'>"
            f"{match.match_date} &middot; {match.season} &middot; "
            f"{match.n_frames:,} frames @ {match.fps:.0f} fps"
            f"</span>"
        )
        title.setTextFormat(Qt.TextFormat.RichText)
        header_row.addWidget(title, stretch=1)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip("Re-run queries against the DB")
        self._refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(self._refresh_btn)
        outer.addLayout(header_row)

        # ---- Summary cards (built once, values updated by refresh) ----
        summary_group = QGroupBox("Match summary")
        summary_outer = QVBoxLayout(summary_group)
        self._cards: dict[str, _StatCard] = {}
        for i, row_defs in enumerate([
            ["total_events", "passes", "shots", "goals", "tackles"],
            ["crosses", "dribbles", "fouls", "corners", "throw_ins"],
        ]):
            row = QHBoxLayout()
            for key in row_defs:
                card = _StatCard(self._card_title(key), "—")
                self._cards[key] = card
                row.addWidget(card, stretch=1)
            summary_outer.addLayout(row)
        outer.addWidget(summary_group)

        # ---- Team toggle for the player table ----
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(QLabel("Show players for:"))
        self._home_radio = QRadioButton(self._home.name)
        self._away_radio = QRadioButton(self._away.name)
        self._home_radio.setChecked(True)
        self._home_radio.toggled.connect(self._on_team_toggle)
        self._away_radio.toggled.connect(self._on_team_toggle)
        toggle_row.addWidget(self._home_radio)
        toggle_row.addWidget(self._away_radio)
        toggle_row.addStretch(1)
        outer.addLayout(toggle_row)

        # ---- Player table ----
        table_group = QGroupBox("Players")
        table_layout = QVBoxLayout(table_group)
        self._table = QTableWidget(0, len(self._PLAYER_HEADERS))
        self._table.setHorizontalHeaderLabels(self._PLAYER_HEADERS)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        # Clicking any cell selects the row; emits a signal we use to
        # drill the heatmap into that player.
        self._table.itemSelectionChanged.connect(self._on_table_row_selected)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # Player-name column stretches to fill remaining horizontal space.
        h.setSectionResizeMode(self._NAME_COLUMN, QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self._table)
        table_layout.addWidget(QLabel(
            "<span style='color:#888;'>Click a row to drill the heatmap "
            "into that player.</span>"
        ))
        outer.addWidget(table_group, stretch=1)

        # ---- Heatmap + event scatter ----
        heatmap_group = QGroupBox("Heatmap & events")
        heatmap_layout = QVBoxLayout(heatmap_group)

        # Top strip — current filter description + "Show team" button.
        # When a player is drilled into, the button becomes enabled and
        # clears the filter back to the team view.
        filter_row = QHBoxLayout()
        self._heatmap_filter_label = QLabel("")
        self._heatmap_filter_label.setTextFormat(Qt.TextFormat.RichText)
        filter_row.addWidget(self._heatmap_filter_label, stretch=1)
        self._show_team_btn = QPushButton("Show team")
        self._show_team_btn.setToolTip(
            "Clear the player filter and show the full team heatmap"
        )
        self._show_team_btn.clicked.connect(self._on_clear_player_filter)
        self._show_team_btn.setEnabled(False)
        filter_row.addWidget(self._show_team_btn)
        heatmap_layout.addLayout(filter_row)

        self._heatmap = HeatmapCanvas(self)
        self._heatmap.setMinimumHeight(360)
        heatmap_layout.addWidget(self._heatmap)
        outer.addWidget(heatmap_group, stretch=2)

        self.refresh()

    # ============================================================ public
    def refresh(self) -> None:
        """Re-query stats from the DB and re-render every widget."""
        self._refresh_summary()
        self._refresh_player_table()
        self._refresh_heatmap()

    # ============================================================ helpers
    @staticmethod
    def _card_title(key: str) -> str:
        return {
            "total_events": "Events tagged",
            "passes":       "Passes",
            "shots":        "Shots",
            "goals":        "Goals",
            "tackles":      "Tackles",
            "crosses":      "Crosses",
            "dribbles":     "Dribbles",
            "fouls":        "Fouls",
            "corners":      "Corners",
            "throw_ins":    "Throw-ins",
        }[key]

    def _refresh_summary(self) -> None:
        s: MatchStats = compute_match_stats(self._con, self._match.id)
        self._cards["total_events"].set_value(f"{s.total_events}")
        self._cards["passes"].set_value(
            _ratio(s.passes_completed, s.passes_attempted)
        )
        self._cards["shots"].set_value(
            _ratio(s.shots_on_target, s.shots)
        )
        self._cards["goals"].set_value(f"{s.goals}")
        self._cards["tackles"].set_value(
            _ratio(s.tackles_won, s.tackles_attempted)
        )
        self._cards["crosses"].set_value(
            _ratio(s.crosses_completed, s.crosses_attempted)
        )
        self._cards["dribbles"].set_value(
            _ratio(s.dribbles_completed, s.dribbles_attempted)
        )
        self._cards["fouls"].set_value(f"{s.fouls_committed}")
        self._cards["corners"].set_value(f"{s.corners}")
        self._cards["throw_ins"].set_value(f"{s.throw_ins}")

    def _refresh_player_table(self) -> None:
        rows = compute_player_stats(
            self._con, self._match.id, self._team_for_player_table.id,
        )
        self._table.setRowCount(len(rows))
        for row_idx, ps in enumerate(rows):
            for col_idx, text in enumerate(self._render_row(ps)):
                item = QTableWidgetItem(text)
                if col_idx != self._NAME_COLUMN:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                if col_idx == 0:
                    # Stash player_id for any future drill-down.
                    item.setData(Qt.ItemDataRole.UserRole, ps.player_id)
                self._table.setItem(row_idx, col_idx, item)

    @staticmethod
    def _render_row(ps: PlayerStats) -> list[str]:
        """One row of the player table, columns in the same order as
        :data:`_PLAYER_HEADERS`. Centralised here so the header list
        and the row renderer can never drift apart silently."""
        kit = str(ps.kit_number) if ps.kit_number is not None else "—"
        top_speed = (
            f"{ps.top_speed_kmh:.1f}" if ps.top_speed_kmh is not None else "—"
        )
        return [
            kit,
            ps.name,
            f"{ps.distance_m:.0f}",
            top_speed,
            f"{ps.minutes_on_pitch:.1f}",
            _ratio(ps.passes_completed, ps.passes_attempted),
            _ratio(ps.shots_on_target, ps.shots),
            str(ps.goals),
            _ratio(ps.tackles_won, ps.tackles_total),
            str(ps.fouls_committed),
        ]

    def _on_team_toggle(self, _checked: bool) -> None:
        # Both radios emit; act on the now-checked one.
        self._team_for_player_table = (
            self._home if self._home_radio.isChecked() else self._away
        )
        # Switching the team radio is an implicit "I want the team view"
        # gesture — clear any drill-down filter that might be active.
        self._filtered_player_id = None
        self._filtered_player_label = ""
        self._refresh_player_table()
        self._refresh_heatmap()

    def _on_table_row_selected(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        first_cell = self._table.item(rows[0].row(), 0)
        if first_cell is None:
            return
        player_id = first_cell.data(Qt.ItemDataRole.UserRole)
        if player_id is None:
            return
        # Cache the player's display name from the table itself — keeps
        # the heatmap title in sync with the table's truncation rules.
        kit_cell = self._table.item(rows[0].row(), 0)
        name_cell = self._table.item(rows[0].row(), self._NAME_COLUMN)
        kit = kit_cell.text() if kit_cell else "—"
        name = name_cell.text() if name_cell else "(unnamed)"
        self._filtered_player_id = int(player_id)
        self._filtered_player_label = f"#{kit} {name}".replace("#—", "")
        self._refresh_heatmap()

    def _on_clear_player_filter(self) -> None:
        self._filtered_player_id = None
        self._filtered_player_label = ""
        self._table.clearSelection()
        self._refresh_heatmap()

    def _refresh_heatmap(self) -> None:
        team = self._team_for_player_table

        # --- Drill-down view: a specific player ---
        if self._filtered_player_id is not None:
            positions = get_player_world_positions(
                self._con, self._match.id, self._filtered_player_id,
            )
            events = get_player_event_locations(
                self._con, self._match.id, self._filtered_player_id,
            )
            self._heatmap_filter_label.setText(
                f"Showing: <b>{self._filtered_player_label}</b> "
                f"<span style='color:#888;'>· "
                f"{len(positions):,} positions · {len(events)} events</span>"
            )
            self._show_team_btn.setEnabled(True)
            self._heatmap.render(
                positions, events,
                title=(
                    f"{self._filtered_player_label} — positions + events "
                    f"({len(events)} on pitch)"
                ),
            )
            return

        # --- Team view (default) ---
        # CV team_side is derived from the operator's track→player
        # mappings — that's the only place we know which kit colour
        # corresponds to which roster team. Fall back to home=1, away=2
        # if no tracks have been mapped yet.
        side = derive_team_side(self._con, self._match.id, team.id)
        if side is None:
            side = 1 if team.is_home else 2
        positions = get_team_world_positions(self._con, self._match.id, side)
        events = get_event_locations(self._con, self._match.id, team_side=side)
        self._heatmap_filter_label.setText(
            f"Showing: <b>{team.name}</b> "
            f"<span style='color:#888;'>(full team)</span>"
        )
        self._show_team_btn.setEnabled(False)
        self._heatmap.render(
            positions, events,
            title=f"{team.name} — occupancy + events ({len(events)} on pitch)",
        )
