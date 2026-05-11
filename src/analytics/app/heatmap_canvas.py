"""Pitch heatmap + event-scatter overlay, embedded as a Qt widget.

Wraps a matplotlib ``FigureCanvasQTAgg`` so the reports view can hand
it raw (x, y) world coords and event records and get a rendered pitch
back. No file I/O — figure renders in-memory and repaints on demand.

The visual language mirrors what ``src.team_stats.team_stats``
produces for the offline PNG reports: dark-green pitch underlay,
white pitch lines, ``hot`` colormap heatmap with light gaussian
smoothing. Adding the scatter on top of the heatmap gives the
operator both "where the team spent time" AND "where specific events
happened" in one view.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .repository import EventLocation


# Canonical pitch — pulled from src.pitch.PITCH_LENGTH_M / PITCH_WIDTH_M,
# hard-coded here to avoid the import dragging in the whole CV stack.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Heatmap bin size in metres. 1 m bins are coarse enough that a single
# noisy frame can't dominate a cell, fine enough to show defensive line
# / attacking-third structure on a 105×68 canvas.
HEATMAP_BIN_M = 1.0
HEATMAP_SMOOTH_SIGMA = 1.5

# Per-event-type marker colours. Picked so the success/fail variants
# stay readable on top of the warm-colour heatmap.
_EVENT_COLORS = {
    "pass":     "#3aa7ff",   # blue
    "shot":     "#ff3838",   # red
    "goal":     "#ffd13a",   # yellow
    "cross":    "#a070ff",   # purple
    "dribble":  "#3affc1",   # cyan
    "tackle":   "#ff8c3a",   # orange
    "foul":     "#ff3a8c",   # pink
    "save":     "#56ff3a",   # green
    "throw_in": "#888888",   # grey
    "corner":   "#bbbbbb",   # light grey
    "offside":  "#666666",   # dark grey
}


# --------------------------------------------------------------------------
# Pure functions — no matplotlib imports needed.
# --------------------------------------------------------------------------
def _occupancy_histogram(
    positions: list[tuple[float, float]],
) -> np.ndarray:
    """2D histogram of (x, y) world points binned at 1 m × 1 m."""
    if not positions:
        nx = int(round(PITCH_LENGTH_M / HEATMAP_BIN_M))
        ny = int(round(PITCH_WIDTH_M / HEATMAP_BIN_M))
        return np.zeros((ny, nx), dtype=np.float32)
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    # Clip to the pitch — extrapolated CV positions sometimes land off-pitch
    # and we don't want them as rogue cells far outside.
    xs = np.clip(xs, 0, PITCH_LENGTH_M)
    ys = np.clip(ys, 0, PITCH_WIDTH_M)
    H, _xe, _ye = np.histogram2d(
        xs, ys,
        bins=[int(round(PITCH_LENGTH_M / HEATMAP_BIN_M)),
              int(round(PITCH_WIDTH_M / HEATMAP_BIN_M))],
        range=[[0, PITCH_LENGTH_M], [0, PITCH_WIDTH_M]],
    )
    return H.T.astype(np.float32)   # imshow expects (rows=y, cols=x)


def _smooth_histogram(h: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or h.sum() == 0:
        return h
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
        return gaussian_filter(h, sigma=sigma)
    except ImportError:
        # Box-filter fallback if scipy isn't around. Slightly more
        # pixelated heatmap; same metric.
        k = max(1, int(round(sigma * 2)))
        kernel = np.ones((2 * k + 1,), dtype=np.float32) / (2 * k + 1)
        tmp = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="same"), 1, h,
        )
        return np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="same"), 0, tmp,
        )


def _draw_pitch_lines(ax) -> None:
    """Minimal FIFA-spec pitch markings on a 105×68 metric pitch:
    outline, halfway line, centre circle, penalty boxes, 6-yard boxes,
    penalty spots. White on dark green so it stays readable on top of
    the warm heatmap."""
    import matplotlib.patches as mp
    L, W = PITCH_LENGTH_M, PITCH_WIDTH_M
    line = dict(color="white", linewidth=1.2, fill=False)
    ax.add_patch(mp.Rectangle((0, 0), L, W, **line))
    ax.plot([L / 2, L / 2], [0, W], color="white", linewidth=1.2)
    ax.add_patch(mp.Circle((L / 2, W / 2), 9.15, **line))
    ax.plot([L / 2], [W / 2], marker="o", color="white", markersize=2)
    pa_w, ga_w = 40.32, 18.32
    for x0 in (0.0, L - 16.5):
        ax.add_patch(mp.Rectangle((x0, (W - pa_w) / 2), 16.5, pa_w, **line))
    for x0 in (0.0, L - 5.5):
        ax.add_patch(mp.Rectangle((x0, (W - ga_w) / 2), 5.5, ga_w, **line))
    for spot_x in (11.0, L - 11.0):
        ax.plot([spot_x], [W / 2], marker="o", color="white", markersize=2)


# --------------------------------------------------------------------------
# Qt widget.
# --------------------------------------------------------------------------
class HeatmapCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas pre-configured for pitch heatmaps.

    Single method ``render(positions, events)`` re-renders the whole
    figure — cheap enough to call on every team-toggle / refresh.
    """

    def __init__(self, parent=None, *, figsize=(9.5, 6.2)) -> None:
        fig = Figure(figsize=figsize, dpi=100, facecolor="#0a3a13")
        super().__init__(fig)
        if parent is not None:
            self.setParent(parent)
        self._ax = fig.add_subplot(111)
        # The figure has a hard-tinted background; the axes get the
        # pitch-green underlay.
        self._ax.set_facecolor("#0e6b1f")
        self._ax.set_aspect("equal")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_xlim(-1, PITCH_LENGTH_M + 1)
        self._ax.set_ylim(-1, PITCH_WIDTH_M + 1)
        fig.tight_layout()

    def render(
        self,
        positions: list[tuple[float, float]],
        events: list[EventLocation] | None = None,
        title: str | None = None,
    ) -> None:
        self._ax.clear()
        self._ax.set_facecolor("#0e6b1f")
        self._ax.set_aspect("equal")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_xlim(-1, PITCH_LENGTH_M + 1)
        self._ax.set_ylim(-1, PITCH_WIDTH_M + 1)

        # Heatmap layer.
        if positions:
            h = _smooth_histogram(
                _occupancy_histogram(positions), HEATMAP_SMOOTH_SIGMA,
            )
            self._ax.imshow(
                h, origin="lower",
                extent=(0, PITCH_LENGTH_M, 0, PITCH_WIDTH_M),
                cmap="hot", alpha=0.65, interpolation="bilinear",
            )

        _draw_pitch_lines(self._ax)

        # Event scatter layer. Successful events get a filled marker;
        # failed events get an outline-only marker of the same colour
        # so the operator can see at-a-glance where things broke down.
        if events:
            for et_code, color in _EVENT_COLORS.items():
                sub = [e for e in events if e.event_type == et_code]
                if not sub:
                    continue
                xs = [e.x_world for e in sub]
                ys = [e.y_world for e in sub]
                # Successful vs failed vs no-success-concept.
                successes = [e.success for e in sub]
                for i, s in enumerate(successes):
                    edge = color
                    if s == 1:
                        face = color
                    elif s == 0:
                        face = "none"
                    else:
                        face = color
                    self._ax.scatter(
                        xs[i], ys[i],
                        c=face if face != "none" else "none",
                        edgecolors=edge,
                        linewidths=1.2,
                        s=42,
                        marker="o",
                        zorder=5,
                    )

        if title:
            self._ax.set_title(title, color="white", fontsize=11)
        self.figure.tight_layout()
        self.draw_idle()
