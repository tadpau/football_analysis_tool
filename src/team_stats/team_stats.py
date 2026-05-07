"""Team-level analytics: total distance + spatial heatmaps.

Phase 8 — robust team-level metrics.

Why team-level (rather than per-player) matters here
----------------------------------------------------
Per-player numbers are sensitive to ID flicker — a ByteTrack ID swap during a
crossing splits one player's distance into two tracks. Team-level aggregates
are immune to that: a swap *within* a team doesn't change the team total, and
the team-aware stitcher already prevents cross-team merges.

Two functions in here:

* :func:`compute_team_distances` — sums each player's `total_distance_m` from
  :class:`SpeedAndDistanceEstimator.summary`, grouped by the player's dominant
  team across the clip. Returns kilometres per team.

* :func:`compute_team_heatmaps` + :func:`save_team_heatmap_png` — builds a 2D
  histogram of `position_transformed` (metric pitch coords) per team, then
  renders it as a PNG with a stylised pitch underlay. The histogram resolution
  is configurable; default 1 m × 1 m bins → 105×68 grid.

All three depend on a "dominant team per track" assignment computed from the
already-stamped `info["team"]` field on each detection (set by TeamAssigner).
We deliberately don't expose `dom_team` from the stitcher module because that
returns it transiently inside its merge loop — easier to recompute here than
to thread it through the call chain.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M


# Default histogram resolution. 1 m bins are coarse enough that a single
# noisy frame can't dominate a cell, and fine enough to show formation
# structure (defensive line, midfield band, attacking third).
HEATMAP_BIN_M = 1.0


# ---------------------------------------------------------------- dominant team
def compute_dominant_teams(tracks: list[dict]) -> dict[int, int]:
    """For each player track_id, return the dominant team across the clip.

    Mirrors the logic in :func:`team_aware_stitch._dominant_team` but applied
    to the full set of player track ids in one shot. Tracks with no team votes
    (refs that got mislabeled briefly, very short fragments) are simply absent
    from the returned dict, so callers should treat missing keys as
    "unassigned".
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    for ft in tracks:
        for tid, info in ft.get("player", {}).items():
            t = info.get("team")
            if t is not None:
                votes[tid][t] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items() if c}


# --------------------------------------------------------------- total distance
def compute_team_distances(
    sd_summary: dict[int, dict[str, float]],
    dominant_teams: dict[int, int],
) -> dict[int, float]:
    """Total distance covered by each team, in **kilometres**.

    Args:
        sd_summary: from :attr:`SpeedAndDistanceEstimator.summary` —
            ``{tid: {"total_distance_m": float, ...}}``.
        dominant_teams: track_id → 1 / 2 mapping (see
            :func:`compute_dominant_teams`).

    Returns:
        ``{1: km_team1, 2: km_team2}``. Teams with zero distance are still
        present in the dict (value 0.0) so the caller can iterate without
        defensive lookups.
    """
    totals: dict[int, float] = {1: 0.0, 2: 0.0}
    for tid, s in sd_summary.items():
        team = dominant_teams.get(tid)
        if team not in (1, 2):
            continue
        totals[team] += s.get("total_distance_m", 0.0)
    return {t: m / 1000.0 for t, m in totals.items()}


# -------------------------------------------------------------------- heatmaps
def compute_team_heatmaps(
    tracks: list[dict],
    dominant_teams: dict[int, int],
    bin_m: float = HEATMAP_BIN_M,
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
) -> dict[int, np.ndarray]:
    """2D occupancy histogram per team in metric pitch coords.

    The histogram is shape ``(ny, nx)`` with ``nx = pitch_length / bin_m`` and
    ``ny = pitch_width / bin_m``, oriented so that index ``[y, x]`` corresponds
    to pitch metric ``(x_m, y_m)`` along the long edge × the short edge. We
    intentionally clip points to the pitch rectangle: occasional small
    overshoots from extrapolation outside the calibrated quad (see
    ``ViewTransformer.clip_to_quad=False``) shouldn't create rogue cells far
    off the field.

    Args:
        tracks: per-frame tracks (post-Phase-7a so each player detection has
            ``position_transformed``).
        dominant_teams: track_id → team mapping.
        bin_m: bin size, metres per cell. 1.0 = 105×68 grid.

    Returns:
        ``{1: ndarray, 2: ndarray}``. Counts (not probabilities). Both teams
        always present so the caller can save each unconditionally.
    """
    nx = max(1, int(round(pitch_length_m / bin_m)))
    ny = max(1, int(round(pitch_width_m / bin_m)))
    hm = {1: np.zeros((ny, nx), dtype=np.float32),
          2: np.zeros((ny, nx), dtype=np.float32)}

    for ft in tracks:
        for tid, info in ft.get("player", {}).items():
            team = dominant_teams.get(tid)
            if team not in (1, 2):
                continue
            pos = info.get("position_transformed")
            if pos is None:
                continue
            x_m, y_m = float(pos[0]), float(pos[1])
            if not (0.0 <= x_m <= pitch_length_m):
                continue
            if not (0.0 <= y_m <= pitch_width_m):
                continue
            ix = min(nx - 1, int(x_m / bin_m))
            iy = min(ny - 1, int(y_m / bin_m))
            hm[team][iy, ix] += 1.0

    return hm


# ------------------------------------------------------------------- rendering
def _draw_pitch_lines(ax, pitch_length_m: float, pitch_width_m: float) -> None:
    """Minimal pitch markings — outline, halfway line, centre circle, both
    penalty boxes + 6-yard boxes. Numbers in metres on a 105×68 canonical
    pitch (FIFA spec). Drawn in white on top of the heatmap colormap so the
    pitch is always legible regardless of cell density."""
    import matplotlib.patches as mp

    L, W = pitch_length_m, pitch_width_m
    line = dict(color="white", linewidth=1.2, fill=False)

    # Outline
    ax.add_patch(mp.Rectangle((0, 0), L, W, **line))
    # Halfway line
    ax.plot([L / 2, L / 2], [0, W], color="white", linewidth=1.2)
    # Centre circle (9.15 m radius)
    ax.add_patch(mp.Circle((L / 2, W / 2), 9.15, **line))
    ax.plot([L / 2], [W / 2], marker="o", color="white", markersize=2)

    # Penalty boxes (16.5 × 40.32 m), 6-yard boxes (5.5 × 18.32 m), penalty spots
    pa_w = 40.32
    ga_w = 18.32
    for x0 in (0.0, L - 16.5):
        ax.add_patch(mp.Rectangle((x0, (W - pa_w) / 2), 16.5, pa_w, **line))
    for x0 in (0.0, L - 5.5):
        ax.add_patch(mp.Rectangle((x0, (W - ga_w) / 2), 5.5, ga_w, **line))
    for spot_x in (11.0, L - 11.0):
        ax.plot([spot_x], [W / 2], marker="o", color="white", markersize=2)


def save_team_heatmap_png(
    heatmap: np.ndarray,
    out_path: Path,
    title: str = "",
    team_color_bgr: tuple[int, int, int] | None = None,
    smooth_sigma: float = 1.5,
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
) -> None:
    """Render one team's occupancy heatmap to PNG with a pitch underlay.

    The raw histogram is heavily peaked (a stationary player adds ~24 counts/s
    to one cell). A small Gaussian blur (``smooth_sigma`` in cells) gives the
    cloud-of-presence look people associate with football heatmaps. Set
    ``smooth_sigma=0`` for the raw discrete histogram.

    ``team_color_bgr`` (B, G, R) tints the heatmap using a dark→bright ramp
    in that team's kit colour, so the two team PNGs are visually
    distinguishable at a glance. Falls back to ``hot`` if the kit colour
    would be too dark to read on the green pitch (e.g. all-black GK kits).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless — no GUI required
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    smoothed = heatmap
    if smooth_sigma > 0 and heatmap.sum() > 0:
        try:
            from scipy.ndimage import gaussian_filter  # type: ignore
            smoothed = gaussian_filter(heatmap, sigma=smooth_sigma)
        except ImportError:
            # scipy is optional; fall back to a simple separable box filter
            # whose width approximates the requested sigma. Heatmap reads
            # slightly more pixelated but the metric is the same.
            k = max(1, int(round(smooth_sigma * 2)))
            kernel = np.ones((2 * k + 1,), dtype=np.float32) / (2 * k + 1)
            tmp = np.apply_along_axis(
                lambda v: np.convolve(v, kernel, mode="same"), 1, heatmap
            )
            smoothed = np.apply_along_axis(
                lambda v: np.convolve(v, kernel, mode="same"), 0, tmp
            )

    # Build a per-team colormap that ramps transparent → kit-colour. Bright
    # enough to read on the dark-green pitch underlay; if the kit is too dark
    # (luminance < 60/255) we fall back to matplotlib's "hot" so the heatmap
    # stays legible.
    cmap = "hot"
    if team_color_bgr is not None:
        b, g, r = (max(0, min(255, int(c))) for c in team_color_bgr)
        # BT.601 luma — same weighting OpenCV uses.
        luma = 0.114 * b + 0.587 * g + 0.299 * r
        if luma >= 60:
            rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
            cmap = LinearSegmentedColormap.from_list(
                "team_kit",
                [(0, 0, 0, 0), (rn, gn, bn, 1.0)],
            )

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=120)
    ax.set_facecolor("#0e6b1f")  # pitch green underlay
    # Origin is bottom-left so y grows upward like real pitch coords.
    ax.imshow(
        smoothed,
        origin="lower",
        extent=(0, pitch_length_m, 0, pitch_width_m),
        cmap=cmap,
        alpha=0.85,
        interpolation="bilinear",
    )
    _draw_pitch_lines(ax, pitch_length_m, pitch_width_m)

    ax.set_xlim(-1, pitch_length_m + 1)
    ax.set_ylim(-1, pitch_width_m + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, color="white")
    fig.patch.set_facecolor("#0a3a13")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
