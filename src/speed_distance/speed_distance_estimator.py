"""Per-player distance + speed (meters, km/h) — noise-robust version.

Phase 7b. Runs AFTER ViewTransformer has stamped `position_transformed`
(metric coords) on every detection.

Why the original 5-frame non-overlapping-window version was noisy:
  * Bbox bottom jitters ±2–4 px per frame due to detector + tracker noise.
  * The homography amplifies that jitter *perspectively*: 3 px at the far
    touchline ≈ 1–2 m of "motion" in world coords. Over a 0.2 s window that
    is ~25 km/h of phantom speed.
  * Non-overlapping windows also meant neighbouring windows could report very
    different numbers for the same player → visible jump when the window
    boundary crossed.

Fixes in this version:
  1. Gather per-track (frame_idx, world_pos) series — skipping None positions.
  2. Rolling *median* on world x / y over SMOOTH_WINDOW samples (median is
     robust to single-frame jitter, unlike mean).
  3. Total distance is cumulative over the smoothed series, so a player's
     total keeps growing even across short gaps (same track_id reappearing).
  4. Per-frame speed uses a **sliding lookback window** (SPEED_WINDOW_S of real
     time) — same lookback every frame, no boundary artifacts.
  5. Speed clipped to MAX_PLAUSIBLE_KMH (≈37 km/h elite sprint) so a residual
     spike doesn't render as a 200 km/h label.
  6. Sprint = any window where smoothed speed ≥ SPRINT_THRESHOLD_KMH; longest
     sprint tracked as max contiguous distance in sprint state.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict


WINDOW = 5                       # legacy name, retained for import compat
SPRINT_THRESHOLD_KMH = 20.0
MAX_PLAUSIBLE_KMH = 37.0         # Usain-Bolt territory; clip anything above
MIN_DISPLAY_KMH = 1.5            # below this, label speed 0.0 — it's detection jitter
# Per-step motion deadband for the distance accumulator. Any step between two
# smoothed samples whose implied speed is below this threshold is treated as
# jitter and NOT added to `distance_m`. This fixes the user-reported issue of
# a stationary player's meter counter climbing while they're just standing
# over the ball looking for a pass — the smoothing window can't fully
# suppress sub-pixel bbox bottom jitter amplified by perspective homography
# near the far touchline.
#
# 2.5 km/h (~0.7 m/s) is slower than a casual walk, so any real walking
# movement still accumulates; only genuine standing-still jitter is cut.
MIN_MOVEMENT_KMH = 2.5
SPEED_WINDOW_S = 1.0             # real-time lookback for speed calc
SMOOTH_WINDOW = 9                # samples (per track, not global frames)
TARGETED_CLASSES = ("player",)   # extend to "goalkeeper" to include keepers


class SpeedAndDistanceEstimator:
    def __init__(
        self,
        frame_window: int = WINDOW,           # kept for API compat, unused
        frame_rate: float = 24.0,
        speed_window_s: float = SPEED_WINDOW_S,
        smooth_window: int = SMOOTH_WINDOW,
        max_kmh: float = MAX_PLAUSIBLE_KMH,
    ):
        self.frame_window = frame_window
        self.frame_rate = frame_rate
        self.speed_window_frames = max(2, int(round(speed_window_s * frame_rate)))
        self.smooth_window = max(1, smooth_window)
        self.max_kmh = max_kmh
        # track_id -> {"total_distance_m": float, "longest_sprint_m": float}
        self.summary: dict[int, dict[str, float]] = {}

    # ------------------------------------------------------------ annotation
    def add_speed_and_distance_to_tracks(self, tracks: list[dict]) -> None:
        if not tracks or self.frame_rate <= 0:
            return

        for cls in TARGETED_CLASSES:
            series = self._gather_series(tracks, cls)
            for tid, pts in series.items():
                self._process_track(tracks, cls, tid, pts)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _gather_series(
        tracks: list[dict], cls: str
    ) -> dict[int, list[tuple[int, tuple[float, float]]]]:
        """Build, per track_id, a list of (frame_idx, (x_m, y_m)) — skipping
        frames where homography returned None (player outside calibrated quad)."""
        out: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
        for i, ft in enumerate(tracks):
            for tid, info in ft.get(cls, {}).items():
                pos = info.get("position_transformed")
                if pos is None:
                    continue
                out[tid].append((i, (float(pos[0]), float(pos[1]))))
        return out

    def _smooth(
        self, pts: list[tuple[int, tuple[float, float]]]
    ) -> list[tuple[int, tuple[float, float]]]:
        """Rolling median on x / y independently. Window is in *sample* count
        along the track, not global frames — handles gaps gracefully."""
        half = self.smooth_window // 2
        out: list[tuple[int, tuple[float, float]]] = []
        for j in range(len(pts)):
            lo, hi = max(0, j - half), min(len(pts), j + half + 1)
            xs = [pts[k][1][0] for k in range(lo, hi)]
            ys = [pts[k][1][1] for k in range(lo, hi)]
            out.append((pts[j][0], (statistics.median(xs), statistics.median(ys))))
        return out

    def _process_track(
        self,
        tracks: list[dict],
        cls: str,
        tid: int,
        pts: list[tuple[int, tuple[float, float]]],
    ) -> None:
        if len(pts) < 2:
            return

        smoothed = self._smooth(pts)

        # Cumulative total distance over the smoothed trajectory, with a
        # motion deadband: steps below MIN_MOVEMENT_KMH are treated as
        # standing-still jitter and not accumulated. Without this, a player
        # standing over the ball deliberating a pass still racks up meters
        # because sub-pixel bbox jitter x perspective = small world motion.
        cum_dist_by_frame: dict[int, float] = {smoothed[0][0]: 0.0}
        cum = 0.0
        for k in range(1, len(smoothed)):
            (fi0, p0), (fi1, p1) = smoothed[k - 1], smoothed[k]
            d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            dt = (fi1 - fi0) / self.frame_rate
            if dt > 0 and (d / dt) * 3.6 >= MIN_MOVEMENT_KMH:
                cum += d
            cum_dist_by_frame[fi1] = cum
        total_distance = cum

        # Per-sample speed via sliding lookback.
        W = self.speed_window_frames
        pos_by_frame = {fi: p for fi, p in smoothed}
        frames_sorted = [fi for fi, _ in smoothed]

        longest_sprint = 0.0
        sprint_start_cum: float | None = None   # cum_dist at the frame sprint began

        for idx, fi in enumerate(frames_sorted):
            target = fi - W
            prev_idx = None
            for k in range(idx - 1, -1, -1):
                if frames_sorted[k] <= target:
                    prev_idx = k
                    break
            if prev_idx is None:
                speed_kmh = 0.0          # not enough history yet
            else:
                fi_prev = frames_sorted[prev_idx]
                p_prev = pos_by_frame[fi_prev]
                p_cur = pos_by_frame[fi]
                dt = (fi - fi_prev) / self.frame_rate
                d = math.hypot(p_cur[0] - p_prev[0], p_cur[1] - p_prev[1])
                speed_kmh = (d / dt) * 3.6 if dt > 0 else 0.0
                speed_kmh = min(speed_kmh, self.max_kmh)
                if speed_kmh < MIN_DISPLAY_KMH:
                    speed_kmh = 0.0     # suppress jitter-floor "slow walk"

            # Sprint distance = delta of cumulative distance between the
            # frame the sprint began and the current frame. This avoids
            # double-counting the sliding lookback window.
            cum_here = cum_dist_by_frame.get(fi, 0.0)
            if speed_kmh >= SPRINT_THRESHOLD_KMH:
                if sprint_start_cum is None:
                    sprint_start_cum = cum_here
            else:
                if sprint_start_cum is not None:
                    run = cum_here - sprint_start_cum
                    if run > longest_sprint:
                        longest_sprint = run
                    sprint_start_cum = None

            det = tracks[fi].get(cls, {}).get(tid)
            if det is not None:
                det["speed_kmh"] = speed_kmh
                det["distance_m"] = cum_here

        if sprint_start_cum is not None:
            run = cum_dist_by_frame.get(frames_sorted[-1], 0.0) - sprint_start_cum
            if run > longest_sprint:
                longest_sprint = run

        self.summary[tid] = {
            "total_distance_m": total_distance,
            "longest_sprint_m": longest_sprint,
        }
