"""Ball position interpolation.

Phase 5. YOLO often misses the ball in 10-30% of frames; the ball moves
near-linearly between missed frames, so pandas.DataFrame.interpolate()
(method='linear') is good enough. Edge frames use bfill/ffill.

Input:  list of per-frame dicts with ball bbox or None.
Output: same list with all ball slots filled.
"""
from __future__ import annotations

import pandas as pd


def interpolate_ball_positions(
    ball_positions: list[dict[int, dict[str, list[float]]]],
) -> list[dict[int, dict[str, list[float]]]]:
    raise NotImplementedError("Ball interpolation — to be implemented in Phase 5.")
