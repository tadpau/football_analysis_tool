"""Post-process tracker output to drop spurious ball detections.

The custom YOLO sometimes misclassifies bright/white things on a player (shoes,
socks, jersey stripes) as the ball. Two signals identify these:

  * the detection center sits *inside* a player's bbox; AND
  * it jumps further from the previous accepted ball position than is
    physically plausible between frames.

Real dribbles satisfy (1) but not (2) — the ball stays near the dribbler. So we
only drop a detection when both fire. Dropped frames are emptied; downstream
interpolation will fill them in.
"""
from __future__ import annotations

import math


def _bbox_center(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _inside(point, bbox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def filter_ball_outliers(
    tracks: list[dict],
    max_jump_px: float = 180.0,
    max_gap_frames: int = 20,
) -> int:
    """Mutates tracks in place. Returns count of detections dropped.

    Args:
        max_jump_px: anything farther than this from the last accepted position
                     AND inside a player bbox is treated as misclassification.
        max_gap_frames: if it's been this many frames since the last accepted
                     ball, allow a far jump (the dribble may have moved on).
    """
    dropped = 0
    last_pos: tuple[float, float] | None = None
    last_seen_idx: int | None = None

    for i, ft in enumerate(tracks):
        ball = ft.get("ball", {}).get(1)
        if not ball:
            last_pos = None if (last_seen_idx is not None and i - last_seen_idx > max_gap_frames) else last_pos
            continue

        bcx, bcy = _bbox_center(ball["bbox"])

        inside_player = any(
            _inside((bcx, bcy), info["bbox"])
            for info in ft.get("player", {}).values()
        )

        if last_pos is None or last_seen_idx is None or (i - last_seen_idx) > max_gap_frames:
            jump = 0.0
        else:
            jump = math.hypot(bcx - last_pos[0], bcy - last_pos[1])

        if inside_player and jump > max_jump_px:
            # Looks like a shoe/jersey misclassification — drop it.
            ft["ball"] = {}
            dropped += 1
            continue

        last_pos = (bcx, bcy)
        last_seen_idx = i

    return dropped
