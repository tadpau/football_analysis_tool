"""Ball position interpolation.

YOLO often misses the ball in 10-30% of frames (tiny, motion-blurred, occluded).
The ball moves near-linearly between frames, so pandas linear interpolation is
good enough and keeps the implementation trivial.

Input shape:
    ball_positions[frame_idx] is one of:
        {}                                      — ball missed this frame
        {1: {"bbox": [x1, y1, x2, y2], ...}}    — ball detected (id always 1)

Output shape: same structure, but every frame has the key. Leading/trailing
gaps are filled with bfill/ffill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def interpolate_ball_positions(
    ball_positions: list[dict],
) -> list[dict]:
    rows = []
    for fp in ball_positions:
        bbox = fp.get(1, {}).get("bbox")
        if bbox is None:
            rows.append([np.nan] * 4)
        else:
            rows.append(list(bbox))
    df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])
    df = df.interpolate(method="linear").bfill().ffill()

    out: list[dict] = []
    for (_, row), original in zip(df.iterrows(), ball_positions):
        if row.isna().any():
            out.append({})
            continue
        # Preserve original confidence if we had one, else mark as interpolated.
        original_info = original.get(1, {})
        was_detected = "bbox" in original_info
        out.append({
            1: {
                "bbox": [float(x) for x in row.tolist()],
                "confidence": original_info.get("confidence", 0.0),
                "interpolated": not was_detected,
            }
        })
    return out
