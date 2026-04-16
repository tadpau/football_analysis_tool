"""Draw detection + tracking overlays onto frames.

Visual language (matches the Roboflow reference video):
  * player/GK/ref → ellipse at the feet + track-ID label
  * ball          → triangle pointing down at the ball
  * color coding  → per-class default; team colors will override in Phase 4b.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.bbox_utils import get_center, get_bbox_width


# BGR (OpenCV convention)
COLORS = {
    "player": (0, 0, 255),       # red
    "goalkeeper": (255, 0, 255), # magenta
    "referee": (0, 255, 255),    # yellow
    "ball": (0, 255, 0),         # green
}


def _draw_ellipse(
    frame: np.ndarray, bbox, color, track_id: int | None = None
) -> None:
    x1, _y1, x2, y2 = bbox
    cx, _ = get_center(bbox)
    width = get_bbox_width(bbox)
    cv2.ellipse(
        frame,
        center=(int(cx), int(y2)),
        axes=(int(width * 0.5), int(width * 0.18)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )
    if track_id is not None:
        label = str(track_id)
        rect_w, rect_h = 36, 18
        rx1 = int(cx - rect_w / 2)
        ry1 = int(y2 + 5)
        cv2.rectangle(
            frame, (rx1, ry1), (rx1 + rect_w, ry1 + rect_h), color, cv2.FILLED
        )
        cv2.putText(
            frame,
            label,
            (rx1 + 4, ry1 + rect_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )


def _draw_triangle(frame: np.ndarray, bbox, color) -> None:
    x1, y1, x2, _y2 = bbox
    cx = int((x1 + x2) / 2)
    top = int(y1) - 6
    pts = np.array(
        [[cx, top + 18], [cx - 10, top], [cx + 10, top]], dtype=np.int32
    )
    cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)


def draw_annotations(
    frames: list[np.ndarray], tracks: list[dict]
) -> list[np.ndarray]:
    """Return a new list of annotated frames; originals are unchanged."""
    out = []
    for frame, frame_tracks in zip(frames, tracks):
        canvas = frame.copy()
        for name in ("player", "goalkeeper", "referee"):
            for tid, info in frame_tracks.get(name, {}).items():
                _draw_ellipse(canvas, info["bbox"], COLORS[name], tid)
        for _tid, info in frame_tracks.get("ball", {}).items():
            _draw_triangle(canvas, info["bbox"], COLORS["ball"])
        out.append(canvas)
    return out
