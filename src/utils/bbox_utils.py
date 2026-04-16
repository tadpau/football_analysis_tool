"""Bounding-box geometry helpers.

All bboxes are (x1, y1, x2, y2) in pixel coords unless stated otherwise.
"""
from __future__ import annotations

import math
from typing import Sequence

Bbox = Sequence[float]  # (x1, y1, x2, y2)
Point = tuple[float, float]


def get_center(bbox: Bbox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_foot_position(bbox: Bbox) -> Point:
    """Point on the ground where the player stands — bottom-center of the box."""
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def get_bbox_width(bbox: Bbox) -> float:
    x1, _y1, x2, _y2 = bbox
    return float(x2 - x1)


def measure_distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
