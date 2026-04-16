"""Pitch perspective transform.

Phase 7. All clips are side-camera, so the pitch is a trapezoid in image space.
We define 4 reference corners of a well-known pitch region (e.g. the visible
half of a touchline + the halfway line) in pixel coords, map them to metric
coords on a canonical 105 m × 68 m pitch, and compute a cv2.getPerspectiveTransform
homography H.

Every player's foot-position is then warped through H to get world coords in
meters — distance/speed metrics are computed in that space, not pixel space.

Two implementation paths:
  (a) Manual — hand-pick 4 corners per clip (fast, works for prototype).
  (b) Pitch-keypoint detector — second YOLO model trained to find pitch line
      intersections per frame (robust to camera motion, required for long matches).

Start with (a); upgrade to (b) once metrics are wired end-to-end.
"""
from __future__ import annotations

import numpy as np


class ViewTransformer:
    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M = 68.0

    def __init__(self):
        self.perspective_transform: np.ndarray | None = None
        raise NotImplementedError("View transformer — to be implemented in Phase 7.")

    def transform_point(self, point) -> np.ndarray | None:
        raise NotImplementedError
