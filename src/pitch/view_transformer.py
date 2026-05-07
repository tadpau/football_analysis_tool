"""Pitch perspective transform — pixel coords → metric pitch coords.

Phase 7a. Side-camera broadcast: the visible patch of the pitch is a trapezoid
in image space. Given 4 reference points whose **real-world metric** position
on the pitch is known, OpenCV's getPerspectiveTransform gives us a homography
H such that  H · (px, py, 1)ᵀ ∝ (mx, my, 1)ᵀ.

Two ways to get the 4 image-space points:
  (a) Hand-pick once per clip (this implementation, fast for prototype).
  (b) Pitch-keypoint detector — second YOLO model that finds line intersections
      every frame (planned upgrade for clips with significant zoom changes).

This class doesn't care which path produced the points — feed it
`image_corners` (4 px coords) + `world_corners` (matching 4 metric coords) and
it precomputes H once. Then `transform_point(px)` returns metric (mx, my) or
None if the point lies outside the calibrated quad.

`add_transformed_positions_to_tracks` pulls `position_adjusted` (camera-
compensated foot position from Phase 6) and stores `position_transformed` —
metric coords on the canonical 105×68 pitch.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


class ViewTransformer:
    PITCH_LENGTH_M = PITCH_LENGTH_M
    PITCH_WIDTH_M = PITCH_WIDTH_M

    def __init__(
        self,
        image_corners: Sequence[Sequence[float]],
        world_corners: Sequence[Sequence[float]],
        clip_to_quad: bool = False,
    ):
        """
        Args:
            image_corners: 4 (px_x, px_y) points in the source image, ordered
                consistently with `world_corners` (e.g. TL, TR, BR, BL).
            world_corners: matching 4 (m_x, m_y) points on a canonical pitch
                whose origin is the bottom-left corner, x along the long edge.
            clip_to_quad: if True, points outside the calibration quadrilateral
                return None (legacy strict behaviour). Default False — a planar
                homography extrapolates cleanly beyond the quad, and strict
                clipping meant players outside the penalty box got no distance
                tracked at all, which is usually NOT what we want.
        """
        if len(image_corners) != 4 or len(world_corners) != 4:
            raise ValueError("Need exactly 4 image and 4 world corners.")

        self.image_corners = np.array(image_corners, dtype=np.float32)
        self.world_corners = np.array(world_corners, dtype=np.float32)
        self.clip_to_quad = clip_to_quad

        self.perspective_transform = cv2.getPerspectiveTransform(
            self.image_corners, self.world_corners
        )

    # ------------------------------------------------------------ point-wise
    def transform_point(self, point: Sequence[float]) -> tuple[float, float] | None:
        """Map an image-space point to world (m).

        When ``clip_to_quad`` is True, returns None outside the calibration
        quadrilateral (the old strict behaviour). By default, extrapolates
        beyond the quad — the pitch is planar so this stays well-behaved, but
        expect a few percent error at the far touchline.
        """
        px = (float(point[0]), float(point[1]))
        if self.clip_to_quad:
            inside = cv2.pointPolygonTest(
                self.image_corners.astype(np.int32).reshape(-1, 1, 2),
                px,
                measureDist=False,
            )
            if inside < 0:
                return None

        src = np.array([[px]], dtype=np.float32)        # shape (1, 1, 2)
        dst = cv2.perspectiveTransform(src, self.perspective_transform)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

    # ------------------------------------------------------- apply to tracks
    def add_transformed_positions_to_tracks(self, tracks: list[dict]) -> None:
        """For every detection with `position_adjusted`, add `position_transformed`.

        Detections outside the calibrated quad get `position_transformed = None`.
        Speed/distance step then knows to skip those frames for that track.
        """
        for ft in tracks:
            for cls, dets in ft.items():
                if not isinstance(dets, dict):
                    continue
                for _tid, info in dets.items():
                    pos = info.get("position_adjusted")
                    if pos is None:
                        continue
                    info["position_transformed"] = self.transform_point(pos)
