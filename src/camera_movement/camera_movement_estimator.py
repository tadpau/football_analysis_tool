"""Camera-motion compensation via sparse optical flow.

Phase 6. Broadcast cameras pan/zoom — a player standing still has bbox motion
from the camera alone. We estimate per-frame camera shift (dx, dy) by tracking
good features on static pitch regions (corners, line intersections) with
Lucas-Kanade, then subtract that shift from every player trajectory before
computing distance/speed.

Output: list[(dx, dy)] of length n_frames, plus helper to apply correction to
adjusted_position = (x - sum_dx_to_now, y - sum_dy_to_now).
"""
from __future__ import annotations


class CameraMovementEstimator:
    def __init__(self, frame):
        raise NotImplementedError("Camera movement — to be implemented in Phase 6.")

    def get_camera_movement(
        self, frames, read_from_stub: bool = False, stub_path=None
    ):
        raise NotImplementedError
