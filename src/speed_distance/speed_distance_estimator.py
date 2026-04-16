"""Distance + speed per player in meters, km/h.

Phase 7. Runs AFTER ViewTransformer has added metric coords to each track.

Algorithm:
  - For each track, compute rolling distance over a WINDOW of frames (e.g. 5)
    to smooth out detection jitter.
  - speed_kmh = (distance_m / (WINDOW / fps)) * 3.6
  - total_distance_m accumulates across the clip.
  - Longest sprint = max contiguous run where speed > SPRINT_THRESHOLD_KMH.
"""
from __future__ import annotations


WINDOW = 5               # frames
SPRINT_THRESHOLD_KMH = 20.0


class SpeedAndDistanceEstimator:
    def __init__(self, frame_window: int = WINDOW, frame_rate: float = 24.0):
        self.frame_window = frame_window
        self.frame_rate = frame_rate
        raise NotImplementedError("Speed/distance — to be implemented in Phase 7.")

    def add_speed_and_distance_to_tracks(self, tracks) -> None:
        raise NotImplementedError
