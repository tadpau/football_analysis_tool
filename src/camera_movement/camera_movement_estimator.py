"""Camera-motion compensation via sparse optical flow.

Phase 6. Broadcast cameras pan/zoom — a player standing still has bbox motion
from the camera alone. We estimate per-frame camera shift (dx, dy) by tracking
"good features to track" on **static** image regions (top + bottom strips,
where banners, advertising boards and crowd live) with Lucas-Kanade pyramidal
optical flow.

Why only top/bottom strips:
  * The middle of the frame is mostly grass and players — both are either
    featureless (grass) or themselves moving (players). Tracking features there
    biases the estimate toward player motion.
  * Side banners, the crowd, and goal frames are rigidly bolted to the world
    so any motion they show is camera motion, by definition.

Output:
  * `get_camera_movement(frames)` → list[(dx, dy)] of length n_frames; entry i
    is the per-frame *delta* (frame i camera position minus frame i-1).
    Entry 0 is (0, 0).
  * `add_adjusted_positions_to_tracks(tracks, movement)` → adds
    `position_adjusted` (foot position minus cumulative camera shift up to that
    frame) to every detection. Downstream homography uses *adjusted* positions
    so the world-coord mapping computed once on frame 0 stays valid as the
    camera pans.

Caching: the estimator is deterministic given input frames, so we pickle the
result like Tracker does — re-running team/possession iteration shouldn't
re-do optical flow.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np


# Lucas-Kanade params — pyramid levels handle larger pans without losing track.
_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# goodFeaturesToTrack params — tuned for broadcast-cam crowd/banner texture.
_FEATURE_PARAMS = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=3,
    blockSize=7,
)

# A camera shift smaller than this (px) is rounding noise — clamp to 0 so a
# locked-off camera reports exactly zero movement.
_MIN_DISTANCE_PX = 5.0

# Fraction of frame height treated as "static" top/bottom strip.
_STATIC_STRIP_FRAC = 0.10


class CameraMovementEstimator:
    def __init__(self, frame: np.ndarray):
        """Build the static-region mask once from the first frame's shape."""
        h, w = frame.shape[:2]
        strip = max(1, int(h * _STATIC_STRIP_FRAC))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:strip, :] = 255          # top strip — banner / score graphic
        mask[-strip:, :] = 255         # bottom strip — pitch-side ads
        self._feature_mask = mask
        self._feature_params = {**_FEATURE_PARAMS, "mask": mask}

        # Streaming state for feed_chunk — populated lazily on the first frame.
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._stream_started = False

    # ------------------------------------------------------------------ core
    def get_camera_movement(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | Path | None = None,
    ) -> list[list[float]]:
        """Per-frame [dx, dy] camera displacement (deltas, not cumulative).

        First entry is [0.0, 0.0] — frame 0 is the reference.
        """
        if read_from_stub and stub_path and Path(stub_path).exists():
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        movement: list[list[float]] = [[0.0, 0.0] for _ in frames]
        if len(frames) < 2:
            return movement

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, **self._feature_params)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            if prev_pts is None or len(prev_pts) < 4:
                # No reliable features — refresh and skip motion this frame.
                prev_pts = cv2.goodFeaturesToTrack(curr_gray, **self._feature_params)
                prev_gray = curr_gray
                continue

            new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **_LK_PARAMS
            )
            if new_pts is None or status is None:
                prev_pts = cv2.goodFeaturesToTrack(curr_gray, **self._feature_params)
                prev_gray = curr_gray
                continue

            good_old = prev_pts[status.flatten() == 1]
            good_new = new_pts[status.flatten() == 1]
            if len(good_old) < 4:
                prev_pts = cv2.goodFeaturesToTrack(curr_gray, **self._feature_params)
                prev_gray = curr_gray
                continue

            # Per-feature displacement — take the median for outlier robustness.
            deltas = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
            dx = float(np.median(deltas[:, 0]))
            dy = float(np.median(deltas[:, 1]))

            # Suppress sub-pixel jitter on a locked-off camera.
            if (dx * dx + dy * dy) ** 0.5 < _MIN_DISTANCE_PX:
                dx, dy = 0.0, 0.0
            else:
                # Re-seed features when the camera actually moved — old points
                # might have left the static strips.
                prev_pts = cv2.goodFeaturesToTrack(curr_gray, **self._feature_params)

            movement[i] = [dx, dy]
            prev_gray = curr_gray
            if (dx, dy) == (0.0, 0.0):
                # Reuse tracked points next frame — cheaper than re-detecting.
                prev_pts = good_new.reshape(-1, 1, 2)

        if stub_path:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(movement, f)

        return movement

    # ------------------------------------------------------------- streaming
    def feed_chunk(self, frames: list[np.ndarray]) -> list[list[float]]:
        """Stateful per-chunk camera-shift estimation.

        Returns ``[[dx, dy], ...]`` of length ``len(frames)`` — same semantics
        as :meth:`get_camera_movement` but designed to be called repeatedly on
        successive slices of a long video without holding all frames in memory.

        The very first frame of the very first chunk is the reference (returns
        ``[0.0, 0.0]``). Subsequent frames — within and across chunks — are
        compared against the previously seen frame, so estimates flow through
        chunk boundaries identically to a single big call.
        """
        out: list[list[float]] = []
        for frame in frames:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not self._stream_started:
                # First frame ever — establish the reference; no displacement.
                self._prev_gray = curr_gray
                self._prev_pts = cv2.goodFeaturesToTrack(
                    curr_gray, **self._feature_params
                )
                self._stream_started = True
                out.append([0.0, 0.0])
                continue

            dx, dy = 0.0, 0.0
            prev_pts = self._prev_pts
            if prev_pts is None or len(prev_pts) < 4:
                # No reliable features — refresh and report no motion.
                self._prev_pts = cv2.goodFeaturesToTrack(
                    curr_gray, **self._feature_params
                )
                self._prev_gray = curr_gray
                out.append([dx, dy])
                continue

            new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, curr_gray, prev_pts, None, **_LK_PARAMS
            )
            if new_pts is None or status is None:
                self._prev_pts = cv2.goodFeaturesToTrack(
                    curr_gray, **self._feature_params
                )
                self._prev_gray = curr_gray
                out.append([dx, dy])
                continue

            good_old = prev_pts[status.flatten() == 1]
            good_new = new_pts[status.flatten() == 1]
            if len(good_old) < 4:
                self._prev_pts = cv2.goodFeaturesToTrack(
                    curr_gray, **self._feature_params
                )
                self._prev_gray = curr_gray
                out.append([dx, dy])
                continue

            deltas = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
            dx = float(np.median(deltas[:, 0]))
            dy = float(np.median(deltas[:, 1]))
            if (dx * dx + dy * dy) ** 0.5 < _MIN_DISTANCE_PX:
                dx, dy = 0.0, 0.0
                # Static frame — reuse tracked points next iter.
                self._prev_pts = good_new.reshape(-1, 1, 2)
            else:
                # Real motion — re-seed because old points may have drifted
                # out of the static strips.
                self._prev_pts = cv2.goodFeaturesToTrack(
                    curr_gray, **self._feature_params
                )

            self._prev_gray = curr_gray
            out.append([dx, dy])

        return out

    # -------------------------------------------------------- apply to tracks
    @staticmethod
    def add_adjusted_positions_to_tracks(
        tracks: list[dict], camera_movement: list[list[float]]
    ) -> None:
        """Add `position_adjusted = foot_pos - cumulative_camera_shift` per detection.

        Cumulative shift at frame i = sum of all per-frame deltas up to and
        including frame i. Subtracting it gives positions in the frame-0
        coordinate frame, which is what the homography was computed against.
        """
        cum_x, cum_y = 0.0, 0.0
        for i, ft in enumerate(tracks):
            if i < len(camera_movement):
                cum_x += camera_movement[i][0]
                cum_y += camera_movement[i][1]
            for cls, dets in ft.items():
                if not isinstance(dets, dict):
                    continue
                for _tid, info in dets.items():
                    bbox = info.get("bbox")
                    if not bbox:
                        continue
                    x1, _y1, x2, y2 = bbox
                    foot = ((x1 + x2) / 2.0, float(y2))
                    info["position_adjusted"] = (foot[0] - cum_x, foot[1] - cum_y)
