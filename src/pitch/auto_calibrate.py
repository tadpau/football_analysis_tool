"""Per-frame homography from detected pitch landmarks.

If the YOLO model has been trained to detect named pitch landmarks (corner
flags, centre spot, penalty spots, etc.), every frame carries enough
information to recover the camera→pitch homography on its own — no manual
``--calibration`` JSON needed, no fixed-camera assumption, robust to pan/
zoom/cuts.

The flow:

  1. Each detected landmark is paired with its **known world coordinate**
     from :data:`LANDMARK_WORLD_COORDS`. Singletons (centre spot, penalty
     spots) map directly. ``corner_flag`` is ambiguous — there are four on
     the pitch — so we disambiguate by image quadrant (top-left in image
     ⇒ far-left in world for our broadcast-style camera angles).

  2. Once we have ≥ 4 (img → world) point pairs, ``cv2.findHomography``
     with RANSAC fits a 3×3 H matrix robust to a single mis-detection.

  3. A short ring buffer averages the most recent N H matrices to suppress
     per-frame jitter. One genuinely-bad fit can't yank distances around;
     a sustained change (genuine pan) propagates within ~½ second.

The module degrades gracefully — if a frame has < 4 landmark detections
the function returns ``None`` and the caller is expected to fall back to
the previous frame's H matrix or a manual calibration if available.

Usage::

    from src.pitch.auto_calibrate import LandmarkCalibrator

    cal = LandmarkCalibrator(frame_width=1920, frame_height=1080)
    for frame_tracks in tracks:
        H = cal.update(frame_tracks)            # 3x3 or None
        if H is not None:
            ... apply H to player foot positions ...

The class is a thin wrapper around the pure function
:func:`compute_homography_from_landmarks` plus the temporal-smoothing
state. Use the function directly for one-shot fits.
"""
from __future__ import annotations

from collections import deque

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "src.pitch.auto_calibrate requires opencv-python (already a "
        "project dep — install it in your venv)."
    ) from e


# Canonical 105×68 m pitch — origin at bottom-left, +x along the long axis.
# These coordinates are the *fixed* world positions of every landmark we
# might detect. The detector's job is to find them in image space; the
# homography solves for the mapping.
#
# Corner flags are stored as a list (4 of them) and disambiguated by the
# detection's image-space quadrant — see _resolve_corner_world_coord below.
LANDMARK_WORLD_COORDS: dict[str, tuple[float, float]] = {
    "center_spot": (52.5, 34.0),
    "penalty_spot_left": (11.0, 34.0),
    "penalty_spot_right": (94.0, 34.0),
    # Tier-2 line-intersection landmarks — included so the table is
    # complete; the detector doesn't have to predict all of them. Only
    # the names that appear in frame_tracks contribute matches.
    "halfway_far": (52.5, 68.0),
    "halfway_near": (52.5, 0.0),
    "circle_far": (52.5, 43.15),
    "circle_near": (52.5, 24.85),
    "pbox_L_far_outer": (0.0, 54.16),
    "pbox_L_far_inner": (16.5, 54.16),
    "pbox_L_near_inner": (16.5, 13.84),
    "pbox_L_near_outer": (0.0, 13.84),
    "pbox_R_far_outer": (105.0, 54.16),
    "pbox_R_far_inner": (88.5, 54.16),
    "pbox_R_near_inner": (88.5, 13.84),
    "pbox_R_near_outer": (105.0, 13.84),
}


# All four corner-flag world positions. The matching one is chosen by the
# corner detection's image quadrant.
CORNER_FLAG_WORLD = {
    "far_left":  (0.0, 68.0),
    "far_right": (105.0, 68.0),
    "near_left": (0.0, 0.0),
    "near_right": (105.0, 0.0),
}


# Default RANSAC reprojection threshold. 3 px of slack is a comfortable
# fit at 1080p — landmark detections rarely drift more than that, but
# occasional mis-localised detections are excluded as outliers.
DEFAULT_RANSAC_THRESHOLD_PX = 3.0


# Default temporal-smoothing window. 15 frames @ 30 fps = ½ s — long
# enough to average out one-frame jitter, short enough that a genuine
# pan propagates fast.
DEFAULT_SMOOTHING_FRAMES = 15


# Minimum match count for a homography fit. cv2.findHomography needs ≥ 4
# but at exactly 4 there's no redundancy for RANSAC — one bad point and
# the whole fit fails. 5 gives one degree of robustness; production
# systems aim for ≥ 6.
MIN_MATCHES_FOR_FIT = 4


def _resolve_corner_world_coord(
    cx: float, cy: float, frame_width: int, frame_height: int,
) -> tuple[float, float]:
    """Map a corner-flag detection at image-space ``(cx, cy)`` to the
    world coord of one of the four pitch corners using the image quadrant.

    Assumes the broadcast camera convention: the *far* touchline is in
    the upper half of the image, the *near* touchline in the lower half.
    Works for centred cameras; FAILS for side-of-pitch cameras where
    both far corners can sit in the same image half. Use
    :func:`_resolve_corner_via_bootstrap` instead when a bootstrap
    homography is available.
    """
    far = cy < frame_height / 2
    left = cx < frame_width / 2
    if far and left:
        return CORNER_FLAG_WORLD["far_left"]
    if far and not left:
        return CORNER_FLAG_WORLD["far_right"]
    if not far and left:
        return CORNER_FLAG_WORLD["near_left"]
    return CORNER_FLAG_WORLD["near_right"]


def _resolve_corner_via_bootstrap(
    cx: float, cy: float,
    bootstrap_H_i2w: np.ndarray,
    max_world_dist_m: float,
) -> tuple[float, float] | None:
    """Match a detected corner_flag at ``(cx, cy)`` to the nearest world
    corner, by projecting it INTO world space via ``bootstrap_H``.

    This is the inverse of the more obvious "project world corners to
    image and match nearest pixel" approach — and it's significantly
    more robust to bootstrap-H extrapolation error. Reasoning:

      * ``bootstrap_H`` is well-conditioned for image points *inside*
        the manually-calibrated quad. It can extrapolate to arbitrary
        image points but the resulting world coords drift with distance
        from the quad.
      * Even with that drift, the projected world coord typically lands
        much closer to the *correct* world corner than to any of the
        other three (corners are 68–105 m apart). So nearest-of-four
        matching tolerates large absolute error in the projection.
      * Conversely, the forward direction (project world corners to
        image, match detection by pixel distance) requires precise
        image-space projection of points *outside* the cal quad —
        exactly where ``bootstrap_H`` is least reliable. Detections
        get dropped because the expected pixel position is wildly off.

    Returns the matched world coord, or ``None`` if the projection is
    too far from any of the 4 corners (likely a false-positive
    detection on a goalpost or banner pole).
    """
    pt = np.array([cx, cy, 1.0], dtype=np.float64)
    wp = bootstrap_H_i2w @ pt
    if abs(wp[2]) < 1e-9:
        return None
    wx, wy = float(wp[0] / wp[2]), float(wp[1] / wp[2])

    best_name: str | None = None
    best_dist = float("inf")
    for name, world in CORNER_FLAG_WORLD.items():
        d = ((wx - world[0]) ** 2 + (wy - world[1]) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_name is None or best_dist > max_world_dist_m:
        return None
    return CORNER_FLAG_WORLD[best_name]


def compute_homography_from_landmarks(
    frame_tracks: dict,
    frame_width: int,
    frame_height: int,
    ransac_threshold_px: float = DEFAULT_RANSAC_THRESHOLD_PX,
    min_matches: int = MIN_MATCHES_FOR_FIT,
    bootstrap_H: np.ndarray | None = None,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
    sanity_pad_m: float = 30.0,
) -> tuple[np.ndarray, int] | tuple[None, int]:
    """One-shot homography fit from a single frame's landmark detections.

    Args:
        frame_tracks: one element of the pipeline's ``tracks`` list.
        bootstrap_H: optional 3×3 image→world homography from a manual
            calibration. When provided, corner_flag detections are
            disambiguated by projecting the 4 world corners through
            ``H^-1`` and matching each detection to its nearest expected
            image position. Without bootstrap, the cruder
            image-quadrant heuristic is used (works for centred cameras
            but fails on side-of-pitch broadcast cameras like academy
            footage). The bootstrap doesn't have to be perfect — it's
            only used to *identify* corners; the per-frame H is then
            re-fitted from scratch.
        sanity_pad_m: image-centre tolerance for the post-fit sanity
            check. The fitted H must place the centre of the image
            within ``[-pad, pitch_dim + pad]`` on both axes — otherwise
            it's a degenerate fit (e.g. mismatched corners).

    Returns:
        ``(H, n_matches)`` on success; ``(None, n_matches)`` if too few
        landmarks, RANSAC failed, or the H failed the sanity check.
    """
    img_pts: list[list[float]] = []
    world_pts: list[list[float]] = []

    # Singletons — direct lookup. We loop over the full coord table so
    # adding a new landmark class only requires extending the table.
    for cls_name, world in LANDMARK_WORLD_COORDS.items():
        for det in frame_tracks.get(cls_name, {}).values():
            x1, y1, x2, y2 = det["bbox"]
            img_pts.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            world_pts.append(list(world))

    # Corner flags — bootstrap-aware disambiguation when manual H is
    # available, image-quadrant fallback otherwise. Tolerance of 30 m
    # in world space is generous (corners are 68–105 m apart, so any
    # projection within 30 m of the right one is unambiguous) but
    # still rejects detections that fired on non-corner objects
    # (goalposts, banner poles).
    if bootstrap_H is not None:
        max_world_dist_m = 30.0
        for det in frame_tracks.get("corner_flag", {}).values():
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            world = _resolve_corner_via_bootstrap(cx, cy, bootstrap_H, max_world_dist_m)
            if world is None:
                continue   # projection too far from any world corner — drop
            img_pts.append([cx, cy])
            world_pts.append(list(world))
    else:
        for det in frame_tracks.get("corner_flag", {}).values():
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            img_pts.append([cx, cy])
            world_pts.append(list(_resolve_corner_world_coord(
                cx, cy, frame_width, frame_height,
            )))

    n = len(img_pts)
    if n < min_matches:
        return None, n

    src = np.asarray(img_pts, dtype=np.float64)
    dst = np.asarray(world_pts, dtype=np.float64)
    H, _mask = cv2.findHomography(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold_px,
    )
    if H is None:
        return None, n

    # Sanity check: the centre of the image should map to a point on or
    # near the pitch. A degenerate H (mismatched corners, etc.) tends to
    # produce wildly off-pitch projections — those are the matrices that
    # explode distances when fed to the rolling buffer.
    cw = H @ np.array([frame_width / 2.0, frame_height / 2.0, 1.0])
    if abs(cw[2]) < 1e-9:
        return None, n
    cx_w, cy_w = cw[0] / cw[2], cw[1] / cw[2]
    if not (-sanity_pad_m <= cx_w <= pitch_length_m + sanity_pad_m):
        return None, n
    if not (-sanity_pad_m <= cy_w <= pitch_width_m + sanity_pad_m):
        return None, n

    return H, n


class LandmarkCalibrator:
    """Stateful per-frame calibrator with temporal smoothing.

    Wraps :func:`compute_homography_from_landmarks` and keeps a ring
    buffer of recent successful H matrices. Each call to :meth:`update`
    fits a fresh H for the new frame; the returned H is the average of
    the buffer, so one bad frame can't jolt distances downstream.

    Frames where the fit fails (too few landmarks, RANSAC didn't agree)
    return the previous smoothed H — the assumption is that the camera
    didn't teleport between frames, so a stale-but-recent H is closer
    to truth than no H at all.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        smoothing_frames: int = DEFAULT_SMOOTHING_FRAMES,
        ransac_threshold_px: float = DEFAULT_RANSAC_THRESHOLD_PX,
        min_matches: int = MIN_MATCHES_FOR_FIT,
        bootstrap_H: np.ndarray | None = None,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ransac_threshold_px = ransac_threshold_px
        self.min_matches = min_matches
        # Optional manual-calibration H matrix (image→world). Used to
        # disambiguate corner_flag identities — see
        # :func:`compute_homography_from_landmarks` docstring.
        self.bootstrap_H = bootstrap_H
        # Ring buffer of the most recent successful H matrices (3×3 each).
        self._buffer: deque[np.ndarray] = deque(maxlen=smoothing_frames)
        # Last *smoothed* H — what we hand back when the current frame
        # can't fit. None until at least one fit has succeeded.
        self._last_smoothed: np.ndarray | None = None
        # Telemetry for end-of-clip diagnostics.
        self.frames_seen = 0
        self.frames_fit = 0
        self.frames_used_fallback = 0

    def update(self, frame_tracks: dict) -> np.ndarray | None:
        """Process one frame; return the smoothed H matrix to use, or
        ``None`` if no fit has ever succeeded yet.
        """
        self.frames_seen += 1
        H, _n_matches = compute_homography_from_landmarks(
            frame_tracks,
            self.frame_width, self.frame_height,
            self.ransac_threshold_px, self.min_matches,
            bootstrap_H=self.bootstrap_H,
        )
        if H is not None:
            self.frames_fit += 1
            self._buffer.append(H)
            self._last_smoothed = self._average_buffer()
            return self._last_smoothed

        # Fit failed for this frame. Re-use the most recent smoothed H if
        # we have one — the camera doesn't teleport between frames.
        if self._last_smoothed is not None:
            self.frames_used_fallback += 1
        return self._last_smoothed

    def _average_buffer(self) -> np.ndarray:
        """Element-wise average of the H matrices in the buffer.

        Averaging 3×3 homography matrices isn't the *most* principled
        approach — strictly you'd want to interpolate in SE(3) or fit
        in a continuous parameterisation. But for small frame-to-frame
        camera changes (≤ a few px of pan), element-wise averaging
        produces visually identical distance/heatmap outputs and costs
        almost nothing. We can swap in a smarter interpolator later if
        needed.
        """
        stacked = np.stack(list(self._buffer), axis=0)
        avg = stacked.mean(axis=0)
        # Re-normalise so H[2, 2] == 1 (homographies are equivalence
        # classes up to scale; averaging knocks the scale slightly off).
        scale = avg[2, 2]
        if abs(scale) > 1e-9:
            avg = avg / scale
        return avg

    def stats(self) -> dict:
        """Return diagnostics for end-of-clip logging."""
        denom = max(1, self.frames_seen)
        return {
            "frames_seen": self.frames_seen,
            "frames_fit": self.frames_fit,
            "frames_used_fallback": self.frames_used_fallback,
            "frames_with_no_h": self.frames_seen - self.frames_fit
                              - self.frames_used_fallback,
            "fit_rate": self.frames_fit / denom,
        }


# Classes that get a ``position_transformed`` field filled in. Landmarks
# are intentionally excluded — they're inputs to the calibration, not
# outputs anyone uses downstream.
_TRANSFORMABLE_CLASSES = ("player", "goalkeeper", "referee", "ball")


def apply_auto_calibration_to_tracks(
    tracks: list[dict],
    calibrator: "LandmarkCalibrator",
) -> dict:
    """Per-frame homography pipeline that replaces the static ViewTransformer.

    Walks ``tracks`` once, calls :meth:`LandmarkCalibrator.update` on each
    frame, then transforms every player / GK / referee / ball detection's
    foot position through the resulting H matrix. The transformed metric
    coords land on ``info["position_transformed"]`` — same field name the
    static :class:`ViewTransformer` writes, so the downstream speed /
    distance / heatmap code is unchanged.

    Notes:
      * Uses **raw** bbox foot position, not ``position_adjusted``. The
        per-frame H matrix already accounts for camera pan/zoom
        intrinsically, so the ``CameraMovementEstimator`` shift would
        be applied twice if we used ``position_adjusted``.
      * Frames where the calibrator can't fit AND has no fallback get
        ``position_transformed = None`` for every detection — same
        contract as ``ViewTransformer`` for points outside the quad.

    Returns ``calibrator.stats()`` for end-of-clip telemetry.
    """
    for ft in tracks:
        H = calibrator.update(ft)
        if H is None:
            for cls in _TRANSFORMABLE_CLASSES:
                for info in ft.get(cls, {}).values():
                    info["position_transformed"] = None
            continue
        for cls in _TRANSFORMABLE_CLASSES:
            for info in ft.get(cls, {}).values():
                bbox = info.get("bbox")
                if not bbox:
                    continue
                x1, _y1, x2, y2 = bbox
                fx = (x1 + x2) / 2.0
                fy = float(y2)
                pt = np.array([fx, fy, 1.0])
                world = H @ pt
                if abs(world[2]) < 1e-9:
                    info["position_transformed"] = None
                    continue
                info["position_transformed"] = (
                    float(world[0] / world[2]),
                    float(world[1] / world[2]),
                )
    return calibrator.stats()
