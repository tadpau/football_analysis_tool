"""YOLOv8 + ByteTrack wrapper.

One Tracker instance handles detection → tracking → stub caching for a clip.

Class mapping
-------------
Two supported modes:
  * "coco"   — use COCO-pretrained yolov8*.pt: `person` (0) becomes everything on
              the pitch (players + GK + referees merged), `sports ball` (32)
              becomes ball. Useful while a custom model is still training.
  * "custom" — expect a 4-class model in Roboflow order:
              0=player, 1=goalkeeper, 2=referee, 3=ball.

The output schema is identical either way, so downstream code is mode-agnostic.

Tracker tuning
--------------
Defaults are aimed at 24 fps broadcast football:
  * `lost_track_buffer=90`  — keep IDs alive ~3.75 s through occlusions/dropped
    detections. Default 30 (~1 s) was way too short: players behind others or
    crowded near set-pieces kept getting new IDs, which in turn reset their
    cumulative distance to 0.
  * `minimum_matching_threshold=0.7` — slightly more forgiving IoU match when
    re-associating a reappearing track. Default 0.8 is tuned for pedestrians
    with little inter-frame motion; football players move much faster.

After ByteTrack finishes, we run `_stitch_tracks` to merge IDs across short
gaps at nearby foot-positions. This is what preserves *total distance covered*
across reappearances — which is the metric that matters most for this project.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO


# Stitcher defaults.
#
# Max gap in frames: how long a track ID can be missing before we refuse to
# stitch it to a new one. Raised from 72 → 150 (~6 s @ 24 fps) because
# ByteTrack itself keeps tracks alive for ~5 s, so the stitcher needs a
# wider window than ByteTrack to catch the cases ByteTrack gave up on.
#
# Max foot-distance in pixels: how far the player can plausibly have moved
# during the gap. Raised from 120 → 200. A sprinter at 8 m/s moving through
# a 2 s occlusion covers ~16 m, which near the camera can be 200+ px — the
# old 120 px cap was biased against fast players.
STITCH_MAX_GAP_FRAMES = 150
STITCH_MAX_DIST_PX = 200.0


# Output schema — every frame is a dict mapping class_name -> {track_id: {"bbox": [x1,y1,x2,y2]}}
TrackFrame = dict[str, dict[int, dict]]


# Minimum confidence per class. Ball is lower because it's small and flickery.
# COCO's generic "sports ball" class rarely exceeds 0.15 for broadcast football —
# we keep the threshold low in COCO mode so we at least get *some* ball triangles;
# the custom-trained model will have genuinely confident ball detections at 0.3+.
DEFAULT_CONF = {"player": 0.3, "goalkeeper": 0.3, "referee": 0.3, "ball": 0.05}


@dataclass
class ClassMap:
    """Maps detector class indices to our class schema.

    Person/ball classes are first-class fields. Pitch landmarks (used by
    the auto-calibration module) live in a separate dict so adding a new
    landmark class doesn't require code changes elsewhere — only an extra
    entry in the dict and a matching entry in
    :data:`src.pitch.auto_calibrate.LANDMARK_WORLD_COORDS`.
    """
    player: list[int]
    goalkeeper: list[int]
    referee: list[int]
    ball: list[int]
    landmarks: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def coco(cls) -> "ClassMap":
        # COCO yolov8m: person=0, sports ball=32. No GK/ref distinction.
        return cls(player=[0], goalkeeper=[], referee=[], ball=[32])

    @classmethod
    def custom(cls) -> "ClassMap":
        # Roboflow football-players-detection-3zvbc v20 order (confirmed from data.yaml):
        #   0: ball, 1: goalkeeper, 2: player, 3: referee
        # If you re-train on a dataset with a different order, update this.
        return cls(player=[2], goalkeeper=[1], referee=[3], ball=[0])

    @classmethod
    def custom_landmarks(cls) -> "ClassMap":
        """7-class extension of :meth:`custom` with Tier-1 pitch landmarks.

        Alphabetical YOLO order — note ``penalty_spot_left`` was *dropped*
        from this round's taxonomy because there were only 19 labelled
        instances (camera angle favours the right penalty box). Add it
        back when there's enough left-half coverage in a future mining
        round.

          0: ball
          1: center_spot
          2: corner_flag
          3: goalkeeper
          4: penalty_spot_right
          5: player
          6: referee

        Use ``--mode custom_landmarks`` once you've trained a model that
        knows these classes. Existing 4-class weights stay on
        ``--mode custom``.
        """
        return cls(
            player=[5], goalkeeper=[3], referee=[6], ball=[0],
            landmarks={
                "center_spot": [1],
                "corner_flag": [2],
                "penalty_spot_right": [4],
            },
        )

    def name_of(self, cls_id: int) -> str | None:
        for name, ids in (
            ("player", self.player),
            ("goalkeeper", self.goalkeeper),
            ("referee", self.referee),
            ("ball", self.ball),
        ):
            if cls_id in ids:
                return name
        for name, ids in self.landmarks.items():
            if cls_id in ids:
                return name
        return None

    def all_keep_ids(self) -> list[int]:
        """Detector class IDs we care about — used to filter raw output."""
        ids = self.player + self.goalkeeper + self.referee + self.ball
        for landmark_ids in self.landmarks.values():
            ids = ids + landmark_ids
        return ids

    def all_landmark_ids(self) -> list[int]:
        out: list[int] = []
        for ids in self.landmarks.values():
            out = out + ids
        return out


class Tracker:
    def __init__(
        self,
        model_path: str | Path,
        mode: Literal["coco", "custom", "custom_landmarks"] = "coco",
        batch_size: int = 16,
        imgsz: int = 1280,
        frame_rate: float = 24.0,
        lost_track_buffer: int = 120,   # ~5 s @ 24 fps — survives most occlusions
        minimum_matching_threshold: float = 0.7,
        stitch_max_gap_frames: int = STITCH_MAX_GAP_FRAMES,
        stitch_max_dist_px: float = STITCH_MAX_DIST_PX,
    ):
        self.model = YOLO(str(model_path))
        self.mode = mode
        if mode == "coco":
            self.class_map = ClassMap.coco()
        elif mode == "custom_landmarks":
            self.class_map = ClassMap.custom_landmarks()
        else:
            self.class_map = ClassMap.custom()
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.stitch_max_gap_frames = stitch_max_gap_frames
        self.stitch_max_dist_px = stitch_max_dist_px

        # Separate tracker for the ball — a single object, no ID switches to worry
        # about. We keep the main ByteTrack for players/GK/refs.
        #
        # supervision's ByteTrack kwargs (>=0.18):
        #   track_activation_threshold, lost_track_buffer,
        #   minimum_matching_threshold, frame_rate, minimum_consecutive_frames.
        # We try the modern signature first and fall back to the older one for
        # compatibility with older supervision installs.
        try:
            self.player_tracker = sv.ByteTrack(
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=int(round(frame_rate)),
            )
        except TypeError:
            # Older supervision (<0.18) uses different kwarg names.
            self.player_tracker = sv.ByteTrack(
                track_buffer=lost_track_buffer,
                match_thresh=minimum_matching_threshold,
                frame_rate=int(round(frame_rate)),
            )

    # ------------------------------------------------------------------ detect
    def _detect(self, frames: list[np.ndarray]) -> list:
        """Run YOLO in batches. Returns list of ultralytics Results."""
        results = []
        for i in tqdm(
            range(0, len(frames), self.batch_size),
            desc="detecting",
            unit="batch",
        ):
            batch = frames[i : i + self.batch_size]
            batch_results = self.model.predict(
                batch,
                conf=0.1,             # low base; we filter per-class below
                imgsz=self.imgsz,
                verbose=False,
            )
            results.extend(batch_results)
        return results

    # -------------------------------------------------- per-Result conversion
    def _track_one_result(self, det) -> TrackFrame:
        """Convert one ultralytics Result into our TrackFrame schema.

        Pure function on top of the persistent ByteTrack state — calling it
        ``N`` times in order is identical to running detection on those ``N``
        frames as a single batch through ``get_object_tracks``. That property
        is what lets ``feed_chunk`` stream frames in slices without changing
        the final track set.
        """
        sv_det = sv.Detections.from_ultralytics(det)

        mask = np.isin(sv_det.class_id, self.class_map.all_keep_ids())
        sv_det = sv_det[mask]

        # Split: ball + landmarks bypass ByteTrack (no ID continuity needed —
        # ball is a singleton handled by the motion tracker downstream;
        # landmarks are stationary anchors for per-frame homography).
        landmark_ids = self.class_map.all_landmark_ids()
        ball_mask = np.isin(sv_det.class_id, self.class_map.ball)
        landmark_mask = np.isin(sv_det.class_id, landmark_ids)
        ball_det = sv_det[ball_mask]
        landmark_det = sv_det[landmark_mask]
        people_det = sv_det[~(ball_mask | landmark_mask)]

        tracked = self.player_tracker.update_with_detections(people_det)

        frame_out: TrackFrame = {
            "player": {},
            "goalkeeper": {},
            "referee": {},
            "ball": {},
        }
        # Pre-create landmark slots so downstream code can do
        # ``frame_out.get(class_name, {})`` without surprises.
        for landmark_name in self.class_map.landmarks:
            frame_out[landmark_name] = {}

        for xyxy, cls_id, track_id, conf in zip(
            tracked.xyxy,
            tracked.class_id,
            tracked.tracker_id,
            tracked.confidence,
        ):
            name = self.class_map.name_of(int(cls_id))
            if name is None or name == "ball":
                continue
            if conf < DEFAULT_CONF[name]:
                continue
            frame_out[name][int(track_id)] = {
                "bbox": xyxy.tolist(),
                "confidence": float(conf),
            }

        if len(ball_det) > 0:
            ball_conf = ball_det.confidence
            valid = ball_conf >= DEFAULT_CONF["ball"]
            if valid.any():
                best_idx = int(np.argmax(ball_conf * valid))
                frame_out["ball"][1] = {
                    "bbox": ball_det.xyxy[best_idx].tolist(),
                    "confidence": float(ball_conf[best_idx]),
                }

        # Pitch landmarks: keep ALL detections per landmark class. Singleton
        # classes (centre spot, penalty spots) typically yield one bbox; the
        # corner_flag class can produce up to four per frame, one per visible
        # corner. We assign incremental "track ids" 1, 2, 3 just so the
        # frame_out value is the same dict-of-dicts shape every other class
        # uses; the auto-calibrate module ignores the ids and walks values().
        if len(landmark_det) > 0:
            for j in range(len(landmark_det)):
                cls_id = int(landmark_det.class_id[j])
                conf = float(landmark_det.confidence[j])
                # Lower confidence floor than people/ball — landmarks are
                # tiny so the model is intrinsically less confident, but
                # geometry is forgiving (RANSAC handles a bad detection).
                if conf < 0.2:
                    continue
                name = self.class_map.name_of(cls_id)
                if name is None:
                    continue
                slot = frame_out[name]
                next_id = (max(slot.keys()) + 1) if slot else 1
                slot[next_id] = {
                    "bbox": landmark_det.xyxy[j].tolist(),
                    "confidence": conf,
                }

        return frame_out

    # ------------------------------------------------------------- streaming
    def feed_chunk(self, frames: list[np.ndarray]) -> list[TrackFrame]:
        """Detect + track a chunk of frames; return per-frame tracks for them.

        Stateful — the underlying ByteTrack instance persists across calls so
        IDs flow through chunk boundaries the same as if all frames had been
        passed in one big call. Does NOT run the stitcher; the caller is
        expected to accumulate tracks across chunks and finally call
        :meth:`finalize_tracks` exactly once at the end.

        This is the streaming entry point used by ``main.run_streaming``.
        """
        detections = self._detect(frames)
        return [self._track_one_result(det) for det in detections]

    def finalize_tracks(
        self,
        tracks: list[TrackFrame],
        stub_path: str | Path | None = None,
    ) -> None:
        """Run the post-pass stitcher + diagnostics on a fully-accumulated
        track list. Mutates ``tracks`` in-place; optionally writes a stub.

        Must be called exactly once after all chunks have been fed — the
        stitcher needs to see the full timeline to merge across long gaps.
        """
        pre = self._count_unique_ids(tracks)
        merges = self._stitch_tracks(
            tracks,
            max_gap_frames=self.stitch_max_gap_frames,
            max_dist_px=self.stitch_max_dist_px,
        )
        post = self._count_unique_ids(tracks)
        if merges:
            print(
                f"  stitched {merges} broken tracks (gap ≤ "
                f"{self.stitch_max_gap_frames}f, foot-dist ≤ "
                f"{self.stitch_max_dist_px:.0f}px)"
            )
        print(
            "  unique IDs after stitching: "
            + ", ".join(f"{cls}={post[cls]} (was {pre[cls]})"
                        for cls in ("player", "goalkeeper", "referee"))
        )

        if stub_path:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

    # ------------------------------------------------------------------ track
    def get_object_tracks(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | Path | None = None,
    ) -> list[TrackFrame]:
        """Batch entry point — kept for backward compat.

        Equivalent to ``feed_chunk(frames)`` followed by
        ``finalize_tracks(tracks)``. New code that wants to stream long
        videos should use those two methods directly.
        """
        if read_from_stub and stub_path and Path(stub_path).exists():
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        tracks = self.feed_chunk(frames)
        self.finalize_tracks(tracks, stub_path=stub_path)
        return tracks

    # ---------------------------------------------------------- diagnostics
    @staticmethod
    def _count_unique_ids(tracks: list[TrackFrame]) -> dict[str, int]:
        seen: dict[str, set[int]] = {
            "player": set(), "goalkeeper": set(), "referee": set()
        }
        for ft in tracks:
            for cls in seen:
                seen[cls].update(ft.get(cls, {}).keys())
        return {cls: len(s) for cls, s in seen.items()}

    # --------------------------------------------------------------- stitching
    @staticmethod
    def _stitch_tracks(
        tracks: list[TrackFrame],
        max_gap_frames: int,
        max_dist_px: float,
    ) -> int:
        """Merge broken tracks in-place. Returns number of merge operations.

        Algorithm, per class in {player, goalkeeper, referee}:
          1. Walk frames, remember (last_frame_seen, last_foot_xy) for every
             track_id.
          2. When a NEW track_id first appears at frame i with foot position f,
             look for an "orphaned" old track_id where
                 last_frame_seen < i
                 i - last_frame_seen <= max_gap_frames
                 euclidean(last_foot_xy, f) <= max_dist_px
                 AND the old id has NOT been reused in [last_seen, i].
             Pick the closest candidate; rename new_id → old_id in all frames
             from i onward.
          3. Repeat. We walk once — stitched IDs are treated as their merged
             target for subsequent bookkeeping.

        Why foot-position, not bbox centre: the foot (bbox bottom-centre) is
        what the homography and distance code actually use downstream, so
        stitching on that metric gives a more self-consistent result.
        """
        if not tracks:
            return 0

        total_merges = 0
        for cls in ("player", "goalkeeper", "referee"):
            # last_seen[tid] = (frame_idx, (fx, fy)) for the most recent sighting
            last_seen: dict[int, tuple[int, tuple[float, float]]] = {}
            # first_seen[tid] = frame_idx of first appearance (original, after merges)
            first_seen: dict[int, int] = {}

            def foot(bbox):
                x1, _y1, x2, y2 = bbox
                return ((x1 + x2) * 0.5, float(y2))

            for fi, ft in enumerate(tracks):
                cls_dict = ft.get(cls, {})
                if not cls_dict:
                    continue

                # First, collect current IDs + their foot positions.
                current = {tid: foot(info["bbox"]) for tid, info in cls_dict.items()}

                # Try to stitch any genuinely-new IDs.
                rename_map: dict[int, int] = {}
                for tid, fxy in current.items():
                    if tid in first_seen:
                        continue   # not new; already being tracked
                    # Candidate = every previously-seen id not currently active
                    # in this frame.
                    best_old: int | None = None
                    best_dist = float("inf")
                    for old_tid, (old_fi, old_fxy) in last_seen.items():
                        if old_tid in current:
                            continue   # old id is ALSO active this frame — can't be same player
                        gap = fi - old_fi
                        if gap <= 0 or gap > max_gap_frames:
                            continue
                        d = ((old_fxy[0] - fxy[0]) ** 2
                             + (old_fxy[1] - fxy[1]) ** 2) ** 0.5
                        if d <= max_dist_px and d < best_dist:
                            best_dist = d
                            best_old = old_tid
                    if best_old is not None and best_old != tid:
                        rename_map[tid] = best_old

                # Apply renames in this frame and propagate: patch every future
                # frame's cls dict entries matching these new_ids.
                if rename_map:
                    total_merges += len(rename_map)
                    for future_fi in range(fi, len(tracks)):
                        fcls = tracks[future_fi].get(cls, {})
                        if not fcls:
                            continue
                        for new_id, old_id in list(rename_map.items()):
                            if new_id in fcls:
                                # If old_id is already used in this future frame,
                                # we can't merge (two players at same time under
                                # same id). Stop propagating this particular merge
                                # at that frame — leave the rest alone.
                                if old_id in fcls and future_fi != fi:
                                    # Collision: abandon further propagation for
                                    # this one rename.
                                    rename_map.pop(new_id)
                                    continue
                                fcls[old_id] = fcls.pop(new_id)

                # Refresh bookkeeping based on the (possibly renamed) frame.
                for tid, info in tracks[fi].get(cls, {}).items():
                    fxy = foot(info["bbox"])
                    last_seen[tid] = (fi, fxy)
                    first_seen.setdefault(tid, fi)

        return total_merges
