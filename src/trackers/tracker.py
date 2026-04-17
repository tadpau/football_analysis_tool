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
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO


# Output schema — every frame is a dict mapping class_name -> {track_id: {"bbox": [x1,y1,x2,y2]}}
TrackFrame = dict[str, dict[int, dict]]


# Minimum confidence per class. Ball is lower because it's small and flickery.
# COCO's generic "sports ball" class rarely exceeds 0.15 for broadcast football —
# we keep the threshold low in COCO mode so we at least get *some* ball triangles;
# the custom-trained model will have genuinely confident ball detections at 0.3+.
DEFAULT_CONF = {"player": 0.3, "goalkeeper": 0.3, "referee": 0.3, "ball": 0.05}


@dataclass
class ClassMap:
    """Maps detector class indices to our 4-class schema."""
    player: list[int]
    goalkeeper: list[int]
    referee: list[int]
    ball: list[int]

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

    def name_of(self, cls_id: int) -> str | None:
        for name, ids in (
            ("player", self.player),
            ("goalkeeper", self.goalkeeper),
            ("referee", self.referee),
            ("ball", self.ball),
        ):
            if cls_id in ids:
                return name
        return None


class Tracker:
    def __init__(
        self,
        model_path: str | Path,
        mode: Literal["coco", "custom"] = "coco",
        batch_size: int = 16,
        imgsz: int = 1280,
    ):
        self.model = YOLO(str(model_path))
        self.mode = mode
        self.class_map = ClassMap.coco() if mode == "coco" else ClassMap.custom()
        self.batch_size = batch_size
        self.imgsz = imgsz

        # Separate tracker for the ball — a single object, no ID switches to worry
        # about. We keep the main ByteTrack for players/GK/refs.
        self.player_tracker = sv.ByteTrack()

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

    # ------------------------------------------------------------------ track
    def get_object_tracks(
        self,
        frames: list[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str | Path | None = None,
    ) -> list[TrackFrame]:
        if read_from_stub and stub_path and Path(stub_path).exists():
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self._detect(frames)
        tracks: list[TrackFrame] = []

        for det in detections:
            # Convert ultralytics Result → supervision Detections
            sv_det = sv.Detections.from_ultralytics(det)

            # In COCO mode only, merge goalkeepers etc. into player class — already
            # handled by ClassMap, but we also filter to classes we care about.
            keep_ids = (
                self.class_map.player
                + self.class_map.goalkeeper
                + self.class_map.referee
                + self.class_map.ball
            )
            mask = np.isin(sv_det.class_id, keep_ids)
            sv_det = sv_det[mask]

            # Separate ball detections; track only non-ball objects with ByteTrack.
            ball_mask = np.isin(sv_det.class_id, self.class_map.ball)
            ball_det = sv_det[ball_mask]
            people_det = sv_det[~ball_mask]

            tracked = self.player_tracker.update_with_detections(people_det)

            frame_out: TrackFrame = {
                "player": {},
                "goalkeeper": {},
                "referee": {},
                "ball": {},
            }

            # tracked is sv.Detections with .tracker_id populated
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

            # Ball: no tracking, just keep highest-conf detection per frame (id=1).
            if len(ball_det) > 0:
                ball_conf = ball_det.confidence
                valid = ball_conf >= DEFAULT_CONF["ball"]
                if valid.any():
                    best_idx = int(np.argmax(ball_conf * valid))
                    frame_out["ball"][1] = {
                        "bbox": ball_det.xyxy[best_idx].tolist(),
                        "confidence": float(ball_conf[best_idx]),
                    }

            tracks.append(frame_out)

        if stub_path:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
