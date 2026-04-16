"""Video I/O helpers built on OpenCV."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def get_video_info(video_path: str | Path) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_video_frames(
    video_path: str | Path,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> list[np.ndarray]:
    """Load frames into a list.

    Args:
        max_frames: stop after this many frames (None = read all).
        start_frame: skip this many frames first.

    1080p @ 24fps fills ~6 MB/frame × 2880 frames = ~17 GB for a full 2-min clip —
    always pass max_frames during development.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def iter_video_frames(video_path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Stream frames one at a time — preferred for long videos."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def save_video(frames: list[np.ndarray], output_path: str | Path, fps: float = 24.0) -> None:
    """Write frames to an MP4. Uses mp4v codec (ships with opencv-python)."""
    if not frames:
        raise ValueError("No frames to write")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()
