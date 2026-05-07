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


def iter_video_frames(
    video_path: str | Path,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream frames one at a time — preferred for long videos.

    Yields ``(global_idx, frame)`` where ``global_idx`` accounts for
    ``start_frame`` (i.e. the first yielded index is ``start_frame``,
    not ``0``). This makes the stream interchangeable with
    :func:`read_video_frames` for any consumer that respects frame indices.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n_yielded = 0
    idx = start_frame
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
            n_yielded += 1
            if max_frames is not None and n_yielded >= max_frames:
                break
    finally:
        cap.release()


def iter_video_chunks(
    video_path: str | Path,
    chunk_size: int,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> Iterator[list[np.ndarray]]:
    """Stream frames in batches of ``chunk_size`` — for memory-bounded loops.

    The last chunk may be shorter. Memory cost is bounded by
    ``chunk_size * frame_size`` (e.g. 600 × 6 MB ≈ 3.6 GB at 1080p),
    which fits comfortably on 16 GB laptops.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    chunk: list[np.ndarray] = []
    for _idx, frame in iter_video_frames(video_path, start_frame, max_frames):
        chunk.append(frame)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class StreamingVideoWriter:
    """Context-managed cv2.VideoWriter wrapper that opens lazily.

    The writer must know frame width/height before opening, but in a streaming
    pipeline we don't know those until we see the first frame. This class
    defers the OpenCV call to the first ``write()`` and uses that frame's
    shape, which is the ergonomic outcome callers want.

    Usage::

        with StreamingVideoWriter(path, fps=30.0) as w:
            for frame in stream:
                w.write(frame)
    """

    def __init__(self, output_path: str | Path, fps: float = 24.0):
        self.output_path = Path(output_path)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def __enter__(self) -> "StreamingVideoWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.output_path), fourcc, self.fps, (w, h)
            )
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"Could not open VideoWriter for {self.output_path}"
                )
        self._writer.write(frame)

    def __exit__(self, *exc_info) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


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
