"""Draw detection + tracking + team + possession overlays onto frames.

Visual language:
  * player       → ellipse at the feet (team color if assigned, else class color) + track ID
  * goalkeeper   → magenta ellipse
  * referee      → yellow ellipse
  * ball         → green triangle pointing at the ball
  * ball-owner   → red triangle above the owning player's head
  * interpolated → lighter green triangle (we know we guessed)
  * possession   → HUD top-left with per-team percentage
"""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.bbox_utils import get_center, get_bbox_width


# BGR defaults — overridden by team_color if TeamAssigner has run
DEFAULT_COLORS = {
    "player": (0, 0, 255),       # red
    "goalkeeper": (255, 0, 255), # magenta
    "referee": (0, 255, 255),    # yellow
    "ball": (0, 255, 0),         # green
    "ball_interp": (100, 220, 100),  # lighter green
    "owner": (0, 0, 255),        # red triangle above head
}


def _draw_ellipse(
    frame: np.ndarray, bbox, color, track_id: int | None = None
) -> None:
    x1, _y1, x2, y2 = bbox
    cx, _ = get_center(bbox)
    width = get_bbox_width(bbox)
    cv2.ellipse(
        frame,
        center=(int(cx), int(y2)),
        axes=(int(width * 0.5), int(width * 0.18)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )
    if track_id is not None:
        label = str(track_id)
        rect_w, rect_h = 36, 18
        rx1 = int(cx - rect_w / 2)
        ry1 = int(y2 + 5)
        cv2.rectangle(
            frame, (rx1, ry1), (rx1 + rect_w, ry1 + rect_h), color, cv2.FILLED
        )
        cv2.putText(
            frame,
            label,
            (rx1 + 4, ry1 + rect_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )


def _draw_triangle(frame: np.ndarray, bbox, color, point_down: bool = True) -> None:
    """Triangle above the bbox. point_down=True → apex touches bbox top."""
    x1, y1, x2, _y2 = bbox
    cx = int((x1 + x2) / 2)
    top = int(y1) - 6
    if point_down:
        pts = np.array([[cx, top + 18], [cx - 10, top], [cx + 10, top]], np.int32)
    else:
        pts = np.array([[cx, top], [cx - 10, top + 18], [cx + 10, top + 18]], np.int32)
    cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)


def _draw_possession_hud(
    frame: np.ndarray, team_share: dict[int, float], team_colors: dict[int, list]
) -> None:
    """Top-left panel showing team 1 / team 2 possession %."""
    h, w = frame.shape[:2]
    pad = 12
    panel_w, panel_h = 280, 78
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, team_id in enumerate((1, 2)):
        color = tuple(int(c) for c in team_colors.get(team_id, (200, 200, 200)))
        y = pad + 22 + i * 28
        cv2.rectangle(frame, (pad + 10, y - 14), (pad + 34, y + 6), color, -1)
        share = team_share.get(team_id, 0.0) * 100.0
        cv2.putText(
            frame,
            f"Team {team_id}:  {share:5.1f}%",
            (pad + 44, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_annotations(
    frames: list[np.ndarray],
    tracks: list[dict],
    per_frame_owner: list[int | None] | None = None,
    team_share: dict[int, float] | None = None,
    team_colors: dict[int, list] | None = None,
) -> list[np.ndarray]:
    out = []
    for i, (frame, frame_tracks) in enumerate(zip(frames, tracks)):
        canvas = frame.copy()

        owner_id = per_frame_owner[i] if per_frame_owner else None

        # players — team color if available, else class default
        for tid, info in frame_tracks.get("player", {}).items():
            color = tuple(int(c) for c in info["team_color"]) if "team_color" in info else DEFAULT_COLORS["player"]
            _draw_ellipse(canvas, info["bbox"], color, tid)
            if tid == owner_id:
                _draw_triangle(canvas, info["bbox"], DEFAULT_COLORS["owner"], point_down=False)

        # GK / refs — default class colors
        for tid, info in frame_tracks.get("goalkeeper", {}).items():
            _draw_ellipse(canvas, info["bbox"], DEFAULT_COLORS["goalkeeper"], tid)
        for tid, info in frame_tracks.get("referee", {}).items():
            _draw_ellipse(canvas, info["bbox"], DEFAULT_COLORS["referee"], tid)

        # ball — triangle; fade color if interpolated
        for _tid, info in frame_tracks.get("ball", {}).items():
            color = DEFAULT_COLORS["ball_interp"] if info.get("interpolated") else DEFAULT_COLORS["ball"]
            _draw_triangle(canvas, info["bbox"], color, point_down=True)

        # HUD
        if team_share is not None and team_colors is not None:
            _draw_possession_hud(canvas, team_share, team_colors)

        out.append(canvas)
    return out
