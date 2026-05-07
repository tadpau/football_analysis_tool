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
    frame: np.ndarray,
    team_share: dict[int, float],
    team_colors: dict[int, list],
    camera_shift: tuple[float, float] | None = None,
    hud_label: str | None = None,
) -> None:
    """Top-left panel: team 1 / team 2 possession %, plus optional camera shift.

    ``hud_label`` (e.g. "last 30s" or "match total") prints above the bars
    so the viewer knows whether they're seeing rolling or aggregate share.
    """
    pad = 12
    panel_w = 280
    has_label = hud_label is not None
    extra = (0 if not has_label else 18) + (0 if camera_shift is None else 24)
    panel_h = 78 + extra
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y_cursor = pad + 6
    if has_label:
        y_cursor += 14
        cv2.putText(
            frame,
            f"Possession ({hud_label})",
            (pad + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        y_cursor += 8

    for i, team_id in enumerate((1, 2)):
        color = tuple(int(c) for c in team_colors.get(team_id, (200, 200, 200)))
        y = y_cursor + 16 + i * 28
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

    if camera_shift is not None:
        dx, dy = camera_shift
        cv2.putText(
            frame,
            f"Cam shift: dx={dx:+5.1f}  dy={dy:+5.1f}",
            (pad + 10, y_cursor + 16 + 2 * 28 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


def _draw_player_metric(
    frame: np.ndarray, bbox, speed_kmh: float | None, distance_m: float | None,
    show_speed: bool = False,
) -> None:
    """Render total distance (and optionally speed) underneath the player bbox.

    Speed is hidden by default: it's noisy on a per-frame basis and the user's
    priority is cumulative distance (km covered), not instantaneous speed.
    Pass ``show_speed=True`` to re-enable the speed readout for debugging.
    """
    if distance_m is None and (not show_speed or speed_kmh is None):
        return
    x1, _y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    text = ""
    if distance_m is not None:
        text = f"{distance_m:5.1f} m"
    if show_speed and speed_kmh is not None:
        text = (f"{speed_kmh:4.1f} km/h  " + text) if text else f"{speed_kmh:4.1f} km/h"
    cv2.putText(
        frame,
        text,
        (cx - 36, int(y2) + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_one_frame(
    frame: np.ndarray,
    frame_tracks: dict,
    owner_id: int | None = None,
    team_share: dict[int, float] | None = None,
    team_colors: dict[int, list] | None = None,
    camera_shift: tuple[float, float] | None = None,
    hud_label: str | None = None,
) -> np.ndarray:
    """Render overlays for a single frame and return the annotated copy.

    Streaming-friendly: takes only what's needed for THIS frame, no
    full-clip lists. The original :func:`draw_annotations` is now a thin
    wrapper that calls this for every frame.
    """
    canvas = frame.copy()

    # players — team color if available, else class default
    for tid, info in frame_tracks.get("player", {}).items():
        color = (
            tuple(int(c) for c in info["team_color"])
            if "team_color" in info
            else DEFAULT_COLORS["player"]
        )
        _draw_ellipse(canvas, info["bbox"], color, tid)
        _draw_player_metric(
            canvas, info["bbox"],
            info.get("speed_kmh"), info.get("distance_m"),
        )
        if tid == owner_id:
            _draw_triangle(
                canvas, info["bbox"], DEFAULT_COLORS["owner"], point_down=False
            )

    # GK / refs — default class colors
    for tid, info in frame_tracks.get("goalkeeper", {}).items():
        _draw_ellipse(canvas, info["bbox"], DEFAULT_COLORS["goalkeeper"], tid)
    for tid, info in frame_tracks.get("referee", {}).items():
        _draw_ellipse(canvas, info["bbox"], DEFAULT_COLORS["referee"], tid)

    # ball — triangle; fade color if interpolated
    for _tid, info in frame_tracks.get("ball", {}).items():
        color = (
            DEFAULT_COLORS["ball_interp"]
            if info.get("interpolated")
            else DEFAULT_COLORS["ball"]
        )
        _draw_triangle(canvas, info["bbox"], color, point_down=True)

    # HUD
    if team_share is not None and team_colors is not None:
        _draw_possession_hud(
            canvas, team_share, team_colors,
            camera_shift=camera_shift, hud_label=hud_label,
        )

    return canvas


def draw_annotations(
    frames: list[np.ndarray],
    tracks: list[dict],
    per_frame_owner: list[int | None] | None = None,
    team_share: dict[int, float] | list[dict[int, float]] | None = None,
    team_colors: dict[int, list] | None = None,
    camera_movement: list[list[float]] | None = None,
    hud_label: str | None = None,
) -> list[np.ndarray]:
    """Batch wrapper around :func:`draw_one_frame` — kept for backward compat.

    ``team_share`` accepts either a single dict (static aggregate, used for
    every frame) or a per-frame list of dicts (rolling/dynamic possession).
    """
    out = []
    share_is_per_frame = isinstance(team_share, list)
    for i, (frame, frame_tracks) in enumerate(zip(frames, tracks)):
        owner_id = per_frame_owner[i] if per_frame_owner else None
        cam_shift = None
        if camera_movement is not None and i < len(camera_movement):
            cam_shift = (camera_movement[i][0], camera_movement[i][1])
        if share_is_per_frame:
            share_i = team_share[i] if i < len(team_share) else None
        else:
            share_i = team_share
        out.append(
            draw_one_frame(
                frame,
                frame_tracks,
                owner_id=owner_id,
                team_share=share_i,
                team_colors=team_colors,
                camera_shift=cam_shift,
                hud_label=hud_label,
            )
        )
    return out
