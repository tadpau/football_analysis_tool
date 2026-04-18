"""Ball-possession assignment.

For each frame with a known ball position, find the player whose nearest foot
(bottom-left or bottom-right corner of bbox) is closest to the ball center.
If the minimum distance exceeds `MAX_BALL_DIST_PX`, possession is "loose" — no
player is assigned (neutral frame).

We approximate feet by the bbox's bottom corners instead of running pose
estimation; at broadcast-camera scale this is accurate enough and 100× cheaper.

Downstream metric:
    team_possession[team_id] = possessed_frames / attributed_frames
"""
from __future__ import annotations

from src.utils.bbox_utils import get_center, measure_distance

MAX_BALL_DIST_PX = 70   # tune per imgsz/resolution; 70 is sane at 1920x1080


def assign_ball_to_player(players: dict, ball_bbox) -> int | None:
    """Return track_id of the player with the ball, or None if nobody close."""
    ball_center = get_center(ball_bbox)
    best_id: int | None = None
    best_dist = MAX_BALL_DIST_PX

    for pid, info in players.items():
        x1, y1, x2, y2 = info["bbox"]
        left_foot = (x1, y2)
        right_foot = (x2, y2)
        d = min(
            measure_distance(ball_center, left_foot),
            measure_distance(ball_center, right_foot),
        )
        if d < best_dist:
            best_dist = d
            best_id = pid
    return best_id


def compute_team_possession(
    tracks: list[dict],
) -> tuple[list[int | None], dict[int, float]]:
    """Per-frame owning player + team possession ratios.

    Returns:
        per_frame_owner: list[int | None] — track_id owning the ball, or None.
        team_share: {team_id: share_of_attributed_frames}.
    """
    per_frame_owner: list[int | None] = []
    counts = {1: 0, 2: 0}
    for frame_tracks in tracks:
        ball = frame_tracks.get("ball", {}).get(1)
        players = frame_tracks.get("player", {})
        if not ball or not players:
            per_frame_owner.append(None)
            continue
        owner = assign_ball_to_player(players, ball["bbox"])
        per_frame_owner.append(owner)
        if owner is None:
            continue
        team = players[owner].get("team")
        if team in counts:
            counts[team] += 1

    total = counts[1] + counts[2]
    if total == 0:
        return per_frame_owner, {1: 0.0, 2: 0.0}
    return per_frame_owner, {1: counts[1] / total, 2: counts[2] / total}
