"""Ball-possession assignment.

Phase 5. For each frame with a ball detection:
  - compute the min of distance(ball_center, player.left_foot) and
    distance(ball_center, player.right_foot) for every player
  - player whose min is smallest (and below MAX_BALL_DIST_PX) owns the ball
  - aggregate per-team possession ratio across the full clip

We approximate left/right foot by the bottom-left and bottom-right corners of
the player bbox — accurate enough at stadium-camera scale without a pose model.
"""
from __future__ import annotations


MAX_BALL_DIST_PX = 70  # tune per video resolution


class PossessionAssigner:
    def assign_ball_to_player(self, players: dict, ball_bbox) -> int | None:
        raise NotImplementedError("Possession — to be implemented in Phase 5.")
