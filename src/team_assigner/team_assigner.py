"""Team assignment by shirt color.

Phase 4b. Algorithm (kit-color baseline):
  1. Crop each player bbox to the top half (shirt region, excludes shorts/legs).
  2. Run a 2-cluster KMeans on pixel colors; the cluster closest to the border
     is treated as background, the other as the shirt.
  3. Aggregate the shirt colors of all players in the first K frames → 2-cluster
     KMeans across players gives the two team centroids.
  4. For each frame thereafter, assign each player to whichever centroid is
     closer in RGB space. Cache per track_id so assignments are stable.

Upgrade path: swap the color heuristic for SigLIP image embeddings (see second
reference video). Same interface — only `_get_player_color` changes.
"""
from __future__ import annotations

import numpy as np


class TeamAssigner:
    def __init__(self):
        self.team_colors: dict[int, np.ndarray] = {}
        self.player_team_dict: dict[int, int] = {}
        raise NotImplementedError("TeamAssigner — to be implemented in Phase 4b.")

    def assign_team_color(self, frame: np.ndarray, player_detections: dict) -> None:
        raise NotImplementedError

    def get_player_team(
        self, frame: np.ndarray, player_bbox, player_id: int
    ) -> int:
        raise NotImplementedError
