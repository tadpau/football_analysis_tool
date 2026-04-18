"""Team assignment by shirt color (kit-color baseline).

Algorithm (matches the Roboflow reference video):
  1. For each player box, crop the TOP HALF (shirt region; excludes shorts/legs).
  2. KMeans(n_clusters=2) on the pixels. One cluster is background (grass/sky);
     the other is the kit. We pick the non-background cluster by looking at the
     crop's four corners — majority corner label ≈ background.
  3. Bootstrap: in the first frame with enough players, collect all kit colors
     and KMeans(n_clusters=2) across them → two team centroids (team 1 + team 2).
  4. For every subsequent frame, classify each player's kit color against the
     two team centroids and cache per track_id (colors flicker, IDs shouldn't).

Upgrade path: replace `get_player_color()` with SigLIP embeddings — same
interface, better robustness to shadow/lighting.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self, min_bootstrap_players: int = 8):
        self.team_colors: dict[int, np.ndarray] = {}   # team_id → BGR centroid
        self.player_team: dict[int, int] = {}          # track_id → team_id (1 or 2)
        self.kmeans: KMeans | None = None              # fitted across-player KMeans
        self.min_bootstrap_players = min_bootstrap_players
        self._bootstrapped = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _get_player_color(frame: np.ndarray, bbox) -> np.ndarray:
        """Average shirt color (BGR) for one bbox."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0, 0], dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        top = crop[: crop.shape[0] // 2]                 # shirt, not shorts
        if top.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)

        pixels = top.reshape(-1, 3).astype(np.float32)
        km = KMeans(n_clusters=2, n_init=1, random_state=0).fit(pixels)

        # Reshape labels back to (h, w) to identify the four corners.
        labels_2d = km.labels_.reshape(top.shape[:2])
        corners = [
            labels_2d[0, 0], labels_2d[0, -1],
            labels_2d[-1, 0], labels_2d[-1, -1],
        ]
        bg = max(set(corners), key=corners.count)
        player_cluster = 1 - bg
        return km.cluster_centers_[player_cluster]

    # -------------------------------------------------------------- bootstrap
    def bootstrap_teams(self, frame: np.ndarray, players: dict) -> None:
        """Call once with a frame that contains both teams in view."""
        if len(players) < self.min_bootstrap_players:
            return
        colors = np.array(
            [self._get_player_color(frame, info["bbox"]) for info in players.values()]
        )
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(colors)
        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]
        self._bootstrapped = True

    # ------------------------------------------------------------------ query
    def get_player_team(
        self, frame: np.ndarray, bbox, track_id: int
    ) -> int:
        """Return team_id (1 or 2). Result cached per track_id."""
        if not self._bootstrapped:
            raise RuntimeError("Call bootstrap_teams() before get_player_team()")
        if track_id in self.player_team:
            return self.player_team[track_id]

        color = self._get_player_color(frame, bbox)
        team = int(self.kmeans.predict(color.reshape(1, -1))[0]) + 1
        self.player_team[track_id] = team
        return team

    # ----------------------------------------------------------- batch helper
    def assign_all(
        self, frames: list[np.ndarray], tracks: list[dict]
    ) -> None:
        """Bootstrap on first viable frame, then tag every track with team."""
        for frame, frame_tracks in zip(frames, tracks):
            players = frame_tracks.get("player", {})
            if not self._bootstrapped:
                self.bootstrap_teams(frame, players)
                if not self._bootstrapped:
                    continue
            for tid, info in players.items():
                info["team"] = self.get_player_team(frame, info["bbox"], tid)
                info["team_color"] = self.team_colors[info["team"]].tolist()
