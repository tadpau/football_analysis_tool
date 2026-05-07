"""Team assignment by shirt color (per-detection classification, trailing vote).

Previous approach (median-per-track) was fragile in two ways:
  * When ByteTrack swapped IDs during a crossing (two bboxes overlap, IDs
    briefly exchange), the same physical player would keep the other track's
    team assignment — their kit colour was right, but we'd locked the team to
    the ID, not the player.
  * A referee or goalkeeper detected as `player` has a kit colour roughly
    equidistant from both team centroids. 2-cluster KMeans force-assigned them
    to whichever centroid happened to be marginally closer, and that choice
    flipped frame-to-frame as the colour sample wobbled.

Current approach:

  1. **Fit** (once): extract a kit-colour sample per detection, take the
     median per long-lived track, fit 2-cluster KMeans → team centroids.
     Also compute per-cluster spread (median within-cluster distance).

  2. **Classify** (per detection, at render time):
       - Crop torso, HSV-mask grass, take median colour.
       - Distance to nearest centroid vs. the other: if ambiguous (ratio ≥
         AMBIGUITY_RATIO) → no team this frame.
       - ABSOLUTE distance to nearest centroid vs. cluster spread: if the
         colour is >3× the cluster spread from the nearest centroid, it's a
         different class entirely (referee, keeper) → no team this frame.
       - Else: vote for the nearer centroid's team.

  3. **Smooth** (per track): keep a rolling history of the last 7 votes for
     each track_id. The rendered team is the majority vote of that history.
     One noisy frame can't flip the colour.

Consequences this fixes directly:
  * A crossing that swaps IDs doesn't swap teams. Each detection's colour
    places it correctly regardless of which track_id it landed under.
  * On-field refs' yellow kit consistently fails the absolute-distance check
    → no team → render red. No more team-colour flicker on the main ref.
  * Short fragmented tracks get the right team from their own colour, no
    need for heuristic propagation passes.

`assign_all(frames, tracks)` keeps the same signature, so main.py is unchanged.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans


# If a colour sample's distance to the nearest centroid is greater than
# AMBIGUITY_RATIO × distance to the *other* centroid, it's ambiguous — sitting
# roughly between the two team colours. Leave unassigned for that frame.
AMBIGUITY_RATIO = 0.85

# If a colour sample's distance to the nearest centroid exceeds
# OUTLIER_SPREAD_MULT × that cluster's median within-cluster spread, the sample
# is WAY off — almost certainly a different class (referee, goalkeeper). Leave
# unassigned for that frame, no matter how it compares to the other centroid.
# Tightened 3.0 → 2.5 after seeing sideline refs occasionally slip through on
# clips where one team's kit is dark (narrow bbox + stands background means
# rare colour samples drift closer to a centroid). 2.5× median still passes
# legitimate team members comfortably — the median is robust to outliers, so
# a real player sits well within 2.5× their cluster's median spread.
OUTLIER_SPREAD_MULT = 2.5

# Fallback spread if a cluster has too few members to measure (shouldn't
# happen with ≥5 members per team, but guard anyway).
DEFAULT_SPREAD = 30.0

# Majority-vote window on per-track team history. 15 frames ≈ 0.6 s @ 24 fps.
# Up from 7 because some clips have close-together team centroids (both kits
# dark) which means more per-detection classifier noise leaks through. 15 is
# still fast enough to resolve a legitimate post-crossing swap within a
# half-second, but a short noise burst (3–4 frames) can't flip a track
# anymore. Combined with hysteresis below, this is the main anti-flicker lever.
HISTORY_WINDOW = 15

# Hysteresis on team transitions. Once a track has been rendered as team X,
# flipping to team Y requires Y to hold ≥60 % of the history window (i.e. 9
# of 15 votes). Plain majority (>50 %) would still let a 4–3 vote split in a
# 7-window flip the rendered colour every time a noisy run happened to
# clear half + one. This biases toward stability, which matches the user's
# mental model ("the player didn't actually change teams between frames").
# It does NOT lock team to track ID — a genuine crossing that swaps IDs
# still flips cleanly because the NEW id starts with no prev_team and
# bootstraps on plain majority.
HYSTERESIS_RATIO = 0.6

# HSV grass mask. Pitch grass is very strongly saturated green under stadium
# lighting. Hue 35–85 (OpenCV 0–180 range) covers yellow-green through
# blue-green. S ≥ 40 kicks out washed-out pixels (shirt highlights, stadium
# floodlight glare). V ≥ 20 excludes near-black shadows (which could be either
# grass shadow or dark shorts — let KMeans sort those).
GRASS_HSV_LO = np.array([35, 40, 20], dtype=np.uint8)
GRASS_HSV_HI = np.array([85, 255, 255], dtype=np.uint8)


class TeamAssigner:
    def __init__(self, min_samples_per_track: int = 3):
        self.team_colors: dict[int, np.ndarray] = {}     # team_id → BGR centroid
        self.player_team: dict[int, int] = {}            # track_id → fallback team
        self.kmeans: KMeans | None = None
        self.min_samples_per_track = min_samples_per_track
        self.cluster_spread: dict[int, float] = {}       # cluster_idx → median dist

    # --------------------------------------------------------------- color
    @staticmethod
    def _get_player_color(frame: np.ndarray, bbox) -> np.ndarray | None:
        """Median torso color (BGR) for one bbox, or None if too small/edge.

        Pipeline:
          1. Crop the bbox.
          2. Take the vertical range [0.15h .. 0.55h] — chest/torso only,
             avoiding head/hair at the top and shorts at the bottom.
          3. Build an HSV green mask of the torso and DROP those pixels. This
             removes grass bleed-in from the edges of a loose bbox, which was
             the main reason blue-kit players' medians were drifting toward
             white (the old KMeans corner-detection heuristic guessed wrong
             on tight crops and sometimes returned the grass as the kit
             colour).
          4. Return the median BGR of the surviving pixels — more robust than
             the mean against stripes, numbers and sponsor logos.

        Returns None if the crop is too small or almost entirely grass (bad
        bbox or a player standing in a weird lighting pocket).
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 - x1 < 4 or y2 - y1 < 8:
            return None

        crop = frame[y1:y2, x1:x2]
        h = crop.shape[0]
        torso = crop[int(0.15 * h) : int(0.55 * h)]
        if torso.size == 0:
            return None

        # Drop grass pixels via HSV.
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, GRASS_HSV_LO, GRASS_HSV_HI)
        non_grass = torso[grass_mask == 0]
        # Fewer than ~20 non-grass pixels: torso is almost all green — either
        # the player isn't actually inside the bbox, or the bbox was mostly
        # background. Skip rather than return a meaningless value.
        if non_grass.shape[0] < 20:
            return None

        return np.median(non_grass.astype(np.float32), axis=0)

    # ----------------------------------------------------------------- fit
    def fit(self, frames: list[np.ndarray], tracks: list[dict]) -> None:
        """Sample kit colours → 2-cluster KMeans → record team centroids +
        per-cluster spread, plus a per-track fallback team (used only if the
        per-detection classifier can't extract a colour for a given frame)."""
        per_track_colors: dict[int, list[np.ndarray]] = defaultdict(list)

        for frame, ft in zip(frames, tracks):
            for tid, info in ft.get("player", {}).items():
                c = self._get_player_color(frame, info["bbox"])
                if c is not None:
                    per_track_colors[tid].append(c)

        # Drop tracks with too few observations — likely flicker IDs.
        per_track_colors = {
            tid: cs for tid, cs in per_track_colors.items()
            if len(cs) >= self.min_samples_per_track
        }
        if len(per_track_colors) < 2:
            return  # not enough players to form two teams

        median_per_track: dict[int, np.ndarray] = {
            tid: np.median(np.stack(cs, axis=0), axis=0)
            for tid, cs in per_track_colors.items()
        }

        X = np.stack(list(median_per_track.values()), axis=0).astype(np.float32)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

        # Per-cluster spread = median within-cluster distance to centroid.
        # Used as the scale for the absolute-distance outlier test. Median is
        # more robust than mean/std when a referee or GK got force-clustered
        # into a team (their large residual won't inflate the threshold).
        labels = self.kmeans.labels_
        centers = self.kmeans.cluster_centers_
        dists = np.linalg.norm(X - centers[labels], axis=1)
        for k in (0, 1):
            members = dists[labels == k]
            self.cluster_spread[k] = (
                float(np.median(members)) if members.size >= 3 else DEFAULT_SPREAD
            )

        # Per-track fallback team — the median colour's nearest centroid, IF
        # the colour isn't an outlier. Used only when per-detection
        # classification can't extract a colour (tiny bbox, total occlusion)
        # so we still render something rather than nothing.
        for tid, color in median_per_track.items():
            team = self._classify_color(color)
            if team is not None:
                self.player_team[tid] = team

    # ------------------------------------------------------------- classify
    def _classify_color(self, color: np.ndarray) -> int | None:
        """Return team (1 or 2) for a single BGR colour, or None if ambiguous.

        Two gates:
          * Ratio gate: if the nearest centroid is only marginally nearer than
            the other (d_own / d_oth > AMBIGUITY_RATIO), the colour is
            between teams — refuse.
          * Absolute gate: if the nearest centroid is still far in ABSOLUTE
            terms relative to that cluster's typical spread
            (d_own > OUTLIER_SPREAD_MULT × spread), the colour is a different
            class entirely (ref, GK) — refuse. This is the gate that finally
            nails the on-field ref: yellow is far from both blue and white
            centroids, so every frame the ref is visible we refuse the
            assignment and render them in the neutral referee colour.
        """
        if self.kmeans is None:
            return None
        c = color.reshape(1, -1).astype(np.float32)
        cluster = int(self.kmeans.predict(c)[0])
        d_own = float(np.linalg.norm(color - self.kmeans.cluster_centers_[cluster]))
        d_oth = float(np.linalg.norm(color - self.kmeans.cluster_centers_[1 - cluster]))

        if d_oth <= 1e-6 or d_own / d_oth > AMBIGUITY_RATIO:
            return None

        spread = self.cluster_spread.get(cluster, DEFAULT_SPREAD)
        if d_own > OUTLIER_SPREAD_MULT * spread:
            return None

        return cluster + 1

    # ------------------------------------------------------------- streaming
    def attach_color_samples(
        self, frame: np.ndarray, ft: dict
    ) -> None:
        """Frame-bound: stash a kit-colour sample on each player detection.

        Stores the BGR median onto ``info["_color_sample"]`` (or ``None`` if
        the bbox was too small / all-grass). After this returns, the caller
        is free to drop ``frame`` — colour information now travels with the
        track entry, so it survives Tracker._stitch_tracks (which renames
        keys but keeps the value dict).

        Pair with :meth:`fit_from_samples` + :meth:`assign_all_from_samples`
        once the full track list is finalised, instead of the all-in-memory
        :meth:`assign_all`.
        """
        for info in ft.get("player", {}).values():
            info["_color_sample"] = self._get_player_color(frame, info["bbox"])

    def fit_from_samples(self, tracks: list[dict]) -> None:
        """KMeans fit using cached ``_color_sample`` values on each detection.

        Mirrors :meth:`fit` but reads colours from the tracks themselves
        rather than re-extracting from frames. Safe to call after the
        Tracker stitcher has run — stitched-away IDs have already had their
        colour samples merged into the surviving track via the dict-rename.
        """
        per_track_colors: dict[int, list[np.ndarray]] = defaultdict(list)
        for ft in tracks:
            for tid, info in ft.get("player", {}).items():
                c = info.get("_color_sample")
                if c is not None:
                    per_track_colors[tid].append(c)

        per_track_colors = {
            tid: cs for tid, cs in per_track_colors.items()
            if len(cs) >= self.min_samples_per_track
        }
        if len(per_track_colors) < 2:
            return

        median_per_track: dict[int, np.ndarray] = {
            tid: np.median(np.stack(cs, axis=0), axis=0)
            for tid, cs in per_track_colors.items()
        }

        X = np.stack(list(median_per_track.values()), axis=0).astype(np.float32)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

        labels = self.kmeans.labels_
        centers = self.kmeans.cluster_centers_
        dists = np.linalg.norm(X - centers[labels], axis=1)
        for k in (0, 1):
            members = dists[labels == k]
            self.cluster_spread[k] = (
                float(np.median(members)) if members.size >= 3 else DEFAULT_SPREAD
            )

        for tid, color in median_per_track.items():
            team = self._classify_color(color)
            if team is not None:
                self.player_team[tid] = team

    def assign_all_from_samples(self, tracks: list[dict]) -> None:
        """Streaming counterpart to :meth:`assign_all` — uses cached samples.

        Runs the same per-detection classify + history-window majority vote +
        hysteresis pipeline, but without needing frames in memory. Drops the
        ``_color_sample`` keys after stamping so the tracks list is clean.
        """
        if self.kmeans is None:
            self.fit_from_samples(tracks)
        if self.kmeans is None:
            # Still nothing — drop helper keys and return cleanly.
            for ft in tracks:
                for info in ft.get("player", {}).values():
                    info.pop("_color_sample", None)
            return

        history: dict[int, list[int]] = defaultdict(list)
        prev_team: dict[int, int] = {}
        per_frame_votes = 0
        per_frame_refused = 0

        for ft in tracks:
            for tid, info in ft.get("player", {}).items():
                color = info.get("_color_sample")
                vote = self._classify_color(color) if color is not None else None

                if vote is not None:
                    per_frame_votes += 1
                    hist = history[tid]
                    hist.append(vote)
                    if len(hist) > HISTORY_WINDOW:
                        del hist[0]
                else:
                    per_frame_refused += 1

                hist = history.get(tid, [])
                if hist:
                    ones = sum(1 for v in hist if v == 1)
                    twos = len(hist) - ones
                    n = len(hist)
                    prev = prev_team.get(tid)
                    if prev is None:
                        if ones > twos:
                            team = 1
                        elif twos > ones:
                            team = 2
                        else:
                            team = hist[-1]
                    else:
                        other = 2 if prev == 1 else 1
                        other_votes = twos if prev == 1 else ones
                        if other_votes / n >= HYSTERESIS_RATIO:
                            team = other
                        else:
                            team = prev
                    prev_team[tid] = team
                else:
                    team = self.player_team.get(tid)

                if team is not None:
                    info["team"] = team
                    info["team_color"] = self.team_colors[team].tolist()

        # Drop the temporary samples — downstream code shouldn't see them.
        for ft in tracks:
            for info in ft.get("player", {}).values():
                info.pop("_color_sample", None)

        total = per_frame_votes + per_frame_refused
        if total:
            refused_pct = 100.0 * per_frame_refused / total
            print(
                f"  team-assigner: classified {per_frame_votes}/{total} "
                f"detections ({refused_pct:.1f}% refused — refs/GK/ambiguous)"
            )

    # ------------------------------------------------------------- annotate
    def assign_all(self, frames: list[np.ndarray], tracks: list[dict]) -> None:
        """Stamp each player detection with team + team_color (in place).

        Per-detection classification decouples team colour from track_id.
        When two players cross and ByteTrack swaps their IDs, each bbox is
        still classified by its own pixels — so the team colour stays with
        the physical player, not the numeric ID.

        A trailing majority vote over HISTORY_WINDOW recent frames protects
        against a single bad sample (player back-turned, half-occluded, etc.)
        flipping the rendered colour.
        """
        if self.kmeans is None:
            self.fit(frames, tracks)
        if self.kmeans is None:
            return  # fit gave up; leave team unset

        # Per-track history of recent classification votes (excluding None).
        # Small list because we only keep the last HISTORY_WINDOW entries.
        history: dict[int, list[int]] = defaultdict(list)
        # Per-track currently-rendered team (for hysteresis).
        prev_team: dict[int, int] = {}

        per_frame_votes = 0
        per_frame_refused = 0

        for frame, ft in zip(frames, tracks):
            for tid, info in ft.get("player", {}).items():
                color = self._get_player_color(frame, info["bbox"])
                vote: int | None
                if color is None:
                    vote = None
                else:
                    vote = self._classify_color(color)

                if vote is not None:
                    per_frame_votes += 1
                    hist = history[tid]
                    hist.append(vote)
                    if len(hist) > HISTORY_WINDOW:
                        del hist[0]
                else:
                    per_frame_refused += 1

                # Decide rendered team from history + hysteresis, or fall back
                # to the per-track median team when we haven't collected any
                # votes yet, or leave unset (renders as neutral referee colour).
                hist = history.get(tid, [])
                if hist:
                    ones = sum(1 for v in hist if v == 1)
                    twos = len(hist) - ones
                    n = len(hist)
                    prev = prev_team.get(tid)

                    if prev is None:
                        # Bootstrap — plain majority, ties go to newest vote.
                        if ones > twos:
                            team = 1
                        elif twos > ones:
                            team = 2
                        else:
                            team = hist[-1]
                    else:
                        # Hysteresis — only flip if the OTHER team has a
                        # ≥HYSTERESIS_RATIO supermajority of the window. Prev
                        # team wins all ties and near-ties, which is what
                        # kills the walking/turning flicker: a few noisy
                        # frames can't clear the 60 % bar.
                        other = 2 if prev == 1 else 1
                        other_votes = twos if prev == 1 else ones
                        if other_votes / n >= HYSTERESIS_RATIO:
                            team = other
                        else:
                            team = prev

                    prev_team[tid] = team
                else:
                    team = self.player_team.get(tid)

                if team is None:
                    continue
                info["team"] = team
                info["team_color"] = self.team_colors[team].tolist()

        total = per_frame_votes + per_frame_refused
        if total:
            refused_pct = 100.0 * per_frame_refused / total
            print(
                f"  team-assigner: classified {per_frame_votes}/{total} "
                f"detections ({refused_pct:.1f}% refused — refs/GK/ambiguous)"
            )
