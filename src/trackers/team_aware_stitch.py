"""Team-aware second-pass ID stitcher.

Runs AFTER TeamAssigner.assign_all() has stamped each player detection with
`team` (1 or 2). Uses team agreement as a strong extra signal so we can push
gap / distance tolerances much further than the vanilla foot-position
stitcher inside Tracker without risking cross-team merges.

Why this exists
---------------
The baseline stitcher in `tracker.py` merges purely on foot-distance + frame
gap. That's conservative for good reason — without any class-conditional
signal, merging a blue and a white player who happen to cross near the same
foot position would be a disaster. So its thresholds are capped at
150f / 200px.

After team assignment runs, each track carries a dominant team label. That's
a much stronger signal than position alone:

  * **Team-gate**: refuse to merge two tracks whose dominant teams disagree.
    Addresses the exact failure case user reported — two players of opposite
    teams crossing, ByteTrack swaps/drops IDs, the old stitcher couldn't
    distinguish, so team colour briefly flipped with the new ID.
  * **Extended reach**: when teams agree, allow much wider gap and distance
    (300 f / 350 px) — a player turning or being occluded for a second can
    produce a new ID that's both temporally and spatially farther away than
    the baseline stitcher allows.

Tracks without a stable team (refs, GK misclassified as player, very short
fragments) are simply skipped by this pass — they're still stitched (or not)
by the baseline stitcher that ran earlier inside Tracker.
"""
from __future__ import annotations

from collections import Counter


# Team-aware thresholds. These are deliberately much wider than the baseline
# stitcher's 150f / 200px — the team-gate is strong enough to justify it.
# 300 frames ≈ 12.5 s @ 24 fps: covers the longest realistic "player off
# camera and back" event. 350 px covers a sprinter who moves across nearly a
# quarter of the frame during that gap, which is plausible.
TEAM_STITCH_MAX_GAP_FRAMES = 300
TEAM_STITCH_MAX_DIST_PX = 350.0


def _foot(bbox) -> tuple[float, float]:
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, float(y2))


def _dominant_team(tracks: list[dict], tid: int) -> int | None:
    """Majority team across this track's lifetime. None if the track has no
    team assignments at all (refs, too-short fragments)."""
    votes: Counter = Counter()
    for ft in tracks:
        info = ft.get("player", {}).get(tid)
        if info is None:
            continue
        t = info.get("team")
        if t is not None:
            votes[t] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def stitch_tracks_team_aware(
    tracks: list[dict],
    max_gap_frames: int = TEAM_STITCH_MAX_GAP_FRAMES,
    max_dist_px: float = TEAM_STITCH_MAX_DIST_PX,
) -> dict:
    """In-place merge of player track IDs that share a dominant team and are
    plausibly the same physical player (gap + foot-distance gates).

    Returns a stats dict: {"merges": int, "skipped_team_mismatch": int,
    "skipped_no_team": int}.

    Only operates on the "player" class. Goalkeeper / referee classes are
    left to the baseline stitcher — the team centroids don't apply to them.
    """
    if not tracks:
        return {"merges": 0, "skipped_team_mismatch": 0, "skipped_no_team": 0}

    # Precompute dominant team per current player track_id.
    all_tids: set[int] = set()
    for ft in tracks:
        all_tids.update(ft.get("player", {}).keys())
    dom_team = {tid: _dominant_team(tracks, tid) for tid in all_tids}

    merges = 0
    skipped_team_mismatch = 0
    skipped_no_team = 0

    # last_seen[tid] = (frame_idx, (fx, fy)) — most recent sighting
    last_seen: dict[int, tuple[int, tuple[float, float]]] = {}
    first_seen: dict[int, int] = {}

    for fi, ft in enumerate(tracks):
        cls_dict = ft.get("player", {})
        if not cls_dict:
            continue

        current = {tid: _foot(info["bbox"]) for tid, info in cls_dict.items()}
        rename_map: dict[int, int] = {}

        for tid, fxy in current.items():
            if tid in first_seen:
                continue   # already tracking; not new

            new_team = dom_team.get(tid)
            if new_team is None:
                # Track has no stable team — don't try to stitch it with this
                # pass. The baseline stitcher already had its chance.
                skipped_no_team += 1
                continue

            best_old: int | None = None
            best_dist = float("inf")
            for old_tid, (old_fi, old_fxy) in last_seen.items():
                if old_tid in current:
                    continue   # still active this frame — can't be the same player
                gap = fi - old_fi
                if gap <= 0 or gap > max_gap_frames:
                    continue

                # Team gate — the critical new rule.
                old_team = dom_team.get(old_tid)
                if old_team is None:
                    # Don't stitch a team-bearing track onto an unassigned one
                    # (could be a ref or GK mislabeled early).
                    continue
                if old_team != new_team:
                    # Opposite team — explicit refusal, this is exactly the
                    # cross-team ID swap the old stitcher couldn't prevent.
                    skipped_team_mismatch += 1
                    continue

                d = ((old_fxy[0] - fxy[0]) ** 2
                     + (old_fxy[1] - fxy[1]) ** 2) ** 0.5
                if d <= max_dist_px and d < best_dist:
                    best_dist = d
                    best_old = old_tid

            if best_old is not None and best_old != tid:
                rename_map[tid] = best_old

        if rename_map:
            merges += len(rename_map)
            # Propagate each rename from this frame forward.
            for future_fi in range(fi, len(tracks)):
                fcls = tracks[future_fi].get("player", {})
                if not fcls:
                    continue
                for new_id, old_id in list(rename_map.items()):
                    if new_id in fcls:
                        # Collision: old_id already occupies this future frame
                        # by some other detection → abandon further propagation
                        # of this one rename (same guard as baseline stitcher).
                        if old_id in fcls and future_fi != fi:
                            rename_map.pop(new_id)
                            continue
                        fcls[old_id] = fcls.pop(new_id)
            # After applying, keep dom_team coherent for the merged ID so any
            # subsequent candidate this frame still team-gates correctly.
            for new_id, old_id in rename_map.items():
                # Merged track keeps the older id's team (same by definition
                # since the rule required them equal).
                dom_team.pop(new_id, None)

        # Refresh bookkeeping based on the (possibly renamed) frame.
        for tid, info in tracks[fi].get("player", {}).items():
            last_seen[tid] = (fi, _foot(info["bbox"]))
            first_seen.setdefault(tid, fi)

    return {
        "merges": merges,
        "skipped_team_mismatch": skipped_team_mismatch,
        "skipped_no_team": skipped_no_team,
    }
