"""Desktop event-tagger application.

Sits on top of the analytics DB (``src.analytics.db``) and the CV
pipeline's analysis stubs. Operator workflow:

  1. Launch with ``python -m src.analytics.app --db data/analytics.db``
  2. Pick a match from the list of ingested matches.
  3. Scrub through the rendered video, watching CV-derived overlays
     (player ellipses, ball triangle) drawn from the DB at every frame.
  4. (Phase 2c) Click a track once to map it to a roster player.
  5. (Phase 2d) Click a player on screen + press a hotkey to tag an
     event (pass / shot / foul / …). Or hotkey-first if the player
     isn't visible in the click area.

This file (``__init__.py``) just exposes the public window classes for
test scripts that want to import the app programmatically.
"""
from .main_window import MainWindow

__all__ = ["MainWindow"]
