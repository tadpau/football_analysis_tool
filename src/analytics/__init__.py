"""Analytics layer — sits on top of the CV pipeline and turns its output
into a queryable per-match / per-season database.

Modules
-------
* ``db/`` — SQLite schema + connection helpers. The schema is in
  ``db/schema.sql`` and is the canonical reference for what shape the
  data takes.
* (planned) ``ingest.py`` — reads an analysis stub (pickled tracks +
  camera_movement from ``main.run_streaming``) and writes it into the DB
  as one match.
* (planned) ``repository.py`` — typed CRUD operations on the DB. The
  desktop event-tagger app and the reporting code both go through this.
* (planned) ``reports.py`` — match- and player-level aggregate queries
  used by the dashboard / PDF export.

Design choice: SQLite, single file, single writer. This is a single-
operator local tool, so we deliberately skip the auth / sync /
multi-process complexity of a server DB. Schema is portable to Postgres
later if the workflow ever goes multi-user.
"""
