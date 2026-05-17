-- =============================================================================
-- Football analytics SQLite schema (v1)
-- =============================================================================
--
-- Three logical layers, all in one .db file:
--
--   1. IDENTITY layer — clubs, seasons, teams, players. Stable across
--      matches. The academy's roster lives here.
--
--   2. MATCH layer — one row per analysed match. Holds CV-derived
--      per-frame state (every player + ball position, every analysed
--      frame) plus the operator's track-id-to-roster mapping.
--
--   3. EVENTS + STATS layer — operator-tagged events (passes, shots,
--      tackles, …) plus pre-computed aggregates (match stats, season
--      stats per player). Aggregates are derivable from raw events +
--      frames; they're stored only to keep dashboard queries fast.
--
-- Conventions
-- -----------
--   * All FK columns end with `_id` and use ON DELETE CASCADE so deleting
--     a match wipes its frames/events cleanly.
--   * Coordinates are stored in BOTH image space (px) and world/pitch
--     space (metres on a 105×68 canonical pitch). Image-space is needed
--     for the event tagger UI (drawing overlays); world-space is needed
--     for distance/heatmap analytics.
--   * Per-frame data is intentionally normalised (one row per (frame,
--     track_id)) rather than denormalised JSON blobs, so we can query
--     "all players within 10 m of the ball" or "all events in the
--     attacking third" with vanilla SQL.
--   * `track_id` is the per-match CV ID assigned by ByteTrack/stitcher.
--     It's NOT stable across matches — the operator maps each track to
--     a roster `player_id` once per match via `match_track_to_player`.
--
-- Storage estimate
-- ----------------
--   A 90-minute match at 30 fps with 22 players ≈ 162k frames × 22 rows
--   ≈ 3.6M rows in `frame_player_positions`. SQLite handles this fine.
--   100 matches ≈ 360M rows; if you ever cross that, partition by season
--   or move to Postgres.
-- =============================================================================

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- Layer 1: identity (stable across matches)
-- -----------------------------------------------------------------------------

CREATE TABLE clubs (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    notes       TEXT
);

CREATE TABLE seasons (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,           -- e.g. "2025-2026 U17"
    start_date  TEXT,                           -- ISO YYYY-MM-DD
    end_date    TEXT
);

CREATE TABLE teams (
    id          INTEGER PRIMARY KEY,
    club_id     INTEGER NOT NULL REFERENCES clubs(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,                  -- "VFA U17 A"
    age_group   TEXT,                           -- "U17", "U19", "Senior"
    notes       TEXT,
    UNIQUE(club_id, name)
);

CREATE TABLE players (
    id              INTEGER PRIMARY KEY,
    team_id         INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    first_name      TEXT,
    last_name       TEXT,
    -- Default kit number for the player. Can be overridden per match in
    -- `match_track_to_player.kit_number_in_match` (kits sometimes change
    -- between matches, especially for academy teams).
    default_kit_number  INTEGER,
    position        TEXT,                       -- "GK"/"DEF"/"MID"/"FWD"/NULL
    -- Helpful for analyst filtering ("show me only attackers").
    UNIQUE(team_id, default_kit_number)
);

-- -----------------------------------------------------------------------------
-- Layer 2: match-level
-- -----------------------------------------------------------------------------

CREATE TABLE matches (
    id                  INTEGER PRIMARY KEY,
    season_id           INTEGER NOT NULL REFERENCES seasons(id) ON DELETE CASCADE,
    home_team_id        INTEGER NOT NULL REFERENCES teams(id),
    away_team_id        INTEGER NOT NULL REFERENCES teams(id),
    match_date          TEXT NOT NULL,          -- ISO YYYY-MM-DD
    -- Source-video metadata, captured at ingest time.
    video_path          TEXT NOT NULL,
    analysis_stub_path  TEXT,                   -- pickled stub from main.py
    calibration_path    TEXT,                   -- per-clip calibration JSON
    model_version       TEXT,                   -- "v6", "v7", ...
    fps                 REAL NOT NULL,
    frame_width         INTEGER NOT NULL,
    frame_height        INTEGER NOT NULL,
    n_frames_analysed   INTEGER NOT NULL,
    -- Audit trail
    ingested_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes               TEXT
);

CREATE INDEX idx_matches_season ON matches(season_id, match_date);

-- Operator's track-id → roster-player mapping. Done ONCE per match, in
-- the event tagger app, before tagging events. Rows for tracks the
-- operator hasn't yet identified are simply absent — the UI shows them
-- as "Player ?" until mapped.
CREATE TABLE match_track_to_player (
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    track_id            INTEGER NOT NULL,
    player_id           INTEGER NOT NULL REFERENCES players(id),
    -- Which side of the pitch this track is on. 1 or 2; matches the
    -- TeamAssigner's output. Useful for "show only home-team players"
    -- without joining all the way back to teams.
    team_side           INTEGER NOT NULL CHECK(team_side IN (1, 2)),
    kit_number_in_match INTEGER,                -- override default_kit_number
    PRIMARY KEY (match_id, track_id)
);

CREATE INDEX idx_track_player ON match_track_to_player(match_id, player_id);

-- -----------------------------------------------------------------------------
-- Layer 2: per-frame CV state
-- -----------------------------------------------------------------------------

CREATE TABLE frames (
    id              INTEGER PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    frame_number    INTEGER NOT NULL,           -- 0-based index in source video
    timestamp_ms    INTEGER NOT NULL,           -- frame_number / fps × 1000
    UNIQUE(match_id, frame_number)
);

CREATE INDEX idx_frames_ts ON frames(match_id, timestamp_ms);

-- One row per (frame, person-class detection). Covers players, GKs and
-- referees — the `cls` column distinguishes them. Landmarks/ball live in
-- separate tables because they have different schemas.
CREATE TABLE frame_player_positions (
    frame_id        INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    track_id        INTEGER NOT NULL,
    cls             TEXT NOT NULL CHECK(cls IN ('player', 'goalkeeper', 'referee')),
    -- Image-space bbox — used by the event-tagger to draw overlays
    -- straight on the video without recomputing.
    bbox_x1         REAL NOT NULL,
    bbox_y1         REAL NOT NULL,
    bbox_x2         REAL NOT NULL,
    bbox_y2         REAL NOT NULL,
    confidence      REAL,
    -- Foot position (bbox bottom-centre) in image AND world space.
    -- Splitting them out avoids re-deriving on every query.
    foot_x_image    REAL NOT NULL,
    foot_y_image    REAL NOT NULL,
    foot_x_world    REAL,                       -- NULL when no homography
    foot_y_world    REAL,
    -- Team assignment from src.team_assigner. NULL for unmapped detections
    -- (refs/GK with kit colour ambiguity, very short tracks, etc.).
    team            INTEGER CHECK(team IS NULL OR team IN (1, 2)),
    -- Per-frame motion. NULL until SpeedAndDistanceEstimator has run.
    speed_kmh       REAL,
    -- Composite PK includes ``cls`` because ByteTrack can output the
    -- same track_id under two classes in a single frame when the model
    -- can't decide between them (e.g. a referee briefly looking like a
    -- goalkeeper in their kit). Treating each class as its own row
    -- preserves the data; the event tagger's track→player mapping
    -- resolves which "true" class the operator considers it to be.
    PRIMARY KEY (frame_id, track_id, cls)
);

CREATE INDEX idx_player_pos_track ON frame_player_positions(track_id, frame_id);
CREATE INDEX idx_player_pos_team ON frame_player_positions(frame_id, team);

CREATE TABLE frame_ball_positions (
    frame_id        INTEGER PRIMARY KEY REFERENCES frames(id) ON DELETE CASCADE,
    bbox_x1         REAL NOT NULL,
    bbox_y1         REAL NOT NULL,
    bbox_x2         REAL NOT NULL,
    bbox_y2         REAL NOT NULL,
    -- Centre in image AND world space.
    cx_image        REAL NOT NULL,
    cy_image        REAL NOT NULL,
    x_world         REAL,
    y_world         REAL,
    confidence      REAL,
    -- Flags from the ball motion tracker: was this frame a real
    -- detection, or was it filled in by extrapolation / carrier-anchor?
    interpolated    INTEGER NOT NULL DEFAULT 0,
    extrapolated    INTEGER NOT NULL DEFAULT 0,
    carrier_anchored INTEGER NOT NULL DEFAULT 0,
    -- Possession owner at this frame (the track holding the ball, or NULL).
    -- Wired up from possession.compute_team_possession.
    owner_track_id  INTEGER
);

-- -----------------------------------------------------------------------------
-- Layer 3: events (operator-tagged)
-- -----------------------------------------------------------------------------

-- Vocabulary for `event_type`. Stored as a separate table so we can
-- evolve it via INSERT instead of schema migrations, and so the UI can
-- bind hotkeys to event types.
CREATE TABLE event_types (
    code            TEXT PRIMARY KEY,           -- e.g. "pass", "shot"
    label           TEXT NOT NULL,              -- human-readable: "Pass", "Shot"
    hotkey          TEXT,                       -- e.g. "p", "s"
    has_success     INTEGER NOT NULL DEFAULT 0, -- bool: needs success/fail flag
    has_secondary   INTEGER NOT NULL DEFAULT 0, -- bool: needs a 2nd player (receiver/fouled-by)
    sort_order      INTEGER NOT NULL DEFAULT 100
);

-- Seed the vocabulary. Easy to add more later via a normal INSERT.
INSERT INTO event_types (code, label, hotkey, has_success, has_secondary, sort_order) VALUES
    ('pass',       'Pass',           'p', 1, 1, 10),   -- success = completed; secondary = receiver
    ('shot',       'Shot',           's', 1, 0, 20),   -- success = on target / scored
    ('cross',      'Cross',          'c', 1, 1, 30),
    ('dribble',    'Dribble',        'd', 1, 0, 40),
    ('tackle',     'Tackle',         't', 1, 1, 50),   -- success = won the ball; secondary = opponent
    ('foul',       'Foul',           'f', 0, 1, 60),   -- secondary = fouled player
    ('lost_ball',  'Lost ball',      'l', 0, 0, 62),   -- player lost possession (not a tackle)
    ('won_ball',   'Won ball',       'w', 0, 0, 64),   -- player gained possession (not a tackle)
    ('throw_in',   'Throw-in',       'i', 0, 0, 70),
    ('corner',     'Corner',         'k', 0, 0, 80),
    ('goal',       'Goal',           'g', 0, 1, 90),   -- secondary = assist provider
    ('save',       'Save',           'v', 1, 0, 100),
    ('offside',    'Offside',        'o', 0, 0, 110);

CREATE TABLE events (
    id                  INTEGER PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    -- Frame/time anchor. `frame_id` is for spatial JOIN; timestamp_ms
    -- is duplicated so reports can sort/window without joining.
    frame_id            INTEGER NOT NULL REFERENCES frames(id),
    timestamp_ms        INTEGER NOT NULL,
    event_type          TEXT NOT NULL REFERENCES event_types(code),
    -- The player who took the action. Stored as track_id so events can
    -- be tagged BEFORE the operator finishes the track→roster mapping.
    -- Reports JOIN through match_track_to_player to get the roster name.
    primary_track_id    INTEGER,
    -- Optional second player (pass receiver, fouler's victim, …).
    secondary_track_id  INTEGER,
    -- Outcome flag for events with `has_success = 1`. NULL otherwise.
    success             INTEGER CHECK(success IS NULL OR success IN (0, 1)),
    -- Free-text notes the operator can attach.
    notes               TEXT,
    -- Audit trail.
    created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- An operator-friendly soft-delete (keep the row for audit, hide in UI).
    deleted_at          TEXT
);

CREATE INDEX idx_events_match_time ON events(match_id, timestamp_ms);
CREATE INDEX idx_events_player    ON events(match_id, primary_track_id);
CREATE INDEX idx_events_type      ON events(match_id, event_type);

-- -----------------------------------------------------------------------------
-- Layer 3: cached aggregates (rebuilt from raw on demand)
-- -----------------------------------------------------------------------------
-- These tables are derived. The `repository` module rebuilds them
-- whenever events change; reports read from them for fast queries.
-- Drop + repopulate is fine — they're caches, not source of truth.

CREATE TABLE match_team_stats (
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_side           INTEGER NOT NULL CHECK(team_side IN (1, 2)),
    -- CV-derived (no operator input needed)
    total_distance_km   REAL,
    possession_pct      REAL,
    -- Event-derived (built from `events`)
    shots               INTEGER NOT NULL DEFAULT 0,
    shots_on_target     INTEGER NOT NULL DEFAULT 0,
    goals               INTEGER NOT NULL DEFAULT 0,
    passes_attempted    INTEGER NOT NULL DEFAULT 0,
    passes_completed    INTEGER NOT NULL DEFAULT 0,
    tackles             INTEGER NOT NULL DEFAULT 0,
    fouls               INTEGER NOT NULL DEFAULT 0,
    corners             INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (match_id, team_side)
);

CREATE TABLE match_player_stats (
    match_id            INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    player_id           INTEGER REFERENCES players(id),       -- NULL until mapped
    track_id            INTEGER NOT NULL,
    -- CV-derived
    total_distance_m    REAL,
    longest_sprint_m    REAL,
    top_speed_kmh       REAL,
    minutes_on_pitch    REAL,
    -- Event-derived
    passes_attempted    INTEGER NOT NULL DEFAULT 0,
    passes_completed    INTEGER NOT NULL DEFAULT 0,
    shots               INTEGER NOT NULL DEFAULT 0,
    shots_on_target     INTEGER NOT NULL DEFAULT 0,
    goals               INTEGER NOT NULL DEFAULT 0,
    assists             INTEGER NOT NULL DEFAULT 0,
    tackles_won         INTEGER NOT NULL DEFAULT 0,
    fouls_committed     INTEGER NOT NULL DEFAULT 0,
    fouls_drawn         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (match_id, track_id)
);

CREATE INDEX idx_player_stats_player ON match_player_stats(player_id);

-- -----------------------------------------------------------------------------
-- Schema versioning
-- -----------------------------------------------------------------------------
-- Single-row table. ingest.py and repository.py compare against this on
-- startup; if a future schema bump changes column shapes, a migration
-- step runs.
CREATE TABLE schema_meta (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
);

INSERT INTO schema_meta (key, value) VALUES
    ('version', '1'),
    ('created_at', CURRENT_TIMESTAMP);
