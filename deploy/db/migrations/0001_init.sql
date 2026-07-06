-- MotionCaddie — initial schema (implementation-guide §A made real)
-- Target: Aurora Serverless v2 / PostgreSQL 15+
-- Conventions: text + CHECK instead of native enums (cheaper to evolve);
--              uuid PKs via pgcrypto's gen_random_uuid().
-- Apply order: this file is idempotent-safe on a fresh DB only (no IF NOT EXISTS
-- on tables by design — a failed partial apply should be dropped, not patched).

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ---------------------------------------------------------------------------
-- Layer 1 · Reference data (seeded from repo JSON; versioned by pipeline)
-- ---------------------------------------------------------------------------

CREATE TABLE indicator_catalog (
    indicator_key      text PRIMARY KEY,          -- e.g. shoulder_turn_top_deg
    label              text NOT NULL,             -- "Shoulder turn"
    plain_name         text NOT NULL,             -- golfer-facing phrasing
    event_ref          text NOT NULL,             -- swing event it is read at
    unit               text NOT NULL CHECK (unit IN ('deg','pct','ratio')),
    depends_on_joints  text[] NOT NULL DEFAULT '{}'
);

CREATE TABLE reference_bands (
    club           text NOT NULL DEFAULT 'all',   -- NOTE: current corpus bands are
                                                  -- club-agnostic; seeded as 'all'.
                                                  -- Column exists so per-club bands
                                                  -- can land without a migration.
    indicator_key  text NOT NULL REFERENCES indicator_catalog(indicator_key),
    p10            real NOT NULL,
    p25            real NOT NULL,
    p50            real NOT NULL,                 -- pro_median in the scorecard
    p75            real NOT NULL,
    p90            real NOT NULL,
    mean           real NOT NULL,
    std            real NOT NULL,
    PRIMARY KEY (club, indicator_key),
    CHECK (p10 <= p25 AND p25 <= p50 AND p50 <= p75 AND p75 <= p90)
);

CREATE TABLE indicator_confidence (
    indicator_key      text PRIMARY KEY REFERENCES indicator_catalog(indicator_key),
    confidence         real NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    tier               text NOT NULL CHECK (tier IN ('high','med','low')),
    worst_joint_z_std  real
);

CREATE TABLE kb_cards (
    term           text PRIMARY KEY,              -- glossary term or indicator key
    kind           text NOT NULL CHECK (kind IN ('glossary','indicator_card')),
    plain_name     text,
    gloss          text NOT NULL,                 -- the ONLY text the LLM may gloss from
    indicator_key  text REFERENCES indicator_catalog(indicator_key),
    card           jsonb                          -- full card (measures/why/…) for indicator_card rows
);

-- ---------------------------------------------------------------------------
-- Layer 2 · Core app data
-- ---------------------------------------------------------------------------

CREATE TABLE users (
    user_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cognito_sub   text UNIQUE,                    -- null for seeded/system rows
    email         text,
    display_name  text,
    handedness    char(1) CHECK (handedness IN ('L','R')),
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- Event-sourced: one row per grant; revoke sets revoked_at. Never UPDATE granted
-- history away — current state = latest row per (user_id, scope).
CREATE TABLE consents (
    consent_id  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid NOT NULL REFERENCES users(user_id),
    scope       text NOT NULL CHECK (scope IN ('analytics','model_training','storage')),
    granted     boolean NOT NULL,
    granted_at  timestamptz NOT NULL DEFAULT now(),
    revoked_at  timestamptz
);
CREATE INDEX idx_consents_user_scope ON consents (user_id, scope, granted_at DESC);

CREATE TABLE swings (
    swing_id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           uuid REFERENCES users(user_id),      -- null for seeded GolfDB
    source            text NOT NULL CHECK (source IN ('upload','golfdb')),
    golfdb_clip_id    integer,                             -- set when source='golfdb'
    raw_video_s3_key  text,
    club              text,
    view              text CHECK (view IN ('face_on','down_the_line')),
    fps               real,
    n_frames          integer,
    status            text NOT NULL DEFAULT 'uploaded'
                      CHECK (status IN ('uploaded','queued','processing','done','failed')),
    error             text,
    uploaded_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT golfdb_clip_unique UNIQUE (golfdb_clip_id)
);
CREATE INDEX idx_swings_user ON swings (user_id, uploaded_at DESC);

CREATE TABLE analyses (
    analysis_id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    swing_id               uuid NOT NULL REFERENCES swings(swing_id),
    pipeline_version       text NOT NULL,                  -- e.g. mixste+oneeuro@2026-07
    lifter                 text NOT NULL,                  -- golfpose3d | motionbert_full | ...
    smoothing_cfg          jsonb NOT NULL DEFAULT '{}'::jsonb,
    landmarks_3d_s3_key    text,
    replay_html_s3_key     text,
    scorecard_png_s3_key   text,
    scorecard_json         jsonb NOT NULL,                 -- exact blob for reproduction
    status                 text NOT NULL DEFAULT 'done'
                           CHECK (status IN ('processing','done','failed')),
    latency_ms             integer,
    created_at             timestamptz NOT NULL DEFAULT now(),
    -- one analysis per swing per pipeline version = the optimization/backfill loop
    CONSTRAINT analyses_swing_version_unique UNIQUE (swing_id, pipeline_version)
);
CREATE INDEX idx_analyses_swing ON analyses (swing_id, created_at DESC);

CREATE TABLE swing_events (
    analysis_id  uuid NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    event_name   text NOT NULL CHECK (event_name IN
                 ('address','toe_up','mid_backswing','top',
                  'mid_downswing','impact','mid_follow_through','finish')),
    frame_idx    integer NOT NULL CHECK (frame_idx >= 0),
    PRIMARY KEY (analysis_id, event_name)
);

CREATE TABLE indicators (
    analysis_id      uuid NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    indicator_key    text NOT NULL REFERENCES indicator_catalog(indicator_key),
    value            real NOT NULL,
    percentile       integer CHECK (percentile BETWEEN 0 AND 100),
    pro_median       real,                                  -- snapshotted at run time
    pro_band_low     real,
    pro_band_high    real,
    in_range         boolean,
    confidence_tier  text CHECK (confidence_tier IN ('high','med','low')),
    PRIMARY KEY (analysis_id, indicator_key)
);
CREATE INDEX idx_indicators_key ON indicators (indicator_key);

CREATE TABLE feedback_notes (
    note_id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id      uuid NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    indicator_key    text NOT NULL REFERENCES indicator_catalog(indicator_key),
    label            text NOT NULL,
    event_name       text,
    value            real,
    percentile       integer,
    pro_median       real,
    confidence_tier  text,
    message          text NOT NULL,
    severity         text NOT NULL CHECK (severity IN ('review','info','low_confidence'))
);
CREATE INDEX idx_feedback_analysis ON feedback_notes (analysis_id);

CREATE TABLE explanations (
    analysis_id  uuid PRIMARY KEY REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    backend      text NOT NULL,                             -- anthropic | codex
    model        text NOT NULL,                             -- claude-opus-4-8
    text         text NOT NULL,
    grounded     boolean NOT NULL,
    violations   jsonb NOT NULL DEFAULT '[]'::jsonb,
    tokens_in    integer,
    tokens_out   integer,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- Audit only — the chat runtime stays stateless (browser holds the transcript).
CREATE TABLE chat_turns (
    turn_id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id       uuid REFERENCES analyses(analysis_id),
    user_id           uuid REFERENCES users(user_id),
    question          text NOT NULL,
    answer            text NOT NULL,
    grounded          boolean NOT NULL,
    tools_used        jsonb NOT NULL DEFAULT '[]'::jsonb,
    compare_swing_id  uuid REFERENCES swings(swing_id),
    iterations        integer,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_chat_analysis ON chat_turns (analysis_id, created_at);

-- ---------------------------------------------------------------------------
-- results_history — rollup view feeding the optimization loop / progress UI.
-- Per user, per indicator, ordered by analysis time, with delta vs the user's
-- previous analysis of the same indicator (same pipeline_version).
-- ---------------------------------------------------------------------------

CREATE VIEW results_history AS
SELECT
    s.user_id,
    s.swing_id,
    a.analysis_id,
    a.pipeline_version,
    a.created_at,
    i.indicator_key,
    i.value,
    i.percentile,
    i.in_range,
    i.value - lag(i.value) OVER (
        PARTITION BY s.user_id, i.indicator_key, a.pipeline_version
        ORDER BY a.created_at
    ) AS delta_vs_prev
FROM analyses a
JOIN swings s      ON s.swing_id = a.swing_id
JOIN indicators i  ON i.analysis_id = a.analysis_id
WHERE a.status = 'done';

COMMIT;
