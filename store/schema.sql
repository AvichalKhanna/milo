-- Milo memory store schema. memory.py's init_db() also embeds this and
-- runs light migrations for existing DBs — this file is the reference copy.

CREATE TABLE IF NOT EXISTS people (
    person_id          TEXT PRIMARY KEY,
    name               TEXT,
    relation           TEXT,
    first_mentioned    TEXT,
    last_mentioned     TEXT,
    paused             INTEGER DEFAULT 0,
    checkin_propensity REAL DEFAULT 0.6,   -- add-on 6
    baseline_mean       REAL DEFAULT 3.0,   -- add-on: Welford running mean of intensity
    baseline_var        REAL DEFAULT 0.0,   -- add-on: Welford running variance
    baseline_n           INTEGER DEFAULT 0,
    last_metaphor        TEXT               -- add-on: verbatim client metaphor for callback
);

CREATE TABLE IF NOT EXISTS situations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT,
    area_of_life  TEXT,
    description   TEXT,
    intensity     INTEGER,
    status        TEXT DEFAULT 'open',   -- open | stale | resolved   (add-on 3)
    created_at    TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT,
    situation_id  INTEGER,
    need          TEXT,      -- hold | explore | move_forward
    created_at    TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT,
    started_at    TEXT,
    summary       TEXT
);

CREATE TABLE IF NOT EXISTS checkins (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT,
    situation_id  INTEGER,
    sent_at       TEXT,
    message       TEXT,
    replied       INTEGER DEFAULT 0,
    stopped       INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT,
    person_id     TEXT,
    input_hash    TEXT,
    decision      TEXT,
    confidence    REAL,
    reason        TEXT,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS trace (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT,
    person_id     TEXT,
    event_type    TEXT,     -- confidence_band | latency_computed | stale_flip |
                              -- metaphor_captured | escalation_flagged |
                              -- risk_gate_triggered | checkin_evaluated | resolution_detected
    details       TEXT       -- JSON blob
);

CREATE TABLE IF NOT EXISTS risk_flags (      -- add-on 5, audit trail separate from trace
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT,
    person_id     TEXT,
    input_hash    TEXT,
    reason        TEXT
);