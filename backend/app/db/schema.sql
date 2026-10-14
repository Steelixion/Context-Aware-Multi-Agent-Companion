PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,                 -- UUID
    display_name   TEXT NOT NULL,
    preferred_lang TEXT NOT NULL DEFAULT 'en',
    cloud_account  TEXT,                             -- NULL until linked
    settings_json  TEXT NOT NULL DEFAULT '{}',       -- thresholds, voice, etc.
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    deleted_at     REAL                              -- tombstone for sync
);

-- One row per drive/session.
CREATE TABLE IF NOT EXISTS monitoring_records (
    record_id        TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    started_at       REAL NOT NULL,
    ended_at         REAL,
    avg_fatigue      REAL,
    max_fatigue      REAL,
    alert_count      INTEGER NOT NULL DEFAULT 0,
    summary          TEXT,
    updated_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_user_time ON monitoring_records(user_id, started_at DESC);

-- Event-level log (state changes and safety prompts, not per-frame).
CREATE TABLE IF NOT EXISTS monitoring_logs (
    log_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id        TEXT NOT NULL REFERENCES monitoring_records(record_id) ON DELETE CASCADE,
    ts               REAL NOT NULL,
    driver_state     TEXT NOT NULL,                  -- e.g. ATTENTIVE, YAWNING, EYES_CLOSED, DISTRACTED, NO_FACE
    drowsiness_level TEXT NOT NULL CHECK (drowsiness_level IN ('ALERT','MILD','MODERATE','HIGH')),
    fatigue_score    REAL,
    ear              REAL,
    mar              REAL,
    safety_prompt    TEXT,                           -- text/ID of warning issued, NULL if none
    emotion          TEXT,                           -- speech-emotion label, if available
    synced           INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_logs_record_ts ON monitoring_logs(record_id, ts);
CREATE INDEX IF NOT EXISTS idx_logs_unsynced ON monitoring_logs(synced) WHERE synced = 0;

-- Sync bookkeeping.
CREATE TABLE IF NOT EXISTS sync_state (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    last_success_at  REAL,
    last_attempt_at  REAL,
    last_bundle_hash TEXT,
    last_error       TEXT
);
INSERT OR IGNORE INTO sync_state(id) VALUES (1);

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version(version) SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
