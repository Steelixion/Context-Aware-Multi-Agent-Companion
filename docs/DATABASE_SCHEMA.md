# Database schemas

The app uses two local stores under one data directory:

- SQLite relational store: `data/driver.db`
- LanceDB vector store: `data/vectors/`

`DATA_DIR` overrides the parent directory. Timestamps are Unix epoch seconds stored as `REAL`.

## 1. SQLite — `data/driver.db`

Source of truth: [`backend/app/db/schema.sql`](../backend/app/db/schema.sql).

```mermaid
erDiagram
  USERS {
    TEXT user_id PK "UUID"
    TEXT display_name
    TEXT preferred_lang "default en"
    TEXT cloud_account "NULL until linked"
    TEXT settings_json "JSON: thresholds, voice, etc."
    REAL created_at
    REAL updated_at
    REAL deleted_at "tombstone for sync"
  }

  MONITORING_RECORDS {
    TEXT record_id PK "UUID; one row per drive/session"
    TEXT user_id FK
    REAL started_at
    REAL ended_at
    REAL avg_fatigue
    REAL max_fatigue
    INTEGER alert_count "default 0"
    TEXT summary
    REAL updated_at
  }

  MONITORING_LOGS {
    INTEGER log_id PK "AUTOINCREMENT"
    TEXT record_id FK
    REAL ts
    TEXT driver_state "ATTENTIVE/EYES_CLOSED/HEAD_DROP/YAWNING/DISTRACTED/POSTURE/NO_FACE/SENSOR_LOST"
    TEXT drowsiness_level "CHECK: ALERT/MILD/MODERATE/HIGH"
    REAL fatigue_score
    REAL ear
    REAL mar
    TEXT safety_prompt "NULL if none"
    TEXT emotion "speech-emotion label"
    INTEGER synced "default 0"
  }

  CALIBRATION_PROFILES {
    INTEGER profile_id PK "AUTOINCREMENT"
    TEXT user_id FK
    REAL created_at
    INTEGER frames
    TEXT profile_json "JSON: base, sigma, frames, created_at, calibrated"
  }

  SYNC_STATE {
    INTEGER id PK "singleton row id=1"
    REAL last_success_at
    REAL last_attempt_at
    TEXT last_bundle_hash
    TEXT last_error
  }

  SCHEMA_VERSION {
    INTEGER version "currently 1"
  }

  USERS ||--o{ MONITORING_RECORDS : "has sessions"
  MONITORING_RECORDS ||--o{ MONITORING_LOGS : "has events"
  USERS ||--o{ CALIBRATION_PROFILES : "has calibration profiles"
```

Indexes and pragmas:

- `PRAGMA journal_mode = WAL`, `foreign_keys = ON`, `synchronous = NORMAL`
- `idx_records_user_time(user_id, started_at DESC)`
- `idx_logs_record_ts(record_id, ts)`
- `idx_logs_unsynced(synced) WHERE synced = 0`
- `idx_calibration_user_time(user_id, created_at DESC)`

Notes:

- The newest `calibration_profiles` row is the active driver baseline; only the latest 10 rows per user are kept.
- `profile_json` contains robust neutral baselines: `base` and `sigma` values for `ear`, `mar`, head pose, gaze and posture features.
- `sync_state` is a singleton bookkeeping row. The actual encrypted sync bundle is a separate file, not a DB table.

## 2. LanceDB — `data/vectors/`

Source of truth: [`backend/app/db/vector_store.py`](../backend/app/db/vector_store.py). The table is named `memories`; this is the long-term memory store.

```mermaid
erDiagram
  MEMORIES {
    string memory_id PK "UUID"
    string user_id "logical partition; matches SQLite users.user_id"
    string context_text "fact or preference text"
    vector384 embedding "384-dim embedding, cosine distance"
    string category "preference/personal_fact/routine/health/navigation/other"
    string topic "short topic label"
    float created_at
    float updated_at
  }
```

Notes:

- `user_id` is not an enforced foreign key in LanceDB, but it logically matches `USERS.user_id` in SQLite.
- Inserts are skipped when cosine similarity to an existing memory for the same user is `>= 0.92`.
- Retrieval keeps results with cosine score `>= 0.35`; the dashboard limit is 200 rows per user.
- Embeddings come from the configured sentence-transformer model; `EMBED_DIM = 384`.

## Cross-store sync bundle

The sync manager packages both stores into one encrypted `DCP1` blob (`data/cloud_outbox/<user_id>.dcp` by default, or `CLOUD_URL` when configured).

```mermaid
flowchart LR
  SQLITE[("data/driver.db")] --> BUNDLE["encrypted bundle<br/>driver.db + vectors/**"]
  VECTORS[("data/vectors/**")] --> BUNDLE
  KEY["data/.profile.key<br/>or OS keyring"] --> BUNDLE
  BUNDLE --> OUTBOX[("data/cloud_outbox/user.dcp")]
  BUNDLE --> CLOUD["CLOUD_URL/profiles/user_id"]
```
