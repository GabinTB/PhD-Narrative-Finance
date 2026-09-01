-- Datalake index schema.
--
-- This is the only mutable file in the datalake; everything else is
-- append-only parquet plus immutable meta.json sidecars.  The index is a
-- cache over those sidecars: `datalake reindex` can rebuild it entirely by
-- walking the tree, so losing index.db loses nothing permanent.
--
-- WAL mode is set by index.py at connect time, not here (PRAGMA journal_mode
-- is a connection-level setting, not part of the schema).

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id         TEXT PRIMARY KEY,
    layer               TEXT NOT NULL CHECK (layer IN ('raw', 'derived', 'output')),
    kind                TEXT NOT NULL,
    path                TEXT NOT NULL UNIQUE,

    pipeline            TEXT NOT NULL,
    pipeline_version    TEXT NOT NULL,
    pipeline_commit     TEXT,
    pipeline_repo       TEXT,

    -- Model identity is denormalised onto the artifact row so `latest(...)`
    -- can filter on model/version without parsing the JSON blob.
    model_id            TEXT,
    model_version       TEXT,
    model_commit        TEXT,

    hyperparams_json    TEXT NOT NULL DEFAULT '{}',
    model_card_json     TEXT,

    run_start           TEXT NOT NULL,
    run_end             TEXT,

    partial             INTEGER NOT NULL DEFAULT 1 CHECK (partial IN (0, 1)),
    deprecated          INTEGER NOT NULL DEFAULT 0 CHECK (deprecated IN (0, 1)),
    deprecation_reason  TEXT,
    notes               TEXT NOT NULL DEFAULT ''
);

-- `latest(kind, model=..., version=...)` is the hottest query in the API.
CREATE INDEX IF NOT EXISTS idx_artifacts_lookup
    ON artifacts (kind, model_id, model_version, partial, deprecated, run_start DESC);

CREATE INDEX IF NOT EXISTS idx_artifacts_layer
    ON artifacts (layer, kind);


CREATE TABLE IF NOT EXISTS file_hashes (
    artifact_id  TEXT NOT NULL,
    filename     TEXT NOT NULL,
    algorithm    TEXT NOT NULL DEFAULT 'blake2b',
    digest       TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    PRIMARY KEY (artifact_id, filename),
    FOREIGN KEY (artifact_id) REFERENCES artifacts (artifact_id) ON DELETE CASCADE
);


-- Lineage: child depends on parent.  Populated from RunMeta.sources.
--
-- parent_id is intentionally NOT a foreign key: an artifact may legitimately
-- cite a source that is not itself registered (an external dataset, a raw
-- vintage tracked outside the index).  `datalake verify --lineage` reports
-- such dangling parents rather than preventing them at write time.
CREATE TABLE IF NOT EXISTS lineage (
    child_id   TEXT NOT NULL,
    parent_id  TEXT NOT NULL,
    PRIMARY KEY (child_id, parent_id),
    FOREIGN KEY (child_id) REFERENCES artifacts (artifact_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage (parent_id);
