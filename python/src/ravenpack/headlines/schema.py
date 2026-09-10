"""RavenPack Annotations 1.0 schemas.

Three layers:

  RAW_SCHEMA        -- columns selected from the raw CSV (one row per
                       entity-event detection).  Tunable: add columns here
                       to pull them at ingestion time without reopening zips.

  STRUCTURED_SCHEMA -- output of the ingestion pipeline: one row per
                       RP_STORY_ID, entity-level columns collapsed to aligned
                       lists.  Derived deterministically from RAW_SCHEMA.

  EMBEDDING_SCHEMA  -- output of the embedding pipeline (and of the archive
                       migration): exactly two columns, RP_STORY_ID plus a
                       fixed-size float16 EMBEDDING vector.  This is a separate
                       parquet, joined back to the structured parquet on
                       RP_STORY_ID -- the two are never merged into one file.

Column-level notes
------------------
- RP_ENTITY_ID / ENTITY_TYPE / ENTITY_NAME / COUNTRY_CODE / EVENT_SENTIMENT_SCORE
  are entity-event level in the raw CSV (one row per detection).  After
  deduplication these become aligned lists -- index i in each list refers to
  the same entity detection.  Never aggregate or reorder them independently.

- EVENT_SENTIMENT_SCORE is kept as a list for losslessness.  Downstream code
  that needs a scalar story-level score should collapse explicitly (mean, max,
  relevance-weighted, etc.) rather than assuming a specific aggregation here.

- COUNTRY_CODE is kept as a scalar (first-occurrence value) because it refers
  to the story's primary market geography, not to each entity separately.

- NEWS_TYPE and SOURCE_NAME are story-level in the raw; kept as scalars.
"""
from __future__ import annotations

import polars as pl
import pyarrow as pa

# ---------------------------------------------------------------------------
# Embedding dimensionality
# ---------------------------------------------------------------------------

EMBEDDING_DIM: int = 384

# ---------------------------------------------------------------------------
# Raw CSV schema (one row per entity-event detection)
#
# To pull additional columns at ingestion time: add them here.  The ingestion
# pipeline derives its usecols list from this schema automatically.
#
# Column name -> PyArrow type.  Types are used for casting after pd.read_csv;
# keep them consistent with what RavenPack actually emits.
# ---------------------------------------------------------------------------

RAW_SCHEMA: dict[str, pa.DataType] = {
    "TIMESTAMP_UTC":          pa.string(),
    "RP_STORY_ID":            pa.string(),
    "RP_ENTITY_ID":           pa.string(),
    "ENTITY_TYPE":            pa.string(),
    "ENTITY_NAME":            pa.string(),
    "COUNTRY_CODE":           pa.string(),
    "NEWS_TYPE":              pa.string(),
    "SOURCE_NAME":            pa.string(),
    "HEADLINE":               pa.string(),
    "EVENT_SENTIMENT_SCORE":  pa.float32(),
}

# Derived helpers -- use these instead of hardcoding column names elsewhere.
RAW_COLUMNS: list[str] = list(RAW_SCHEMA.keys())

# Entity-level columns that collapse to aligned lists after deduplication.
# Order matters: index i in each list must refer to the same detection.
ENTITY_LIST_COLS: list[str] = [
    "RP_ENTITY_ID",
    "ENTITY_TYPE",
    "ENTITY_NAME",
    "EVENT_SENTIMENT_SCORE",
]

# Columns handled explicitly at the top of STRUCTURED_SCHEMA (not repeated
# in SCALAR_COLS to avoid duplicate fields).
_EXPLICIT_COLS: frozenset[str] = frozenset({"RP_STORY_ID", "TIMESTAMP_UTC"})

# Scalar columns: first-occurrence value kept after deduplication.
SCALAR_COLS: list[str] = [
    c for c in RAW_COLUMNS
    if c not in ENTITY_LIST_COLS and c not in _EXPLICIT_COLS
]

# ---------------------------------------------------------------------------
# Structured parquet schema (one row per RP_STORY_ID)
# ---------------------------------------------------------------------------

def _structured_fields() -> list[pa.Field]:
    fields: list[pa.Field] = [
        pa.field("TIMESTAMP_UTC", pa.string()),
        pa.field("RP_STORY_ID",   pa.string()),
    ]
    for col in ENTITY_LIST_COLS:
        raw_type = RAW_SCHEMA[col]
        fields.append(pa.field(col, pa.list_(raw_type)))
    for col in SCALAR_COLS:
        fields.append(pa.field(col, RAW_SCHEMA[col]))
    return fields


STRUCTURED_SCHEMA: pa.Schema = pa.schema(_structured_fields())

# Polars equivalent -- used for reading structured parquets in downstream code.
# Polars does not have a direct list(float32); use pl.List(pl.Float32) etc.
_PA_TO_PL: dict[pa.DataType, pl.DataType] = {
    pa.string():  pl.String,
    pa.float32(): pl.Float32,
    pa.float64(): pl.Float64,
    pa.int32():   pl.Int32,
    pa.int64():   pl.Int64,
    pa.bool_():   pl.Boolean,
}


def _pa_to_polars(t: pa.DataType) -> pl.DataType:
    if pa.types.is_list(t):
        return pl.List(_pa_to_polars(t.value_type))
    if t in _PA_TO_PL:
        return _PA_TO_PL[t]
    raise NotImplementedError(f"no Polars mapping for PyArrow type {t!r}")


STRUCTURED_SCHEMA_POLARS: pl.Schema = pl.Schema(
    {f.name: _pa_to_polars(f.type) for f in STRUCTURED_SCHEMA}
)

# ---------------------------------------------------------------------------
# Embedding parquet schema (one row per RP_STORY_ID, two columns only)
#
# Canonical output schema for every embedding artifact -- both the fresh
# RavenBERT pipeline (embed.py) and the archive migration write exactly these
# two columns and nothing else.  EMBEDDING is a fixed-size float16 vector;
# polars round-trips pl.Array(pl.Float16, N) to arrow fixed_size_list<halffloat>.
# ---------------------------------------------------------------------------

EMBEDDING_SCHEMA: pl.Schema = pl.Schema(
    {
        "RP_STORY_ID": pl.String,
        "EMBEDDING":   pl.Array(pl.Float16, EMBEDDING_DIM),
    }
)
