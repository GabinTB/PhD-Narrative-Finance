"""Polars schemas for every artifact narrative_scoring reads and writes."""
from __future__ import annotations

import polars as pl

EMBEDDING_DIM: int = 384

# ---------------------------------------------------------------------------
# taxonomy_embeddings: one row per (primitive, paraphrase index); PARAPHRASE_I
# is -1 for a pooled centroid row, 0..K-1 for an individual paraphrase vector
# (MAX / MEDIAN pooling keeps all K rows per primitive; CENTROID pooling
# writes a single -1 row per primitive).
# ---------------------------------------------------------------------------

TAXONOMY_EMBEDDINGS_SCHEMA: pl.Schema = pl.Schema(
    {
        "PRIMITIVE":    pl.String,
        "PARAPHRASE_I": pl.Int32,
        "EMBEDDING":    pl.Array(pl.Float32, EMBEDDING_DIM),
    }
)

# ---------------------------------------------------------------------------
# primitive_scores: one row per (day, primitive), one parquet per month.
# ---------------------------------------------------------------------------

PRIMITIVE_SCORES_SCHEMA: pl.Schema = pl.Schema(
    {
        "DATE":      pl.Date,
        "PRIMITIVE": pl.String,
        "INTENSITY": pl.Float32,
        "SUPPORT":   pl.Int32,
        "PEAK":      pl.Float32,
    }
)

# ---------------------------------------------------------------------------
# aggregated_scores: primitive_scores rolled up to narrative / dimension /
# reservoir, produced on demand by aggregate.py (not a standalone artifact
# kind -- callers write it wherever they need it).
# ---------------------------------------------------------------------------

AGGREGATED_SCORES_SCHEMA: pl.Schema = pl.Schema(
    {
        "DATE":      pl.Date,
        "LEVEL":     pl.String,
        "NODE":      pl.String,
        "INTENSITY": pl.Float32,
        "SUPPORT":   pl.Int32,
        "PEAK":      pl.Float32,
    }
)
