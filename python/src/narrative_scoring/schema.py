"""Polars schemas for every artifact narrative_scoring reads and writes."""
from __future__ import annotations

import polars as pl

EMBEDDING_DIM: int = 384

# ---------------------------------------------------------------------------
# narrative_daily: the canonical artifact of the scorer (pipeline.py): one row
# per (day, narrative, sentiment label). Null, never zero, where a narrative
# had no retained match (SUPPORT == 0). N_HEADLINES is the DAY's scored
# headline count, repeated per row so ATTENTION = TOTAL_SCORE / N_HEADLINES
# needs no join; N_LABELLED is the number of those headlines carrying the
# row's sentiment label (== N_HEADLINES on "all" rows).
# ---------------------------------------------------------------------------

NARRATIVE_DAILY_SCHEMA: pl.Schema = pl.Schema(
    {
        "DATE":          pl.Date,
        "SENTIMENT":     pl.String,
        "reservoir":     pl.String,
        "dimension":     pl.String,
        "narrative":     pl.String,
        "pole":          pl.String,
        "narrative_key": pl.String,
        "n_primitives":  pl.UInt32,
        "SUPPORT":       pl.Int32,
        "TOTAL_SCORE":   pl.Float64,
        "INTENSITY":     pl.Float32,
        "STD_SCORE":     pl.Float32,
        "PEAK":          pl.Float32,
        "N_HEADLINES":   pl.Int32,
        "N_LABELLED":    pl.Int32,
    }
)

# primitive_daily: optional diagnostics at primitive grain, same statistics.
# Never the datalake artifact of record (owner ruling R1).
PRIMITIVE_DAILY_SCHEMA: pl.Schema = pl.Schema(
    {
        "DATE":                  pl.Date,
        "SENTIMENT":             pl.String,
        "reservoir":             pl.String,
        "dimension":             pl.String,
        "narrative":             pl.String,
        "pole":                  pl.String,
        "sub_mechanism":         pl.String,
        "observability_channel": pl.String,
        "primitive":             pl.String,
        "narrative_key":         pl.String,
        "SUPPORT":               pl.Int32,
        "TOTAL_SCORE":           pl.Float64,
        "INTENSITY":             pl.Float32,
        "STD_SCORE":             pl.Float32,
        "PEAK":                  pl.Float32,
        "N_HEADLINES":           pl.Int32,
        "N_LABELLED":            pl.Int32,
    }
)

# day_diagnostics: one row per scored day (its own artifact family). Selection
# funnel counts, the point-in-time inputs actually used, and memory telemetry.
#   N_F0_SURVIVORS_PRE_Q      scores >= tau over the FULL row (before the q budget)
#   N_Q_CANDIDATES            scores in the per-row top-q candidate set
#   N_RETAINED_PRE_JUMP       candidates that also clear tau (before the jump cut)
#   N_RETAINED_POST_Q_TAU     final retained (headline, primitive) pairs
#   PCT_TAU_PRUNED_WITHIN_Q   1 - N_RETAINED_PRE_JUMP / N_Q_CANDIDATES
#   PCT_JUMP_APPLIED          share of headlines whose retained set the jump cut shortened
#   MEAN_JUMP_GAP             mean cutting gap over those headlines (null when none)
DAY_DIAGNOSTICS_SCHEMA: pl.Schema = pl.Schema(
    {
        "DATE":                      pl.Date,
        "N_HEADLINES":               pl.Int64,
        "N_UNASSIGNED":              pl.Int64,
        "N_F0_SURVIVORS_PRE_Q":      pl.Int64,
        "N_Q_CANDIDATES":            pl.Int64,
        "N_RETAINED_PRE_JUMP":       pl.Int64,
        "N_RETAINED_POST_Q_TAU":     pl.Int64,
        "MEAN_RETAINED_PER_HEADLINE": pl.Float64,
        "PCT_TAU_PRUNED_WITHIN_Q":   pl.Float64,
        "PCT_JUMP_APPLIED":          pl.Float64,
        "MEAN_JUMP_GAP":             pl.Float64,
        "NARRATIVES_TOUCHED":        pl.Int32,
        "N_WITH_SENTIMENT":          pl.Int64,
        "MU_DATE":                   pl.Date,
        "MU_NORM":                   pl.Float64,
        "TAU":                       pl.Float64,
        "TAU_MONTH_END":             pl.Date,
        "N_EFF":                     pl.Float64,
        "CONFIG_ID":                 pl.String,
        "F0_CONFIG_ID":              pl.String,
        "TAU_SOURCE_ID":             pl.String,
        "MU_ASOF_ID":                pl.String,
        "SENTIMENT_SOURCE_ID":       pl.String,     # headline_sentiment artifact:column
        "RSS_GB_BEFORE":             pl.Float64,
        "RSS_GB_AFTER":              pl.Float64,
        "SECONDS":                   pl.Float64,
    }
)

# The day_diagnostics schema before SENTIMENT_SOURCE_ID existed; artifacts written with it
# stay valid (the verifier accepts both).
DAY_DIAGNOSTICS_SCHEMA_V1: pl.Schema = pl.Schema(
    {k: v for k, v in DAY_DIAGNOSTICS_SCHEMA.items() if k != "SENTIMENT_SOURCE_ID"})

# f0_monthly_partitions: one row (one file) per closed calendar month.
F0_PARTITION_SCHEMA: pl.Schema = pl.Schema(
    {
        "MONTH_END":          pl.Date,
        "F0_CONFIG_ID":       pl.String,
        "TAXONOMY_SHA1":      pl.String,
        "PARAPHRASE_SHA1":    pl.String,
        "SEED":               pl.Int64,
        "COMPRESSION":        pl.Float64,
        "N_HEADLINES":        pl.Int64,
        "N_DAYS_CLOSED":      pl.Int32,
        "N_DAYS_IN_MONTH":    pl.Int32,
        "COVERAGE":           pl.Float64,
        "N_DRAWS_AVAILABLE":  pl.Int64,
        "N_DRAWS_SAMPLED":    pl.Int64,
        "WELFORD_COUNT":      pl.Int64,
        "WELFORD_MEAN":       pl.Float64,
        "WELFORD_M2":         pl.Float64,
        "DIGEST_MEANS":       pl.List(pl.Float64),
        "DIGEST_WEIGHTS":     pl.List(pl.Float64),
        "DIGEST_MIN":         pl.Float64,
        "DIGEST_MAX":         pl.Float64,
        "RSS_PEAK_GB":        pl.Float64,
        "INPUT_IDS":          pl.String,     # JSON: source artifact ids
        "CODE_VERSION":       pl.String,
    }
)

# tau_asof: one row per month close. N_EFF & co. are recomputed at every cutoff
# from the mu_asof row available then (MU_DATE / MU_ARTIFACT_ID), so a tau
# change decomposes into null-pool drift vs geometry (N_eff) drift.
# WINDOW_MONTHS_USED > the nominal window means the min_month_draws safeguard
# extended it backwards; N_PARTITIONS < nominal identifies short-window
# (cold-start) rows, which are valid.
TAU_ASOF_SCHEMA: pl.Schema = pl.Schema(
    {
        "MONTH_END":            pl.Date,
        "TAU_EMPIRICAL":        pl.Float64,
        "TAU_GAUSS":            pl.Float64,
        "ABS_GAP":              pl.Float64,
        "POOL_Q99":             pl.Float64,
        "POOL_Q999":            pl.Float64,
        "POOL_Q9993":           pl.Float64,
        "POOL_Q9999":           pl.Float64,
        "P_TAIL":               pl.Float64,
        "POOL_COUNT":           pl.Int64,
        "POOL_MEAN":            pl.Float64,
        "POOL_STD":             pl.Float64,
        "N_PARTITIONS":         pl.Int32,
        "WINDOW_START":         pl.Date,
        "WINDOW":               pl.String,     # "5Y" | "expanding"
        "WINDOW_MONTHS_USED":   pl.Int32,
        "WINDOW_EXTENDED_BY":   pl.Int32,      # months added by the min_month_draws safeguard
        "MIN_MONTH_DRAWS":      pl.Int64,
        "N_EFF":                pl.Float64,
        "EFFECTIVE_RANK":       pl.Float64,
        "LAMBDA1_SHARE":        pl.Float64,
        "MU_DATE":              pl.Date,
        "MU_ARTIFACT_ID":       pl.String,
        "ALPHA":                pl.Float64,
        "MODE":                 pl.String,
        "POOLING":              pl.String,
        "SEED":                 pl.Int64,
        "F0_CONFIG_ID":         pl.String,
        "TAXONOMY_SHA1":        pl.String,
        "PARAPHRASE_SHA1":      pl.String,
        "EMBEDDINGS_DIGEST":    pl.String,
        "PARTITIONS_ID":        pl.String,
        "CODE_VERSION":         pl.String,
    }
)
