"""Sentiment buckets: which headlines a filtered scoring run reads.

A run with ``config.sentiment`` = positive / neutral / negative / unscored scores only the
headlines of that bucket, selected at load time in the day query (streaming.py) from a
``headline_sentiment`` artifact:

    SENT, CONF   model sources (grid tables): ``nlp.sentiment.ordinal_sql`` rule
                 ``config.sentiment_rule`` on P_00..P_40; vendor source: SENT_CSS, no CONF
    unscored     SENT null OR CONF < min_conf
    negative     SENT <= neg_max            (and not unscored)
    positive     SENT >= pos_min            (and not unscored)
    neutral      neg_max < SENT < pos_min   (and not unscored)

With -1 <= neg_max < pos_min <= 1 (checked by ScoringConfig) every headline falls in
exactly one bucket. ``bucket_of`` is the numpy twin of ``predicate_sql`` (tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from narrative_scoring.config import (
    VENDOR_SOURCES,
    ScoringConfig,
    SentimentFilter,
)
from nlp.sentiment.ordinal_sql import grid_select_sql, score_select_sql

VENDOR_COLUMN = {"ravenpack": "SENT_CSS"}


def _lit(x: float) -> str:
    return f"CAST({float(x)!r} AS DOUBLE)"


def predicate_sql(config: ScoringConfig, sent: str = "s.SENT", conf: str = "s.CONF") -> str:
    """The SQL condition selecting ``config.sentiment``'s bucket over SENT / CONF columns."""
    unscored = f"({sent} IS NULL OR coalesce({conf} < {_lit(config.min_conf)}, FALSE))"
    bucket = config.sentiment
    if bucket is SentimentFilter.UNSCORED:
        return unscored
    if bucket is SentimentFilter.NEGATIVE:
        cond = f"{sent} <= {_lit(config.neg_max)}"
    elif bucket is SentimentFilter.POSITIVE:
        cond = f"{sent} >= {_lit(config.pos_min)}"
    elif bucket is SentimentFilter.NEUTRAL:
        cond = f"{sent} > {_lit(config.neg_max)} AND {sent} < {_lit(config.pos_min)}"
    else:
        raise ValueError("no bucket predicate for sentiment=none")
    return f"(NOT {unscored} AND {cond})"


def bucket_of(sent: np.ndarray, conf: np.ndarray, config: ScoringConfig) -> np.ndarray:
    """numpy twin of ``predicate_sql``: the bucket name of every row (NaN = null)."""
    sent, conf = np.asarray(sent, dtype=np.float64), np.asarray(conf, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        unscored = np.isnan(sent) | (~np.isnan(conf) & (conf < config.min_conf))
        out = np.where(sent <= config.neg_max, SentimentFilter.NEGATIVE.value,
                       np.where(sent >= config.pos_min, SentimentFilter.POSITIVE.value,
                                SentimentFilter.NEUTRAL.value))
    return np.where(unscored, SentimentFilter.UNSCORED.value, out)


def sentiment_select_sql(config: ScoringConfig, relation: str) -> str:
    """``SELECT RP_STORY_ID, SENT, CONF`` over a sentiment relation for ``config``."""
    if config.sentiment_source in VENDOR_SOURCES:
        return score_select_sql(relation, VENDOR_COLUMN[config.sentiment_source])
    return grid_select_sql(relation, config.sentiment_rule)


@dataclass
class SentimentBucket:
    """The bucket a filtered run reads, over the files of one headline_sentiment artifact."""

    config: ScoringConfig
    sentiment_dir: Path
    artifact_id: str = ""

    def __post_init__(self) -> None:
        if not self.config.filtered:
            raise ValueError("SentimentBucket needs a config with a sentiment filter")
        self.sentiment_dir = Path(self.sentiment_dir)

    def select_sql(self, sentiment_file: Path, day_ids_sql: str) -> str:
        """SENT / CONF of the stories in ``day_ids_sql`` only (semi join before the rule's
        arithmetic, so a day costs its own rows, not the whole partition's)."""
        relation = (f"(SELECT s0.* FROM read_parquet('{sentiment_file}') s0 "
                    f"WHERE s0.RP_STORY_ID IN ({day_ids_sql}))")
        return sentiment_select_sql(self.config, relation)

    def predicate(self) -> str:
        return predicate_sql(self.config)

    def describe(self) -> str:
        c = self.config
        where = f"{self.artifact_id or self.sentiment_dir.name}"
        tail = "" if c.sentiment_source in VENDOR_SOURCES else f", min_conf={c.min_conf}"
        return (f"{where}:{c.sentiment_rule}:{c.sentiment.value}"
                f"(neg_max={c.neg_max}, pos_min={c.pos_min}{tail})")
