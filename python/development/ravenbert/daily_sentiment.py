"""Daily headline-sentiment aggregates for the study notebooks of this folder.

One DuckDB pass per month joins the headlines of that month (``ravenpack_headlines``) with
the model's grid distribution (``headline_sentiment``, P_00..P_40 on s = linspace(-1, 1, 41))
and aggregates per (UTC day, NEWS_TYPE). Scores come from ``nlp.sentiment.ordinal_sql``:

    SENT       ordinal mean of the distribution (rule "mean"), in [-1, 1]
    CONF       1 - its standard deviation
    SENT_ARGMAX, SENT_MEDIAN   the two other rules, for robustness
    ESS        RavenPack's own EVENT_SENTIMENT_SCORE, averaged over the story's entities
               (already in [-1, 1]; present on ~30% of stories)

Stored as sums and counts, never as means, so a notebook can recombine them over news types
or weeks exactly: N (headlines), N_SCORED (a model output), SUM_SENT, SUM_SENT2, SUM_CONF,
N_NEG (SENT <= neg_max), N_POS (SENT >= pos_min), SUM_SENT_ARGMAX, SUM_SENT_MEDIAN, N_ESS,
SUM_ESS, and SUM_P_00..SUM_P_40 (the day's summed predicted distribution).

Each month is cached as ``{cache_dir}/{artifact id}/{thresholds}/{YYYY-MM}.parquet`` and read
back on later runs; the datalake is only read.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from nlp.sentiment.base import GRID_COLUMNS
from nlp.sentiment.ordinal_sql import grid_select_sql

NEG_MAX, POS_MIN = -1.0 / 3.0, 1.0 / 3.0      # the scoring's default bucket thresholds
SUM_COLUMNS = ["N", "N_SCORED", "SUM_SENT", "SUM_SENT2", "SUM_CONF", "N_NEG", "N_POS",
               "SUM_SENT_ARGMAX", "SUM_SENT_MEDIAN", "N_ESS", "SUM_ESS",
               *[f"SUM_{c}" for c in GRID_COLUMNS]]


@dataclass(frozen=True)
class Sources:
    sentiment_id: str
    headlines_id: str
    sentiment_dir: Path
    headlines_dir: Path


def locate(dl, source: str = "ravenbert", sentiment_id: str | None = None) -> Sources:
    """The headline_sentiment artifact (exact id, or the latest of ``source``) and the
    headlines artifact it was computed from."""
    if sentiment_id:
        art = dl.get(sentiment_id)
    else:
        cands = [a for a in dl.list("headline_sentiment", include_partial=False)
                 if a.meta.hyperparams.get("source") == source and not a.deprecated]
        if not cands:
            raise LookupError(f"no complete headline_sentiment artifact for source {source!r}")
        art = cands[0]
    hl = dl.get(art.meta.hyperparams["headlines_id"])
    return Sources(art.artifact_id, hl.artifact_id, art.path, hl.path)


def _month_sql(hl_file: Path, sent_file: Path, neg_max: float, pos_min: float) -> str:
    rel = f"read_parquet('{sent_file}')"
    p_sums = ", ".join(f"sum(CAST(s.{c} AS DOUBLE)) AS SUM_{c}" for c in GRID_COLUMNS)
    return f"""
    WITH h AS (
        SELECT RP_STORY_ID, CAST(CAST(TIMESTAMP_UTC AS TIMESTAMP) AS DATE) AS DATE, NEWS_TYPE,
               list_avg(list_filter(EVENT_SENTIMENT_SCORE, x -> x IS NOT NULL)) AS ESS
        FROM read_parquet('{hl_file}')),
    m AS ({grid_select_sql(rel, "mean")}),
    a AS ({grid_select_sql(rel, "argmax")}),
    md AS ({grid_select_sql(rel, "median")}),
    s AS (SELECT * FROM {rel})
    SELECT h.DATE, coalesce(h.NEWS_TYPE, '') AS NEWS_TYPE,
           count(*) AS N, count(m.SENT) AS N_SCORED,
           sum(m.SENT) AS SUM_SENT, sum(m.SENT * m.SENT) AS SUM_SENT2, sum(m.CONF) AS SUM_CONF,
           count(*) FILTER (WHERE m.SENT <= {neg_max!r}) AS N_NEG,
           count(*) FILTER (WHERE m.SENT >= {pos_min!r}) AS N_POS,
           sum(a.SENT) AS SUM_SENT_ARGMAX, sum(md.SENT) AS SUM_SENT_MEDIAN,
           count(h.ESS) AS N_ESS, sum(h.ESS) AS SUM_ESS, {p_sums}
    FROM h
    LEFT JOIN m USING (RP_STORY_ID) LEFT JOIN a USING (RP_STORY_ID)
    LEFT JOIN md USING (RP_STORY_ID) LEFT JOIN s USING (RP_STORY_ID)
    GROUP BY ALL ORDER BY 1, 2"""


def month_daily(src: Sources, month: str, *, neg_max: float = NEG_MAX,
                pos_min: float = POS_MIN, threads: int = 8) -> pl.DataFrame:
    """One month (``YYYY-MM``) of (DATE, NEWS_TYPE) sums."""
    hl_file, sent_file = src.headlines_dir / f"{month}.parquet", src.sentiment_dir / f"{month}.parquet"
    con = duckdb.connect()
    try:
        con.execute(f"SET threads={int(threads)}")
        con.execute("SET enable_progress_bar=false")
        return con.sql(_month_sql(hl_file, sent_file, neg_max, pos_min)).pl()
    finally:
        con.close()


def load_daily(src: Sources, start: date, end: date, cache_dir: Path, *,
               neg_max: float = NEG_MAX, pos_min: float = POS_MIN, threads: int = 8,
               verbose: bool = True) -> pl.DataFrame:
    """(DATE, NEWS_TYPE) sums for every month overlapping [start, end], cached per month,
    cut to [start, end]."""
    folder = Path(cache_dir) / src.sentiment_id / f"neg{neg_max:+.4f}_pos{pos_min:+.4f}"
    folder.mkdir(parents=True, exist_ok=True)
    months = pl.date_range(start.replace(day=1), end, "1mo", eager=True).dt.strftime("%Y-%m")
    frames = []
    for m in months:
        path = folder / f"{m}.parquet"
        if not path.exists():
            if verbose:
                print(f"aggregating {m} ...", flush=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            month_daily(src, m, neg_max=neg_max, pos_min=pos_min, threads=threads) \
                .write_parquet(tmp)
            tmp.replace(path)
        frames.append(pl.read_parquet(path))
    return pl.concat(frames).filter(pl.col("DATE").is_between(start, end)).sort("DATE", "NEWS_TYPE")


def summarise(sums: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Means and shares from summed rows grouped by ``by`` (e.g. ["DATE"], ["WEEK"],
    ["DATE", "NEWS_TYPE"]): MEAN_SENT, STD_SENT, MEAN_CONF, SHARE_NEG, SHARE_POS,
    NET_TONE = SHARE_POS - SHARE_NEG, MEAN_SENT_ARGMAX, MEAN_SENT_MEDIAN, MEAN_ESS,
    SHARE_SCORED, SHARE_ESS, and the mean predicted distribution Q_00..Q_40."""
    g = sums.group_by(by).agg([pl.col(c).sum() for c in SUM_COLUMNS])
    n = pl.col("N_SCORED")
    mean = pl.col("SUM_SENT") / n
    q_tot = pl.sum_horizontal([pl.col(f"SUM_{c}") for c in GRID_COLUMNS])
    return g.with_columns(
        mean.alias("MEAN_SENT"),
        (pl.col("SUM_SENT2") / n - mean ** 2).clip(lower_bound=0).sqrt().alias("STD_SENT"),
        (pl.col("SUM_CONF") / n).alias("MEAN_CONF"),
        (pl.col("N_NEG") / n).alias("SHARE_NEG"), (pl.col("N_POS") / n).alias("SHARE_POS"),
        ((pl.col("N_POS") - pl.col("N_NEG")) / n).alias("NET_TONE"),
        (pl.col("SUM_SENT_ARGMAX") / n).alias("MEAN_SENT_ARGMAX"),
        (pl.col("SUM_SENT_MEDIAN") / n).alias("MEAN_SENT_MEDIAN"),
        (pl.col("SUM_ESS") / pl.col("N_ESS")).alias("MEAN_ESS"),
        (n / pl.col("N")).alias("SHARE_SCORED"), (pl.col("N_ESS") / pl.col("N")).alias("SHARE_ESS"),
        *[(pl.col(f"SUM_{c}") / q_tot).alias(f"Q_{c[2:]}") for c in GRID_COLUMNS],
    ).sort(by)
