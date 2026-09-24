"""RavenPack vendor sentiment -> a ``headline_sentiment`` artifact (source ``ravenpack``).

One read-only pass over the raw Annotations zips (the same files ingest.py read;
the ``ravenpack_headlines`` parquet does not carry CSS or EVENT_RELEVANCE). A raw
row is one entity x event detection; per story (RP_STORY_ID):

    SENT_CSS        the Composite Sentiment Score. Story-level in the vendor feed: it
                    is asserted constant across the story's rows (a violation raises,
                    naming the story); NaN when the story has no CSS at all.
    SENT_ESS_MEAN   unweighted mean of EVENT_SENTIMENT_SCORE over the story's event rows
                    (rows with a non-null ESS, i.e. a detected event CATEGORY).
    SENT_ESS_WMEAN  sum(ESS_r * EVENT_RELEVANCE_r) / sum(EVENT_RELEVANCE_r) over the
                    event rows whose relevance is known.

Both ESS columns are NaN for a story without an event (~60-80% of stories); WMEAN
is also NaN when the known relevances sum to 0. All three are in [-1, 1] (vendor
range, checked by the contract). Every value is emitted with the story at its
TIMESTAMP_UTC, so the scores are available at publication.

Aggregation is exact and order-free: each streamed CSV batch is reduced to
per-story partial sums (min/max/count/sum), and the partials are combined at the
end of the month, so a story split across batches (or not contiguous in the file)
is still aggregated once. The month's story set is the headlines artifact's
(sentiment.align_to_stories); a raw story absent from it raises.

    uv run python -m ravenpack.headlines.sentiment_vendor run --start-year 2000 \
        --end-year 2025 [--headlines-artifact ID] [--raw-dir DIR] [--temp]
    uv run python -m ravenpack.headlines.sentiment_vendor resume <partial artifact id>
"""
from __future__ import annotations

import argparse
import logging
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv

from ravenpack.headlines.sentiment import (
    ID_COL,
    SOURCE_KIND,
    MonthProducer,
    ingest_to_datalake,
    resume_partial,
)

log = logging.getLogger(__name__)

SOURCE = "ravenpack"
COLUMNS = ["SENT_CSS", "SENT_ESS_MEAN", "SENT_ESS_WMEAN"]
RAW_FIELDS = {ID_COL: pa.string(), "CSS": pa.float64(), "EVENT_SENTIMENT_SCORE": pa.float64(),
              "EVENT_RELEVANCE": pa.float64()}
ZIP_NAME = "RavenPackAnalytics_AllEntities_1.0_{year}.zip"      # as ingest.py
MEMBER_NAME = "{stem}/{year}-{month:02d}.csv"
BLOCK_BYTES = 64 << 20                                           # CSV bytes per batch


def raw_month_batches(raw_dir: Path, year: int, month: int,
                      block_bytes: int = BLOCK_BYTES) -> Iterator[pa.RecordBatch]:
    """Stream the four needed columns of one raw month CSV, straight out of the zip."""
    zip_path = Path(raw_dir) / ZIP_NAME.format(year=year)
    if not zip_path.exists():
        raise FileNotFoundError(f"raw zip missing: {zip_path}")
    member = MEMBER_NAME.format(stem=zip_path.stem, year=year, month=month)
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"{member} not in {zip_path.name}")
        with zf.open(member) as fh:
            reader = pacsv.open_csv(
                fh, read_options=pacsv.ReadOptions(block_size=block_bytes),
                convert_options=pacsv.ConvertOptions(include_columns=list(RAW_FIELDS),
                                                     column_types=RAW_FIELDS))
            yield from reader


def _partials(batch: pl.DataFrame) -> pl.DataFrame:
    """Per-story partial sums of one batch (nulls are ignored by every aggregate)."""
    ess, rel = pl.col("EVENT_SENTIMENT_SCORE"), pl.col("EVENT_RELEVANCE")
    return batch.group_by(ID_COL).agg(
        pl.col("CSS").min().alias("css_min"), pl.col("CSS").max().alias("css_max"),
        pl.col("CSS").count().alias("css_n"),
        ess.sum().alias("ess_sum"), ess.count().alias("ess_n"),
        pl.when(ess.is_not_null()).then(rel).sum().alias("w_sum"),
        (ess * rel).sum().alias("wess_sum"),
    )


def story_scores(batches: Iterator[pa.RecordBatch | pl.DataFrame], *,
                 where: str = "") -> pl.DataFrame:
    """RP_STORY_ID + the three SENT_* columns (Float32, NaN = none) from raw row batches."""
    parts: list[pl.DataFrame] = []
    for b in batches:
        df = b if isinstance(b, pl.DataFrame) else pl.from_arrow(b)
        if not df.height:
            continue
        if (df["EVENT_RELEVANCE"] < 0).any():
            raise ValueError(f"{where}: negative EVENT_RELEVANCE")
        parts.append(_partials(df))
    if not parts:
        return pl.DataFrame(schema={ID_COL: pl.String, **{c: pl.Float32 for c in COLUMNS}})
    p = pl.concat(parts).group_by(ID_COL).agg(
        pl.col("css_min").min(), pl.col("css_max").max(), pl.col("css_n").sum(),
        pl.col("ess_sum").sum(), pl.col("ess_n").sum(), pl.col("w_sum").sum(),
        pl.col("wess_sum").sum())
    varying = p.filter(pl.col("css_min") != pl.col("css_max"))
    if varying.height:
        raise ValueError(f"{where}: CSS varies within {varying.height} stor(y/ies), e.g. "
                         f"{varying[ID_COL].head(3).to_list()}; CSS is expected story-level")
    nan = pl.lit(float("nan"))
    return p.select(
        ID_COL,
        pl.when(pl.col("css_n") > 0).then(pl.col("css_min")).otherwise(nan)
        .cast(pl.Float32).alias("SENT_CSS"),
        pl.when(pl.col("ess_n") > 0).then(pl.col("ess_sum") / pl.col("ess_n")).otherwise(nan)
        .cast(pl.Float32).alias("SENT_ESS_MEAN"),
        pl.when(pl.col("w_sum") > 0).then(pl.col("wess_sum") / pl.col("w_sum")).otherwise(nan)
        .cast(pl.Float32).alias("SENT_ESS_WMEAN"),
    )


def producer(raw_dir: Path, block_bytes: int = BLOCK_BYTES) -> MonthProducer:
    """The month producer for sentiment.fill_months: headlines ``YYYY-MM.parquet`` -> scores."""
    def produce(headlines_month: Path, tag: str) -> pl.DataFrame:
        year, month = (int(x) for x in headlines_month.stem.split("-"))
        return story_scores(raw_month_batches(raw_dir, year, month, block_bytes), where=tag)
    return produce


def main(argv: list[str] | None = None) -> int:
    from dotenv import find_dotenv, load_dotenv

    from datalake import DatalakeIndex

    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=None,
                    help="default: $RAW_DATA_PATH/RavenPack/headlines_edge_v1.0")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--start-year", type=int, required=True)
    run.add_argument("--end-year", type=int, required=True)
    run.add_argument("--headlines-artifact", default=None)
    run.add_argument("--temp", action="store_true", help="mark the artifact agent-created")
    res = sub.add_parser("resume")
    res.add_argument("artifact_id")
    args = ap.parse_args(argv)

    raw_dir = (Path(args.raw_dir) if args.raw_dir
               else Path(os.environ["RAW_DATA_PATH"]) / "RavenPack" / "headlines_edge_v1.0")
    if not raw_dir.is_dir():
        raise SystemExit(f"raw directory does not exist: {raw_dir}")
    repo_dir = Path(__file__).resolve().parents[4]
    with DatalakeIndex(os.environ["DATALAKE_ROOT"]) as dl:
        if args.cmd == "resume":
            art = resume_partial(dl, args.artifact_id, producer(raw_dir), repo_dir=repo_dir)
        else:
            hl = (dl.get(args.headlines_artifact) if args.headlines_artifact
                  else dl.latest(SOURCE_KIND))
            art = ingest_to_datalake(
                dl, source=SOURCE, columns=COLUMNS, produce=producer(raw_dir), headlines=hl,
                start_year=args.start_year, end_year=args.end_year,
                extra_hyperparams={"rule": "css+ess_mean+ess_wmean"},
                notes=f"raw_dir={raw_dir}", temp=args.temp, repo_dir=repo_dir)
    print(art.artifact_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
