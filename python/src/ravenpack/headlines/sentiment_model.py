"""Model sentiment (RavenBERT, FinBERT) -> a ``headline_sentiment`` artifact.

RavenPack I/O around an ``nlp.sentiment.Sentimeter``: every headline of a
``ravenpack_headlines`` month is scored by the sentimeter (``clean_text``
preprocessing, one inference pass per read batch, on TEI by default or a
local model). The month's table:

    RP_STORY_ID   String
    SENT_SCORE    Float32  the sentimeter's score in [-1, 1]; NaN below
                           ``--min-confidence`` ("no score", never 0)
    SENT_CONF     Float32  its confidence in [0, 1]
    P_*           Float16  optional (``--canonical-columns``): the canonical
                           probabilities (RavenBERT P_00..P_40, FinBERT
                           P_NEG / P_NEU / P_POS), not selectable as sentiment

The score maps, confidence measures and compatibility checks live in
``nlp.sentiment`` (``RavenbertSentimeter``: EV or median of 41 ordinal
classes; ``FinbertSentimeter``: band-consistent score). The artifact's
model card carries the backend, its full serving metadata, the sentimeter
settings and the check results; backend, dtype, score rule and the
confidence threshold are also hyperparams, so they are part of the id.

    uv run jobs start headline_sentiment --source finbert --start-year 2000 \\
        --end-year 2025 [--temp]
    uv run jobs start headline_sentiment --source ravenbert --backend local \\
        --dtype float32 --device cuda --start-year 2026 --end-year 2026
    uv run jobs resume <partial artifact id> [--opt batch_size=64 --opt device=cuda]

``resume`` takes everything from the artifact: the Sentimeter (family, score rule,
confidence threshold) and its backend are rebuilt from the model card, and the
backend must still serve the recorded model (a remote server's metadata is read
first); the canonical-column choice and the months come from the hyperparams.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from nlp.sentiment import (
    CONFIDENCE_COLUMN,
    SCORE_COLUMN,
    SENTIMETERS,
    load_sentimeter,
    sentimeter_from_card,
)
from ravenpack.headlines.sentiment import (
    ID_COL,
    SOURCE_KIND,
    HeadlineSentimentJob,
    MonthProducer,
    job_from_recorded,
    sentiment_layout,
)

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex, ModelCard
    from datalake.layout import Layout
    from nlp.sentiment import Sentimeter

log = logging.getLogger(__name__)


def producer(sentimeter: Sentimeter, *, canonical_columns: bool = False,
             read_rows: int = 200_000) -> MonthProducer:
    """headlines ``YYYY-MM.parquet`` -> RP_STORY_ID + SENT_SCORE + SENT_CONF (+ P_*), streamed."""
    names = sentimeter.canonical_columns()

    def produce(headlines_month: Path, tag: str) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        t0, n = time.monotonic(), 0
        for batch in pq.ParquetFile(headlines_month).iter_batches(
                batch_size=read_rows, columns=[ID_COL, "HEADLINE"]):
            out = sentimeter.evaluate(batch.column("HEADLINE").to_pylist())
            cols: dict[str, Any] = {SCORE_COLUMN: out["score"],
                                    CONFIDENCE_COLUMN: out["confidence"]}
            if canonical_columns:
                P16 = out["canonical"].astype(np.float16)
                cols.update({name: P16[:, i] for i, name in enumerate(names)})
            frames.append(pl.DataFrame({ID_COL: batch.column(ID_COL).to_pylist(), **cols}))
            n += batch.num_rows
            log.info("%s  %d headlines scored | %.0f/s", tag, n,
                     n / max(time.monotonic() - t0, 1e-9))
        if not frames:
            schema = {ID_COL: pl.String, SCORE_COLUMN: pl.Float32, CONFIDENCE_COLUMN: pl.Float32}
            if canonical_columns:
                schema.update({name: pl.Float16 for name in names})
            return pl.DataFrame(schema=schema)
        return pl.concat(frames)

    return produce


def run_hyperparams(sentimeter: Sentimeter, canonical_columns: bool = False) -> dict[str, Any]:
    """What changes the table: model spec version, engine, dtype, score rule, threshold,
    and whether the canonical probabilities are written."""
    hp: dict[str, Any] = {"model_version": sentimeter.version,
                          "backend": sentimeter.backend.name,
                          "dtype": sentimeter.backend.dtype,
                          "score_rule": sentimeter.score_rule,
                          "canonical_columns": canonical_columns}
    if sentimeter.min_confidence is not None:
        hp["min_confidence"] = sentimeter.min_confidence
    return hp


def check_resume_compatible(card: ModelCard | None, hyperparams: dict[str, Any],
                            sentimeter: Sentimeter) -> None:
    """Refuse to finish an artifact with a sentimeter that scores differently."""
    if card is None:
        raise ValueError("artifact has no model card; cannot verify the sentimeter")
    want = {"source": sentimeter.name,
            **run_hyperparams(sentimeter, bool(hyperparams.get("canonical_columns")))}
    got = {k: hyperparams.get(k) for k in want}
    if got != want or card.backend != sentimeter.backend.name:
        raise ValueError(f"artifact was scored with {got} (backend {card.backend}); "
                         f"this run uses {want}; refusing to mix them in one artifact")


def job(headlines: Artifact, layout: Layout, sentimeter: Sentimeter, *,
        canonical_columns: bool = False, temp: bool = False,
        read_rows: int = 200_000) -> HeadlineSentimentJob:
    """The model-sentiment job; the backend is re-checked before every partition."""
    return HeadlineSentimentJob(
        headlines, layout, source=sentimeter.name, columns=sentimeter.columns(),
        produce=producer(sentimeter, canonical_columns=canonical_columns, read_rows=read_rows),
        extra_hyperparams=run_hyperparams(sentimeter, canonical_columns),
        model_card=sentimeter.model_card(), backends=[sentimeter.backend], temp=temp)


def _backend_kwargs(batch_size: int | None, device: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if batch_size:
        kwargs["batch_size"] = batch_size
    if device:
        kwargs["device"] = device
    return kwargs


def job_from_args(args, index: DatalakeIndex, headlines: Artifact,
                  layout: Layout) -> HeadlineSentimentJob:
    if args.device and args.backend != "local":
        raise ValueError("--device applies to the local backend only")
    sentimeter = load_sentimeter(args.source, args.backend, args.dtype,
                                 score_rule=args.score_rule,
                                 min_confidence=args.min_confidence,
                                 **_backend_kwargs(args.batch_size, args.device))
    return job(headlines, layout, sentimeter, canonical_columns=args.canonical_columns,
               temp=args.temp)


def job_from_artifact(artifact: Artifact, index: DatalakeIndex, *,
                      batch_size: int | None = None, device: str | None = None,
                      read_rows: int = 200_000) -> HeadlineSentimentJob:
    """Resume: the Sentimeter and its backend are rebuilt from the model card, and
    the backend must serve the recorded model (a server's metadata is read first)."""
    card, hp = artifact.meta.model_card, artifact.meta.hyperparams
    if card is None:
        raise ValueError(f"{artifact.artifact_id} has no model card; cannot resume it")
    sentimeter = sentimeter_from_card(card, **_backend_kwargs(batch_size, device))
    check_resume_compatible(card, hp, sentimeter)
    produce = producer(sentimeter, canonical_columns=bool(hp.get("canonical_columns")),
                       read_rows=read_rows)
    return job_from_recorded(artifact, index, produce, backends=[sentimeter.backend])


def main(argv: list[str] | None = None) -> int:
    """Deprecated entry point: ``jobs start headline_sentiment --source <model> ...``."""
    from dotenv import find_dotenv, load_dotenv

    from datalake import DatalakeIndex
    from datalake.jobs import JobRunner

    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(SENTIMETERS), help="run only (resume reads it)")
    ap.add_argument("--backend", default="tei", choices=["tei", "local"])
    ap.add_argument("--dtype", default="float16", help="float16 (default) | float32")
    ap.add_argument("--device", default=None, help="local backend: cuda|mps|cpu (default auto)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="texts per request / forward pass (default: the backend's)")
    ap.add_argument("--score-rule", default=None,
                    help="ravenbert: mean (default) | median; finbert: band")
    ap.add_argument("--min-confidence", type=float, default=None,
                    help="score NaN below this confidence (default: keep every score)")
    ap.add_argument("--canonical-columns", action="store_true",
                    help="also write the canonical probabilities as P_* (Float16)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--start-year", type=int, required=True)
    run.add_argument("--end-year", type=int, required=True)
    run.add_argument("--headlines-artifact", default=None)
    run.add_argument("--temp", action="store_true", help="mark the artifact agent-created")
    res = sub.add_parser("resume")
    res.add_argument("artifact_id")
    args = ap.parse_args(argv)
    log.warning("deprecated: use `jobs start headline_sentiment --source <model>` / "
                "`jobs resume <id>`")

    repo_dir = Path(__file__).resolve().parents[4]
    with DatalakeIndex(os.environ["DATALAKE_ROOT"]) as dl:
        runner = JobRunner(dl, repo_dir=repo_dir, allow_dirty=True)
        try:
            if args.cmd == "resume":
                art = runner.resume(args.artifact_id, batch_size=args.batch_size,
                                    device=args.device)
            else:
                if not args.model:
                    raise SystemExit("run needs --model")
                args.source = args.model
                hl = (dl.get(args.headlines_artifact) if args.headlines_artifact
                      else dl.latest(SOURCE_KIND))
                layout = sentiment_layout(hl, date(args.start_year, 1, 1),
                                          date(args.end_year, 12, 31))
                art = runner.start(job_from_args(args, dl, hl, layout))
        except ValueError as exc:
            raise SystemExit(f"{args.cmd}: {exc}") from None
    print(art.artifact_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
