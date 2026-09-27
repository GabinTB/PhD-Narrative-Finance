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

    uv run python -m ravenpack.headlines.sentiment_model --model finbert \\
        run --start-year 2000 --end-year 2025 [--temp]
    uv run python -m ravenpack.headlines.sentiment_model --model ravenbert \\
        --backend local --dtype float32 --device cuda run --start-year 2026 --end-year 2026
    uv run python -m ravenpack.headlines.sentiment_model resume <partial artifact id>

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
    MonthProducer,
    ingest_to_datalake,
    resume_partial,
)

if TYPE_CHECKING:
    from datalake import ModelCard
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


def main(argv: list[str] | None = None) -> int:
    from dotenv import find_dotenv, load_dotenv

    from datalake import DatalakeIndex

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

    kwargs: dict[str, Any] = {}
    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    if args.device:
        kwargs["device"] = args.device
    repo_dir = Path(__file__).resolve().parents[4]
    with DatalakeIndex(os.environ["DATALAKE_ROOT"]) as dl:
        if args.cmd == "resume":
            art = dl.get(args.artifact_id)
            if art.meta.model_card is None:
                raise SystemExit(f"{args.artifact_id} has no model card; cannot resume it")
            try:        # rebuilt from the card; the backend must serve the recorded model
                sentimeter = sentimeter_from_card(art.meta.model_card, **kwargs)
                check_resume_compatible(art.meta.model_card, art.meta.hyperparams, sentimeter)
            except ValueError as exc:
                raise SystemExit(f"cannot resume {args.artifact_id}: {exc}") from None
            produce = producer(sentimeter, canonical_columns=bool(
                art.meta.hyperparams.get("canonical_columns")))
            art = resume_partial(dl, args.artifact_id, produce, repo_dir=repo_dir)
        else:
            if not args.model:
                raise SystemExit("run needs --model")
            if args.device and args.backend != "local":
                raise SystemExit("--device applies to the local backend only")
            sentimeter = load_sentimeter(args.model, args.backend, args.dtype,
                                         score_rule=args.score_rule,
                                         min_confidence=args.min_confidence, **kwargs)
            produce = producer(sentimeter, canonical_columns=args.canonical_columns)
            hl = (dl.get(args.headlines_artifact) if args.headlines_artifact
                  else dl.latest(SOURCE_KIND))
            art = ingest_to_datalake(
                dl, source=sentimeter.name, columns=sentimeter.columns(), produce=produce,
                headlines=hl, start_year=args.start_year, end_year=args.end_year,
                extra_hyperparams=run_hyperparams(sentimeter, args.canonical_columns),
                model_card=sentimeter.model_card(), temp=args.temp, repo_dir=repo_dir)
    print(art.artifact_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
