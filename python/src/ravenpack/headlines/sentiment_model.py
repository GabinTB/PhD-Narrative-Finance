"""Model sentiment (RavenBERT, FinBERT) -> a ``headline_sentiment`` artifact.

RavenPack I/O around an ``nlp.sentiment.Sentimeter``: every headline of a
``rp_headlines`` partition goes through the sentimeter (``clean_text``
preprocessing, one inference pass per read batch, on TEI by default or a local
model). The table stores the model's output only, as a distribution on the 41-point
grid s = linspace(-1, 1, 41):

    RP_STORY_ID   String
    P_00..P_40    Float16, the grid distribution (RavenBERT: its 41 classes; FinBERT:
                  its 3 classes spread on the bands [-1, -1/3], [-1/3, 1/3], [1/3, 1]);
                  all null for an empty cleaned headline (no model output)

No score, confidence or label is stored: they are computed at load time
(``nlp.sentiment.ordinal_sql``), so thresholds and rules stay scoring choices. The
artifact's model card carries the backend, its full serving metadata, the sentimeter
settings (band edges, cleaning) and the check results; backend, dtype and the output
spec version are also hyperparams, so they are part of the id.

    uv run jobs start headline_sentiment --source finbert --start-year 2000 \\
        --end-year 2025 [--temp]
    uv run jobs start headline_sentiment --source ravenbert --backend local \\
        --dtype float32 --device cuda --start-year 2026 --end-year 2026
    uv run jobs resume <partial artifact id> [--opt batch_size=64 --opt device=cuda]

``resume`` takes everything from the artifact: the Sentimeter family and its backend
are rebuilt from the model card, and the backend must still serve the recorded model
(a remote server's metadata is read first).
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
    GRID_COLUMNS,
    SENTIMETERS,
    load_sentimeter,
    sentimeter_from_card,
)
from ravenpack.annotations.access import latest_headlines
from ravenpack.headlines.sentiment import (
    ID_COL,
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


def grid_frame(story_ids: list[str], G: np.ndarray) -> pl.DataFrame:
    """RP_STORY_ID + P_00..P_40 (Float16); an all-NaN row of ``G`` becomes all null."""
    import pyarrow as pa

    G16 = np.asarray(G, dtype=np.float16)
    empty = np.isnan(G16).all(axis=1)
    cols: dict[str, Any] = {ID_COL: pa.array(story_ids, type=pa.string())}
    for j, name in enumerate(GRID_COLUMNS):
        cols[name] = pa.array(G16[:, j], type=pa.float16(), mask=empty)
    return pl.from_arrow(pa.table(cols))


def producer(sentimeter: Sentimeter, *, read_rows: int = 200_000) -> MonthProducer:
    """headlines partition -> RP_STORY_ID + P_00..P_40 (Float16 grid), streamed."""

    def produce(headlines_month: Path, tag: str) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        t0, n = time.monotonic(), 0
        for batch in pq.ParquetFile(headlines_month).iter_batches(
                batch_size=read_rows, columns=[ID_COL, "HEADLINE"]):
            G = sentimeter.grid_batch(batch.column("HEADLINE").to_pylist())
            frames.append(grid_frame(batch.column(ID_COL).to_pylist(), G))
            n += batch.num_rows
            log.info("%s  %d headlines scored | %.0f/s", tag, n,
                     n / max(time.monotonic() - t0, 1e-9))
        if not frames:
            return pl.DataFrame(schema={ID_COL: pl.String,
                                        **{name: pl.Float16 for name in GRID_COLUMNS}})
        return pl.concat(frames)

    return produce


def run_hyperparams(sentimeter: Sentimeter) -> dict[str, Any]:
    """What changes the table: output spec version, engine and dtype."""
    return {"model_version": sentimeter.version, "backend": sentimeter.backend.name,
            "dtype": sentimeter.backend.dtype}


def check_resume_compatible(card: ModelCard | None, hyperparams: dict[str, Any],
                            sentimeter: Sentimeter) -> None:
    """Refuse to finish an artifact with a sentimeter that scores differently."""
    if card is None:
        raise ValueError("artifact has no model card; cannot verify the sentimeter")
    want = {"source": sentimeter.name, **run_hyperparams(sentimeter)}
    got = {k: hyperparams.get(k) for k in want}
    if got != want or card.backend != sentimeter.backend.name:
        raise ValueError(f"artifact was scored with {got} (backend {card.backend}); "
                         f"this run uses {want}; refusing to mix them in one artifact")


def job(headlines: Artifact, layout: Layout, sentimeter: Sentimeter, *, temp: bool = False,
        read_rows: int = 200_000) -> HeadlineSentimentJob:
    """The model-sentiment job; the backend is re-checked before every partition."""
    return HeadlineSentimentJob(
        headlines, layout, source=sentimeter.name, columns=sentimeter.columns(),
        produce=producer(sentimeter, read_rows=read_rows),
        extra_hyperparams=run_hyperparams(sentimeter),
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
                                 model_path=getattr(args, "model_path", None),
                                 **_backend_kwargs(args.batch_size, args.device))
    return job(headlines, layout, sentimeter, temp=args.temp)


def job_from_artifact(artifact: Artifact, index: DatalakeIndex, *,
                      batch_size: int | None = None, device: str | None = None,
                      model_path: str | Path | None = None,
                      read_rows: int = 200_000) -> HeadlineSentimentJob:
    """Resume: the Sentimeter and its backend are rebuilt from the model card, and
    the backend must serve the recorded model (a server's metadata is read first)."""
    card, hp = artifact.meta.model_card, artifact.meta.hyperparams
    if card is None:
        raise ValueError(f"{artifact.artifact_id} has no model card; cannot resume it")
    sentimeter = sentimeter_from_card(card, model_path=str(model_path) if model_path else None,
                                      **_backend_kwargs(batch_size, device))
    check_resume_compatible(card, hp, sentimeter)
    produce = producer(sentimeter, read_rows=read_rows)
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
                      else latest_headlines(dl))
                layout = sentiment_layout(hl, date(args.start_year, 1, 1),
                                          date(args.end_year, 12, 31))
                art = runner.start(job_from_args(args, dl, hl, layout))
        except ValueError as exc:
            raise SystemExit(f"{args.cmd}: {exc}") from None
    print(art.artifact_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
