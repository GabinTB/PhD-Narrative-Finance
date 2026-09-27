"""The ``headline_sentiment`` family: a score database contract shared by every producer.

An artifact of this kind is one ``YYYY-MM.parquet`` per month of its source
``ravenpack_headlines`` artifact, each holding:

    RP_STORY_ID   String, unique, EXACTLY the story set of that headlines month
    SENT_*        one or more Float32 score columns, finite in [-1, 1] or NaN
    (anything)    other columns are allowed (e.g. FinBERT's P_NEG/P_NEU/P_POS) but a
                  consumer may only select a ``SENT_*`` column as a sentiment

NaN means "no score" (no vendor event, empty headline, ...), never zero; nulls are
not allowed in score columns. Producers (sentiment_vendor.py, sentiment_model.py)
only compute a month's frame; this module aligns it to the headlines story set,
validates it, writes it atomically and registers the artifact, so every producer
obeys the same contract.

Point-in-time: a score is a function of the story as published (vendor fields
emitted with the story at TIMESTAMP_UTC, or a frozen model applied to the
headline text), so it is available at the headline's own timestamp; the
scorer joins it on RP_STORY_ID within the headline's day.

Registration follows the datalake run pattern (``index.run``); an interrupted
run is resumed with ``resume=<partial artifact id>``, which writes the missing
months into the same directory and finalises the record, exactly like
scripts/embed_headlines.py. Months already written are never recomputed.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from datalake.layout import Layout, layout_from_hyperparams
from datalake.periods import period_of

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex, ModelCard
    from datalake.verify import Finding

log = logging.getLogger(__name__)

KIND = "headline_sentiment"
SOURCE_KIND = "ravenpack_headlines"
ID_COL = "RP_STORY_ID"
SCORE_PREFIX = "SENT_"

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"
PIPELINE_VERSION = "v0.2.0"   # v0.2.0: partition layout (partition_freq/start/end) in the id
TEMP_SUFFIX = "__TEMP"

MonthProducer = Callable[[Path, str], pl.DataFrame]
"""(headlines month parquet, tag) -> frame with RP_STORY_ID + SENT_* (+ extras)."""


class SentimentContractError(ValueError):
    """A frame or file violates the headline_sentiment contract."""


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def score_columns(columns: Sequence[str]) -> list[str]:
    """The selectable score columns, in file order."""
    return [c for c in columns if c.startswith(SCORE_PREFIX)]


def validate_sentiment_frame(df: pl.DataFrame, *, where: str = "") -> list[str]:
    """Raise ``SentimentContractError`` unless ``df`` satisfies the contract.

    Returns the score column names. Checks: RP_STORY_ID String, no null, unique;
    at least one SENT_* column; every SENT_* column Float32, no null, every
    value NaN or finite in [-1, 1].
    """
    tag = f"{where}: " if where else ""
    if ID_COL not in df.columns:
        raise SentimentContractError(f"{tag}missing {ID_COL}")
    if df.schema[ID_COL] != pl.String:
        raise SentimentContractError(f"{tag}{ID_COL} is {df.schema[ID_COL]}, expected String")
    if df[ID_COL].null_count():
        raise SentimentContractError(f"{tag}{df[ID_COL].null_count()} null {ID_COL}")
    if df[ID_COL].n_unique() != df.height:
        raise SentimentContractError(f"{tag}{df.height - df[ID_COL].n_unique()} duplicate "
                                     f"{ID_COL}")
    cols = score_columns(df.columns)
    if not cols:
        raise SentimentContractError(f"{tag}no {SCORE_PREFIX}* score column")
    for c in cols:
        if df.schema[c] != pl.Float32:
            raise SentimentContractError(f"{tag}{c} is {df.schema[c]}, expected Float32")
        s = df[c]
        if s.null_count():
            raise SentimentContractError(f"{tag}{c} has {s.null_count()} null(s); use NaN")
        v = s.to_numpy()
        bad = ~np.isnan(v) & ~((v >= -1.0) & (v <= 1.0))     # inf fails the range test too
        if bad.any():
            raise SentimentContractError(
                f"{tag}{c} has {int(bad.sum())} value(s) outside [-1, 1] or infinite")
    return cols


def align_to_stories(scores: pl.DataFrame, story_ids: pl.Series, *,
                     where: str = "") -> pl.DataFrame:
    """One row per ``story_ids`` entry, in that order; stories without a score get NaN.

    Raises on duplicate ids (either side) and on scored ids that are not in
    ``story_ids``: a score for a story the headlines month does not hold means the
    two inputs disagree on the month's content, which must not be papered over.
    Float columns of missing stories become NaN; other columns stay null.
    """
    tag = f"{where}: " if where else ""
    if story_ids.n_unique() != story_ids.len():
        raise SentimentContractError(f"{tag}duplicate {ID_COL} in the headlines month")
    if scores[ID_COL].n_unique() != scores.height:
        raise SentimentContractError(f"{tag}duplicate {ID_COL} in the scores")
    base = pl.DataFrame({ID_COL: story_ids.cast(pl.String)})
    extra = scores.join(base, on=ID_COL, how="anti")
    if extra.height:
        raise SentimentContractError(
            f"{tag}{extra.height} scored stor(y/ies) absent from the headlines month, "
            f"e.g. {extra[ID_COL].head(3).to_list()}")
    out = base.join(scores, on=ID_COL, how="left", maintain_order="left")
    floats = [c for c, t in out.schema.items() if c != ID_COL and t.is_float()]
    return out.with_columns(pl.col(c).fill_null(float("nan")) for c in floats)


def headline_story_ids(headlines_month: Path) -> pl.Series:
    """RP_STORY_ID of one headlines month, in file order."""
    return pl.read_parquet(headlines_month, columns=[ID_COL])[ID_COL]


def write_month(df: pl.DataFrame, out_path: Path) -> None:
    """Atomic zstd parquet write (``.parquet.tmp`` + rename)."""
    tmp = out_path.with_suffix(".parquet.tmp")
    try:
        df.write_parquet(tmp, compression="zstd")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out_path)


def partition_names(layout: Layout) -> list[str]:
    """The partition file names a layout declares (``2008-01.parquet``, ``2008Q1.parquet``...)."""
    return [f"{p.key}.parquet" for p in layout.expected()]


def month_names(start_year: int, end_year: int) -> list[str]:
    """Legacy monthly names of [start_year, end_year] (the M layout of those years)."""
    return partition_names(Layout("M", date(start_year, 1, 1), date(end_year, 12, 31)))


# ---------------------------------------------------------------------------
# The shared ingest driver (fresh run or resume)
# ---------------------------------------------------------------------------

def fill_months(
    produce: MonthProducer, headlines_dir: Path, out_dir: Path, months: Sequence[str],
    columns: Sequence[str],
) -> int:
    """Produce, align, validate and write every month of ``months`` not yet in ``out_dir``.

    ``columns`` is the declared score-column list; every month must carry exactly
    those SENT_* columns. Headlines months absent from the source are skipped with
    a warning (some months genuinely have no data). Returns the months written.
    """
    todo = [m for m in months if (headlines_dir / m).exists() and not (out_dir / m).exists()]
    missing = [m for m in months if not (headlines_dir / m).exists()]
    if missing:
        log.warning("%d headlines month(s) absent from the source, skipped (e.g. %s)",
                    len(missing), missing[0])
    log.info("sentiment: %d month(s) to write, %d already present", len(todo),
             sum((out_dir / m).exists() for m in months))
    for i, name in enumerate(todo, 1):
        tag = f"[{i}/{len(todo)}] {name[:7]}"
        t0 = time.monotonic()
        ids = headline_story_ids(headlines_dir / name)
        frame = align_to_stories(produce(headlines_dir / name, tag), ids, where=tag)
        got = validate_sentiment_frame(frame, where=tag)
        if got != list(columns):
            raise SentimentContractError(f"{tag}: score columns {got} != declared {list(columns)}")
        write_month(frame, out_dir / name)
        n_scored = int(frame.select(pl.col(columns[0]).is_nan().not_().sum()).item())
        log.info("%s  wrote %d stories (%d with %s) in %.0fs", tag, frame.height, n_scored,
                 columns[0], time.monotonic() - t0)
    return len(todo)


def ingest_to_datalake(
    index: DatalakeIndex, *, source: str, columns: Sequence[str], produce: MonthProducer,
    headlines: Artifact, start_year: int | None = None, end_year: int | None = None,
    extra_hyperparams: dict[str, Any], model_card: ModelCard | None = None, notes: str = "",
    temp: bool = False, repo_dir: Path | None = None, start: date | None = None,
    end: date | None = None,
) -> Artifact:
    """Fresh ``headline_sentiment`` run over the partitions of ``headlines``.

    ``source`` names the producer (``ravenpack``, ``ravenbert``, ``finbert``);
    ``temp`` marks the artifact agent-created (``__TEMP`` in its id). The output
    has the headlines' layout (same partition keys), restricted to [start, end]
    (or legacy [start_year, end_year]) rounded out to whole periods, default the
    headlines' declared range.
    """
    src = layout_from_hyperparams(headlines.meta.hyperparams)
    if start is None and start_year is not None:
        start = date(start_year, 1, 1)
    if end is None and end_year is not None:
        end = date(end_year, 12, 31)
    start, end = start or src.start, end or src.end
    if start is None or end is None:
        raise ValueError(f"no range given and none declared by {headlines.artifact_id}")
    layout = Layout(src.freq, period_of(start, src.freq).first, period_of(end, src.freq).last)
    hp: dict[str, Any] = {"source": source, "columns": ",".join(columns),
                          "headlines_id": headlines.artifact_id, **layout.hyperparams(),
                          **extra_hyperparams, "agent_created": temp}
    version = PIPELINE_VERSION + (TEMP_SUFFIX if temp else "")
    with index.run(kind=KIND, pipeline=PIPELINE, pipeline_version=version,
                   pipeline_repo=PIPELINE_REPO, hyperparams=hp, sources=[headlines],
                   model_card=model_card, verifier=KIND, hash_pattern="*.parquet",
                   repo_dir=repo_dir, notes=notes) as run:
        n = fill_months(produce, headlines.path, run.out_dir, partition_names(layout), columns)
        if not any(run.out_dir.glob("*.parquet")):
            raise RuntimeError(f"no sentiment partition written for {layout.start}..{layout.end}"
                               f" from {headlines.artifact_id}")
        run.note(f"{n} {layout.freq} partition(s) written")
    return index.get(run.artifact_id)


def resume_partial(index: DatalakeIndex, artifact_id: str, produce: MonthProducer, *,
                   repo_dir: Path | None = None) -> Artifact:
    """Finish a partial ``headline_sentiment`` artifact in place, then mark it complete.

    Parameters come from the artifact's own hyperparams (headlines id, years,
    columns), so a resume cannot silently change what the artifact means; the
    datalake refuses any other hyperparams (``DatalakeIndex.run(resume=...)``).
    Months already written are skipped.
    """
    art = index.get(artifact_id)
    if art.kind != KIND:
        raise ValueError(f"{artifact_id} is a {art.kind}, not a {KIND}")
    if not art.partial:
        raise ValueError(f"{artifact_id} is complete; start a new run instead")
    hp = art.meta.hyperparams
    headlines = index.get(hp["headlines_id"])
    columns = hp["columns"].split(",")
    with index.run(kind=KIND, pipeline=art.meta.pipeline,
                   pipeline_version=art.meta.pipeline_version,
                   pipeline_repo=art.meta.pipeline_repo, hyperparams=hp, repo_dir=repo_dir,
                   verifier=KIND, hash_pattern="*.parquet", resume=artifact_id) as run:
        n = fill_months(produce, headlines.path, run.out_dir,
                        partition_names(layout_from_hyperparams(hp)), columns)
        if not any(run.out_dir.glob("*.parquet")):
            raise RuntimeError(f"still no output in {run.out_dir} after resume")
        run.note(f"{n} month(s) written")
    return index.get(artifact_id)


# ---------------------------------------------------------------------------
# Content verifier (entry point in pyproject.toml)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: Artifact) -> list[Finding]:
    """Contract + story-set bijection with the source headlines month.

    Every month file is checked for schema, declared columns and value range
    (parquet footers only for the schema; values on a first/middle/last sample,
    as in embed.py); the sampled months are also checked for an exact
    RP_STORY_ID bijection with the headlines artifact recorded in the hyperparams.
    """
    from datalake import DatalakeError, DatalakeIndex
    from datalake.verify import Finding, Severity

    aid = artifact.artifact_id
    findings: list[Finding] = []

    def err(msg: str) -> None:
        findings.append(Finding(Severity.ERROR, aid, msg))

    hp = artifact.meta.hyperparams
    for key in ("source", "columns", "headlines_id"):
        if key not in hp:
            err(f"hyperparams missing {key}")
    layout = layout_from_hyperparams(hp)
    if layout.start is None or layout.end is None:
        err("hyperparams declare no range (start/end or start_year/end_year)")
    if findings:
        return findings
    columns = hp["columns"].split(",")
    present = sorted(p.name for p in artifact.path.glob("*.parquet"))
    if not present:
        err("no partition files")
        return findings
    outside = set(present) - set(partition_names(layout))
    if outside:
        err(f"{len(outside)} file(s) outside the declared range: {sorted(outside)[:3]}")
    for name in present:
        names = pq.read_schema(artifact.path / name).names
        if ID_COL not in names or score_columns(names) != columns:
            err(f"{name}: columns {names} do not match declared {columns}")

    headlines = None
    try:
        root = artifact.path.parents[2]            # {root}/{layer}/{kind}/{artifact_id}
        with DatalakeIndex(root, create=False) as dl:
            headlines = dl.get(hp["headlines_id"])
    except (DatalakeError, OSError, IndexError) as exc:
        findings.append(Finding(Severity.WARNING, aid,
                                f"source headlines not resolvable ({exc}); bijection unchecked"))
    for name in sorted({present[0], present[len(present) // 2], present[-1]}):
        try:
            frame = pl.read_parquet(artifact.path / name)
            validate_sentiment_frame(frame, where=name)
        except (SentimentContractError, OSError) as exc:
            err(str(exc))
            continue
        if headlines is not None:
            src = headlines.path / name
            if not src.exists():
                err(f"{name}: no such month in {headlines.artifact_id}")
                continue
            ids = headline_story_ids(src)
            if ids.len() != frame.height or not ids.sort().equals(frame[ID_COL].sort()):
                err(f"{name}: RP_STORY_ID set differs from {headlines.artifact_id}")
    return findings


__all__ = [
    "KIND", "SCORE_PREFIX", "ID_COL", "SentimentContractError", "score_columns",
    "validate_sentiment_frame", "align_to_stories", "headline_story_ids", "write_month",
    "month_names", "partition_names", "fill_months", "ingest_to_datalake", "resume_partial",
    "verify_artifact",
]
