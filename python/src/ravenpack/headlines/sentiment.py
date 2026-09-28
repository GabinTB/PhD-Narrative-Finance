"""The ``headline_sentiment`` family: a score database contract shared by every producer.

An artifact of this kind is one parquet file per partition of its source
``ravenpack_headlines`` artifact, each holding RP_STORY_ID (String, unique, EXACTLY
the story set of that headlines partition) and ONE of:

    SENT_*        vendor scores: one or more Float32 columns, finite in [-1, 1] or NaN
                  (NaN = "no score", never zero; nulls not allowed)
    P_00..P_40    a model's output as a distribution on s = linspace(-1, 1, 41)
                  (``nlp.sentiment``): Float16, per row either all null (no model
                  output, e.g. an empty headline) or all non-null in [0, 1] with a row
                  sum in [0.99, 1.01] (Float16 carries 2^-11 relative error per value).
                  Scores and confidences are computed from it at load time
                  (``nlp.sentiment.ordinal_sql``); none is stored.

Producers (sentiment_vendor.py, sentiment_model.py) only compute a month's frame;
this module aligns it to the headlines story set, validates it, writes it atomically
and registers the artifact, so every producer obeys the same contract.

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

from datalake.jobs import Job, JobContext, Unit, register_job
from datalake.layout import Layout, layout_from_hyperparams
from datalake.periods import partition_file, period_of
from nlp.sentiment.base import GRID_COLUMNS

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex, ModelCard
    from datalake.verify import Finding

log = logging.getLogger(__name__)

KIND = "headline_sentiment"
SOURCE_KIND = "ravenpack_headlines"
ID_COL = "RP_STORY_ID"
SCORE_PREFIX = "SENT_"
GRID_ROW_SUM = (0.99, 1.01)                  # Float16 grid: accepted row-sum range
SUMMARY_FILE = "summary.json"
DECILES = tuple(round(0.1 * i, 1) for i in range(1, 10))

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"
PIPELINE_VERSION = "v0.2.0"   # v0.2.0: partition layout (partition_freq/start/end) in the id
TEMP_SUFFIX = "__TEMP"                        # = datalake.jobs.TEMP_SUFFIX

MonthProducer = Callable[[Path, str], pl.DataFrame]
"""(headlines month parquet, tag) -> frame with RP_STORY_ID + SENT_* (+ extras)."""


class SentimentContractError(ValueError):
    """A frame or file violates the headline_sentiment contract."""


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def score_columns(columns: Sequence[str]) -> list[str]:
    """The vendor score columns (SENT_*), in file order."""
    return [c for c in columns if c.startswith(SCORE_PREFIX)]


def is_grid(columns: Sequence[str]) -> bool:
    """True when the columns are a model grid table (P_00..P_40)."""
    return any(c in GRID_COLUMNS for c in columns)


def sentiment_columns(columns: Sequence[str]) -> list[str]:
    """The sentiment columns of a table: the grid P_00..P_40 when present, else SENT_*."""
    return list(GRID_COLUMNS) if is_grid(columns) else score_columns(columns)


def _validate_grid(df: pl.DataFrame, tag: str) -> list[str]:
    missing = [c for c in GRID_COLUMNS if c not in df.columns]
    if missing:
        raise SentimentContractError(f"{tag}grid table lacks {missing[:3]}")
    if score_columns(df.columns):
        raise SentimentContractError(f"{tag}a grid table carries no SENT_* column")
    for c in GRID_COLUMNS:
        if df.schema[c] != pl.Float16:
            raise SentimentContractError(f"{tag}{c} is {df.schema[c]}, expected Float16")
    if not df.height:
        return list(GRID_COLUMNS)
    nulls = df.select(pl.sum_horizontal(pl.col(c).is_null().cast(pl.Int32)
                                        for c in GRID_COLUMNS)).to_series().to_numpy()
    mixed = (nulls > 0) & (nulls < len(GRID_COLUMNS))
    if mixed.any():
        raise SentimentContractError(f"{tag}{int(mixed.sum())} row(s) with some but not all "
                                     "P_* null")
    full = df.filter(pl.col(GRID_COLUMNS[0]).is_not_null()).select(GRID_COLUMNS)
    if full.height:
        P = full.to_numpy().astype(np.float64)
        if not np.isfinite(P).all() or (P < 0).any() or (P > 1).any():
            raise SentimentContractError(f"{tag}P_* values must be finite in [0, 1]")
        tot = P.sum(axis=1)
        bad = (tot < GRID_ROW_SUM[0]) | (tot > GRID_ROW_SUM[1])
        if bad.any():
            raise SentimentContractError(f"{tag}{int(bad.sum())} row sum(s) outside "
                                         f"{list(GRID_ROW_SUM)}, e.g. {tot[bad][:3].tolist()}")
    return list(GRID_COLUMNS)


def validate_sentiment_frame(df: pl.DataFrame, *, where: str = "") -> list[str]:
    """Raise ``SentimentContractError`` unless ``df`` satisfies the contract.

    Returns the sentiment column names. Checks: RP_STORY_ID String, no null, unique;
    then either the grid rules (P_00..P_40, see the module docstring) or: at least
    one SENT_* column, every one Float32, no null, every value NaN or finite in [-1, 1].
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
    if is_grid(df.columns):
        return _validate_grid(df, tag)
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
    # a vendor score of a missing story is NaN ("no score"); a grid row stays all null
    return out.with_columns(pl.col(c).fill_null(float("nan"))
                            for c in score_columns(out.columns))


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

def write_partition(produce: MonthProducer, headlines_part: Path, out_path: Path,
                    columns: Sequence[str], tag: str) -> None:
    """Produce, align, validate and atomically write one partition."""
    t0 = time.monotonic()
    ids = headline_story_ids(headlines_part)
    frame = align_to_stories(produce(headlines_part, tag), ids, where=tag)
    got = validate_sentiment_frame(frame, where=tag)
    if got != list(columns):
        raise SentimentContractError(f"{tag}: score columns {got} != declared {list(columns)}")
    write_month(frame, out_path)
    first = pl.col(columns[0])
    n_scored = int(frame.select((first.is_not_null() if is_grid(columns)
                                 else first.is_nan().not_()).sum()).item())
    log.info("%s  wrote %d stories (%d with %s) in %.0fs", tag, frame.height, n_scored,
             columns[0], time.monotonic() - t0)


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
        write_partition(produce, headlines_dir / name, out_dir / name, columns,
                        f"[{i}/{len(todo)}] {name.removesuffix('.parquet')}")
    return len(todo)


def write_summary(out_dir: Path, columns: Sequence[str]) -> dict[str, Any]:
    """Per year: the null share and the deciles of SENT and CONF of every rule (grid
    tables: ``nlp.sentiment.ordinal_sql.RULES``) or of every SENT_* column (vendor; no
    CONF). Thresholds are scoring choices, so no bucket shares are reported here: the
    deciles show where any threshold would fall. Written to ``summary.json``."""
    import json

    import duckdb

    from nlp.sentiment.ordinal_sql import RULES, grid_select_sql, score_select_sql

    by_year: dict[str, list[Path]] = {}
    for f in sorted(out_dir.glob("*.parquet")):
        by_year.setdefault(f.stem[:4], []).append(f)
    grid = is_grid(columns)
    levels = "[" + ", ".join(str(q) for q in DECILES) + "]"
    years: dict[str, Any] = {}
    con = duckdb.connect()
    try:
        con.execute("SET enable_progress_bar=false")
        for year, files in by_year.items():
            rel = "read_parquet([" + ", ".join(f"'{f}'" for f in files) + "])"
            measures = ({rule: grid_select_sql(rel, rule) for rule in RULES} if grid
                        else {c: score_select_sql(rel, c) for c in columns})
            entry: dict[str, Any] = {}
            for name, sql in measures.items():
                n, n_null, sent_q, conf_q = con.sql(
                    f"SELECT count(*), count(*) - count(SENT), quantile_cont(SENT, {levels}), "
                    f"quantile_cont(CONF, {levels}) FROM ({sql})").fetchone()
                entry[name] = {"null_share": n_null / n if n else None,
                               "sent_deciles": sent_q, "conf_deciles": conf_q}
            first = next(iter(entry.values()))
            entry["n"], entry["null_share"] = n, first["null_share"]
            years[year] = entry
    finally:
        con.close()
    summary = {"columns": "P_00..P_40 (grid)" if grid else list(columns),
               "deciles": list(DECILES), "years": years}
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def sentiment_layout(headlines: Artifact, start: date | None = None,
                     end: date | None = None) -> Layout:
    """The headlines' partitioning over [start, end] rounded out to whole periods
    (default: the headlines' declared range)."""
    src = layout_from_hyperparams(headlines.meta.hyperparams)
    start, end = start or src.start, end or src.end
    if start is None or end is None:
        raise ValueError(f"no range given and none declared by {headlines.artifact_id}")
    return Layout(src.freq, period_of(start, src.freq).first, period_of(end, src.freq).last)


# source -> module providing ``job_from_artifact`` / ``job_from_args`` (imported lazily:
# the model producers pull in nlp backends, the vendor one reads raw zips)
SOURCE_MODULES = {"ravenpack": "ravenpack.headlines.sentiment_vendor",
                  "ravenbert": "ravenpack.headlines.sentiment_model",
                  "finbert": "ravenpack.headlines.sentiment_model"}


def _source_module(source: str) -> Any:
    import importlib

    if source not in SOURCE_MODULES:
        raise ValueError(f"unknown sentiment source {source!r}; known: {sorted(SOURCE_MODULES)}")
    return importlib.import_module(SOURCE_MODULES[source])


@register_job
class HeadlineSentimentJob(Job):
    """Headline scores (vendor or model) aligned to a ravenpack_headlines artifact."""

    kind = KIND
    pipeline_version = PIPELINE_VERSION

    def __init__(self, headlines: Artifact, layout: Layout, *, source: str,
                 columns: Sequence[str], produce: MonthProducer,
                 extra_hyperparams: dict[str, Any] | None = None,
                 model_card: ModelCard | None = None, backends: Sequence[Any] = (),
                 notes: str = "", temp: bool = False) -> None:
        self.headlines, self.layout, self.source = headlines, layout, source
        self.columns, self.produce = list(columns), produce
        self.extra_hyperparams = dict(extra_hyperparams or {})
        self._card, self._backends, self._notes, self.temp = model_card, list(backends), \
            notes, temp

    def params(self) -> dict[str, Any]:
        return {"source": self.source, "columns": ",".join(self.columns),
                "headlines_id": self.headlines.artifact_id, **self.layout.hyperparams(),
                **self.extra_hyperparams, "agent_created": self.temp}

    def sources(self) -> list[Any]:
        return [self.headlines]

    def model_card(self) -> ModelCard | None:
        return self._card

    def backends(self) -> list[Any]:
        return self._backends

    def notes(self) -> str:
        return self._notes

    def units(self) -> list[Unit]:
        keys = [p.key for p in self.layout.expected()]
        missing = [k for k in keys if not (self.headlines.path / partition_file(k)).exists()]
        if missing:
            log.warning("%d headlines partition(s) absent from %s, skipped (e.g. %s)",
                        len(missing), self.headlines.artifact_id, missing[0])
        return [Unit(k) for k in keys if k not in set(missing)]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return (out_dir / partition_file(unit.key)).exists()

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        name = partition_file(unit.key)
        write_partition(self.produce, self.headlines.path / name, ctx.out_dir / name,
                        self.columns, unit.key)

    def finalize(self, ctx: JobContext) -> None:
        n = len(list(ctx.out_dir.glob("*.parquet")))
        if not n:
            raise RuntimeError(f"no sentiment partition written for {self.layout.start}.."
                               f"{self.layout.end} from {self.headlines.artifact_id}")
        summary = write_summary(ctx.out_dir, self.columns)
        null_share = {y: v["null_share"] for y, v in summary["years"].items()}
        ctx.note(f"{n} {self.layout.freq} partition(s) written; null share per year "
                 f"{null_share}; per-year deciles of SENT / CONF in {SUMMARY_FILE}")

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: DatalakeIndex,
                      **kwargs: Any) -> HeadlineSentimentJob:
        """The source's module rebuilds its producer (and model) from the artifact."""
        hp = artifact.meta.hyperparams
        return _source_module(hp["source"]).job_from_artifact(artifact, index, **kwargs)

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        from datalake.layout import add_layout_args

        parser.add_argument("--source", required=True, choices=sorted(SOURCE_MODULES))
        parser.add_argument("--headlines-artifact", default=None,
                            help="default: the latest ravenpack_headlines")
        add_layout_args(parser, default_freq=None, required=False)
        parser.add_argument("--temp", action="store_true", help="agent-created (__TEMP)")
        vendor = parser.add_argument_group("ravenpack (vendor) source")
        vendor.add_argument("--raw-dir", default=None,
                            help="default: $RAW_DATA_PATH/RavenPack/headlines_edge_v1.0")
        model = parser.add_argument_group("model sources (ravenbert, finbert)")
        model.add_argument("--backend", default="tei", choices=["tei", "local"])
        model.add_argument("--dtype", default="float16", help="float16 (default) | float32")
        model.add_argument("--device", default=None, help="local backend: cuda|mps|cpu")
        model.add_argument("--model-path", default=None,
                           help="local backend: default the family's model path variable")
        model.add_argument("--batch-size", type=int, default=None)

    @classmethod
    def from_args(cls, args: Any, index: DatalakeIndex) -> HeadlineSentimentJob:
        from datalake.layout import layout_from_args

        headlines = (index.get(args.headlines_artifact) if args.headlines_artifact
                     else index.latest(SOURCE_KIND))
        given = layout_from_args(args, default=layout_from_hyperparams(
            headlines.meta.hyperparams))
        src_freq = layout_from_hyperparams(headlines.meta.hyperparams).freq
        if args.partition_freq and args.partition_freq != src_freq:
            raise ValueError(f"sentiment follows its headlines' partitioning ({src_freq}); "
                             "re-partitioning is its own job")
        layout = sentiment_layout(headlines, given.start, given.end)
        return _source_module(args.source).job_from_args(args, index, headlines, layout)


def ingest_to_datalake(
    index: DatalakeIndex, *, source: str, columns: Sequence[str], produce: MonthProducer,
    headlines: Artifact, start_year: int | None = None, end_year: int | None = None,
    extra_hyperparams: dict[str, Any], model_card: ModelCard | None = None, notes: str = "",
    temp: bool = False, repo_dir: Path | None = None, start: date | None = None,
    end: date | None = None, backends: Sequence[Any] = (),
) -> Artifact:
    """Fresh ``headline_sentiment`` run over the partitions of ``headlines`` (library
    entry point; runs ``HeadlineSentimentJob``).

    ``source`` names the producer (``ravenpack``, ``ravenbert``, ``finbert``);
    ``temp`` marks the artifact agent-created (``__TEMP`` in its id). The output
    has the headlines' layout (same partition keys), restricted to [start, end]
    (or legacy [start_year, end_year]) rounded out to whole periods, default the
    headlines' declared range.
    """
    from datalake.jobs import JobRunner

    if start is None and start_year is not None:
        start = date(start_year, 1, 1)
    if end is None and end_year is not None:
        end = date(end_year, 12, 31)
    job = HeadlineSentimentJob(headlines, sentiment_layout(headlines, start, end),
                               source=source, columns=columns, produce=produce,
                               extra_hyperparams=extra_hyperparams, model_card=model_card,
                               backends=backends, notes=notes, temp=temp)
    return JobRunner(index, repo_dir=repo_dir, allow_dirty=True,
                     handle_signals=False).start(job)


def resume_partial(index: DatalakeIndex, artifact_id: str, produce: MonthProducer, *,
                   repo_dir: Path | None = None, backends: Sequence[Any] = ()) -> Artifact:
    """Finish a partial ``headline_sentiment`` artifact in place with ``produce``.

    Parameters come from the artifact's own hyperparams (headlines id, range,
    columns), so a resume cannot silently change what the artifact means; the
    datalake refuses any other hyperparams. Partitions already written are
    skipped. (``jobs resume <id>`` does the same and rebuilds ``produce`` itself.)
    """
    from datalake.jobs import JobRunner

    art = index.get(artifact_id)
    if art.kind != KIND:
        raise ValueError(f"{artifact_id} is a {art.kind}, not a {KIND}")
    if not art.partial:
        raise ValueError(f"{artifact_id} is complete; start a new run instead")
    job = job_from_recorded(art, index, produce, backends=backends)
    runner = JobRunner(index, repo_dir=repo_dir, allow_dirty=True, handle_signals=False)
    return runner.resume_job(artifact_id, job)


def job_from_recorded(art: Artifact, index: DatalakeIndex, produce: MonthProducer, *,
                      backends: Sequence[Any] = ()) -> HeadlineSentimentJob:
    """The job an artifact records, with a producer supplied by its source module."""
    hp = dict(art.meta.hyperparams)
    base = {"source", "columns", "headlines_id", "agent_created",
            *layout_from_hyperparams(hp).hyperparams(), "start_year", "end_year"}
    extra = {k: v for k, v in hp.items() if k not in base}
    layout = layout_from_hyperparams(hp)
    job = HeadlineSentimentJob(index.get(hp["headlines_id"]), layout, source=hp["source"],
                               columns=hp["columns"].split(","), produce=produce,
                               extra_hyperparams=extra, model_card=art.meta.model_card,
                               backends=backends,
                               temp=art.meta.pipeline_version.endswith(TEMP_SUFFIX))
    return job


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
        if ID_COL not in names or sentiment_columns(names) != columns:
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
