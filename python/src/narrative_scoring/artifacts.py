"""Datalake registration and the chronological replay driver.

Families (registered exactly like mu_asof / headline_embeddings: slug id,
per-file content hashes, hyperparams carrying the config hashes, source
artifact ids, git revision, timestamps):

    narrative_taxonomy      (raw layer) one dir per taxonomy version: authored CSV + the
                            headline and semantic paraphrase JSONLs, validated before
                            registration; the scorer loads the taxonomy from here
    f0_monthly_partitions   one dir per F0 config, EXTENDED month by month (YYYY-MM.parquet)
    tau_asof                one dir per (F0 config, window), EXTENDED cutoff by cutoff
    narrative_daily         one dir per scoring run (YYYY-MM.parquet, run_metadata.json)
    day_diagnostics         one dir per scoring run, same file layout

There is exactly one scoring path, ``score_range_to_datalake``: it walks the
months of [start, end] in order and, for each month, first runs the monthly
tau_asof job as of the first day of that month (delay enforced there), then
scores the month's days through ``pipeline.score_dates`` with the tau/mu
providers resolving as of each day. Days for which no tau row is old enough
(cold start) feed the null partitions only. Feeding this driver historical
dates is the live loop replayed; nothing else exists.

Agent/owner boundary: anything registered with ``temp=True`` carries
``__TEMP`` in its artifact id (via the pipeline version) and
``agent_created=True`` in its hyperparams. This module never deletes an
artifact; ``mark_temp_deprecated`` only annotates and deprecates.

Readers resolve an artifact by family + config hashes (``latest_matching``),
never by path. ``upstream`` is the index holding the headline / embedding /
mu_asof families when the scorer's own families live in a sandbox root.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from datalake import Artifact, DatalakeError, DatalakeIndex
from datalake.meta import git_commit
from narrative_scoring.calibration import load_mu_asof
from narrative_scoring.config import ScoringConfig
from narrative_scoring.corrections import Correction
from narrative_scoring.partitions import (
    MonthlyNullPartitionWriter,
    load_partitions,
    month_range_days,
    months_between,
)
from narrative_scoring.pipeline import ParquetMonthWriter, score_dates
from narrative_scoring.primitives import (
    PARAPHRASE_JSONL,
    TAXONOMY_CSV,
    PrimitiveTable,
    file_sha1,
    load_primitive_table,
)
from narrative_scoring.schema import (
    DAY_DIAGNOSTICS_SCHEMA,
    F0_PARTITION_SCHEMA,
    NARRATIVE_DAILY_SCHEMA,
    TAU_ASOF_SCHEMA,
)
from narrative_scoring.streaming import HeadlineSource, ParquetHeadlineSource
from narrative_scoring.tau_asof import (
    WINDOW_DEFAULT,
    TauSeriesProvider,
    build_tau_rows,
    latest_cutoff,
)

log = logging.getLogger(__name__)

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"
PIPELINE_VERSION = "v2.1.0"
TEMP_SUFFIX = "__TEMP"

KIND_TAXONOMY = "narrative_taxonomy"
KIND_PARTITIONS = "f0_monthly_partitions"
KIND_TAU_ASOF = "tau_asof"
KIND_NARRATIVE_DAILY = "narrative_daily"
KIND_DAY_DIAGNOSTICS = "day_diagnostics"
KIND_HEADLINES = "ravenpack_headlines"
KIND_EMBEDDINGS = "headline_embeddings"
KIND_MU_ASOF = "mu_asof"
AGENT_KINDS = (KIND_PARTITIONS, KIND_TAU_ASOF, KIND_NARRATIVE_DAILY, KIND_DAY_DIAGNOSTICS)

TAU_ASOF_FILE = "tau_asof.parquet"
PARAPHRASE_STYLES = ("headline", "semantic")
X_MIN_MONTHS = 2          # earliest scorable start = earliest data + X_MIN_MONTHS (spec)
X_FULL_MONTHS = 61        # 60-partition rolling window + the 1M delay


def _repo_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def code_version() -> str | None:
    return git_commit(_repo_dir())


def _version(temp: bool) -> str:
    return PIPELINE_VERSION + TEMP_SUFFIX if temp else PIPELINE_VERSION


# ---------------------------------------------------------------------------
# Lookup by family + config; agent marking
# ---------------------------------------------------------------------------

def latest_matching(dl: DatalakeIndex, kind: str, **hyperparams: Any) -> Artifact:
    """Complete artifact of ``kind`` whose hyperparams contain ``hyperparams``, with the most
    recent finished execution (so an extended series beats an older rebuild and vice versa)."""
    matches = [art for art in dl.list(kind)
               if all(art.meta.hyperparams.get(k) == v for k, v in hyperparams.items())]
    if not matches:
        raise DatalakeError(f"no complete {kind} artifact matching {hyperparams}")
    return max(matches, key=lambda a: (a.meta.run_end or a.meta.run_start, a.artifact_id))


TEMP_NOTE = "TEMP: agent_created=true, safe to delete"


def mark_temp_deprecated(dl: DatalakeIndex, artifact_id: str,
                         reason: str = "superseded by owner run") -> Artifact:
    """Tag an agent-created artifact TEMP (safe to delete) and deprecate it. Never deletes.

    An existing artifact's id and hyperparameters are frozen (the id is derived
    from them), so the retroactive marker lives in ``notes`` and in the
    deprecation reason; artifacts created with ``temp=True`` carry the marker
    in their id and hyperparams from the start.
    """
    dl.annotate(artifact_id, TEMP_NOTE)
    dl.deprecate(artifact_id, f"{reason} [{TEMP_NOTE}]")
    return dl.get(artifact_id)


def is_agent_created(art: Artifact) -> bool:
    return (bool(art.meta.hyperparams.get("agent_created")) or TEMP_SUFFIX in art.artifact_id
            or TEMP_NOTE in (art.meta.notes or ""))


def _taxonomy_params(table: PrimitiveTable) -> dict[str, Any]:
    return {"taxonomy_name": table.name, "taxonomy_sha1": table.taxonomy_sha1[:12],
            "paraphrase_sha1": table.paraphrase_sha1[:12], "paraphrase_style": table.style}


# ---------------------------------------------------------------------------
# narrative_taxonomy: the scorer's taxonomy input, registered and versioned
# ---------------------------------------------------------------------------

def taxonomy_files(name: str) -> list[str]:
    return [TAXONOMY_CSV.format(name=name)] + [
        PARAPHRASE_JSONL.format(name=name, style=st) for st in PARAPHRASE_STYLES]


def register_taxonomy(dl: DatalakeIndex, source_root: Path, name: str, *,
                      temp: bool = False) -> Artifact:
    """Validate and register ``{name}`` (authored CSV + both paraphrase JSONLs).

    Both styles are loaded through ``load_primitive_table`` first (vendor columns, CSV<->JSONL
    bijection on the sha1 path, uniform K, key uniqueness), so an invalid taxonomy is never
    registered. Registering byte-identical files again returns the existing artifact.
    """
    source_root = Path(source_root)
    files = [source_root / f for f in taxonomy_files(name)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"taxonomy {name!r}: missing {missing} under {source_root}")
    tables = {st: load_primitive_table(source_root, name, st) for st in PARAPHRASE_STYLES}
    t = tables["headline"]
    shas = {f.name: file_sha1(f) for f in files}
    hp = {"taxonomy_name": name, "taxonomy_sha1": t.taxonomy_sha1[:12],
          "headline_sha1": shas[files[1].name][:12], "semantic_sha1": shas[files[2].name][:12],
          "n_primitives": t.n_primitives, "n_narratives": t.n_narratives,
          "k": t.k_paraphrases, "has_observability_channel": t.has_observability_channel,
          "agent_created": temp}
    for art in dl.list(KIND_TAXONOMY):
        same = all(art.meta.hyperparams.get(k) == v for k, v in hp.items())
        if same:
            log.info("taxonomy %s already registered as %s", name, art.artifact_id)
            return art
    with dl.run(kind=KIND_TAXONOMY, pipeline=PIPELINE, pipeline_version=_version(temp),
                pipeline_repo=PIPELINE_REPO, hyperparams=hp, repo_dir=_repo_dir(),
                layer="raw", verifier=KIND_TAXONOMY, hash_pattern="*",
                notes=f"copied from {source_root}") as run:
        for f in files:
            shutil.copy2(f, run.out_dir / f.name)
        run.note(f"{t.n_primitives} primitives, {t.n_narratives} narratives, K={t.k_paraphrases}, "
                 f"observability channel {'present' if t.has_observability_channel else 'absent'}")
    return dl.get(run.artifact_id)


def resolve_taxonomy(indexes: list[DatalakeIndex], *, name: str | None = None,
                     artifact_id: str | None = None) -> Artifact:
    """A registered taxonomy: the exact ``artifact_id`` if given, else the latest of ``name``."""
    if (name is None) == (artifact_id is None):
        raise ValueError("give exactly one of name / artifact_id")
    for dl in indexes:
        try:
            art = (dl.get(artifact_id) if artifact_id
                   else latest_matching(dl, KIND_TAXONOMY, taxonomy_name=name))
        except DatalakeError:
            continue
        if art.kind != KIND_TAXONOMY:
            raise DatalakeError(f"{artifact_id} is a {art.kind}, not a {KIND_TAXONOMY}")
        return art
    what = artifact_id or name
    raise DatalakeError(
        f"no registered {KIND_TAXONOMY} {what!r}; register it first:  "
        f"uv run python -m narrative_scoring.jobs register-taxonomy --taxonomy {name or '<NAME>'}")


def load_registered_table(art: Artifact, style: str, *,
                          include_master: bool = True) -> PrimitiveTable:
    """Load the primitive table from a registered taxonomy artifact's own files."""
    name = art.meta.hyperparams["taxonomy_name"]
    return load_primitive_table(art.path, name, style, include_master=include_master)


# ---------------------------------------------------------------------------
# f0_monthly_partitions / tau_asof lookups
# ---------------------------------------------------------------------------

def partitions_params(config: ScoringConfig, table: PrimitiveTable, seed: int,
                      temp: bool) -> dict[str, Any]:
    return {**_taxonomy_params(table), "f0_config_id": config.f0_digest(),
            "mode": config.mode.value, "pooling": config.paraphrase_pooling.value,
            "trim_frac": config.trim_frac, "draws_per_headline": config.null_draws_per_headline,
            "seed": seed, "agent_created": temp}


def find_partitions(dl: DatalakeIndex, config: ScoringConfig, table: PrimitiveTable,
                    seed: int = 0, temp: bool = False) -> Artifact | None:
    try:
        return latest_matching(dl, KIND_PARTITIONS, f0_config_id=config.f0_digest(),
                               taxonomy_sha1=table.taxonomy_sha1[:12],
                               paraphrase_sha1=table.paraphrase_sha1[:12], seed=seed,
                               agent_created=temp)
    except DatalakeError:
        return None


def tau_params(config: ScoringConfig, table: PrimitiveTable, window: str, seed: int,
               temp: bool) -> dict[str, Any]:
    return {**_taxonomy_params(table), "f0_config_id": config.f0_digest(), "window": window,
            "alpha": config.alpha, "mode": config.mode.value,
            "pooling": config.paraphrase_pooling.value, "seed": seed,
            "min_month_draws": config.min_month_draws, "agent_created": temp}


def find_tau_asof(dl: DatalakeIndex, config: ScoringConfig, table: PrimitiveTable,
                  window: str = WINDOW_DEFAULT, seed: int = 0,
                  temp: bool = False) -> Artifact | None:
    try:
        return latest_matching(dl, KIND_TAU_ASOF, **tau_params(config, table, window, seed, temp))
    except DatalakeError:
        return None


def load_tau_series(art: Artifact | None) -> pl.DataFrame:
    if art is None or not (art.path / TAU_ASOF_FILE).exists():
        return pl.DataFrame(schema=TAU_ASOF_SCHEMA)
    return pl.read_parquet(art.path / TAU_ASOF_FILE)


# ---------------------------------------------------------------------------
# Upstream inputs
# ---------------------------------------------------------------------------

def headline_source(upstream: DatalakeIndex, chunk_size: int = 8_192, threads: int = 8,
                    sentiment: Any = None) -> ParquetHeadlineSource:
    hl, em = upstream.latest(KIND_HEADLINES), upstream.latest(KIND_EMBEDDINGS)
    return ParquetHeadlineSource(hl.path, em.path, chunk_size=chunk_size, threads=threads,
                                 source_id=f"{hl.artifact_id}+{em.artifact_id}",
                                 sentiment=sentiment)


def earliest_headline_day(headlines: Artifact) -> date:
    """First calendar day with a headline: min TIMESTAMP_UTC of the first monthly file."""
    files = headlines.files()
    if not files:
        raise DatalakeError(f"{headlines.artifact_id} has no monthly files")
    conn = duckdb.connect()
    try:
        row = conn.sql(
            f"SELECT MIN(CAST(TIMESTAMP_UTC AS DATE)) FROM read_parquet('{files[0]}')"
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None:
        raise DatalakeError(f"{files[0].name} holds no timestamps")
    return row[0]


def _mu_inputs(upstream: DatalakeIndex, config: ScoringConfig):
    if config.mode is Correction.RAW:
        return None, None
    mu_art = upstream.latest(KIND_MU_ASOF)
    return mu_art, load_mu_asof(next(mu_art.path.glob("*.parquet")))


# ---------------------------------------------------------------------------
# The monthly tau_asof job
# ---------------------------------------------------------------------------

def build_tau_asof(
    dl: DatalakeIndex, config: ScoringConfig, table: PrimitiveTable, P: np.ndarray, *,
    upstream: DatalakeIndex | None = None, window: str = WINDOW_DEFAULT, seed: int = 0,
    today: date | None = None, rebuild: bool = False, temp: bool = False,
    partitions_art: Artifact | None = None, taxonomy_id: str | None = None,
) -> Artifact | None:
    """Append every cutoff <= today - 1M not yet in the series (or rebuild it all).

    Self-contained: N_eff and the spectrum are recomputed per cutoff from the
    mu_asof row available then. Returns None when no partition is old enough
    yet (cold start), otherwise the tau_asof artifact. ``partitions_art`` lets
    the replay driver pass the partitions artifact it is still extending
    (partial until its run closes, hence invisible to ``find_partitions``).
    """
    upstream = upstream or dl
    parts_art = partitions_art or find_partitions(dl, config, table, seed, temp)
    if parts_art is None:
        log.info("tau_asof: no %s artifact yet for this config", KIND_PARTITIONS)
        return None
    parts = load_partitions(parts_art.path)
    if parts.is_empty():
        return None
    today = today or date.today()
    cutoff = latest_cutoff(parts, today)
    if cutoff is None:
        log.info("tau_asof: no partition old enough as of %s (needs month_end <= today - 1M)",
                 today)
        return None
    mu_art, mu_df = _mu_inputs(upstream, config)

    hp = tau_params(config, table, window, seed, temp)
    existing: Artifact | None = None
    prior = pl.DataFrame(schema=TAU_ASOF_SCHEMA)
    if rebuild:
        from datalake.artifact import utc_now_iso
        hp = {**hp, "rebuilt_at": utc_now_iso()}
    else:
        existing = find_tau_asof(dl, config, table, window, seed, temp)
        prior = load_tau_series(existing)
    all_cutoffs = parts.filter(pl.col("MONTH_END") <= cutoff)["MONTH_END"].to_list()
    done = set(prior["MONTH_END"].to_list())
    todo = [c for c in all_cutoffs if c not in done]
    if not todo and existing is not None:
        return existing

    sources = [parts_art] + ([mu_art] if mu_art else []) + ([taxonomy_id] if taxonomy_id else [])
    with dl.run(kind=KIND_TAU_ASOF, pipeline=PIPELINE, pipeline_version=_version(temp),
                pipeline_repo=PIPELINE_REPO, hyperparams=hp, repo_dir=_repo_dir(),
                sources=sources, verifier=KIND_TAU_ASOF, hash_pattern="*.parquet",
                extend=existing.artifact_id if existing else None) as run:
        new_rows = build_tau_rows(
            parts, config, table, P, mu_df, mu_asof_id=mu_art.artifact_id if mu_art else None,
            cutoffs=todo, window=window, seed=seed, partitions_id=parts_art.artifact_id,
            code_version=run.record.pipeline_commit)
        out = pl.concat([prior, new_rows]).sort("MONTH_END") if prior.height else new_rows
        tmp = run.out_dir / (TAU_ASOF_FILE + ".tmp")
        out.write_parquet(tmp, compression="zstd")
        tmp.replace(run.out_dir / TAU_ASOF_FILE)
        last = out.row(-1, named=True)
        run.note(f"{len(todo)} cutoff(s) added through {cutoff}; latest tau_empirical="
                 f"{last['TAU_EMPIRICAL']:.4f} tau_gauss={last['TAU_GAUSS']:.4f} "
                 f"n_partitions={last['N_PARTITIONS']} n_eff={last['N_EFF']:.3f}")
    return dl.get(run.artifact_id)


def calibration_as_of(dl: DatalakeIndex, config: ScoringConfig, table: PrimitiveTable, *,
                      upstream: DatalakeIndex | None = None, window: str = WINDOW_DEFAULT,
                      seed: int = 0, temp: bool = False) -> TauSeriesProvider:
    """The CalibrationProvider over whatever tau_asof rows exist right now (possibly none)."""
    upstream = upstream or dl
    mu_art, mu_df = _mu_inputs(upstream, config)
    tau_art = find_tau_asof(dl, config, table, window, seed, temp)
    return TauSeriesProvider(load_tau_series(tau_art), mu_df,
                             tau_source_id=tau_art.artifact_id if tau_art else "",
                             mu_asof_id=mu_art.artifact_id if mu_art else None)


# ---------------------------------------------------------------------------
# THE scoring path: chronological replay of the live loop
# ---------------------------------------------------------------------------

def _next_month_first(y: int, m: int) -> date:
    return date(y + (m == 12), 1 if m == 12 else m + 1, 1)


def score_range_to_datalake(
    dl: DatalakeIndex, start: date, end: date, config: ScoringConfig, *,
    table: PrimitiveTable, P: np.ndarray, upstream: DatalakeIndex | None = None,
    source: HeadlineSource | None = None, window: str = WINDOW_DEFAULT, seed: int = 0,
    keep_primitive_daily: bool = False, threads: int = 8,
    rss_budget_gb: float | None = 30.0, use_kernel: bool | None = None,
    temp: bool = False, label: str = "", taxonomy_id: str | None = None,
) -> dict[str, Any]:
    """Score [start, end] chronologically through the live machinery.

    Per month M in order: (1) the monthly tau_asof job as of the first day of
    M (cutoffs <= that day - 1M); (2) ``score_dates`` over M's requested days
    with providers resolving as of each day; days with no tau row old enough
    feed the partitions only. Months before ``start`` that have no partition
    yet are walked the same way (null-only), because the first tau row needs
    them. ``start`` is shifted to earliest_data + X_MIN_MONTHS when earlier,
    with a loud warning, never an error.

    Returns a summary dict with the artifact ids and counts.
    """
    upstream = upstream or dl
    if end < start:
        raise ValueError("end before start")
    source = source or headline_source(upstream, threads=threads)
    hl, em = upstream.latest(KIND_HEADLINES), upstream.latest(KIND_EMBEDDINGS)
    mu_art, _ = _mu_inputs(upstream, config)

    earliest = earliest_headline_day(hl)
    x_min = (pd.Timestamp(earliest) + pd.DateOffset(months=X_MIN_MONTHS)).date()
    if start < x_min:
        log.warning("requested start %s precedes earliest data %s + %d months; start shifted "
                    "to %s (cold start: earlier months feed the null partitions only)",
                    start, earliest, X_MIN_MONTHS, x_min)
        start = x_min
    if end < start:
        log.warning("nothing to score: end %s precedes the shifted start %s", end, start)

    parts_art = find_partitions(dl, config, table, seed, temp)
    have = (set(load_partitions(parts_art.path)["MONTH_END"].to_list())
            if parts_art else set())
    # months that end before the first mu_asof row can never be corrected, hence never
    # yield a partition: not walked at all
    first_mu = _mu_inputs(upstream, config)[1]
    first_correctable = first_mu["DATE"].min() if first_mu is not None else earliest
    months = [(y, m) for (y, m) in months_between(earliest, end)
              if month_range_days(y, m)[-1] >= first_correctable]
    hp = {**_taxonomy_params(table), "config_id": config.digest(),
          "f0_config_id": config.f0_digest(), "mode": config.mode.value,
          "pooling": config.paraphrase_pooling.value, "q": config.q,
          "split": config.sentiment_split.value, "window": window, "seed": seed,
          "start": start.isoformat(), "end": end.isoformat(), "label": label,
          "taxonomy_artifact_id": taxonomy_id, "agent_created": temp}
    sources: list[Any] = ([hl, em] + ([mu_art] if mu_art else [])
                          + ([taxonomy_id] if taxonomy_id else []))
    summary: dict[str, Any] = {"start": start, "end": end, "earliest_data": earliest,
                               "n_days_scored": 0, "n_days_null_only": 0, "peak_rss_gb": 0.0,
                               "months_finalised": [], "tau_rows": None}

    run_kw = dict(pipeline=PIPELINE, pipeline_version=_version(temp),
                  pipeline_repo=PIPELINE_REPO, repo_dir=_repo_dir())
    with dl.run(kind=KIND_NARRATIVE_DAILY, hyperparams=hp, sources=sources,
                verifier=KIND_NARRATIVE_DAILY, hash_pattern="*", **run_kw) as nd_run, \
         dl.run(kind=KIND_DAY_DIAGNOSTICS, hyperparams=hp, sources=sources,
                verifier=KIND_DAY_DIAGNOSTICS, hash_pattern="*", **run_kw) as dg_run, \
         dl.run(kind=KIND_PARTITIONS, hyperparams=partitions_params(config, table, seed, temp),
                sources=sources, verifier=KIND_PARTITIONS, hash_pattern="*.parquet",
                extend=parts_art.artifact_id if parts_art else None, **run_kw) as pt_run:
        writer = ParquetMonthWriter(nd_run.out_dir, dg_run.out_dir)
        sink = MonthlyNullPartitionWriter(
            pt_run.out_dir, config=config, table=table, seed=seed,
            input_ids={"headlines": hl.artifact_id, "embeddings": em.artifact_id,
                       "mu_asof": mu_art.artifact_id if mu_art else None},
            code_version=pt_run.record.pipeline_commit)
        last_metadata = None
        for (y, m) in months:
            all_days = month_range_days(y, m)
            days = [d for d in all_days if d <= end]
            if date(y, m, 1) < start.replace(day=1):
                if all_days[-1] in have:
                    continue                      # partition already there
                mode = "null-only (pre-start)"
            else:
                days = [d for d in days if d >= start]
                mode = "score"
            if not days:
                continue
            # (1) the monthly tau job, as of the first day of this month
            tau_art = build_tau_asof(dl, config, table, P, upstream=upstream, window=window,
                                     seed=seed, today=date(y, m, 1), temp=temp,
                                     partitions_art=dl.get(pt_run.artifact_id),
                                     taxonomy_id=taxonomy_id)
            cal = calibration_as_of(dl, config, table, upstream=upstream, window=window,
                                    seed=seed, temp=temp)
            if tau_art is not None:
                summary["tau_rows"] = load_tau_series(tau_art)
            log.info("month %d-%02d: %s, %d day(s), tau rows available: %d",
                     y, m, mode, len(days), cal.tau_df.height)
            # (2) the days, through the one pipeline
            res = score_dates(
                days, config, table=table, primitive_embeddings=P, source=source,
                calibration=cal, writer=writer, null_sink=sink, tau_missing="null_only",
                keep_primitive_daily=keep_primitive_daily, collect=False,
                use_kernel=use_kernel, threads=threads, rss_budget_gb=rss_budget_gb,
                seed=seed, code_version=nd_run.record.pipeline_commit,
                extra_metadata={"narrative_daily_id": nd_run.artifact_id,
                                "day_diagnostics_id": dg_run.artifact_id,
                                "partitions_id": pt_run.artifact_id})
            summary["n_days_scored"] += res.n_days
            summary["n_days_null_only"] += len(res.null_only_days)
            summary["peak_rss_gb"] = max(summary["peak_rss_gb"], res.peak_rss_gb)
            if res.metadata is not None:
                last_metadata = res.metadata
            if days[-1] == all_days[-1]:          # the month closed on schedule
                sink.finalise_before(_next_month_first(y, m))
        summary["months_finalised"] = [p.name for p in sink.finalised]
        if last_metadata is not None:
            writer.close(last_metadata)
        pt_run.note(f"{len(sink.finalised)} month(s) finalised in [{earliest}, {end}]")
        nd_run.note(f"{summary['n_days_scored']} day(s) scored, "
                    f"{summary['n_days_null_only']} null-only, peak RSS "
                    f"{summary['peak_rss_gb']:.2f} GB")
        dg_run.note(f"{summary['n_days_scored']} day(s)")
    summary.update(narrative_daily_id=nd_run.artifact_id, day_diagnostics_id=dg_run.artifact_id,
                   partitions_id=pt_run.artifact_id)
    tau_art = find_tau_asof(dl, config, table, window, seed, temp)
    summary["tau_asof_id"] = tau_art.artifact_id if tau_art else None
    return summary


# ---------------------------------------------------------------------------
# Content verifiers (entry points in pyproject.toml)
# ---------------------------------------------------------------------------

def _findings(kind: str, artifact: Artifact, check):
    from datalake.verify import Finding, Severity

    out: list[Finding] = []
    try:
        for msg in check():
            out.append(Finding(Severity.ERROR, artifact.artifact_id, msg))
    except Exception as exc:  # noqa: BLE001
        out.append(Finding(Severity.ERROR, artifact.artifact_id, f"{kind}: {exc}"))
    return out


def _schema_check(files: list[Path], schema: pl.Schema, label: str):
    if not files:
        yield f"{label}: no parquet files"
    for p in files[:6]:
        df = pl.read_parquet(p)
        if df.schema != schema:
            yield f"{p.name}: schema mismatch"
        if df.is_empty():
            yield f"{p.name}: empty"


def verify_taxonomy(artifact: Artifact):
    def check():
        hp = artifact.meta.hyperparams
        name = hp.get("taxonomy_name")
        if not name:
            yield "hyperparams missing taxonomy_name"
            return
        for f in taxonomy_files(name):
            if not (artifact.path / f).exists():
                yield f"{f} missing"
        for st in PARAPHRASE_STYLES:
            t = load_primitive_table(artifact.path, name, st)
            if t.n_primitives != hp.get("n_primitives"):
                yield f"{st}: {t.n_primitives} primitives, hyperparams say {hp.get('n_primitives')}"
            if t.taxonomy_sha1[:12] != hp.get("taxonomy_sha1"):
                yield "authored CSV hash differs from the registered hash"
    return _findings(KIND_TAXONOMY, artifact, check)


def verify_partitions(artifact: Artifact):
    def check():
        yield from _schema_check(artifact.files(), F0_PARTITION_SCHEMA, KIND_PARTITIONS)
    return _findings(KIND_PARTITIONS, artifact, check)


def verify_tau_asof(artifact: Artifact):
    def check():
        yield from _schema_check(artifact.files(), TAU_ASOF_SCHEMA, KIND_TAU_ASOF)
        df = pl.read_parquet(artifact.path / TAU_ASOF_FILE)
        if df["MONTH_END"].n_unique() != df.height:
            yield "duplicate MONTH_END rows"
        if not (df["TAU_EMPIRICAL"].is_finite()).all():
            yield "non-finite tau"
    return _findings(KIND_TAU_ASOF, artifact, check)


def verify_narrative_daily(artifact: Artifact):
    def check():
        yield from _schema_check(artifact.files(), NARRATIVE_DAILY_SCHEMA, KIND_NARRATIVE_DAILY)
        if not (artifact.path / "run_metadata.json").exists():
            yield "run_metadata.json missing"
        else:
            json.loads((artifact.path / "run_metadata.json").read_text())
    return _findings(KIND_NARRATIVE_DAILY, artifact, check)


def verify_day_diagnostics(artifact: Artifact):
    def check():
        yield from _schema_check(artifact.files(), DAY_DIAGNOSTICS_SCHEMA, KIND_DAY_DIAGNOSTICS)
    return _findings(KIND_DAY_DIAGNOSTICS, artifact, check)


__all__ = [
    "KIND_TAXONOMY", "register_taxonomy", "resolve_taxonomy", "load_registered_table",
    "KIND_PARTITIONS", "KIND_TAU_ASOF", "KIND_NARRATIVE_DAILY", "KIND_DAY_DIAGNOSTICS",
    "latest_matching", "mark_temp_deprecated", "is_agent_created", "find_partitions",
    "find_tau_asof",
    "build_tau_asof", "calibration_as_of", "headline_source", "earliest_headline_day",
    "score_range_to_datalake",
]
