"""Datalake registration and the chronological replay driver.

Families (registered exactly like mu_asof / headline_embeddings: slug id,
per-file content hashes, hyperparams carrying the config hashes, source
artifact ids, git revision, timestamps):

    headline_sentiment      (read only) a sentiment database (RP_STORY_ID + a model's 41-grid
                            P_00..P_40, see ravenpack/headlines/sentiment.py); a run with
                            sentiment tags reads its grid to give each headline its tag
    universe                (read only) a registered universe with RavenPack entity ids: the
                            asset layer of a run (assets.py)
    narrative_taxonomy      (raw layer) one dir per taxonomy version: authored CSV + the
                            headline and semantic paraphrase JSONLs, validated before
                            registration; the scorer loads the taxonomy from here
    f0_monthly_partitions   one dir per F0 config, EXTENDED month by month (YYYY-MM.parquet)
    tau_asof                one dir per (F0 config, window), EXTENDED cutoff by cutoff
    narrative_daily         one dir per scoring run (YYYY-MM.parquet, run_metadata.json),
                            one SENTIMENT label per tag ("all" without tags)
    day_diagnostics         one dir per scoring run, same file layout, one row per label
    asset_attention_daily   with an asset layer: day x label x asset counts (sparse)
    narrative_asset_daily   with an asset layer: day x label x narrative x asset sums (sparse)

There is exactly one scoring path, ``score_range_to_datalake``: it walks the
months of [start, end] in order and, for each month, first runs the monthly
tau_asof job as of the first day of that month (delay enforced there), then
scores the month's days through ``pipeline.score_dates`` with the tau/mu
providers resolving as of each day, in ONE pass: every family above is written
from the same read of each headlines partition. Days for which no tau row is old enough
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
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from datalake import Artifact, DatalakeError, DatalakeIndex
from datalake.jobs import Job, JobContext, Unit, register_job
from datalake.layout import Layout, layout_of
from datalake.meta import git_commit
from narrative_scoring.assets import (
    ASSET_ATTENTION_SCHEMA,
    NARRATIVE_ASSET_SCHEMA,
    AssetUniverse,
)
from narrative_scoring.config import ScoringConfig
from narrative_scoring.partitions import (
    NullPartitionWriter,
    SkipClosedDays,
    load_partitions,
)
from narrative_scoring.pipeline import ParquetMonthWriter, score_dates
from narrative_scoring.primitives import (
    PARAPHRASE_JSONL,
    TAXONOMY_CSV,
    PrimitiveTable,
    embeddings_digest,
    file_sha1,
    load_primitive_table,
)
from narrative_scoring.schema import (
    DAY_DIAGNOSTICS_SCHEMA,
    DAY_DIAGNOSTICS_SCHEMA_V1,
    DAY_DIAGNOSTICS_SCHEMA_V2,
    DAY_DIAGNOSTICS_SCHEMA_V3,
    DAY_DIAGNOSTICS_SCHEMA_V4,
    F0_PARTITION_SCHEMA,
    NARRATIVE_DAILY_SCHEMA,
    NARRATIVE_DAILY_SCHEMA_V1,
    TAU_ASOF_SCHEMA,
)
from narrative_scoring.streaming import HeadlineSource, PartitionHeadlineSource
from narrative_scoring.tau_asof import (
    CALIBRATION_DELAY_DEFAULT,
    CALIBRATION_FREQ_DEFAULT,
    WINDOW_DEFAULT,
    CalibrationCalendar,
    TauSeriesProvider,
    build_tau_rows,
    latest_cutoff,
)
from nlp.corrections import Correction
from nlp.reference_vector import load_reference_series
from ravenpack.annotations.access import (
    ENTITIES_KIND,
    HEADLINES_KIND,
    duckdb_day,
    entities_of,
    pin_utc,
    require_headlines,
)

log = logging.getLogger(__name__)

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"
PIPELINE_VERSION = "v2.4.0"   # v2.4.0: reads rp_headlines + rp_headline_entities (assets
#                               from the entities table), UTC-pinned DuckDB;
#                               v2.3.0: one pass (each partition read once), sentiment
#                               tags as labels, asset outputs, per-label day_diagnostics;
#                               v2.2.0: nlp embeddings, P digest in ids, provenance
TEMP_SUFFIX = "__TEMP"

KIND_TAXONOMY = "narrative_taxonomy"
KIND_PARTITIONS = "f0_monthly_partitions"
KIND_TAU_ASOF = "tau_asof"
KIND_NARRATIVE_DAILY = "narrative_daily"
KIND_DAY_DIAGNOSTICS = "day_diagnostics"
KIND_HEADLINES = HEADLINES_KIND         # rp_headlines (ravenpack.annotations)
KIND_ENTITIES = ENTITIES_KIND           # rp_headline_entities, its sibling
KIND_EMBEDDINGS = "headline_embeddings"
KIND_MU_ASOF = "mu_asof"
KIND_SENTIMENT = "headline_sentiment"
KIND_UNIVERSE = "universe"
KIND_ASSET_ATTENTION = "asset_attention_daily"
KIND_NARRATIVE_ASSET = "narrative_asset_daily"
AGENT_KINDS = (KIND_PARTITIONS, KIND_TAU_ASOF, KIND_NARRATIVE_DAILY, KIND_DAY_DIAGNOSTICS,
               KIND_ASSET_ATTENTION, KIND_NARRATIVE_ASSET)

TAU_ASOF_FILE = "tau_asof.parquet"
DEFAULT_CALENDAR = CalibrationCalendar()
RUN_CONFIG_FILE = "run_config.json"      # everything a resume needs, written at run start
# Fixed per run (recorded, enforced on resume): a block's float32 scores can move by ~1e-7
# with the block's row count (BLAS blocking), which can flip a retention at tau.
# Benchmark (2022-03-15, 371,818 headlines, three tags + MSCI World assets, 64 GB machine):
# 8192 -> 35.1k headlines/s, 32768 -> 43.2k, 65536 -> 45.5k, 131072 -> 42.5k; peak RSS
# 14.5-15.9 GB with the month held in memory.
CHUNK_SIZE = 65_536
RSS_BUDGET_GB = 48.0         # ~75% of the 64 GB machine: room for the OS and page cache
PROVENANCE_FILE = "embeddings_provenance.json"
PARAPHRASE_STYLES = ("headline", "semantic")


def _repo_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def code_version() -> str | None:
    return git_commit(_repo_dir())


def _version(temp: bool) -> str:
    return PIPELINE_VERSION + TEMP_SUFFIX if temp else PIPELINE_VERSION


# ---------------------------------------------------------------------------
# Lookup by family + config; agent marking
# ---------------------------------------------------------------------------

def latest_matching(dl: DatalakeIndex, kind: str, *, partial: bool = False,
                    **hyperparams: Any) -> Artifact:
    """Complete artifact of ``kind`` whose hyperparams contain ``hyperparams``, with the most
    recent finished execution (so an extended series beats an older rebuild and vice versa).
    ``partial=True`` looks for an interrupted (partial) one instead."""
    pool = ([a for a in dl.list(kind, include_partial=True) if a.partial] if partial
            else dl.list(kind))
    matches = [art for art in pool
               if all(art.meta.hyperparams.get(k, HP_DEFAULTS.get(k)) == v
                      for k, v in hyperparams.items())]
    if not matches:
        state = "partial" if partial else "complete"
        raise DatalakeError(f"no {state} {kind} artifact matching {hyperparams}")
    return max(matches, key=lambda a: (a.meta.run_end or a.meta.run_start, a.artifact_id))


TEMP_NOTE = "TEMP: agent_created=true, safe to delete"

# Hyperparams that are recorded only when non-default: an artifact without the key
# was built with the default (keeps every pre-existing id and lookup valid).
HP_DEFAULTS: dict[str, Any] = {"calibration_freq": CALIBRATION_FREQ_DEFAULT,
                               "calibration_delay": CALIBRATION_DELAY_DEFAULT}


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

def embedding_key(P: np.ndarray, primitive_meta: dict[str, Any] | None) -> str:
    """What produced the primitive-text embeddings: the lookup key of every family
    built from scores against them (partitions, tau_asof, narrative_daily).

    The embedder identity digest from the cache sidecar (model, backend, dtype,
    served model, pooling, dim; the texts are already in the taxonomy hashes), so
    a cache rebuilt with the same recipe keeps extending the same artifacts, and
    another recipe (e.g. bf16 -> fp16, TEI -> local) never does. An in-memory P
    without a sidecar falls back to its byte digest.
    """
    if primitive_meta and primitive_meta.get("identity_digest"):
        return str(primitive_meta["identity_digest"])
    return "bytes-" + embeddings_digest(P)


def partitions_params(config: ScoringConfig, table: PrimitiveTable, emb_key: str, seed: int,
                      temp: bool,
                      calendar: CalibrationCalendar = DEFAULT_CALENDAR) -> dict[str, Any]:
    return {**_taxonomy_params(table), "embeddings_id": emb_key,
            "f0_config_id": config.f0_digest(),
            "mode": config.mode.value, "pooling": config.paraphrase_pooling.value,
            "trim_frac": config.trim_frac, "draws_per_headline": config.null_draws_per_headline,
            "seed": seed, "agent_created": temp, **calendar.hyperparams()}


def find_partitions(dl: DatalakeIndex, config: ScoringConfig, table: PrimitiveTable,
                    emb_key: str, seed: int = 0, temp: bool = False,
                    calendar: CalibrationCalendar = DEFAULT_CALENDAR) -> Artifact | None:
    """The partitions artifact for this config, taxonomy, embedding recipe
    (``embedding_key``: null draws are scores against P) and calibration calendar."""
    try:
        return latest_matching(dl, KIND_PARTITIONS, f0_config_id=config.f0_digest(),
                               embeddings_id=emb_key,
                               taxonomy_sha1=table.taxonomy_sha1[:12],
                               paraphrase_sha1=table.paraphrase_sha1[:12], seed=seed,
                               agent_created=temp, calibration_freq=calendar.freq,
                               calibration_delay=calendar.delay)
    except DatalakeError:
        return None


def tau_params(config: ScoringConfig, table: PrimitiveTable, emb_key: str, window: str,
               seed: int, temp: bool,
               calendar: CalibrationCalendar = DEFAULT_CALENDAR) -> dict[str, Any]:
    return {**_taxonomy_params(table), "embeddings_id": emb_key,
            "f0_config_id": config.f0_digest(), "window": window,
            "alpha": config.alpha, "mode": config.mode.value,
            "pooling": config.paraphrase_pooling.value, "seed": seed,
            "min_month_draws": config.min_month_draws, "agent_created": temp,
            **calendar.hyperparams()}


def _lookup(hp: dict[str, Any], calendar: CalibrationCalendar) -> dict[str, Any]:
    """Lookup keys: the calendar is always compared (missing on an artifact = default)."""
    return {**hp, "calibration_freq": calendar.freq, "calibration_delay": calendar.delay}


def find_tau_asof(dl: DatalakeIndex, config: ScoringConfig, table: PrimitiveTable,
                  emb_key: str, window: str = WINDOW_DEFAULT, seed: int = 0,
                  temp: bool = False,
                  calendar: CalibrationCalendar = DEFAULT_CALENDAR) -> Artifact | None:
    try:
        return latest_matching(dl, KIND_TAU_ASOF, **_lookup(
            tau_params(config, table, emb_key, window, seed, temp, calendar), calendar))
    except DatalakeError:
        return None


def embeddings_provenance(P: np.ndarray, primitive_meta: dict[str, Any] | None,
                          headline_embeddings: Artifact | None) -> dict[str, Any]:
    """What produced both sides of S = H @ P.T: the primitive-text embeddings (the
    cache sidecar: backend, serving metadata, checks) and the headline embeddings
    artifact with its model card."""
    card = headline_embeddings.meta.model_card if headline_embeddings is not None else None
    return {
        "embedding_key": embedding_key(P, primitive_meta),
        "primitive_embeddings": {"digest": embeddings_digest(P), "shape": list(P.shape),
                                 **(primitive_meta or {"note": "in-memory P, no sidecar"})},
        "headline_embeddings": None if headline_embeddings is None else {
            "artifact_id": headline_embeddings.artifact_id,
            "model_card": card.to_dict() if card is not None else None},
    }


def _latest_or_none(dl: DatalakeIndex, kind: str) -> Artifact | None:
    try:
        return dl.latest(kind)
    except DatalakeError:
        return None


def record_provenance(out_dir: Path, provenance: dict[str, Any]) -> None:
    """Append this run's embedding provenance to the artifact's history.

    An extended artifact keeps one entry per run. When the recipe is the same but
    the bytes of P differ from the previous run (e.g. the cache was rebuilt through
    a non-bit-reproducible server), the artifact keeps extending and a warning is
    logged, so the change is visible in the log and in the history.
    """
    path = Path(out_dir) / PROVENANCE_FILE
    history: list[dict[str, Any]] = []
    if path.exists():
        previous = json.loads(path.read_text())
        history = previous.get("runs", [previous])
    if history:
        before = history[-1].get("primitive_embeddings", {}).get("digest")
        after = provenance["primitive_embeddings"]["digest"]
        if before and before != after:
            log.warning("%s: extended with primitive embeddings of the same recipe (%s) but "
                        "different bytes (digest %s -> %s)", Path(out_dir).name,
                        provenance["embedding_key"], before, after)
    from datalake.artifact import utc_now_iso

    history.append({"recorded_at": utc_now_iso(), **provenance})
    path.write_text(json.dumps({"runs": history}, indent=2, sort_keys=True, default=str))


def write_run_config(out_dir: Path, run_config: dict[str, Any]) -> None:
    """Everything ``resume_scoring`` needs to rerun the same pipeline, at run start."""
    (Path(out_dir) / RUN_CONFIG_FILE).write_text(
        json.dumps(run_config, indent=2, sort_keys=True, default=str))


def load_tau_series(art: Artifact | None) -> pl.DataFrame:
    if art is None or not (art.path / TAU_ASOF_FILE).exists():
        return pl.DataFrame(schema=TAU_ASOF_SCHEMA)
    return pl.read_parquet(art.path / TAU_ASOF_FILE)


# ---------------------------------------------------------------------------
# Upstream inputs
# ---------------------------------------------------------------------------

def headline_source(upstream: DatalakeIndex, chunk_size: int = 8_192, threads: int = 8,
                    sentiment: Artifact | None = None, config: ScoringConfig | None = None, *,
                    headlines: Artifact | None = None, embeddings: Artifact | None = None,
                    assets: AssetUniverse | None = None) -> HeadlineSource:
    """Headlines + embeddings (the latest, or the given ones), each partition read once
    (``PartitionHeadlineSource``), with the tag codes of ``config.tags`` from
    ``sentiment`` (a ``headline_sentiment`` grid artifact) and the assets of ``assets``
    when given. ``threads`` is unused (kept for callers)."""
    hl = require_headlines(headlines or upstream.latest(KIND_HEADLINES))
    em = embeddings or upstream.latest(KIND_EMBEDDINGS)
    tags = config.tags if config is not None else None
    if (sentiment is None) != (tags is None):
        raise ValueError("a sentiment artifact is needed iff the config has sentiment tags")
    ent = entities_of(upstream, hl) if assets is not None else None
    layout = shared_layout(hl, em, *([sentiment] if sentiment is not None else []),
                           *([ent] if ent is not None else []))
    if sentiment is not None:
        check_sentiment(sentiment, headlines_id=hl.artifact_id, config=config)
    return PartitionHeadlineSource(
        hl.path, em.path, chunk_size=chunk_size, source_id=f"{hl.artifact_id}+{em.artifact_id}",
        layout=layout, sentiment_dir=sentiment.path if sentiment is not None else None,
        tags=tags, assets=assets, entities_dir=ent.path if ent is not None else None)


def shared_layout(*artifacts: Artifact) -> Layout:
    """The one partition layout of artifacts read together file by file (headlines,
    embeddings, sentiment); refuses artifacts partitioned differently."""
    layouts = [layout_of(a) for a in artifacts]
    freqs = {lay.freq for lay in layouts}
    if len(freqs) != 1:
        raise DatalakeError("artifacts read together must share a partition frequency: "
                            + ", ".join(f"{a.artifact_id}={lay.freq}"
                                        for a, lay in zip(artifacts, layouts)))
    return Layout(freqs.pop())


def sentiment_columns(art: Artifact) -> list[str]:
    return [c for c in str(art.meta.hyperparams.get("columns", "")).split(",") if c]


def check_sentiment(art: Artifact, *, headlines_id: str, config: ScoringConfig) -> None:
    """Refuse a sentiment artifact that cannot serve ``config.tags``: wrong kind, built on
    another headlines artifact (story sets would differ), another source, or not a grid
    table (P_00..P_40, which every ordinal rule reads)."""
    hp = art.meta.hyperparams
    if config.tags is None:
        raise ValueError("the config has no sentiment tags")
    if art.kind != KIND_SENTIMENT:
        raise DatalakeError(f"{art.artifact_id} is a {art.kind}, not a {KIND_SENTIMENT}")
    if hp.get("headlines_id") != headlines_id:
        raise DatalakeError(f"{art.artifact_id} was built on {hp.get('headlines_id')}, the "
                            f"scorer reads {headlines_id}")
    if hp.get("source") != config.tags.source:
        raise DatalakeError(f"{art.artifact_id} is source {hp.get('source')!r}, the tags say "
                            f"{config.tags.source!r}")
    cols = sentiment_columns(art)
    if not ("P_00" in cols and "P_40" in cols):
        raise DatalakeError(f"{art.artifact_id} is not a grid table (has {cols[:5]}...); the "
                            "tags read P_00..P_40 (sentiment spec 2.1)")


def resolve_sentiment(indexes: list[DatalakeIndex], *, headlines_id: str, config: ScoringConfig,
                      source: str | None = None, artifact_id: str | None = None) -> Artifact:
    """A ``headline_sentiment`` able to serve ``config``: the exact ``artifact_id`` if given,
    else the latest of ``source`` built on ``headlines_id``; checked by ``check_sentiment``."""
    if (source is None) == (artifact_id is None):
        raise ValueError("give exactly one of source / artifact_id")
    for dl in indexes:
        try:
            art = (dl.get(artifact_id) if artifact_id
                   else latest_matching(dl, KIND_SENTIMENT, source=source,
                                        headlines_id=headlines_id))
        except DatalakeError:
            continue
        check_sentiment(art, headlines_id=headlines_id, config=config)
        return art
    raise DatalakeError(f"no {KIND_SENTIMENT} {artifact_id or source!r} built on "
                        f"{headlines_id}; run `jobs start headline_sentiment` first")


def earliest_headline_day(headlines: Artifact) -> date:
    """First calendar day with a headline: min TIMESTAMP_UTC of the first partition."""
    files = list(layout_of(headlines).existing(headlines.path).values())
    if not files:
        raise DatalakeError(f"{headlines.artifact_id} has no partition files")
    conn = pin_utc(duckdb.connect())
    try:
        row = conn.sql(
            f"SELECT MIN({duckdb_day('TIMESTAMP_UTC')}) FROM read_parquet('{files[0]}')"
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None:
        raise DatalakeError(f"{files[0].name} holds no timestamps")
    return row[0]


def _mu_inputs(upstream: DatalakeIndex, config: ScoringConfig, mu_id: str | None = None):
    if config.mode is Correction.RAW:
        return None, None
    mu_art = upstream.get(mu_id) if mu_id else upstream.latest(KIND_MU_ASOF)
    return mu_art, load_reference_series(next(mu_art.path.glob("*.parquet")))


# ---------------------------------------------------------------------------
# The monthly tau_asof job
# ---------------------------------------------------------------------------

def build_tau_asof(
    dl: DatalakeIndex, config: ScoringConfig, table: PrimitiveTable, P: np.ndarray, *,
    upstream: DatalakeIndex | None = None, window: str = WINDOW_DEFAULT, seed: int = 0,
    today: date | None = None, rebuild: bool = False, temp: bool = False,
    partitions_art: Artifact | None = None, taxonomy_id: str | None = None,
    primitive_meta: dict[str, Any] | None = None, mu_id: str | None = None,
    calendar: CalibrationCalendar = DEFAULT_CALENDAR,
) -> Artifact | None:
    """Append every cutoff <= today - 1M not yet in the series (or rebuild it all).

    Self-contained: N_eff and the spectrum are recomputed per cutoff from the
    mu_asof row available then. Returns None when no partition is old enough
    yet (cold start), otherwise the tau_asof artifact. ``partitions_art`` lets
    the replay driver pass the partitions artifact it is still extending
    (partial until its run closes, hence invisible to ``find_partitions``).
    """
    upstream = upstream or dl
    emb_key = embedding_key(P, primitive_meta)
    parts_art = partitions_art or find_partitions(dl, config, table, emb_key, seed, temp,
                                                  calendar)
    if parts_art is None:
        log.info("tau_asof: no %s artifact yet for this config", KIND_PARTITIONS)
        return None
    parts = load_partitions(parts_art.path)
    if parts.is_empty():
        return None
    today = today or date.today()
    cutoff = latest_cutoff(parts, today, calendar.delay_offset)
    if cutoff is None:
        log.info("tau_asof: no partition old enough as of %s (needs period end <= today - "
                 + calendar.delay + ")",
                 today)
        return None
    mu_art, mu_df = _mu_inputs(upstream, config, mu_id)

    hp = tau_params(config, table, emb_key, window, seed, temp, calendar)
    existing: Artifact | None = None
    prior = pl.DataFrame(schema=TAU_ASOF_SCHEMA)
    if rebuild:
        from datalake.artifact import utc_now_iso
        hp = {**hp, "rebuilt_at": utc_now_iso()}
    else:
        existing = find_tau_asof(dl, config, table, emb_key, window, seed, temp, calendar)
        interrupted = None
        if existing is None:                       # a tau job killed mid-run: continue it
            try:
                interrupted = latest_matching(dl, KIND_TAU_ASOF, partial=True,
                                              **_lookup(hp, calendar))
            except DatalakeError:
                interrupted = None
        prior = load_tau_series(existing or interrupted)
    all_cutoffs = parts.filter(pl.col("MONTH_END") <= cutoff)["MONTH_END"].to_list()
    done = set(prior["MONTH_END"].to_list())
    todo = [c for c in all_cutoffs if c not in done]
    if not todo and existing is not None:
        return existing

    sources = [parts_art] + ([mu_art] if mu_art else []) + ([taxonomy_id] if taxonomy_id else [])
    if rebuild:
        interrupted = None
    with dl.run(kind=KIND_TAU_ASOF, pipeline=PIPELINE, pipeline_version=_version(temp),
                pipeline_repo=PIPELINE_REPO, hyperparams=hp, repo_dir=_repo_dir(),
                sources=sources, verifier=KIND_TAU_ASOF, hash_pattern="*.parquet",
                extend=existing.artifact_id if existing else None,
                resume=interrupted.artifact_id if interrupted else None) as run:
        new_rows = build_tau_rows(
            parts, config, table, P, mu_df, mu_asof_id=mu_art.artifact_id if mu_art else None,
            cutoffs=todo, window=window, seed=seed, partitions_id=parts_art.artifact_id,
            code_version=run.record.pipeline_commit, freq=calendar.freq, delay=calendar.delay)
        out = pl.concat([prior, new_rows]).sort("MONTH_END") if prior.height else new_rows
        tmp = run.out_dir / (TAU_ASOF_FILE + ".tmp")
        out.write_parquet(tmp, compression="zstd")
        tmp.replace(run.out_dir / TAU_ASOF_FILE)
        record_provenance(run.out_dir, embeddings_provenance(
            P, primitive_meta, _latest_or_none(upstream, KIND_EMBEDDINGS)))
        last = out.row(-1, named=True)
        run.note(f"{len(todo)} cutoff(s) added through {cutoff}; latest tau_empirical="
                 f"{last['TAU_EMPIRICAL']:.4f} tau_gauss={last['TAU_GAUSS']:.4f} "
                 f"n_partitions={last['N_PARTITIONS']} n_eff={last['N_EFF']:.3f}")
    return dl.get(run.artifact_id)


def check_scoring_lineage(dl: DatalakeIndex, *, hl: Artifact, em: Artifact,
                          mu_art: Artifact | None, parts_art: Artifact | None,
                          tau_hp: dict[str, Any],
                          calendar: CalibrationCalendar = DEFAULT_CALENDAR) -> None:
    """Refuse a run that would combine artifacts built from different inputs (before
    any work): the embeddings must come from the scored headlines; mu_asof from those
    headlines and embeddings; the null partitions and the tau series this run extends
    from those headlines / embeddings / mu_asof (and tau from those partitions). An
    artifact that records no source of a kind (legacy, migrated) is only warned about.
    The partitions / tau caches are keyed by config, taxonomy and primitive recipe, not
    by these inputs: without this check new embeddings would silently extend a null
    distribution and tau series built from the old ones."""
    from datalake.lineage import require_lineage

    mu_id = mu_art.artifact_id if mu_art else None
    tau_art = None
    for partial in (False, True):                 # the series build_tau_asof would extend
        try:
            tau_art = latest_matching(dl, KIND_TAU_ASOF, partial=partial,
                                      **_lookup(tau_hp, calendar))
            break
        except DatalakeError:
            continue
    require_lineage([
        (em, {KIND_HEADLINES: hl.artifact_id}),
        (mu_art, {KIND_HEADLINES: hl.artifact_id, KIND_EMBEDDINGS: em.artifact_id}),
        (parts_art, {KIND_HEADLINES: hl.artifact_id, KIND_EMBEDDINGS: em.artifact_id,
                     KIND_MU_ASOF: mu_id}),
        (tau_art, {KIND_PARTITIONS: parts_art.artifact_id if parts_art else None,
                   KIND_MU_ASOF: mu_id}),
    ], what="scoring")


def calibration_as_of(dl: DatalakeIndex, config: ScoringConfig, table: PrimitiveTable,
                      emb_key: str, *, upstream: DatalakeIndex | None = None,
                      window: str = WINDOW_DEFAULT, seed: int = 0,
                      temp: bool = False, mu_id: str | None = None,
                      as_of: date | None = None,
                      calendar: CalibrationCalendar = DEFAULT_CALENDAR) -> TauSeriesProvider:
    """The CalibrationProvider over the tau_asof rows that exist right now (possibly none).

    ``as_of`` restricts them to the rows the monthly job had built by that day
    (cutoffs <= as_of - 1M, ``latest_cutoff``): a replay, a rerun of an earlier range
    or a resumed run then sees exactly what the live loop saw, even when the series
    already holds later rows.
    """
    upstream = upstream or dl
    mu_art, mu_df = _mu_inputs(upstream, config, mu_id)
    tau_art = find_tau_asof(dl, config, table, emb_key, window, seed, temp, calendar)
    tau_df = load_tau_series(tau_art)
    if as_of is not None:
        limit = (pd.Timestamp(as_of) - calendar.delay_offset).date()
        tau_df = tau_df.filter(pl.col("MONTH_END") <= limit)
    return TauSeriesProvider(tau_df, mu_df,
                             tau_source_id=tau_art.artifact_id if tau_art else "",
                             mu_asof_id=mu_art.artifact_id if mu_art else None)


# ---------------------------------------------------------------------------
# THE scoring path: chronological replay of the live loop
# ---------------------------------------------------------------------------

@register_job
class ScoringJob(Job):
    """Narrative scores over [start, end]: per calibration period, tau then the days.

    The replay of the live loop, ONE pass: one unit = one calibration period. Writes
    narrative_daily (the job's artifact, one label per sentiment tag) with its siblings
    day_diagnostics, the null partitions and, with an asset layer, asset_attention_daily
    and narrative_asset_daily, which ``session`` holds open across units; tau_asof is
    extended by its own nested runs. The chunk size is part of the run (a different one
    can move a score at the tau boundary): recorded, and a resume refuses another."""

    kind = KIND_NARRATIVE_DAILY
    pipeline_version = PIPELINE_VERSION
    hash_pattern = "*"

    def __init__(
        self, dl: DatalakeIndex, start: date, end: date, config: ScoringConfig, *,
        table: PrimitiveTable, P: np.ndarray, upstream: DatalakeIndex | None = None,
        source: HeadlineSource | None = None, window: str = WINDOW_DEFAULT, seed: int = 0,
        keep_primitive_daily: bool = False, threads: int = 8,
        rss_budget_gb: float | None = RSS_BUDGET_GB, use_kernel: bool | None = None,
        temp: bool = False, label: str = "", taxonomy_id: str | None = None,
        sentiment: Artifact | None = None, primitive_meta: dict[str, Any] | None = None,
        inputs: dict[str, str | None] | None = None, resume: dict[str, str] | None = None,
        calendar: CalibrationCalendar = DEFAULT_CALENDAR, chunk_size: int = CHUNK_SIZE,
        assets: AssetUniverse | None = None, universe: Artifact | None = None,
    ) -> None:
        upstream = upstream or dl
        if end < start:
            raise ValueError("end before start")
        if (assets is None) != (universe is None):
            raise ValueError("an asset layer needs its universe artifact, and vice versa")
        inputs = inputs or {}
        hl = require_headlines(upstream.get(inputs["headlines"]) if inputs.get("headlines")
                               else upstream.latest(KIND_HEADLINES))
        em = upstream.get(inputs["embeddings"]) if inputs.get("embeddings") \
            else upstream.latest(KIND_EMBEDDINGS)
        mu_id = inputs.get("mu_asof")
        if (config.tags is not None) != (sentiment is not None):
            raise ValueError("a sentiment artifact is required iff the config has sentiment "
                             "tags")
        if sentiment is not None:
            check_sentiment(sentiment, headlines_id=hl.artifact_id, config=config)
        source = source or headline_source(upstream, chunk_size=chunk_size, sentiment=sentiment,
                                           config=config, headlines=hl, embeddings=em,
                                           assets=assets)
        mu_art, first_mu = _mu_inputs(upstream, config, mu_id)

        earliest = earliest_headline_day(hl)
        x_min = calendar.first_scorable(earliest)
        if start < x_min:
            log.warning("requested start %s precedes earliest data %s + delay %s + one %s "
                        "period; start shifted to %s (cold start: earlier periods feed the "
                        "null partitions only)", start, earliest, calendar.delay,
                        calendar.freq, x_min)
            start = x_min
        if end < start:
            log.warning("nothing to score: end %s precedes the shifted start %s", end, start)

        emb_key = embedding_key(P, primitive_meta)
        parts_art = (dl.get(resume["partitions"]) if resume and resume.get("partitions")
                     else find_partitions(dl, config, table, emb_key, seed, temp, calendar))
        check_scoring_lineage(dl, hl=hl, em=em, mu_art=mu_art, parts_art=parts_art,
                              tau_hp=tau_params(config, table, emb_key, window, seed, temp,
                                                calendar),
                              calendar=calendar)
        have = (set(load_partitions(parts_art.path)["MONTH_END"].to_list())
                if parts_art else set())
        # periods that end before the first mu_asof row can never be corrected, hence
        # never yield a partition: not walked at all
        first_correctable = first_mu["DATE"].min() if first_mu is not None else earliest
        walk = [p for p in calendar.periods(earliest, end) if p.last >= first_correctable]
        tags = config.tags
        hp = {**_taxonomy_params(table), "embeddings_id": emb_key,
              "config_id": config.digest(),
              "f0_config_id": config.f0_digest(), "mode": config.mode.value,
              "pooling": config.paraphrase_pooling.value, "q": config.q,
              "labels": ",".join(config.labels),
              **({"mask_bipolar": True} if config.mask_bipolar else {}),
              "window": window, "seed": seed,
              "start": start.isoformat(), "end": end.isoformat(), "label": label,
              "taxonomy_artifact_id": taxonomy_id, "agent_created": temp,
              **calendar.hyperparams()}
        if tags is not None:
            hp.update(sentiment_source=tags.source, sentiment_rule=tags.rule,
                      tags_id=tags.digest(), sentiment_artifact_id=sentiment.artifact_id)

        self.dl, self.upstream, self.config, self.table, self.P = dl, upstream, config, table, P
        self.source, self.window, self.seed, self.temp = source, window, seed, temp
        self.keep_primitive_daily, self.threads = keep_primitive_daily, threads
        self.rss_budget_gb, self.use_kernel, self.label = rss_budget_gb, use_kernel, label
        self.taxonomy_id, self.sentiment, self.primitive_meta = taxonomy_id, sentiment, \
            primitive_meta
        self.resume, self.calendar, self.start, self.end = resume or {}, calendar, start, end
        self.hl, self.em, self.mu_art, self.mu_id = hl, em, mu_art, mu_id
        self.earliest, self.emb_key, self.parts_art, self.have = earliest, emb_key, \
            parts_art, have
        self.chunk_size, self.assets, self.universe = chunk_size, assets, universe
        self.hp = hp
        self.asset_hp = ({**hp, **assets.params()} if assets is not None else None)
        self.provenance = embeddings_provenance(P, primitive_meta, em)
        self._sources: list[Any] = ([hl, em] + ([mu_art] if mu_art else [])
                                    + ([taxonomy_id] if taxonomy_id else []))
        self.summary: dict[str, Any] = {
            "start": start, "end": end, "earliest_data": earliest, "n_days_scored": 0,
            "n_days_null_only": 0, "peak_rss_gb": 0.0, "months_finalised": [],
            "tau_rows": None}
        start_period = calendar.period_of(start).first
        self._plan: dict[str, tuple[Any, list[date], str]] = {}
        for period in walk:
            days = [d for d in period.days() if d <= end]
            if period.first < start_period:
                mode = "null-only (pre-start)"
            else:
                days = [d for d in days if d >= start]
                mode = "score"
            if days:
                self._plan[period.key] = (period, days, mode)

    # -- identity -------------------------------------------------------------

    def params(self) -> dict[str, Any]:
        return self.hp

    def sources(self) -> list[Any]:
        return self._sources + ([self.sentiment] if self.sentiment is not None else [])

    def asset_sources(self) -> list[Any]:
        return self.sources() + [self.universe]

    # -- units ----------------------------------------------------------------

    def units(self) -> list[Unit]:
        return [Unit(key, mode) for key, (_, _, mode) in self._plan.items()]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        period, _, mode = self._plan[unit.key]
        if mode != "score":
            return period.last in self.have                    # partition already there
        return Layout(self.calendar.freq).path_of(out_dir, period).exists()

    @contextmanager
    def session(self, ctx: JobContext) -> Iterator[None]:
        """Open day_diagnostics, the null partitions and the asset tables next to
        narrative_daily."""
        dl, config, calendar = self.dl, self.config, self.calendar
        run_kw = dict(pipeline=self.pipeline, pipeline_version=self.version,
                      pipeline_repo=self.pipeline_repo, repo_dir=_repo_dir())
        parts_art = self.parts_art
        pt_partial = bool(parts_art) and parts_art.partial
        nd_run = ctx.run
        with ExitStack() as stack:
            dg_run = stack.enter_context(dl.run(
                kind=KIND_DAY_DIAGNOSTICS, hyperparams=self.hp, sources=self.sources(),
                verifier=KIND_DAY_DIAGNOSTICS, hash_pattern="*",
                resume=self.resume.get("day_diagnostics"), **run_kw))
            pt_run = stack.enter_context(dl.run(
                kind=KIND_PARTITIONS,
                hyperparams=partitions_params(config, self.table, self.emb_key, self.seed,
                                              self.temp, calendar),
                sources=self._sources, verifier=KIND_PARTITIONS, hash_pattern="*.parquet",
                extend=parts_art.artifact_id if parts_art and not pt_partial else None,
                resume=parts_art.artifact_id if pt_partial else None, **run_kw))
            att_run = cross_run = None
            if self.assets is not None:
                att_run = stack.enter_context(dl.run(
                    kind=KIND_ASSET_ATTENTION, hyperparams=self.asset_hp,
                    sources=self.asset_sources(), verifier=KIND_ASSET_ATTENTION,
                    hash_pattern="*", resume=self.resume.get("asset_attention"), **run_kw))
                cross_run = stack.enter_context(dl.run(
                    kind=KIND_NARRATIVE_ASSET, hyperparams=self.asset_hp,
                    sources=self.asset_sources(), verifier=KIND_NARRATIVE_ASSET,
                    hash_pattern="*", resume=self.resume.get("narrative_asset"), **run_kw))
            for run in (nd_run, dg_run, pt_run, att_run, cross_run):
                if run is not None:
                    record_provenance(run.out_dir, self.provenance)
            if not self.resume:
                write_run_config(nd_run.out_dir, {
                    "config": config.to_dict(), "start": self.start.isoformat(),
                    "end": self.end.isoformat(), "window": self.window, "seed": self.seed,
                    "label": self.label, "temp": self.temp,
                    "keep_primitive_daily": self.keep_primitive_daily,
                    "chunk_size": self.chunk_size,
                    "taxonomy_artifact_id": self.taxonomy_id,
                    "calibration": {"freq": calendar.freq, "delay": calendar.delay},
                    "sentiment_artifact_id": (self.sentiment.artifact_id if self.sentiment
                                              else None),
                    "assets": ({"universe_artifact_id": self.universe.artifact_id,
                                "min_relevance": self.assets.min_relevance}
                               if self.assets is not None else None),
                    "inputs": {"headlines": self.hl.artifact_id,
                               "embeddings": self.em.artifact_id,
                               "mu_asof": self.mu_art.artifact_id if self.mu_art else None},
                    "primitive_embeddings": {
                        "embedding_key": self.emb_key,
                        "paraphrase_style": config.paraphrase_style.value,
                        "model_card": (self.primitive_meta or {}).get("model_card")},
                    "artifacts": {"narrative_daily": nd_run.artifact_id,
                                  "day_diagnostics": dg_run.artifact_id,
                                  "partitions": pt_run.artifact_id,
                                  "asset_attention": att_run.artifact_id if att_run else None,
                                  "narrative_asset": (cross_run.artifact_id if cross_run
                                                      else None)},
                })
            self.nd_run, self.dg_run, self.pt_run = nd_run, dg_run, pt_run
            self.att_run, self.cross_run = att_run, cross_run
            self.writer = ParquetMonthWriter(
                nd_run.out_dir, dg_run.out_dir, freq=calendar.freq,
                asset_attention_dir=att_run.out_dir if att_run else None,
                narrative_asset_dir=cross_run.out_dir if cross_run else None)
            partition_writer = NullPartitionWriter(
                pt_run.out_dir, config=config, table=self.table, seed=self.seed,
                input_ids={"headlines": self.hl.artifact_id,
                           "embeddings": self.em.artifact_id,
                           "mu_asof": self.mu_art.artifact_id if self.mu_art else None},
                code_version=pt_run.record.pipeline_commit, freq=calendar.freq)
            self.sink = SkipClosedDays(partition_writer) if self.resume else partition_writer
            self.last_metadata = None
            yield
        self.summary.update(narrative_daily_id=nd_run.artifact_id,
                            day_diagnostics_id=dg_run.artifact_id,
                            partitions_id=pt_run.artifact_id,
                            asset_attention_id=att_run.artifact_id if att_run else None,
                            narrative_asset_id=cross_run.artifact_id if cross_run else None)

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        """(1) the tau job as of the period's first day; (2) its days, through the one
        pipeline; (3) the period's null partition closes when the period does."""
        period, days, mode = self._plan[unit.key]
        dl, config, calendar, summary = self.dl, self.config, self.calendar, self.summary
        tau_art = build_tau_asof(dl, config, self.table, self.P, upstream=self.upstream,
                                 window=self.window, seed=self.seed, today=period.first,
                                 temp=self.temp, partitions_art=dl.get(self.pt_run.artifact_id),
                                 taxonomy_id=self.taxonomy_id,
                                 primitive_meta=self.primitive_meta, mu_id=self.mu_id,
                                 calendar=calendar)
        cal = calibration_as_of(dl, config, self.table, self.emb_key, upstream=self.upstream,
                                window=self.window, seed=self.seed, temp=self.temp,
                                mu_id=self.mu_id, as_of=period.first, calendar=calendar)
        if tau_art is not None:
            summary["tau_rows"] = load_tau_series(tau_art)
        log.info("%s %s: %s, %d day(s), tau rows available: %d",
                 "month" if calendar.freq == "M" else "period", period.key, mode, len(days),
                 cal.tau_df.height)
        res = score_dates(
            days, config, table=self.table, primitive_embeddings=self.P, source=self.source,
            calibration=cal, writer=self.writer, null_sink=self.sink,
            tau_missing="null_only", keep_primitive_daily=self.keep_primitive_daily,
            collect=False, use_kernel=self.use_kernel, threads=self.threads,
            rss_budget_gb=self.rss_budget_gb, seed=self.seed,
            code_version=self.nd_run.record.pipeline_commit,
            sentiment_artifact_id=self.sentiment.artifact_id if self.sentiment else None,
            embeddings_provenance=self.provenance, assets=self.assets,
            extra_metadata={"narrative_daily_id": self.nd_run.artifact_id,
                            "day_diagnostics_id": self.dg_run.artifact_id,
                            "partitions_id": self.pt_run.artifact_id,
                            "chunk_size": self.chunk_size,
                            **({"asset_attention_id": self.att_run.artifact_id,
                                "narrative_asset_id": self.cross_run.artifact_id}
                               if self.att_run is not None else {})})
        summary["n_days_scored"] += res.n_days
        summary["n_days_null_only"] += len(res.null_only_days)
        summary["peak_rss_gb"] = max(summary["peak_rss_gb"], res.peak_rss_gb)
        if res.metadata is not None:
            self.last_metadata = res.metadata
        if days[-1] == period.last:                     # the period closed on schedule
            self.sink.finalise_before(period.next().first)

    def finalize(self, ctx: JobContext) -> None:
        summary, calendar = self.summary, self.calendar
        summary["months_finalised"] = [p.name for p in self.sink.finalised]
        if self.last_metadata is not None:
            self.writer.close(self.last_metadata)
        self.pt_run.note(f"{len(self.sink.finalised)} {calendar.freq} partition(s) "
                         f"finalised in [{self.earliest}, {self.end}]")
        self.nd_run.note(f"{summary['n_days_scored']} day(s) scored, "
                         f"{summary['n_days_null_only']} null-only, peak RSS "
                         f"{summary['peak_rss_gb']:.2f} GB")
        self.dg_run.note(f"{summary['n_days_scored']} day(s)")
        for run in (self.att_run, self.cross_run):
            if run is not None:
                run.note(f"{summary['n_days_scored']} day(s)")

    def final_summary(self) -> dict[str, Any]:
        """The run summary (artifact ids, counts) once the job has completed."""
        tau_art = find_tau_asof(self.dl, self.config, self.table, self.emb_key, self.window,
                                self.seed, self.temp, self.calendar)
        return {**self.summary, "tau_asof_id": tau_art.artifact_id if tau_art else None}

    # -- resume / CLI -----------------------------------------------------------

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: DatalakeIndex, *,
                      upstream: DatalakeIndex | None = None, cache_dir: Path | None = None,
                      threads: int = 8, chunk_size: int | None = None,
                      rss_budget_gb: float | None = RSS_BUDGET_GB,
                      use_kernel: bool | None = None, embedder: Any = None) -> ScoringJob:
        """Everything from the run's ``run_config.json`` (written at its start):
        config, dates, window, seed, taxonomy, sentiment, asset layer, chunk size, the exact
        upstream inputs (never "latest") and the sibling artifact ids. The primitive-text
        embedder is rebuilt from the recorded model card and must serve the same model (a
        remote server's metadata is read first); its embeddings come from the recipe-keyed
        cache. Only throughput knobs come from the caller; a ``chunk_size`` other than the
        recorded one is refused."""
        from datalake import ModelCard
        from narrative_scoring.primitives import embed_primitive_texts
        from nlp.embedding import embedder_from_card

        dl = index
        upstream = upstream or _upstream_from_env(dl)
        nd = artifact
        if nd.kind != KIND_NARRATIVE_DAILY:
            raise ValueError(f"{nd.artifact_id} is a {nd.kind}, not a {KIND_NARRATIVE_DAILY}")
        path = nd.path / RUN_CONFIG_FILE
        if not path.exists():
            raise ValueError(f"{nd.artifact_id} has no {RUN_CONFIG_FILE} (written before "
                             "resume support); it cannot be resumed")
        rc = json.loads(path.read_text())
        recorded_chunk = rc.get("chunk_size", 8_192)     # absent: the earlier default
        if chunk_size is not None and chunk_size != recorded_chunk:
            raise ValueError(f"{nd.artifact_id} was scored with chunk_size={recorded_chunk}; "
                             f"resuming with {chunk_size} could move scores at the tau "
                             "boundary. Resume without --chunk-size")
        config = ScoringConfig.from_dict(rc["config"])
        tax_art = resolve_taxonomy([dl, upstream], artifact_id=rc["taxonomy_artifact_id"])
        table = load_registered_table(tax_art, rc["primitive_embeddings"]["paraphrase_style"])
        card = rc["primitive_embeddings"]["model_card"]
        if embedder is None:
            if card is None:
                raise ValueError("the run recorded no primitive-embedding model card")
            embedder = embedder_from_card(ModelCard.from_dict(card))
        if cache_dir is None:
            cache_dir = _cache_dir_from_env()
        P, meta = embed_primitive_texts(table, cache_dir, embedder=embedder)
        if embedding_key(P, meta) != rc["primitive_embeddings"]["embedding_key"]:
            raise ValueError("the primitive-embedding recipe differs from the one the run "
                             "started with")

        def _get(aid: str) -> Artifact:
            return next(ix.get(aid) for ix in (dl, upstream) if ix.exists(aid))

        sentiment = _get(rc["sentiment_artifact_id"]) if rc.get("sentiment_artifact_id") \
            else None
        universe = assets = None
        if rc.get("assets"):
            universe = _get(rc["assets"]["universe_artifact_id"])
            assets = AssetUniverse.from_artifact(
                universe, min_relevance=rc["assets"]["min_relevance"])
        inputs = rc["inputs"]
        source = headline_source(upstream, chunk_size=recorded_chunk, sentiment=sentiment,
                                 config=config, headlines=upstream.get(inputs["headlines"]),
                                 embeddings=upstream.get(inputs["embeddings"]), assets=assets)
        calendar = CalibrationCalendar(**rc.get("calibration", {}))   # absent: the default
        log.info("resuming %s (%s -> %s)", nd.artifact_id, rc["start"], rc["end"])
        return cls(
            dl, date.fromisoformat(rc["start"]), date.fromisoformat(rc["end"]), config,
            table=table, P=P, upstream=upstream, source=source, window=rc["window"],
            seed=rc["seed"], keep_primitive_daily=rc["keep_primitive_daily"], threads=threads,
            rss_budget_gb=rss_budget_gb, use_kernel=use_kernel, temp=rc["temp"],
            label=rc["label"], taxonomy_id=rc["taxonomy_artifact_id"], sentiment=sentiment,
            primitive_meta=meta, inputs=inputs, resume=rc["artifacts"], calendar=calendar,
            chunk_size=recorded_chunk, assets=assets, universe=universe)

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        from narrative_scoring.jobs import add_scoring_args

        add_scoring_args(parser)
        parser.add_argument("--from", dest="start", required=True)
        parser.add_argument("--to", dest="end", required=True)

    @classmethod
    def from_args(cls, args: Any, index: DatalakeIndex) -> ScoringJob:
        from narrative_scoring.jobs import scoring_job_from_args

        return scoring_job_from_args(args, index, _upstream_from_env(index))


def _upstream_from_env(dl: DatalakeIndex) -> DatalakeIndex:
    """$DATALAKE_UPSTREAM_ROOT when set (headlines / embeddings / mu_asof), else ``dl``."""
    import os

    up = os.environ.get("DATALAKE_UPSTREAM_ROOT")
    return DatalakeIndex(up, create=False) if up else dl


def _cache_dir_from_env() -> Path:
    import os

    root = os.environ.get("CACHE_PATH")
    if not root:
        raise ValueError("CACHE_PATH is not set; pass cache_dir")
    return Path(root) / "narrative_scoring"


def _runner(dl: DatalakeIndex) -> Any:
    from datalake.jobs import JobRunner

    return JobRunner(dl, repo_dir=_repo_dir(), allow_dirty=True, handle_signals=False)


def score_range_to_datalake(dl: DatalakeIndex, start: date, end: date, config: ScoringConfig,
                            **kwargs: Any) -> dict[str, Any]:
    """Score [start, end] chronologically through the live machinery (``ScoringJob``).

    The walk follows the tau job's calibration ``calendar`` (default monthly,
    1M delay). Per calibration period P in order: (1) the tau_asof job as of
    the first day of P (cutoffs <= that day - delay); (2) ``score_dates`` over
    P's requested days with providers resolving as of each day; days with no
    tau row old enough feed the partitions only. Periods before ``start`` that
    have no partition yet are walked the same way (null-only), because the
    first tau row needs them. ``start`` is shifted to earliest_data + delay +
    one period (``CalibrationCalendar.first_scorable``) when earlier, with a
    loud warning, never an error. narrative_daily / day_diagnostics are
    partitioned by the same period.

    A sentiment-filtered run (``config.sentiment`` = a bucket) needs ``sentiment``, the
    ``headline_sentiment`` artifact its bucket is read from; it is checked against the
    headlines artifact and cited in the lineage. It scores the same days as the
    all-headlines run with the same tau and mu, reads that run's tau_asof (and refuses
    to start without one covering its dates), and never opens the null partitions.

    ``primitive_meta`` is the primitive-embedding cache sidecar (backend, serving
    metadata, checks); with the headline-embeddings model card it is written to
    every artifact of the run (``embeddings_provenance.json``) and to the run
    metadata. The embedding recipe (``embedding_key``) is part of every lookup key,
    so partitions / tau built from other embeddings are never reused.

    ``inputs`` pins the upstream artifacts (``headlines``, ``embeddings``,
    ``mu_asof`` ids) instead of the latest ones; ``resume`` (``narrative_daily``,
    ``day_diagnostics``, ``partitions`` ids of an interrupted run) continues that
    run in place -- use ``resume_scoring`` (or ``jobs resume <narrative_daily id>``),
    which reads both from the run itself. On resume every period whose narrative_daily
    file exists is skipped, the rest re-scored, and no null draw is fed twice (partitions checkpoint
    their open period at every closed day, with the RNG state; days already closed
    there are re-scored for narrative_daily only). The outputs equal an
    uninterrupted run.

    Keyword arguments are ``ScoringJob``'s. Returns a summary dict with the artifact
    ids and counts.
    """
    job = ScoringJob(dl, start, end, config, **kwargs)
    if job.resume:                     # the run's own ids (resume_scoring reads them)
        _runner(dl).resume_job(job.resume["narrative_daily"], job)
    else:
        _runner(dl).start(job)
    return job.final_summary()


def resume_scoring(
    dl: DatalakeIndex, narrative_daily_id: str, *, upstream: DatalakeIndex | None = None,
    cache_dir: Path, threads: int = 8, chunk_size: int = 8_192,
    rss_budget_gb: float | None = 30.0, use_kernel: bool | None = None, embedder: Any = None,
) -> dict[str, Any]:
    """Continue an interrupted scoring run from the run alone (``ScoringJob.from_artifact``;
    the same as ``jobs resume <id>``). Only throughput knobs come from the caller."""
    nd = dl.get(narrative_daily_id)
    if nd.kind != KIND_NARRATIVE_DAILY:
        raise ValueError(f"{narrative_daily_id} is a {nd.kind}, not a {KIND_NARRATIVE_DAILY}")
    if not nd.partial:
        raise ValueError(f"{narrative_daily_id} is complete; nothing to resume")
    job = ScoringJob.from_artifact(nd, dl, upstream=upstream, cache_dir=cache_dir,
                                   threads=threads, chunk_size=chunk_size,
                                   rss_budget_gb=rss_budget_gb, use_kernel=use_kernel,
                                   embedder=embedder)
    _runner(dl).resume_job(narrative_daily_id, job)
    return job.final_summary()


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


def _schema_check(files: list[Path], schema: pl.Schema, label: str,
                  accepted: tuple[pl.Schema, ...] = ()):
    if not files:
        yield f"{label}: no parquet files"
    for p in files[:6]:
        df = pl.read_parquet(p)
        if df.schema != schema and df.schema not in accepted:
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
        yield from _schema_check(artifact.files(), NARRATIVE_DAILY_SCHEMA, KIND_NARRATIVE_DAILY,
                                 accepted=(NARRATIVE_DAILY_SCHEMA_V1,))
        if not (artifact.path / "run_metadata.json").exists():
            yield "run_metadata.json missing"
        else:
            json.loads((artifact.path / "run_metadata.json").read_text())
    return _findings(KIND_NARRATIVE_DAILY, artifact, check)


def load_day_diagnostics(artifact: Artifact) -> pl.DataFrame:
    """Every day_diagnostics row of an artifact, whatever layout its files were written
    with: columns absent from an older layout come back null, a missing N_SCORED is
    N_HEADLINES (an all-headlines run scores every headline), a missing SENTIMENT is "all"
    and a missing N_UNTAGGED 0 (one row per day, before the tags)."""
    files = sorted(p for p in artifact.files() if p.suffix == ".parquet")
    if not files:
        return pl.DataFrame(schema=DAY_DIAGNOSTICS_SCHEMA)
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    for col, dtype in (("N_SCORED", pl.Int64), ("SENTIMENT", pl.String),
                       ("N_UNTAGGED", pl.Int64)):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
    df = df.with_columns(pl.col("N_SCORED").fill_null(pl.col("N_HEADLINES")),
                         pl.col("SENTIMENT").fill_null("all"),
                         pl.col("N_UNTAGGED").fill_null(0))
    return df.select([pl.col(c).cast(t) if c in df.columns else pl.lit(None, dtype=t).alias(c)
                      for c, t in DAY_DIAGNOSTICS_SCHEMA.items()]).sort("DATE", "SENTIMENT")


def verify_day_diagnostics(artifact: Artifact):
    def check():
        yield from _schema_check(artifact.files(), DAY_DIAGNOSTICS_SCHEMA, KIND_DAY_DIAGNOSTICS,
                                 accepted=(DAY_DIAGNOSTICS_SCHEMA_V4, DAY_DIAGNOSTICS_SCHEMA_V3,
                                           DAY_DIAGNOSTICS_SCHEMA_V2, DAY_DIAGNOSTICS_SCHEMA_V1))
    return _findings(KIND_DAY_DIAGNOSTICS, artifact, check)


def _sparse_check(artifact: Artifact, schema: pl.Schema, kind: str):
    """Schema of every partition (a day with no support writes no row, so an empty file
    is valid), plus run_metadata.json."""
    files = [p for p in artifact.files() if p.suffix == ".parquet"]
    if not files:
        yield f"{kind}: no parquet files"
    for p in files[:6]:
        if pl.read_parquet_schema(p) != schema:
            yield f"{p.name}: schema mismatch"
    if not (artifact.path / "run_metadata.json").exists():
        yield "run_metadata.json missing"


def verify_asset_attention(artifact: Artifact):
    return _findings(KIND_ASSET_ATTENTION, artifact,
                     lambda: _sparse_check(artifact, ASSET_ATTENTION_SCHEMA,
                                           KIND_ASSET_ATTENTION))


def verify_narrative_asset(artifact: Artifact):
    return _findings(KIND_NARRATIVE_ASSET, artifact,
                     lambda: _sparse_check(artifact, NARRATIVE_ASSET_SCHEMA,
                                           KIND_NARRATIVE_ASSET))


__all__ = [
    "KIND_TAXONOMY", "register_taxonomy", "resolve_taxonomy", "load_registered_table",
    "KIND_PARTITIONS", "KIND_TAU_ASOF", "KIND_NARRATIVE_DAILY", "KIND_DAY_DIAGNOSTICS",
    "latest_matching", "mark_temp_deprecated", "is_agent_created", "find_partitions",
    "find_tau_asof",
    "build_tau_asof", "calibration_as_of", "headline_source", "earliest_headline_day",
    "KIND_SENTIMENT", "resolve_sentiment", "check_sentiment", "sentiment_columns",
    "KIND_ASSET_ATTENTION", "KIND_NARRATIVE_ASSET", "CHUNK_SIZE",
    "score_range_to_datalake",
]
