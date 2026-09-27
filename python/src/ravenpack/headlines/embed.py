"""RavenBERT headline embedding pipeline (RavenPack I/O around ``nlp.Embedder``).

Reads the ``ravenpack_headlines`` datalake artifact (one structured parquet per
month, with a ``HEADLINE`` column) and writes one embedding parquet per month
into a new ``headline_embeddings`` artifact.

Output layout (mirrors the ingest layout)::

    {artifact_dir}/{year}-{month:02d}.parquet

Each output parquet has exactly two columns (``schema.EMBEDDING_SCHEMA``):

    RP_STORY_ID   String
    EMBEDDING     Array(Float16, 384)     -- L2-normalized RavenBERT vector

The embedding parquet is a standalone file joined back to the structured parquet
on ``RP_STORY_ID``; the two are never merged into one file.

Processing per month:
    1. Stream the source parquet in row batches (``write_chunk_rows`` at a time)
       to bound memory on high-volume months.
    2. Encode ``HEADLINE`` with an ``nlp.embedding.Embedder`` (any backend:
       TEI by default, local or embedx; passages, no query prefix). The
       embedder returns L2-normalized float32; we cast to float16 before writing.
    3. Write with zstd compression, atomically (``.parquet.tmp`` + rename), one
       row group per chunk.

Resumable: a month whose output parquet already exists is skipped. Before each
month the backend re-checks that the served model has not changed (a model
switched on a remote server mid-run would otherwise mix two models).

Provenance: the run's ``ModelCard`` is ``Embedder.model_card()`` --
``model_id="ravenbert"``, ``version="1.0"``, the backend, its full serving
metadata (TEI ``/info``, embedx model entry, or local device/dtype), the
compatibility-check results, and the weights sha256 when the backend can know
it (local). The backend and dtype are also hyperparams, so they are part of
the artifact id.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from datalake.layout import Layout, layout_from_hyperparams
from datalake.periods import period_of
from ravenpack.headlines.schema import EMBEDDING_DIM, EMBEDDING_SCHEMA

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex
    from datalake.verify import Finding
    from nlp.embedding import Embedder

log = logging.getLogger(__name__)

# Datalake artifact kind produced by this module, and the kind it reads from.
KIND = "headline_embeddings"
SOURCE_KIND = "ravenpack_headlines"

# Model identity -- must match the archive migration so that
# `dl.latest("headline_embeddings", model="ravenbert", version="1.0")` resolves
# both the migrated and the freshly-computed artifacts.
MODEL_ID = "ravenbert"
MODEL_VERSION = "1.0"
RAVENBERT_REPO = "https://github.com/GabinTB/RavenBERT"

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"



# ---------------------------------------------------------------------------
# Arrow schema for the ParquetWriter -- derived from EMBEDDING_SCHEMA so the
# written tables and the writer schema are guaranteed to agree.
# ---------------------------------------------------------------------------

_ARROW_SCHEMA: pa.Schema | None = None


def _arrow_schema() -> pa.Schema:
    global _ARROW_SCHEMA
    if _ARROW_SCHEMA is None:
        _ARROW_SCHEMA = pl.DataFrame(schema=EMBEDDING_SCHEMA).to_arrow().schema
    return _ARROW_SCHEMA


def build_embedder(backend: str = "tei", dtype: str = "float16", *,
                   model_path: Path | str | None = None, batch_size: int | None = None,
                   device: str | None = None) -> Embedder:
    """RavenBERT ``Embedder`` on a named backend, named ``ravenbert`` / ``1.0`` so
    ``dl.latest("headline_embeddings", model="ravenbert", version="1.0")`` keeps
    resolving artifacts from every backend."""
    import os

    from nlp.backends import make_backend
    from nlp.embedding import ENV_EMBEDDING_MODEL_PATH, Embedder

    kwargs: dict[str, Any] = {}
    if batch_size:
        kwargs["batch_size"] = batch_size
    if backend == "local":
        model_path = model_path or os.environ.get(ENV_EMBEDDING_MODEL_PATH)
        if device:
            kwargs["device"] = device
    backend_obj = make_backend(backend, task="embedding", dtype=dtype,
                               model_path=str(model_path) if model_path else None, **kwargs)
    return Embedder(backend_obj, model_id=MODEL_ID, version=MODEL_VERSION, repo=RAVENBERT_REPO)


# ---------------------------------------------------------------------------
# Per-month embedding
# ---------------------------------------------------------------------------

def _prepare_headlines(values: list[Any]) -> tuple[list[str], int]:
    """Replace null headlines with the empty string.

    Rows are never dropped: the embedding parquet must carry one row per story in
    the source month, aligned by ``RP_STORY_ID``.  A null headline (not seen in
    practice across ~1e9 rows) is embedded as ``""`` and counted for logging.
    """
    out: list[str] = []
    n_null = 0
    for v in values:
        if v is None:
            out.append("")
            n_null += 1
        else:
            out.append(v)
    return out, n_null


def embed_month(
    embedder: Any,
    in_path: Path,
    out_path: Path,
    *,
    write_chunk_rows: int = 200_000,
    tag: str = "",
    log_every: int = 100_000,
) -> int:
    """Embed one month's headlines and write the two-column parquet atomically.

    Args:
        embedder:        Object exposing ``encode(list[str]) -> np.ndarray`` of
                         shape ``(n, EMBEDDING_DIM)`` (``nlp.embedding.Embedder``;
                         batching is the backend's business).
        in_path:         Source structured parquet (needs ``RP_STORY_ID``, ``HEADLINE``).
        out_path:        Destination ``YYYY-MM.parquet``.
        write_chunk_rows: Source rows read (and encoded) per row group.
        tag:             Log prefix.
        log_every:       Progress line roughly every N rows embedded.

    Returns:
        Number of rows written (0 if the source month is empty; no file is left).
    """
    tmp = out_path.with_suffix(".parquet.tmp")
    parquet_file = pq.ParquetFile(in_path)
    writer = pq.ParquetWriter(tmp, _arrow_schema(), compression="zstd")

    n_written = 0
    n_null_total = 0
    next_report = log_every
    t0 = time.monotonic()

    try:
        for batch in parquet_file.iter_batches(
            batch_size=write_chunk_rows,
            columns=["RP_STORY_ID", "HEADLINE"],
        ):
            story_ids = batch.column("RP_STORY_ID").to_pylist()
            if not story_ids:
                continue
            headlines, n_null = _prepare_headlines(batch.column("HEADLINE").to_pylist())
            n_null_total += n_null

            emb = np.asarray(embedder.encode(headlines), dtype=np.float32)
            if emb.shape != (len(headlines), EMBEDDING_DIM):
                raise ValueError(
                    f"{tag}: encode returned {emb.shape}, "
                    f"expected {(len(headlines), EMBEDDING_DIM)}"
                )

            df = pl.DataFrame(
                {"RP_STORY_ID": story_ids, "EMBEDDING": list(emb.astype(np.float16))},
                schema=EMBEDDING_SCHEMA,
            )
            writer.write_table(df.to_arrow().cast(_arrow_schema()))
            n_written += len(story_ids)

            if n_written >= next_report:
                rate = n_written / max(time.monotonic() - t0, 1e-9)
                log.info("%s  %d embedded | %.0f/s", tag, n_written, rate)
                next_report = n_written + log_every
    except BaseException:
        writer.close()
        tmp.unlink(missing_ok=True)
        raise

    writer.close()

    if n_written == 0:
        tmp.unlink(missing_ok=True)
        return 0

    if n_null_total:
        log.warning("%s  %d null headline(s) embedded as empty string", tag, n_null_total)

    tmp.replace(out_path)
    return n_written


# ---------------------------------------------------------------------------
# Pure range driver (used by embed_to_datalake and by the resume path)
# ---------------------------------------------------------------------------

def embed_range(
    embedder: Any,
    source_dir: Path,
    out_dir: Path,
    start_year: int | None = None,
    end_year: int | None = None,               # inclusive
    *,
    layout: Layout | None = None,
    write_chunk_rows: int = 200_000,
    log_every: int = 100_000,
    overwrite: bool = False,
) -> None:
    """Embed every partition of ``layout`` present in ``source_dir``.

    The output has the source's layout (one embedding file per headlines
    partition, same key). ``start_year`` / ``end_year`` are the legacy monthly
    call. Existing output partitions are skipped unless ``overwrite`` is set, so
    an interrupted run resumes by re-entering the same output directory. Source
    partitions absent from ``source_dir`` are warned and skipped (some periods
    genuinely have no data). Before each partition the embedder's backend
    re-checks that the served model is unchanged.
    """
    if layout is None:
        if start_year is None or end_year is None:
            raise ValueError("give a layout or start_year/end_year")
        layout = Layout("M", date(start_year, 1, 1), date(end_year, 12, 31))
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = layout.expected()
    todo: list[tuple[str, Path]] = []
    for period in parts:
        src = layout.path_of(source_dir, period)
        name = src.name
        if not src.exists():
            log.warning("source partition missing, skipping: %s", name)
            continue
        if not overwrite and (out_dir / name).exists():
            continue
        todo.append((name, src))

    log.info(
        "embed: %d %s partition(s) to process (of %d in range), overwrite=%s",
        len(todo), layout.freq, len(parts), overwrite,
    )
    total = len(todo)
    for i, (name, src) in enumerate(todo, 1):
        tag = f"[{i}/{total}] {Path(name).stem}"
        backend = getattr(embedder, "backend", None)
        if backend is not None:
            backend.check_unchanged()
        t_month = time.monotonic()
        n = embed_month(
            embedder, src, out_dir / name,
            write_chunk_rows=write_chunk_rows,
            tag=tag,
            log_every=log_every,
        )
        if n == 0:
            log.warning("%s  no rows in source, skipped", tag)
            continue
        elapsed = time.monotonic() - t_month
        log.info(
            "%s  wrote %s (%d stories in %.0fs, %.0f/s)",
            tag, name, n, elapsed, n / max(elapsed, 1e-9),
        )


# ---------------------------------------------------------------------------
# Datalake-aware entry point
# ---------------------------------------------------------------------------

def embed_to_datalake(
    index: "DatalakeIndex",
    pipeline_version: str,
    *,
    embedder: Embedder | None = None,
    backend: str = "tei",
    dtype: str = "float16",
    model_path: Path | str | None = None,
    device: str | None = None,
    source_artifact_id: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    start: date | None = None,
    end: date | None = None,
    pipeline: str = PIPELINE,
    pipeline_repo: str | None = PIPELINE_REPO,
    repo_dir: Path | None = None,
    batch_size: int | None = None,
    write_chunk_rows: int = 200_000,
    log_every: int = 100_000,
) -> "Artifact":
    """Embed a ``ravenpack_headlines`` artifact into a new ``headline_embeddings`` one.

    Wraps ``embed_range`` in a datalake run: the output directory is allocated by
    the index, the source artifact is recorded as lineage, the embedder's full
    model card (backend, serving metadata, checks) is attached, sidecars and
    hashes are written on completion, and a crash leaves the artifact
    registered as partial.

    Args:
        index:              Datalake index to register the artifact in.
        pipeline_version:   Semantic version of this pipeline.
        embedder:           A ready ``Embedder``; built from ``backend`` /
                            ``dtype`` / ``model_path`` / ``device`` /
                            ``batch_size`` when None.
        backend, dtype:     Inference engine ("tei" | "local" | "embedx") and
                            compute dtype; both enter the artifact id.
        model_path:         Local backend only; default
                            ``$RAVENBERT_EMBEDDING_MODEL_PATH``.
        source_artifact_id: Explicit source artifact; defaults to
                            ``index.latest("ravenpack_headlines")``.
        start/end (or legacy start_year/end_year): restrict the partitions
                            embedded (rounded out to whole periods of the
                            source's frequency); default to the source
                            artifact's declared range.
        pipeline, pipeline_repo, repo_dir: Provenance passthrough.
        write_chunk_rows, log_every: Throughput / logging knobs.

    Returns:
        The completed Artifact.
    """
    src = index.get(source_artifact_id) if source_artifact_id else index.latest(SOURCE_KIND)
    hp = src.meta.hyperparams
    src_layout = layout_from_hyperparams(hp)
    if start is None and start_year is not None:
        start = date(start_year, 1, 1)
    if end is None and end_year is not None:
        end = date(end_year, 12, 31)
    start, end = start or src_layout.start, end or src_layout.end
    if start is None or end is None:
        raise ValueError(
            "no start/end given and none declared by the source "
            f"artifact ({src.artifact_id})"
        )
    layout = Layout(src_layout.freq, period_of(start, src_layout.freq).first,
                    period_of(end, src_layout.freq).last)

    if embedder is None:
        embedder = build_embedder(backend, dtype, model_path=model_path,
                                  batch_size=batch_size, device=device)
    card = embedder.model_card()

    # Hyperparams feed the artifact_id slug: the year range plus what changes
    # the numbers (engine and dtype). The source artifact is lineage (sources).
    hyperparams: dict[str, Any] = {
        **layout.hyperparams(),
        "backend": embedder.backend.name,
        "dtype": embedder.backend.dtype,
    }
    notes = f"source_artifact={src.artifact_id}"

    with index.run(
        kind=KIND,
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        repo_dir=repo_dir,
        hyperparams=hyperparams,
        notes=notes,
        sources=[src],
        model_card=card,
        verifier=KIND,
        hash_pattern="*.parquet",
    ) as run:
        embed_range(
            embedder,
            source_dir=src.path,
            out_dir=run.out_dir,
            layout=layout,
            write_chunk_rows=write_chunk_rows,
            log_every=log_every,
            overwrite=False,
        )
        n_months = len(list(run.out_dir.glob("*.parquet")))
        if n_months == 0:
            raise RuntimeError(
                f"embed produced no output for {layout.start}..{layout.end}; "
                f"check source artifact {src.artifact_id} at {src.path}"
            )
        run.note(f"{n_months} monthly embedding files from {src.artifact_id}")

    return index.get(run.artifact_id)


def resume_embedding(
    index: "DatalakeIndex",
    artifact_id: str,
    *,
    embedder: Embedder | None = None,
    model_path: Path | str | None = None,
    write_chunk_rows: int = 200_000,
    log_every: int = 100_000,
    **backend_kwargs: Any,
) -> "Artifact":
    """Finish a partial ``headline_embeddings`` artifact from its own metadata.

    Nothing is taken from the caller but throughput knobs: the embedder is rebuilt
    from the artifact's model card (backend, dtype, naming) and must serve the
    SAME model -- a remote server's metadata is read first and compared with the
    recorded identity; local weights must hash to the recorded sha256. The source
    is the recorded lineage, the year range the recorded hyperparams. Months
    already written are skipped; the artifact is completed in place. A ready
    ``embedder`` may be passed instead of rebuilding one; it must pass the same
    identity check.
    """
    from nlp.backends import assert_same_model
    from nlp.embedding import embedder_from_card

    art = index.get(artifact_id)
    if art.kind != KIND:
        raise ValueError(f"{artifact_id} is a {art.kind}, not a {KIND}")
    if not art.partial:
        raise ValueError(f"{artifact_id} is complete; nothing to resume")
    card = art.meta.model_card
    if card is None:
        raise ValueError(f"{artifact_id} has no model card; cannot identify its embedder")
    if embedder is None:
        embedder = embedder_from_card(card, model_path=str(model_path) if model_path else None,
                                      **backend_kwargs)
    else:
        assert_same_model((card.serving or {}).get("identity") or {}, embedder.backend)
    hp = art.meta.hyperparams
    source_id = next((s for s in art.meta.sources if s.startswith(SOURCE_KIND)), None)
    if source_id is None:
        raise ValueError(f"{artifact_id} records no {SOURCE_KIND} source")
    src = index.get(source_id)
    with index.run(kind=KIND, pipeline=art.meta.pipeline,
                   pipeline_version=art.meta.pipeline_version,
                   pipeline_repo=art.meta.pipeline_repo, hyperparams=hp,
                   verifier=KIND, hash_pattern="*.parquet", resume=artifact_id) as run:
        embed_range(embedder, source_dir=src.path, out_dir=run.out_dir,
                    layout=layout_from_hyperparams(hp),
                    write_chunk_rows=write_chunk_rows, log_every=log_every, overwrite=False)
        run.note(f"{len(list(run.out_dir.glob('*.parquet')))} monthly embedding files")
    return index.get(artifact_id)


# ---------------------------------------------------------------------------
# Content verifier (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: "Artifact") -> "list[Finding]":
    """Verify a headline_embeddings artifact matches its declared scope.

    Registered as the ``headline_embeddings`` entry point.  Checks:
      - one parquet per partition of the declared layout (missing -> WARNING)
      - no parquet outside the declared range (-> ERROR)
      - a first/middle/last sample are non-empty and match EMBEDDING_SCHEMA
        (RP_STORY_ID + EMBEDDING Array(Float16, 384), nothing else)
    """
    from datalake.verify import Finding, Severity

    findings: list[Finding] = []
    aid = artifact.artifact_id
    hp = artifact.meta.hyperparams
    layout = layout_from_hyperparams(hp)
    if layout.start is None or layout.end is None:
        findings.append(Finding(
            Severity.WARNING, aid,
            "no declared range in hyperparams; cannot verify coverage",
        ))
        return findings

    present = {p.name for p in artifact.path.glob("*.parquet")}
    expected = {f"{p.key}.parquet" for p in layout.expected()}

    missing = expected - present
    if missing:
        findings.append(Finding(
            Severity.WARNING, aid,
            f"{len(missing)} of {len(expected)} declared {layout.freq} partitions missing: "
            f"{', '.join(sorted(missing)[:6])}"
            + (" ..." if len(missing) > 6 else ""),
        ))

    unexpected = present - expected
    if unexpected:
        findings.append(Finding(
            Severity.ERROR, aid,
            f"{len(unexpected)} parquet(s) outside declared range "
            f"[{layout.start}, {layout.end}]: {', '.join(sorted(unexpected)[:6])}",
        ))

    sample_names = sorted(present)
    if sample_names:
        sample = {
            sample_names[0],
            sample_names[len(sample_names) // 2],
            sample_names[-1],
        }
        for name in sorted(sample):
            path = artifact.path / name
            try:
                df = pl.read_parquet(path)
            except Exception as exc:  # noqa: BLE001 - report any read failure
                findings.append(Finding(
                    Severity.ERROR, aid, f"{name}: unreadable parquet: {exc}",
                ))
                continue
            if df.is_empty():
                findings.append(Finding(
                    Severity.ERROR, aid, f"{name}: file is empty",
                ))
                continue
            if df.schema != EMBEDDING_SCHEMA:
                findings.append(Finding(
                    Severity.ERROR, aid,
                    f"{name}: schema mismatch (got {dict(df.schema)}, "
                    f"expected {dict(EMBEDDING_SCHEMA)})",
                ))

    return findings
