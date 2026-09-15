"""RavenBERT headline embedding pipeline.

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
    2. Encode ``HEADLINE`` with RavenBERT (passages, no query prefix).  The model
       returns L2-normalized float32; we cast to float16 before writing.
    3. Write with zstd compression, atomically (``.parquet.tmp`` + rename), one
       row group per chunk.

Resumable: a month whose output parquet already exists is skipped.

Model provenance is recorded in the run's ``ModelCard``: ``model_id="ravenbert"``,
``version="1.0"``, ``weights_public=False``, ``weights_sha256`` = a sha256 over
the RavenBERT weights directory (sorted relative paths + file bytes).
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ravenpack.headlines.schema import EMBEDDING_DIM, EMBEDDING_SCHEMA

if TYPE_CHECKING:  # avoids importing torch/ravenbert just to import this module
    from datalake import Artifact, DatalakeIndex, ModelCard
    from datalake.verify import Finding

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

_HASH_CHUNK_BYTES = 8 * 1024 * 1024


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


# ---------------------------------------------------------------------------
# Model weights hashing
# ---------------------------------------------------------------------------

def model_dir_sha256(model_path: Path, *, log_every: int = 20) -> str:
    """Deterministic sha256 over every file in a model directory.

    The digest folds in each file's path relative to ``model_path`` (POSIX form,
    NUL-terminated) followed by its bytes, iterating files in sorted order.  Two
    directories with identical contents and layout produce the same digest;
    changing any file's bytes or renaming any file changes it.

    Args:
        model_path: Directory holding the model weights / tokenizer / config.
        log_every:  Emit a progress line every N files (the RavenBERT weights
                    live on a GDrive FUSE mount and hashing is slow).

    Returns:
        Hex sha256 digest.
    """
    model_path = Path(model_path)
    files = sorted(p for p in model_path.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files under model directory {model_path}")

    h = hashlib.sha256()
    for i, path in enumerate(files, 1):
        rel = path.relative_to(model_path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
                h.update(chunk)
        if i % log_every == 0 or i == len(files):
            log.info("  hashed %d/%d model files", i, len(files))
    return h.hexdigest()


def build_model_card(
    model_path: Path, weights_sha256: str, *, backend_note: str | None = None
) -> "ModelCard":
    """RavenBERT embedding model card, shared by the pipeline and the migration.

    backend_note is appended to notes when the run used a non-local backend
    (e.g. "embedded via embedx at http://10.10.10.2:8477/v1") -- the weights
    hash always describes the LOCAL model_path mirror regardless of backend
    (see embedx_client.py's docstring on this assumption).
    """
    from datalake import ModelCard

    notes = (
        f"Local weights dir: {Path(model_path).name}. Output vectors are "
        "L2-normalized float32 from RavenBERT, stored as float16."
    )
    if backend_note:
        notes = f"{notes} {backend_note}"

    return ModelCard(
        model_id=MODEL_ID,
        version=MODEL_VERSION,
        repo=RAVENBERT_REPO,
        weights_public=False,
        weights_sha256=weights_sha256,
        architecture="BERT-small (12L, hidden 384), sentence-transformers, CLS pooling",
        dim=EMBEDDING_DIM,
        pooling="cls",
        trained_on="RavenPack headlines",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_embedding_model(model_path: Path, device: str | None = None) -> Any:
    """Load the RavenBERT embedding model, locally or via a remote embedx server.

    device="embedx" dispatches to RemoteEmbeddingModel.from_env() (see
    embedx_client.py) instead of loading torch/ravenbert locally at all --
    any other value (None, "cuda", "mps", "cpu") loads RavenBERT in-process
    as before. The ravenbert import stays lazy and conditional on this
    branch so --device embedx needs no GPU stack installed locally.
    """
    if device == "embedx":
        from ravenpack.headlines.embedx_client import RemoteEmbeddingModel

        log.info("using remote embedx backend (device=embedx)")
        return RemoteEmbeddingModel.from_env()

    from ravenbert.embedding.model import EmbeddingModel

    log.info("loading RavenBERT embedding model from %s (device=%s)", model_path, device)
    return EmbeddingModel.from_path(str(model_path), device=device)


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
    model: Any,
    in_path: Path,
    out_path: Path,
    *,
    batch_size: int = 128,
    write_chunk_rows: int = 200_000,
    tag: str = "",
    log_every: int = 100_000,
) -> int:
    """Embed one month's headlines and write the two-column parquet atomically.

    Args:
        model:           Object exposing ``encode(list[str]) -> np.ndarray`` of
                         shape ``(n, EMBEDDING_DIM)`` (RavenBERT ``EmbeddingModel``).
        in_path:         Source structured parquet (needs ``RP_STORY_ID``, ``HEADLINE``).
        out_path:        Destination ``YYYY-MM.parquet``.
        batch_size:      Passed through to ``model.encode``.
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

            emb = np.asarray(
                model.encode(headlines, batch_size=batch_size, show_progress_bar=False),
                dtype=np.float32,
            )
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
    model: Any,
    source_dir: Path,
    out_dir: Path,
    start_year: int,
    end_year: int,               # inclusive
    *,
    batch_size: int = 128,
    write_chunk_rows: int = 200_000,
    log_every: int = 100_000,
    overwrite: bool = False,
) -> None:
    """Embed every month in ``[start_year, end_year]`` present in ``source_dir``.

    Mirrors ``ingest.ingest_range``: existing output months are skipped unless
    ``overwrite`` is set, so an interrupted run resumes by re-entering the same
    output directory.  Source months absent from ``source_dir`` are warned and
    skipped (some months genuinely have no data).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    all_months = [
        (y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)
    ]
    todo: list[tuple[str, Path]] = []
    for y, m in all_months:
        name = f"{y}-{m:02d}.parquet"
        src = source_dir / name
        if not src.exists():
            log.warning("source month missing, skipping: %s", name)
            continue
        if not overwrite and (out_dir / name).exists():
            continue
        todo.append((name, src))

    log.info(
        "embed: %d month(s) to process (of %d in range), overwrite=%s",
        len(todo), len(all_months), overwrite,
    )
    total = len(todo)
    for i, (name, src) in enumerate(todo, 1):
        tag = f"[{i}/{total}] {name[:7]}"
        t_month = time.monotonic()
        n = embed_month(
            model, src, out_dir / name,
            batch_size=batch_size,
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
    model_path: Path | str,
    pipeline_version: str,
    *,
    source_artifact_id: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    device: str | None = None,
    pipeline: str = PIPELINE,
    pipeline_repo: str | None = PIPELINE_REPO,
    repo_dir: Path | None = None,
    batch_size: int = 128,
    write_chunk_rows: int = 200_000,
    log_every: int = 100_000,
) -> "Artifact":
    """Embed a ``ravenpack_headlines`` artifact into a new ``headline_embeddings`` one.

    Wraps ``embed_range`` in a datalake run: the output directory is allocated by
    the index, the source artifact is recorded as lineage, the RavenBERT model
    card (with a weights sha256) is attached, sidecars and hashes are written on
    completion, and a crash leaves the artifact registered as partial.

    Args:
        index:              Datalake index to register the artifact in.
        model_path:         RavenBERT embedding weights directory.
        pipeline_version:   Semantic version of this pipeline.
        source_artifact_id: Explicit source artifact; defaults to
                            ``index.latest("ravenpack_headlines")``.
        start_year/end_year: Restrict the months embedded; default to the source
                            artifact's declared range.
        device:             Force a torch device ('cuda'|'mps'|'cpu'); None = auto.
        pipeline, pipeline_repo, repo_dir: Provenance passthrough.
        batch_size, write_chunk_rows, log_every: Throughput / logging knobs.

    Returns:
        The completed Artifact.
    """
    model_path = Path(model_path)

    src = index.get(source_artifact_id) if source_artifact_id else index.latest(SOURCE_KIND)
    hp = src.meta.hyperparams
    resolved_start = start_year if start_year is not None else hp.get("start_year")
    resolved_end = end_year if end_year is not None else hp.get("end_year")
    if resolved_start is None or resolved_end is None:
        raise ValueError(
            "start_year/end_year not provided and not present in source "
            f"artifact hyperparams ({src.artifact_id})"
        )

    log.info("hashing RavenBERT model directory %s ...", model_path)
    weights_sha256 = model_dir_sha256(model_path)
    backend_note = (
        f"embedded via embedx at {os.environ.get('EMBEDX_BASE_URL', '?')}"
        if device == "embedx" else None
    )
    card = build_model_card(model_path, weights_sha256, backend_note=backend_note)

    # Keep hyperparams minimal -- they feed the artifact_id slug.  batch_size and
    # the source artifact id go into the run notes instead (mirrors
    # ingest_to_datalake, which keeps raw_dir/columns out of the slug).
    hyperparams: dict[str, Any] = {
        "start_year": resolved_start,
        "end_year": resolved_end,
    }
    notes = f"source_artifact={src.artifact_id} batch_size={batch_size}"

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
        model = load_embedding_model(model_path, device=device)
        embed_range(
            model,
            source_dir=src.path,
            out_dir=run.out_dir,
            start_year=resolved_start,
            end_year=resolved_end,
            batch_size=batch_size,
            write_chunk_rows=write_chunk_rows,
            log_every=log_every,
            overwrite=False,
        )
        n_months = len(list(run.out_dir.glob("*.parquet")))
        if n_months == 0:
            raise RuntimeError(
                f"embed produced no output for {resolved_start}-{resolved_end}; "
                f"check source artifact {src.artifact_id} at {src.path}"
            )
        run.note(f"{n_months} monthly embedding files from {src.artifact_id}")

    return index.get(run.artifact_id)


# ---------------------------------------------------------------------------
# Content verifier (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: "Artifact") -> "list[Finding]":
    """Verify a headline_embeddings artifact matches its declared scope.

    Registered as the ``headline_embeddings`` entry point.  Checks:
      - one parquet per month in [start_year, end_year] (missing -> WARNING)
      - no parquet outside the declared range (-> ERROR)
      - a first/middle/last sample are non-empty and match EMBEDDING_SCHEMA
        (RP_STORY_ID + EMBEDDING Array(Float16, 384), nothing else)
    """
    from datalake.verify import Finding, Severity

    findings: list[Finding] = []
    aid = artifact.artifact_id
    hp = artifact.meta.hyperparams
    start_year = hp.get("start_year")
    end_year = hp.get("end_year")

    if start_year is None or end_year is None:
        findings.append(Finding(
            Severity.WARNING, aid,
            "no start_year/end_year in hyperparams; cannot verify date coverage",
        ))
        return findings

    present = {p.name for p in artifact.path.glob("*.parquet")}
    expected = {
        f"{y}-{m:02d}.parquet"
        for y in range(start_year, end_year + 1)
        for m in range(1, 13)
    }

    missing = expected - present
    if missing:
        findings.append(Finding(
            Severity.WARNING, aid,
            f"{len(missing)} of {len(expected)} declared months missing: "
            f"{', '.join(sorted(missing)[:6])}"
            + (" ..." if len(missing) > 6 else ""),
        ))

    unexpected = present - expected
    if unexpected:
        findings.append(Finding(
            Severity.ERROR, aid,
            f"{len(unexpected)} parquet(s) outside declared range "
            f"[{start_year}, {end_year}]: {', '.join(sorted(unexpected)[:6])}",
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
