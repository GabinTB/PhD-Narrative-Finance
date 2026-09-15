"""Taxonomy description embeddings: three pooling modes.

    CENTROID  mean-pool K paraphrase embeddings, renormalize -> (n_prim, 384)
    MAX       keep all K, score against each, take max at score time
    MEDIAN    keep all K, score against each, take median at score time

Centroid construction -- the normalization order matters::

    For primitive p with paraphrase embeddings e_1..e_K (each already unit norm):
        c_p = (1/K) * sum_k e_k        # mean is NOT unit norm
        d_p = c_p / ||c_p||            # renormalize so cosine == dot product

For MAX and MEDIAN, no pooling happens at embed time -- all K vectors are
retained and pooling happens at score time (see scoring.score_chunk).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from narrative_scoring.schema import EMBEDDING_DIM, TAXONOMY_EMBEDDINGS_SCHEMA

if TYPE_CHECKING:
    from datalake import Artifact
    from datalake.verify import Finding

log = logging.getLogger(__name__)


class PoolingMode(str, Enum):
    CENTROID = "centroid"
    MAX = "max"
    MEDIAN = "median"


@dataclass
class DescriptionEmbeddings:
    primitives: list[str]          # row order, length n_prim
    mode: PoolingMode
    vectors: np.ndarray             # (n_prim, 384) for CENTROID; (n_prim, K, 384) otherwise
    k: int


def _centroid(paraphrase_vecs: np.ndarray) -> np.ndarray:
    """paraphrase_vecs: (K, dim) unit-norm rows -> (dim,) unit-norm centroid.

    The mean is NOT unit norm; renormalize after averaging so cosine
    similarity equals a dot product against the result.
    """
    c = paraphrase_vecs.mean(axis=0)
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        return np.zeros_like(c, dtype=np.float32)
    return (c / norm).astype(np.float32)


def embed_descriptions(
    model: Any,
    paraphrases: dict[str, list[str]],
    primitives: list[str],
    mode: PoolingMode,
    batch_size: int = 256,
) -> DescriptionEmbeddings:
    """Embed the K paraphrases of each primitive (in ``primitives`` row order)."""
    ks = {len(paraphrases[p]) for p in primitives}
    if len(ks) > 1:
        raise ValueError(f"inconsistent paraphrase counts across primitives: {sorted(ks)}")
    k = next(iter(ks), 0)
    if k == 0:
        raise ValueError("no paraphrases to embed")

    all_texts: list[str] = []
    for p in primitives:
        all_texts.extend(paraphrases[p])

    emb = np.asarray(model.encode(all_texts, batch_size=batch_size), dtype=np.float32)
    if emb.shape != (len(primitives) * k, EMBEDDING_DIM):
        raise ValueError(
            f"encode returned {emb.shape}, expected {(len(primitives) * k, EMBEDDING_DIM)}"
        )
    emb = emb.reshape(len(primitives), k, EMBEDDING_DIM)

    if mode is PoolingMode.CENTROID:
        vectors = np.stack([_centroid(emb[i]) for i in range(len(primitives))]).astype(np.float32)
    else:
        vectors = emb  # (n_prim, K, dim); pooling deferred to score time

    return DescriptionEmbeddings(primitives=list(primitives), mode=mode, vectors=vectors, k=k)


# ---------------------------------------------------------------------------
# Datalake (de)serialization
# ---------------------------------------------------------------------------

def to_frame(desc: DescriptionEmbeddings) -> pl.DataFrame:
    """DescriptionEmbeddings -> TAXONOMY_EMBEDDINGS_SCHEMA frame.

    CENTROID writes one row per primitive with PARAPHRASE_I = -1; MAX/MEDIAN
    write K rows per primitive with PARAPHRASE_I = 0..K-1.
    """
    if desc.mode is PoolingMode.CENTROID:
        return pl.DataFrame(
            {
                "PRIMITIVE": desc.primitives,
                "PARAPHRASE_I": [-1] * len(desc.primitives),
                "EMBEDDING": list(desc.vectors.astype(np.float32)),
            },
            schema=TAXONOMY_EMBEDDINGS_SCHEMA,
        )

    primitives_col: list[str] = []
    idx_col: list[int] = []
    emb_col: list[np.ndarray] = []
    for i, p in enumerate(desc.primitives):
        for j in range(desc.k):
            primitives_col.append(p)
            idx_col.append(j)
            emb_col.append(desc.vectors[i, j].astype(np.float32))
    return pl.DataFrame(
        {"PRIMITIVE": primitives_col, "PARAPHRASE_I": idx_col, "EMBEDDING": emb_col},
        schema=TAXONOMY_EMBEDDINGS_SCHEMA,
    )


def from_frame(df: pl.DataFrame, mode: PoolingMode) -> DescriptionEmbeddings:
    """TAXONOMY_EMBEDDINGS_SCHEMA frame -> DescriptionEmbeddings, in file order."""
    if mode is PoolingMode.CENTROID:
        df = df.filter(pl.col("PARAPHRASE_I") == -1)
        primitives = df["PRIMITIVE"].to_list()
        vectors = np.asarray(df["EMBEDDING"].to_list(), dtype=np.float32)
        return DescriptionEmbeddings(primitives=primitives, mode=mode, vectors=vectors, k=1)

    df = df.sort(["PRIMITIVE", "PARAPHRASE_I"])
    primitives = df["PRIMITIVE"].unique(maintain_order=True).to_list()
    k = df.filter(pl.col("PRIMITIVE") == primitives[0]).height
    dim = df["EMBEDDING"].to_list()[0].__len__()
    vectors = np.zeros((len(primitives), k, dim), dtype=np.float32)
    for i, p in enumerate(primitives):
        sub = df.filter(pl.col("PRIMITIVE") == p)
        vectors[i] = np.asarray(sub["EMBEDDING"].to_list(), dtype=np.float32)
    return DescriptionEmbeddings(primitives=primitives, mode=mode, vectors=vectors, k=k)


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.write_parquet(tmp, compression="zstd")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Datalake-aware entry point
# ---------------------------------------------------------------------------

KIND = "taxonomy_embeddings"
PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"


def embed_taxonomy_to_datalake(
    index: Any,
    raw_data_path: Path,
    model_path: Path,
    taxonomy_name: str,
    pipeline_version: str,
    *,
    paraphrase_style: str = "headline",
    pooling: PoolingMode = PoolingMode.CENTROID,
    base_schema: list[str] | None = None,
    device: str | None = None,
    pipeline: str = PIPELINE,
    pipeline_repo: str | None = PIPELINE_REPO,
    repo_dir: Path | None = None,
    batch_size: int = 256,
) -> Any:
    """Embed one named taxonomy's descriptions into a taxonomy_embeddings artifact.

    The taxonomy CSV/JSONL files live in raw/, outside the datalake, so this
    artifact has no lineage sources; the taxonomy directory path and a
    sha256 of each input file are recorded in the run notes instead.
    """
    import hashlib

    from narrative_scoring.taxonomy import load_taxonomy
    from ravenpack.headlines.embed import build_model_card, model_dir_sha256

    tv = load_taxonomy(
        raw_data_path, taxonomy_name,
        paraphrase_style=paraphrase_style, base_schema=base_schema,
    )

    log.info("hashing RavenBERT model directory %s ...", model_path)
    weights_sha256 = model_dir_sha256(model_path)
    card = build_model_card(model_path, weights_sha256)

    input_files = [tv.csv_path, tv.jsonl_path("semantic"), tv.jsonl_path("headline")]
    file_digests = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files if p.exists()
    }

    hyperparams: dict[str, Any] = {
        "taxonomy_name": taxonomy_name,
        "paraphrase_style": paraphrase_style,
        "pooling": pooling.value,
    }
    notes = (
        f"taxonomy_dir={tv.path} " + " ".join(f"{k}={v[:12]}" for k, v in file_digests.items())
    )

    with index.run(
        kind=KIND,
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        repo_dir=repo_dir,
        hyperparams=hyperparams,
        notes=notes,
        sources=[],
        model_card=card,
        verifier=KIND,
        hash_pattern="*.parquet",
    ) as run:
        from ravenbert.embedding.model import EmbeddingModel

        model = EmbeddingModel.from_path(str(model_path), device=device)

        primitives = sorted(tv.paraphrases().keys())
        log.info(
            "embedding %d taxonomy primitives (%s pooling)...", len(primitives), pooling.value
        )
        desc = embed_descriptions(
            model, tv.paraphrases(), primitives, pooling, batch_size=batch_size
        )
        _write_parquet_atomic(to_frame(desc), run.out_dir / "taxonomy.parquet")

        run.note(f"{len(primitives)} taxonomy primitives")

    return index.get(run.artifact_id)


# ---------------------------------------------------------------------------
# Content verifier (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: "Artifact") -> "list[Finding]":
    """Verify a taxonomy_embeddings artifact matches its declared scope.

    Registered as the ``taxonomy_embeddings`` entry point. Checks:
      - taxonomy.parquet is present, readable, non-empty, schema-matching
      - CENTROID rows are unit-norm within float32 tolerance
    """
    from datalake.verify import Finding, Severity

    findings: list[Finding] = []
    aid = artifact.artifact_id
    hp = artifact.meta.hyperparams
    pooling = hp.get("pooling", "centroid")

    tax_path = artifact.path / "taxonomy.parquet"
    if not tax_path.is_file():
        findings.append(Finding(Severity.ERROR, aid, "taxonomy.parquet missing"))
    else:
        findings.extend(_check_embeddings_file(aid, tax_path, pooling, "taxonomy.parquet"))

    return findings


def _check_embeddings_file(
    aid: str, path: Path, pooling: str, label: str
) -> "list[Finding]":
    from datalake.verify import Finding, Severity

    findings: list[Finding] = []
    try:
        df = pl.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        return [Finding(Severity.ERROR, aid, f"{label}: unreadable parquet: {exc}")]

    if df.is_empty():
        return [Finding(Severity.ERROR, aid, f"{label}: file is empty")]

    if df.schema != TAXONOMY_EMBEDDINGS_SCHEMA:
        findings.append(Finding(
            Severity.ERROR, aid,
            f"{label}: schema mismatch (got {dict(df.schema)}, "
            f"expected {dict(TAXONOMY_EMBEDDINGS_SCHEMA)})",
        ))
        return findings

    if pooling == "centroid":
        emb = np.asarray(df["EMBEDDING"].to_list(), dtype=np.float64)
        sq_norms = np.sum(emb**2, axis=1)
        n_bad = int(np.sum(np.abs(1.0 - sq_norms) >= 1e-4))
        if n_bad:
            findings.append(Finding(
                Severity.ERROR, aid,
                f"{label}: {n_bad} centroid row(s) not unit-norm",
            ))

    return findings
