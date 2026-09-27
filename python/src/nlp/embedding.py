"""Embedder: text -> unit vectors on any backend, with a compatibility check.

``Embedder(backend, dim=384, pooling="cls")`` checks at construction that the
backend serves an embedding model of that output dim and pooling (the
inference structure the downstream scorer relies on), and raises
``IncompatibleModelError`` otherwise. It does not check model names or
weights: any CLS-pooled 384-d encoder is compatible.

``model_card()`` records what ran: backend, full serving metadata, the
compatibility-check results, and the weights sha256 when the backend can know
it (local). ``embed_texts_cached`` keys a local cache by the exact texts AND
the embedder's identity (backend, dtype, served model, structure), with a
JSON sidecar holding the full card: a change of backend or dtype is a cache
miss, never a silent reuse.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nlp.backends.base import Backend, IncompatibleModelError

if TYPE_CHECKING:
    from datalake import ModelCard

log = logging.getLogger(__name__)

DEFAULT_DIM = 384
DEFAULT_POOLING = "cls"
_NORM_TOL = 1e-3


def _digest(obj: Any, size: int = 8) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.blake2b(raw, digest_size=size).hexdigest()


class Embedder:
    """Unit-norm text embeddings from a compatible backend.

    Args:
        backend: any backend serving embeddings (local / TEI / embedx).
        dim:     required output dimension.
        pooling: required pooling ("cls", "mean", ...).
        model_id / version / repo: the card's model naming (what the vectors
                 are called downstream), e.g. "ravenbert-embedding" / "1.0".
    """

    def __init__(self, backend: Backend, *, dim: int = DEFAULT_DIM,
                 pooling: str = DEFAULT_POOLING, model_id: str = "ravenbert-embedding",
                 version: str = "1.0",
                 repo: str = "https://github.com/GabinTB/RavenBERT") -> None:
        self.backend = backend
        self.dim, self.pooling = dim, pooling
        self.model_id, self.version, self.repo = model_id, version, repo
        self.checks = self._check()

    def _check(self) -> dict[str, Any]:
        cfg = self.backend.embedding_config()
        if cfg.dim != self.dim:
            raise IncompatibleModelError(f"{self.backend.name} serves dim {cfg.dim}, "
                                         f"need {self.dim}")
        if cfg.pooling != self.pooling:
            raise IncompatibleModelError(f"{self.backend.name} serves pooling {cfg.pooling!r}, "
                                         f"need {self.pooling!r}")
        probe = self.backend.embed(["compatibility probe"])
        norm = float(np.linalg.norm(probe[0]))
        if probe.shape != (1, self.dim) or abs(norm - 1.0) > _NORM_TOL:
            raise IncompatibleModelError(f"probe returned shape {probe.shape}, norm {norm:.4f}")
        return {"dim": cfg.dim, "pooling": cfg.pooling, "dtype": self.backend.dtype,
                "probe_norm": round(norm, 6), "passed": True}

    def encode(self, texts: str | Sequence[str]) -> np.ndarray:
        """float32 unit rows: (d,) for one text, (N, d) for N texts."""
        if isinstance(texts, str):
            return self.backend.embed([texts])[0]
        return self.backend.embed(list(texts))

    def identity(self) -> dict[str, Any]:
        """Everything that changes the vectors: the cache key and the provenance digest."""
        return {"embedder": {"model_id": self.model_id, "version": self.version,
                             "dim": self.dim, "pooling": self.pooling},
                "backend": self.backend.identity()}

    def identity_digest(self) -> str:
        return _digest(self.identity())

    def model_card(self) -> ModelCard:
        from datalake import ModelCard

        return ModelCard(
            model_id=self.model_id, version=self.version, repo=self.repo,
            weights_public=False, weights_sha256=self.backend.weights_sha256(),
            architecture="BERT encoder, sentence-transformers",
            dim=self.dim, pooling=self.pooling,
            notes=f"L2-normalised {self.backend.dtype} inference via {self.backend.name}.",
            backend=self.backend.name,
            serving={**self.backend.info(), "checks": self.checks,
                     "identity": self.backend.identity(),
                     "identity_digest": self.identity_digest()},
        )


def embedder_from_card(card: ModelCard, *, model_path: str | None = None,
                       **backend_kwargs: Any) -> Embedder:
    """The Embedder an artifact was produced with, rebuilt from its model card.

    The backend is rebuilt from the recorded identity and must serve the same
    model (remote servers are queried first); the naming, dim and pooling come
    from the card. Local backends read their weights from ``model_path`` or
    ``$RAVENBERT_EMBEDDING_MODEL_PATH`` and must hash to the recorded sha256.
    """
    import os

    from nlp.backends import backend_from_identity

    identity = (card.serving or {}).get("identity") or {}
    if identity.get("backend") == "local":
        model_path = model_path or os.environ.get(ENV_EMBEDDING_MODEL_PATH)
    backend = backend_from_identity(identity, task="embedding", model_path=model_path,
                                    **backend_kwargs)
    return Embedder(backend, dim=card.dim or DEFAULT_DIM, pooling=card.pooling or DEFAULT_POOLING,
                    model_id=card.model_id, version=card.version, repo=card.repo)


def texts_digest(texts: Sequence[str]) -> str:
    return hashlib.blake2b("\x00".join(texts).encode(), digest_size=8).hexdigest()


def embed_texts_cached(texts: Sequence[str], embedder: Embedder, cache_dir: Path,
                       *, prefix: str = "texts") -> tuple[np.ndarray, dict[str, Any]]:
    """Embeddings of ``texts`` from a local cache keyed by texts AND embedder identity.

    Returns ``(vectors float32 (N, d), meta)``; ``meta`` is the sidecar: texts
    digest, identity, identity digest and the full model card (as a dict).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{prefix}__{texts_digest(texts)}__{embedder.identity_digest()}"
    npy, sidecar = cache_dir / f"{key}.npy", cache_dir / f"{key}.json"
    if npy.exists() and sidecar.exists():
        log.info("embedding cache hit %s", npy.name)
        return np.load(npy), json.loads(sidecar.read_text())
    log.info("embedding %d texts via %s (%s) ...", len(texts), embedder.backend.name,
             embedder.backend.dtype)
    vectors = embedder.encode(list(texts)).astype(np.float32)
    meta = {"texts_digest": texts_digest(texts), "n_texts": len(texts),
            "identity": embedder.identity(), "identity_digest": embedder.identity_digest(),
            "model_card": embedder.model_card().to_dict()}
    tmp = npy.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, vectors)
    tmp.replace(npy)
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return vectors, meta


ENV_EMBEDDING_MODEL_PATH = "RAVENBERT_EMBEDDING_MODEL_PATH"


def load_embedder(backend: str = "tei", dtype: str = "float16", *,
                  model_path: str | None = None, dim: int = DEFAULT_DIM,
                  pooling: str = DEFAULT_POOLING, **kwargs: Any) -> Embedder:
    """``Embedder`` on a named backend (default TEI, fp16).

    The local backend reads its weights directory from ``model_path`` or
    ``$RAVENBERT_EMBEDDING_MODEL_PATH``.
    """
    import os

    from nlp.backends import make_backend

    if backend == "local":
        model_path = model_path or os.environ.get(ENV_EMBEDDING_MODEL_PATH)
    return Embedder(make_backend(backend, task="embedding", dtype=dtype,
                                 model_path=model_path, **kwargs), dim=dim, pooling=pooling)
