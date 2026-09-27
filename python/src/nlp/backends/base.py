"""Inference backends: the canonical model operations and nothing else.

A backend runs ONE model and exposes what that model does natively:

    embed(texts)    -> float32 (n, d), L2-normalised rows    (embedding models)
    classify(texts) -> float64 (n, k), softmax probabilities  (classifiers),
                       columns in ``labels()`` order
    labels()        -> the classifier's labels, index order
    embedding_config() -> EmbeddingConfig(dim, pooling) of the served embedder
    info()          -> JSON-able serving metadata recorded in every artifact

Pre/post-processing (text cleaning, score maps, confidence, caching) belongs
to the NLP objects built on top (``Embedder``, ``Sentimeter``); the
compatibility checks between what an object needs and what a backend serves
live there too, so a backend never needs to know which model it is "supposed"
to be. Only the inference structure is checked (dim, pooling, labels, dtype),
never a model name or a weights hash.

Every backend batches in descending text-length order (less padding per
batch) and restores input order, and computes softmax client-side in float64
from raw logits, so local and remote classifiers apply the same maths.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from nlp.corrections import l2_normalise

DTYPES: tuple[str, ...] = ("float16", "float32")
_DTYPE_ALIASES = {"fp16": "float16", "float16": "float16", "half": "float16",
                  "fp32": "float32", "float32": "float32", "float": "float32"}


def canonical_dtype(dtype: str) -> str:
    """'fp16' / 'float16' -> 'float16', 'fp32' / 'float32' -> 'float32'."""
    try:
        return _DTYPE_ALIASES[str(dtype).lower()]
    except KeyError:
        raise ValueError(f"dtype must be one of {DTYPES} (or fp16/fp32), got {dtype!r}") from None


class IncompatibleModelError(ValueError):
    """The served model cannot run the requested task (dim, pooling, labels, dtype)."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """What an embedding model serves: output dim and pooling ('cls', 'mean', ...)."""

    dim: int
    pooling: str | None


class Backend(ABC):
    """One model behind one inference engine (local, TEI, embedx)."""

    name: ClassVar[str]

    def __init__(self, dtype: str) -> None:
        self.dtype = canonical_dtype(dtype)

    # -- canonical operations (a backend implements what its model supports) --

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"the {self.name} backend does not serve embeddings")

    def classify(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"the {self.name} backend does not serve classification")

    def labels(self) -> list[str]:
        raise NotImplementedError(f"the {self.name} backend does not serve classification")

    def embedding_config(self) -> EmbeddingConfig:
        raise NotImplementedError(f"the {self.name} backend does not serve embeddings")

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """JSON-able serving metadata: engine, served model, dtype, limits, versions."""

    def weights_sha256(self) -> str | None:
        """sha256 of the weights actually loaded, when knowable (local only)."""
        return None

    def check_unchanged(self) -> None:
        """Raise if the served model changed since construction (remote servers).

        A local model cannot be switched underneath a job, so this is a no-op here.
        """
        return None

    def identity(self) -> dict[str, Any]:
        """The fields that change the numbers a backend produces.

        Used to key caches and to detect a model switched under a running job:
        engine, served model id, dtype, task structure.
        """
        return {"backend": self.name, "dtype": self.dtype}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def length_sorted_batches(texts: Sequence[str], batch_size: int) -> Iterator[np.ndarray]:
    """Index batches over ``texts`` in descending length order (less padding).

    Yields integer index arrays; the caller writes each batch's results back
    at those indices, which restores input order.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = np.argsort([-len(t) for t in texts], kind="stable")
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def run_in_length_order(texts: Sequence[str], batch_size: int, width: int,
                        run_batch: Callable[[list[str]], np.ndarray],
                        dtype: type = np.float32) -> np.ndarray:
    """Apply ``run_batch`` over length-sorted batches, results in input order.

    ``run_batch`` maps a list of texts to an array of shape (len(batch), width).
    """
    out = np.empty((len(texts), width), dtype=dtype)
    for idx in length_sorted_batches(texts, batch_size):
        result = np.asarray(run_batch([texts[i] for i in idx]), dtype=dtype)
        if result.shape != (len(idx), width):
            raise ValueError(f"batch returned shape {result.shape}, expected {(len(idx), width)}")
        out[idx] = result
    return out


def softmax64(logits: np.ndarray) -> np.ndarray:
    """Row softmax in float64 (max-shifted): the one probability map for every backend."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def unit_rows(X: np.ndarray) -> np.ndarray:
    """float32 L2-normalised rows. Remote servers normalise before rounding, so
    their rows are only approximately unit; every backend renormalises."""
    return l2_normalise(X)
