"""Thin client for github.com/GabinTB/embedx's OpenAI-compatible
``POST /v1/embeddings``, so RavenBERT embedding can be dispatched to a GPU
machine reachable only over the network (e.g. Vertex, 10.10.10.2, over a
direct 2.5G link) instead of running in-process.

embedx replicates one model across GPUs of possibly different speed/memory
and shards inputs at runtime; from this client's point of view it is just
an OpenAI-shaped embeddings endpoint (``EMBEDX_BASE_URL``, e.g.
``http://10.10.10.2:8477/v1``), naming a model by local path or HF id
(``EMBEDX_MODEL``, e.g. ``/models/RavenPack/ravenbert-embedding``).

Always sends ``pooling="cls"`` -- RavenBERT is CLS-pooled (see embed.py's
ModelCard, ``pooling="cls"``), and embedx requires ``pooling`` on a model's
first load on that server (it never guesses: the wrong choice would return
plausible-looking but silently wrong vectors). Resending the SAME value on
every later call is always safe and a no-op; a DIFFERENT value later would
409, which never happens here since this client only ever sends "cls".

embedx L2-normalizes by default (server-side ``normalize=True``), matching
this pipeline's "L2-normalized float32" contract everywhere else.

No fallback: a connection error here is raised, not swallowed -- an offload
path that silently degrades to a multi-hour local CPU run is a worse
failure mode than stopping immediately with a clear message. This module
does not import torch or ravenbert at all, so ``--device embedx`` works
from a machine with no GPU stack installed -- only the ``openai`` and
``httpx`` clients, both already base dependencies of this project.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

EMBEDDING_DIM = 384

_ENV_BASE_URL = "EMBEDX_BASE_URL"
_ENV_MODEL = "EMBEDX_MODEL"
_ENV_API_KEY = "EMBEDX_API_KEY"


class RemoteEmbeddingModel:
    """Drop-in for ``ravenbert.embedding.model.EmbeddingModel``, backed by embedx.

    Exposes the same ``.encode(texts, batch_size=..., show_progress_bar=...)``
    -> ``np.ndarray`` interface so every caller of the local model (embed.py,
    narrative_scoring/descriptions.py) works unmodified.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        pooling: str = "cls",
        api_key: str | None = None,
        client: Any | None = None,
    ):
        self.base_url = base_url
        self.model = model
        self.pooling = pooling
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=base_url, api_key=api_key or "unused-without-EMBEDX_API_KEY"
            )

    @classmethod
    def from_env(cls) -> "RemoteEmbeddingModel":
        """Build from EMBEDX_BASE_URL / EMBEDX_MODEL / EMBEDX_API_KEY.

        Raises RuntimeError naming the specific missing variable, rather
        than a bare KeyError from deep inside a run.
        """
        base_url = os.environ.get(_ENV_BASE_URL)
        if not base_url:
            raise RuntimeError(
                f"{_ENV_BASE_URL} must be set (in .env or environment) to use "
                "--device embedx"
            )
        model = os.environ.get(_ENV_MODEL)
        if not model:
            raise RuntimeError(
                f"{_ENV_MODEL} must be set (in .env or environment) to use "
                "--device embedx"
            )
        return cls(base_url=base_url, model=model, api_key=os.environ.get(_ENV_API_KEY))

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 128,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """(n, EMBEDDING_DIM) float32, L2-normalized -- matches the local model.

        Splits texts into batch_size-sized groups (one HTTP call each; embedx
        does its own length-sorted, token-budget batching server-side across
        whatever GPUs it has, so batch_size here only bounds request size,
        not GPU batching). Each group's response is reassembled by its
        `.index` field rather than assumed to preserve input order.
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        n_batches = (len(texts) + batch_size - 1) // batch_size
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
                extra_body={"pooling": self.pooling},
            )
            ordered = sorted(response.data, key=lambda d: d.index)
            chunk = np.asarray([d.embedding for d in ordered], dtype=np.float32)
            if chunk.shape != (len(batch), EMBEDDING_DIM):
                raise ValueError(
                    f"embedx returned {chunk.shape}, expected {(len(batch), EMBEDDING_DIM)}"
                )
            chunks.append(chunk)
            if show_progress_bar:
                log.info("embedx: batch %d/%d done", i // batch_size + 1, n_batches)

        return np.concatenate(chunks, axis=0)
