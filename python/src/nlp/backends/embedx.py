"""embedx backend (github.com/GabinTB/embedx): OpenAI-shaped ``POST /v1/embeddings``.

embedx replicates one model across GPUs and shards inputs itself; it serves
embeddings only (``classify`` is not available). Load options are decided by
the FIRST request that names a model and inherited by every later one, so
this backend sends ``pooling`` AND ``dtype`` on every request: a model
resident with different options answers 409 instead of silently returning
vectors computed another way. At construction the backend reads embedx's
``/info`` and refuses a resident model whose pooling or dtype differs from
the requested ones (e.g. a model first loaded with ``dtype=auto``, which is
bfloat16 on current GPUs).

embedx normalises server-side before its output is rounded to the compute
dtype, so rows are only approximately unit; they are renormalised here.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from nlp.backends.base import (
    Backend,
    EmbeddingConfig,
    IncompatibleModelError,
    length_sorted_batches,
    unit_rows,
)
from nlp.backends.http import HTTPClient, map_ordered

log = logging.getLogger(__name__)

ENV_BASE_URL = "EMBEDX_BASE_URL"
ENV_MODEL = "EMBEDX_MODEL"
ENV_API_KEY = "EMBEDX_API_KEY"


def _server_root(base_url: str) -> str:
    """``http://host:8477/v1`` -> ``http://host:8477`` (where /info lives)."""
    base_url = base_url.rstrip("/")
    return base_url[:-3] if base_url.endswith("/v1") else base_url


class EmbedxBackend(Backend):
    """An embedx server, one model.

    Args:
        base_url:   ``.../v1`` root; default ``$EMBEDX_BASE_URL``.
        model:      model id (path or hub id); default ``$EMBEDX_MODEL``.
        api_key:    Bearer token; default ``$EMBEDX_API_KEY``.
        dtype:      compute dtype requested from the server ("float16"/"float32").
        pooling:    pooling requested at load ("cls", "mean", "last_token").
        batch_size: texts per request (embedx batches by tokens server-side).
        workers:    concurrent requests; default the server's max_concurrent_requests.
        transport:  injected httpx transport (tests).
    """

    name = "embedx"

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, *, dtype: str = "float16",
                 pooling: str = "cls", batch_size: int = 128, workers: int | None = None,
                 timeout: float = 300.0, transport: Any = None) -> None:
        super().__init__(dtype)
        base_url = base_url or os.environ.get(ENV_BASE_URL)
        model = model or os.environ.get(ENV_MODEL)
        for var, value in ((ENV_BASE_URL, base_url), (ENV_MODEL, model)):
            if not value:
                raise RuntimeError(f"{var} must be set (in .env or environment) "
                                   "to use the embedx backend")
        api_key = api_key if api_key is not None else os.environ.get(ENV_API_KEY)
        self.model, self.pooling, self.batch_size = model, pooling, batch_size
        self._http = HTTPClient(base_url, api_key, timeout=timeout, transport=transport)
        self._root = HTTPClient(_server_root(base_url), api_key, timeout=30.0,
                                transport=transport)
        server = self._root.get("/info")
        self._check_resident(server)
        self.workers = workers or int(server.get("concurrency", {})
                                      .get("max_concurrent_requests") or 4)
        # First request: loads the model with our options (or confirms them).
        self._dim = int(np.asarray(self._embed_batch(["dim"])).shape[1])
        self._server = self._root.get("/info")
        self._check_resident(self._server, must_exist=True)
        log.info("embedx %s: %s (%s, %s), %d workers", self._server.get("version"), model,
                 pooling, self.dtype, self.workers)

    # -- served model -------------------------------------------------------

    def _entry(self, server: dict[str, Any]) -> dict[str, Any] | None:
        return next((m for m in server.get("models", []) if m.get("model_id") == self.model), None)

    def _check_resident(self, server: dict[str, Any], must_exist: bool = False) -> None:
        entry = self._entry(server)
        if entry is None:
            if must_exist:
                raise IncompatibleModelError(f"embedx did not load {self.model}")
            return
        if entry.get("pooling") != self.pooling or entry.get("dtype") != self.dtype:
            raise IncompatibleModelError(
                f"embedx has {self.model} resident with pooling={entry.get('pooling')} "
                f"dtype={entry.get('dtype')}; requested pooling={self.pooling} "
                f"dtype={self.dtype}. Load options are fixed at first load: wait for its "
                f"keep_alive to expire (idle {entry.get('idle_s', 0):.0f}s), or restart embedx.")

    def embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(dim=self._dim, pooling=self.pooling)

    # -- canonical operation ------------------------------------------------

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        payload = {"model": self.model, "input": batch, "pooling": self.pooling,
                   "dtype": self.dtype}
        data = self._http.post("/embeddings", payload)["data"]
        ordered = sorted(data, key=lambda d: d["index"])
        return np.asarray([d["embedding"] for d in ordered], dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        if not texts:
            return out
        batches = list(length_sorted_batches(texts, self.batch_size))
        results = map_ordered(lambda idx: self._embed_batch([texts[i] for i in idx]),
                              batches, self.workers)
        for idx, res in zip(batches, results):
            if res.shape != (len(idx), self._dim):
                raise ValueError(f"embedx returned {res.shape}, expected {(len(idx), self._dim)}")
            out[idx] = res
        return unit_rows(out)

    # -- metadata -----------------------------------------------------------

    def info(self) -> dict[str, Any]:
        server = {k: v for k, v in self._server.items() if k != "models"}
        return {"engine": "embedx", "base_url": self._http.base_url, "server": server,
                "model": self._entry(self._server),
                "client": {"batch_size": self.batch_size, "workers": self.workers,
                           "batching": "length-sorted", "renormalised": True}}

    def identity(self) -> dict[str, Any]:
        return {**super().identity(), "model_id": self.model, "pooling": self.pooling,
                "dim": self._dim, "version": self._server.get("version")}

    def check_unchanged(self) -> None:
        """Re-read /info: the model must still be resident with our options."""
        self._check_resident(self._root.get("/info"))

    def close(self) -> None:
        self._http.close()
        self._root.close()
