"""Hugging Face Text Embeddings Inference (TEI) backend.

TEI serves one encoder model at a time (switched on the server, not here):

    GET  /info      model_id, model_dtype, model_type (embedding pooling or
                    classifier id2label), max_input_length, batch limits, version
    POST /embed     {"inputs": [text, ...]}              -> [[float, ...], ...]
    POST /predict   {"inputs": [[text], [text], ...]}    -> per input, all labels
                    (each text wrapped in its own list: a flat list of two
                    strings would be scored as ONE sentence pair)

The backend reads ``/info`` once at construction and refuses a server whose
``model_dtype`` differs from the requested dtype. What the served model IS
(embedder vs classifier, pooling, labels) is checked by the NLP object that
uses the backend. ``check_unchanged()`` re-reads ``/info`` and raises if the
served model changed since construction (a model switched on the server
mid-job would otherwise silently mix two models in one artifact).

Batches are at most ``max_client_batch_size`` texts, length-sorted, sent on
``max_concurrent_requests // batch`` threads (TEI counts every input against
``max_concurrent_requests``; beyond it the server answers 429, retried with
backoff). ``""`` is rejected by TEI and is sent as ``" "``: BERT-family
tokenizers strip whitespace, so both encode to [CLS] [SEP].
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
    canonical_dtype,
    length_sorted_batches,
    softmax64,
    unit_rows,
)
from nlp.backends.http import HTTPClient, map_ordered

log = logging.getLogger(__name__)

ENV_BASE_URL = "TEI_BASE_URL"
ENV_API_KEY = "TEI_API_KEY"
EMPTY_TEXT_SUBSTITUTE = " "

# /info fields that define what the served model computes
_IDENTITY_FIELDS = ("model_id", "model_dtype", "model_type", "max_input_length", "version")


class TEIBackend(Backend):
    """A TEI server.

    Args:
        base_url:   server root; default ``$TEI_BASE_URL``.
        api_key:    Bearer token; default ``$TEI_API_KEY``.
        dtype:      the dtype the server must be serving ("float16" / "float32").
        batch_size: texts per request, capped at the server's max_client_batch_size.
        workers:    concurrent requests; default max_concurrent_requests // batch.
        timeout:    per-request timeout (s).
        transport:  injected httpx transport (tests).
    """

    name = "tei"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, *,
                 dtype: str = "float16", batch_size: int | None = None,
                 workers: int | None = None, timeout: float = 120.0,
                 transport: Any = None) -> None:
        super().__init__(dtype)
        base_url = base_url or os.environ.get(ENV_BASE_URL)
        if not base_url:
            raise RuntimeError(f"{ENV_BASE_URL} must be set (in .env or environment) "
                               "to use the TEI backend")
        api_key = api_key if api_key is not None else os.environ.get(ENV_API_KEY)
        self._http = HTTPClient(base_url, api_key, timeout=timeout, transport=transport,
                                max_connections=128)
        self._info = self._http.get("/info")
        served = canonical_dtype(self._info.get("model_dtype", "?"))
        if served != self.dtype:
            raise IncompatibleModelError(
                f"TEI serves {self._info.get('model_id')} in {served}, requested {self.dtype}")
        server_cap = int(self._info.get("max_client_batch_size") or 32)
        self.batch_size = min(batch_size or server_cap, server_cap)
        max_conc = int(self._info.get("max_concurrent_requests") or 512)
        self.workers = workers or max(1, max_conc // self.batch_size)
        self._dim: int | None = None
        log.info("TEI %s: %s (%s), batch %d x %d workers", self._info.get("version"),
                 self._info.get("model_id"), served, self.batch_size, self.workers)

    # -- served model structure --------------------------------------------

    @property
    def model_type(self) -> dict[str, Any]:
        return dict(self._info.get("model_type") or {})

    def embedding_config(self) -> EmbeddingConfig:
        embedding = self.model_type.get("embedding")
        if embedding is None:
            raise IncompatibleModelError(
                f"TEI serves a non-embedding model ({list(self.model_type)}): "
                f"{self._info.get('model_id')}")
        if self._dim is None:                              # /info has no dim: measure it
            self._dim = int(np.asarray(self._http.post("/embed", {"inputs": ["dim"]})).shape[1])
        return EmbeddingConfig(dim=self._dim, pooling=embedding.get("pooling"))

    def labels(self) -> list[str]:
        classifier = self.model_type.get("classifier")
        if classifier is None:
            raise IncompatibleModelError(
                f"TEI serves a non-classifier model ({list(self.model_type)}): "
                f"{self._info.get('model_id')}")
        id2label = {int(k): str(v) for k, v in (classifier.get("id2label") or {}).items()}
        if sorted(id2label) != list(range(len(id2label))):
            raise IncompatibleModelError(f"TEI id2label indices are not contiguous: {id2label}")
        return [id2label[i] for i in range(len(id2label))]

    # -- canonical operations ----------------------------------------------

    @staticmethod
    def _prepare(texts: Sequence[str]) -> list[str]:
        return [t if t else EMPTY_TEXT_SUBSTITUTE for t in texts]

    def _batched(self, texts: list[str], call, width: int, dtype: type) -> np.ndarray:
        out = np.empty((len(texts), width), dtype=dtype)
        batches = list(length_sorted_batches(texts, self.batch_size))
        results = map_ordered(lambda idx: call([texts[i] for i in idx]), batches, self.workers)
        for idx, res in zip(batches, results):
            if res.shape != (len(idx), width):
                raise ValueError(f"TEI returned {res.shape}, expected {(len(idx), width)}")
            out[idx] = res
        return out

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        dim = self.embedding_config().dim
        texts = self._prepare(texts)
        if not texts:
            return np.empty((0, dim), dtype=np.float32)

        def call(batch: list[str]) -> np.ndarray:
            payload = {"inputs": batch, "normalize": True, "truncate": True}
            return np.asarray(self._http.post("/embed", payload), dtype=np.float32)

        return unit_rows(self._batched(texts, call, dim, np.float32))

    def classify(self, texts: Sequence[str]) -> np.ndarray:
        labels = self.labels()
        index = {label: i for i, label in enumerate(labels)}
        texts = self._prepare(texts)
        if not texts:
            return np.empty((0, len(labels)), dtype=np.float64)

        def call(batch: list[str]) -> np.ndarray:
            # each text in its own list: a flat list would be ONE sentence pair
            payload = {"inputs": [[t] for t in batch], "raw_scores": True, "truncate": True}
            response = self._http.post("/predict", payload)
            logits = np.full((len(batch), len(labels)), np.nan)
            for row, scores in zip(logits, response):
                for item in scores:
                    row[index[item["label"]]] = item["score"]
            if np.isnan(logits).any():
                raise ValueError("TEI /predict did not return a score for every label")
            return logits

        return softmax64(self._batched(texts, call, len(labels), np.float64))

    # -- metadata -----------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {"engine": "tei", "base_url": self._http.base_url, "info": dict(self._info),
                "client": {"batch_size": self.batch_size, "workers": self.workers,
                           "empty_text_sent_as": EMPTY_TEXT_SUBSTITUTE,
                           "batching": "length-sorted"}}

    def identity(self) -> dict[str, Any]:
        return {**super().identity(), **{k: self._info.get(k) for k in _IDENTITY_FIELDS}}

    def check_unchanged(self) -> None:
        """Re-read /info; raise if the served model changed since construction."""
        now = self._http.get("/info")
        before = {k: self._info.get(k) for k in _IDENTITY_FIELDS}
        after = {k: now.get(k) for k in _IDENTITY_FIELDS}
        if before != after:
            raise IncompatibleModelError(f"TEI model changed mid-run: {before} -> {after}")

    def close(self) -> None:
        self._http.close()
