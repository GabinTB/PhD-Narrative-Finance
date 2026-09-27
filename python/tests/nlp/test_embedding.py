"""Embedder compatibility checks, output shapes, provenance card, and the cache."""
from __future__ import annotations

import json

import numpy as np
import pytest

from nlp.backends.base import Backend, EmbeddingConfig, IncompatibleModelError, unit_rows
from nlp.embedding import Embedder, embed_texts_cached, load_embedder


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, dim=384, pooling="cls", dtype="float16", scale=1.0, served="m"):
        super().__init__(dtype)
        self._cfg = EmbeddingConfig(dim, pooling)
        self.scale, self.served, self.calls = scale, served, 0

    def embedding_config(self):
        return self._cfg

    def embed(self, texts):
        self.calls += 1
        rng = np.random.default_rng(len(texts))
        return unit_rows(rng.normal(size=(len(texts), self._cfg.dim))) * self.scale

    def info(self):
        return {"engine": "fake", "served": self.served}

    def identity(self):
        return {**super().identity(), "served": self.served}


def test_compatible_backend_passes_and_records_checks():
    e = Embedder(FakeBackend())
    assert e.checks["passed"] and e.checks["dim"] == 384 and e.checks["pooling"] == "cls"
    assert e.encode("one").shape == (384,)
    assert e.encode(["a", "b"]).shape == (2, 384)


@pytest.mark.parametrize("backend,match", [
    (FakeBackend(dim=768), "dim"),
    (FakeBackend(pooling="mean"), "pooling"),
    (FakeBackend(scale=2.0), "norm"),
])
def test_incompatible_backend_refused(backend, match):
    with pytest.raises(IncompatibleModelError, match=match):
        Embedder(backend)


def test_model_card_carries_backend_serving_checks():
    card = Embedder(FakeBackend(dtype="float32")).model_card()
    assert card.backend == "fake" and card.model_id == "ravenbert-embedding"
    assert card.serving["served"] == "m" and card.serving["checks"]["dtype"] == "float32"
    assert card.weights_sha256 is None                   # remote: not knowable, not guessed
    assert card.serving["identity_digest"]


def test_cache_hits_same_identity_misses_on_dtype_or_model_change(tmp_path):
    texts = ["alpha", "beta", "gamma"]
    b16 = FakeBackend(dtype="float16")
    v1, meta = embed_texts_cached(texts, Embedder(b16), tmp_path)
    calls = b16.calls
    v2, meta2 = embed_texts_cached(texts, Embedder(b16), tmp_path)
    assert b16.calls == calls + 1                        # only the construction probe
    np.testing.assert_array_equal(v1, v2)
    assert meta == meta2 and meta["n_texts"] == 3
    assert meta["model_card"]["backend"] == "fake"

    b32 = FakeBackend(dtype="float32")
    embed_texts_cached(texts, Embedder(b32), tmp_path)
    b_other = FakeBackend(served="other")
    embed_texts_cached(texts, Embedder(b_other), tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 3
    assert len(list(tmp_path.glob("*.json"))) == 3
    sidecars = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert len({s["identity_digest"] for s in sidecars}) == 3


def test_identity_keeps_embedder_and_backend_fields_apart():
    ident = Embedder(FakeBackend(served="/data/x")).identity()
    assert ident["embedder"]["model_id"] == "ravenbert-embedding"
    assert ident["backend"]["served"] == "/data/x" and ident["backend"]["dtype"] == "float16"


def test_cache_misses_on_text_change(tmp_path):
    e = Embedder(FakeBackend())
    embed_texts_cached(["a"], e, tmp_path)
    embed_texts_cached(["a", "b"], e, tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_local_embedder_end_to_end(embedder_dir):
    e = load_embedder("local", "float32", model_path=str(embedder_dir), device="cpu", dim=16)
    X = e.encode(["stocks rally", ""])
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
    card = e.model_card()
    assert card.backend == "local" and card.weights_sha256
    assert e.checks["pooling"] == "cls"


def test_local_embedder_checks_dim(embedder_dir):
    from nlp.backends.local import LocalBackend

    with pytest.raises(IncompatibleModelError, match="dim"):
        Embedder(LocalBackend(embedder_dir, task="embedding", dtype="float32", device="cpu"))
    Embedder(LocalBackend(embedder_dir, task="embedding", dtype="float32", device="cpu"),
             dim=16)


def test_unknown_backend_name():
    with pytest.raises(ValueError, match="backend must be one of"):
        load_embedder("vllm")
