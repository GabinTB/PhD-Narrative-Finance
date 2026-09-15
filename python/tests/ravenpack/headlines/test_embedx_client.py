"""Tests for ravenpack.headlines.embedx_client.

No real HTTP: a fake client (matching the openai SDK's
`.embeddings.create(**kwargs) -> response` shape) is injected directly, so
these tests never touch the network or import `openai`'s transport layer.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ravenpack.headlines.embedx_client import EMBEDDING_DIM, RemoteEmbeddingModel


class _FakeEmbeddingObject(SimpleNamespace):
    index: int
    embedding: list[float]


class _FakeResponse(SimpleNamespace):
    data: list[_FakeEmbeddingObject]


class _FakeEmbeddingsResource:
    """Stand-in for openai's client.embeddings -- records every call."""

    def __init__(
        self, *, dim: int = EMBEDDING_DIM, shuffle: bool = False, error: Exception | None = None
    ):
        self.dim = dim
        self.shuffle = shuffle
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        texts = kwargs["input"]
        objs = [
            _FakeEmbeddingObject(index=i, embedding=_vector(t, self.dim))
            for i, t in enumerate(texts)
        ]
        if self.shuffle:
            objs = list(reversed(objs))
        return _FakeResponse(data=objs)


class _FakeClient(SimpleNamespace):
    embeddings: _FakeEmbeddingsResource


def _vector(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding for a text (stable, not normalized)."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(size=dim).astype(np.float32).tolist()


def _make_model(**resource_kwargs) -> tuple[RemoteEmbeddingModel, _FakeEmbeddingsResource]:
    resource = _FakeEmbeddingsResource(**resource_kwargs)
    client = _FakeClient(embeddings=resource)
    model = RemoteEmbeddingModel(
        base_url="http://10.10.10.2:8477/v1", model="/models/rb", client=client
    )
    return model, resource


class TestRequestShape:
    def test_sends_model_input_and_cls_pooling(self):
        model, resource = _make_model()
        model.encode(["a", "b", "c"])
        assert len(resource.calls) == 1
        call = resource.calls[0]
        assert call["model"] == "/models/rb"
        assert call["input"] == ["a", "b", "c"]
        assert call["extra_body"] == {"pooling": "cls"}

    def test_custom_pooling_is_sent_every_call(self):
        resource = _FakeEmbeddingsResource()
        client = _FakeClient(embeddings=resource)
        model = RemoteEmbeddingModel(
            base_url="http://x/v1", model="m", pooling="mean", client=client
        )
        model.encode(["a"])
        model.encode(["b"])
        assert resource.calls[0]["extra_body"] == {"pooling": "mean"}
        assert resource.calls[1]["extra_body"] == {"pooling": "mean"}


class TestResponseReassembly:
    def test_out_of_order_data_reassembled_by_index(self):
        model, resource = _make_model(shuffle=True)
        texts = ["first", "second", "third"]
        out = model.encode(texts)
        # Reconstruct what each text's vector should be, independent of order.
        expected = np.asarray([_vector(t, EMBEDDING_DIM) for t in texts], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_output_shape_and_dtype(self):
        model, _ = _make_model()
        out = model.encode(["a", "b"])
        assert out.shape == (2, EMBEDDING_DIM)
        assert out.dtype == np.float32

    def test_empty_input_returns_empty_array_without_a_call(self):
        model, resource = _make_model()
        out = model.encode([])
        assert out.shape == (0, EMBEDDING_DIM)
        assert resource.calls == []


class TestBatching:
    def test_splits_into_batch_size_groups_and_concatenates_in_order(self):
        model, resource = _make_model()
        texts = [f"text-{i}" for i in range(10)]
        out = model.encode(texts, batch_size=3)

        assert len(resource.calls) == 4  # ceil(10/3)
        assert [len(c["input"]) for c in resource.calls] == [3, 3, 3, 1]
        assert resource.calls[0]["input"] == texts[0:3]
        assert resource.calls[3]["input"] == texts[9:10]

        expected = np.asarray([_vector(t, EMBEDDING_DIM) for t in texts], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)


class TestHardFailure:
    def test_connection_error_propagates_unchanged(self):
        boom = ConnectionError("could not reach 10.10.10.2:8477")
        model, _ = _make_model(error=boom)
        with pytest.raises(ConnectionError, match="could not reach"):
            model.encode(["a"])

    def test_shape_mismatch_from_server_raises(self):
        model, resource = _make_model(dim=EMBEDDING_DIM - 1)  # wrong dim
        with pytest.raises(ValueError, match="expected"):
            model.encode(["a"])


class TestFromEnv:
    def test_raises_when_base_url_missing(self, monkeypatch):
        monkeypatch.delenv("EMBEDX_BASE_URL", raising=False)
        monkeypatch.setenv("EMBEDX_MODEL", "/models/rb")
        with pytest.raises(RuntimeError, match="EMBEDX_BASE_URL"):
            RemoteEmbeddingModel.from_env()

    def test_raises_when_model_missing(self, monkeypatch):
        monkeypatch.setenv("EMBEDX_BASE_URL", "http://10.10.10.2:8477/v1")
        monkeypatch.delenv("EMBEDX_MODEL", raising=False)
        with pytest.raises(RuntimeError, match="EMBEDX_MODEL"):
            RemoteEmbeddingModel.from_env()

    def test_builds_from_env_when_both_present(self, monkeypatch):
        monkeypatch.setenv("EMBEDX_BASE_URL", "http://10.10.10.2:8477/v1")
        monkeypatch.setenv("EMBEDX_MODEL", "/models/rb")
        monkeypatch.delenv("EMBEDX_API_KEY", raising=False)
        model = RemoteEmbeddingModel.from_env()
        assert model.base_url == "http://10.10.10.2:8477/v1"
        assert model.model == "/models/rb"
        assert model.pooling == "cls"
