"""EmbedxBackend against an in-process fake embedx (httpx.MockTransport).

The fake enforces embedx's load-once semantics: the first request that names
a model fixes its pooling and dtype; a later request with different options
gets 409. It returns rows slightly off unit norm (as bf16/fp16 rounding after
server-side normalisation does) and in shuffled order.
"""
from __future__ import annotations

import json
import os
import zlib

import httpx
import numpy as np
import pytest

from nlp.backends.base import IncompatibleModelError
from nlp.backends.embedx import EmbedxBackend
from nlp.backends.http import RemoteError

KEY = "embedx-secret"
MODEL = "/models/fake-embedder"
DIM = 8


def _vec(text: str) -> np.ndarray:
    v = np.random.default_rng(zlib.crc32(text.encode())).normal(size=DIM)
    return v / np.linalg.norm(v)


class FakeEmbedx:
    def __init__(self, resident: dict | None = None):
        self.models: dict[str, dict] = {}
        if resident:
            self.models[MODEL] = {"model_id": MODEL, "idle_s": 12.0, **resident}
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"error": {"message": "unauthorized"}})
        if request.url.path == "/info":
            return httpx.Response(200, json={
                "version": "0.3.1", "normalize": True,
                "concurrency": {"max_concurrent_requests": 4},
                "models": list(self.models.values())})
        body = json.loads(request.content)
        self.requests.append(body)
        entry = self.models.get(body["model"])
        if entry is None:
            entry = self.models[body["model"]] = {
                "model_id": body["model"], "pooling": body["pooling"],
                "dtype": body.get("dtype", "auto"), "idle_s": 0.0}
        elif entry["pooling"] != body["pooling"] or entry["dtype"] != body.get("dtype", "auto"):
            return httpx.Response(409, json={"error": {"message": "load options differ"}})
        data = [{"index": i, "embedding": (_vec(t) * 1.003).tolist()}
                for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data[::-1]})


def _backend(fake: FakeEmbedx, **kw) -> EmbedxBackend:
    return EmbedxBackend("http://embedx.test/v1", MODEL, KEY,
                         transport=httpx.MockTransport(fake.handler), **kw)


def test_loads_with_our_options_and_sends_them_every_request():
    fake = FakeEmbedx()
    b = _backend(fake, dtype="float16", batch_size=3)
    texts = [f"headline {i}" + "!" * i for i in range(7)]
    X = b.embed(texts)
    np.testing.assert_allclose(X, np.stack([_vec(t) for t in texts]), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)   # renormalised
    assert all(r["pooling"] == "cls" and r["dtype"] == "float16" for r in fake.requests)
    assert all(len(r["input"]) <= 3 for r in fake.requests)
    assert fake.models[MODEL]["dtype"] == "float16"
    cfg = b.embedding_config()
    assert (cfg.dim, cfg.pooling) == (DIM, "cls")


def test_resident_with_other_dtype_is_refused():
    fake = FakeEmbedx(resident={"pooling": "cls", "dtype": "auto"})
    with pytest.raises(IncompatibleModelError, match="dtype=auto"):
        _backend(fake, dtype="float16")


def test_resident_with_same_options_is_reused():
    fake = FakeEmbedx(resident={"pooling": "cls", "dtype": "float32"})
    b = _backend(fake, dtype="float32")
    assert b.embed(["a"]).shape == (1, DIM)


def test_options_changed_under_us_raise():
    fake = FakeEmbedx()
    b = _backend(fake)
    fake.models[MODEL]["dtype"] = "bfloat16"       # someone reloaded it differently
    with pytest.raises(RemoteError, match="409"):
        b.embed(["a"])
    with pytest.raises(IncompatibleModelError):
        b.check_unchanged()


def test_classification_not_served():
    b = _backend(FakeEmbedx())
    with pytest.raises(NotImplementedError):
        b.classify(["x"])


def test_info_has_model_entry_and_no_key():
    b = _backend(FakeEmbedx())
    info = b.info()
    assert info["model"]["dtype"] == "float16" and info["server"]["version"] == "0.3.1"
    assert KEY not in json.dumps(info) + json.dumps(b.identity())


def test_missing_env_raises(monkeypatch):
    monkeypatch.delenv("EMBEDX_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDX_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="EMBEDX_BASE_URL"):
        EmbedxBackend()


@pytest.mark.live
def test_live_embedx_smoke():
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("EMBEDX_BASE_URL"):
        pytest.skip("EMBEDX_BASE_URL not set")
    try:
        b = EmbedxBackend(dtype="float16")
    except (httpx.HTTPError, IncompatibleModelError, RemoteError) as exc:
        pytest.skip(f"embedx not usable as fp16 here: {exc}")
    X = b.embed(["Fed hikes rates", "Apple beats estimates"])
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
