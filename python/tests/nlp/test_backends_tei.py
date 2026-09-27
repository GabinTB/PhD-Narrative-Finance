"""TEIBackend against an in-process fake TEI server (httpx.MockTransport).

The fake mimics the TEI behaviour the backend relies on: /info, batch cap
(422 above max_client_batch_size), "" rejected (422), 429 on demand, /predict
returning every label per input in arbitrary order. A live smoke test runs
only when TEI_BASE_URL is set and the server is reachable.
"""
from __future__ import annotations

import json
import logging
import os
import zlib

import httpx
import numpy as np
import pytest

from nlp.backends.base import IncompatibleModelError, softmax64
from nlp.backends.http import RemoteError
from nlp.backends.tei import TEIBackend

KEY = "secret-test-key-123"
DIM = 8


def _vec(text: str) -> list[float]:
    rng = np.random.default_rng(zlib.crc32(text.encode()))
    v = rng.normal(size=DIM)
    return (v / np.linalg.norm(v)).tolist()


def _logits(text: str, k: int) -> np.ndarray:
    return np.random.default_rng(zlib.crc32(text.encode()) + 1).normal(size=k)


class FakeTEI:
    def __init__(self, *, model_type=None, dtype="float16", cap=4, fail_429=0):
        self.info = {"model_id": "/data/fake", "model_sha": None, "model_dtype": dtype,
                     "model_type": model_type or {"embedding": {"pooling": "cls"}},
                     "max_concurrent_requests": 16, "max_input_length": 512,
                     "max_client_batch_size": cap, "auto_truncate": True, "version": "1.9.4"}
        self.cap, self.fail_429 = cap, fail_429
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"error": "unauthorized"})
        if request.url.path == "/info":
            return httpx.Response(200, json=self.info)
        body = json.loads(request.content)
        self.requests.append({"path": request.url.path, **body})
        if self.fail_429 > 0:
            self.fail_429 -= 1
            return httpx.Response(429, json={"error": "overloaded"})
        inputs = body["inputs"]
        if len(inputs) > self.cap:
            return httpx.Response(422, json={"error": f"batch size {len(inputs)} > {self.cap}"})
        if request.url.path == "/embed":
            if any(t == "" for t in inputs):
                return httpx.Response(422, json={"error": "`inputs` cannot be empty"})
            return httpx.Response(200, json=[_vec(t) for t in inputs])
        if request.url.path == "/predict":
            id2label = self.info["model_type"]["classifier"]["id2label"]
            out = []
            for item in inputs:
                assert isinstance(item, list) and len(item) == 1     # one text per list
                z = _logits(item[0], len(id2label))
                scores = [{"label": id2label[str(i)], "score": float(z[i])}
                          for i in range(len(id2label))]
                out.append(scores[::-1])                              # arbitrary order
            return httpx.Response(200, json=out)
        return httpx.Response(404, json={"error": "not found"})


def _backend(fake: FakeTEI, **kw) -> TEIBackend:
    return TEIBackend("http://tei.test", KEY, transport=httpx.MockTransport(fake.handler), **kw)


FINBERT_TYPE = {"classifier": {"id2label": {"0": "positive", "1": "negative", "2": "neutral"}}}


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------

def test_embed_batches_at_server_cap_and_restores_order():
    fake = FakeTEI(cap=3)
    b = _backend(fake)
    texts = [f"text number {i} " + "x" * (i % 5) for i in range(10)]
    X = b.embed(texts)
    np.testing.assert_allclose(X, np.array([_vec(t) for t in texts], dtype=np.float32),
                               atol=1e-6)
    embeds = [r for r in fake.requests if r["path"] == "/embed" and r["inputs"] != ["dim"]]
    assert all(len(r["inputs"]) <= 3 for r in embeds)
    assert sum(len(r["inputs"]) for r in embeds) == 10
    assert b.embedding_config().dim == DIM and b.embedding_config().pooling == "cls"


def test_empty_text_sent_as_space():
    fake = FakeTEI()
    X = _backend(fake).embed(["", "a"])
    np.testing.assert_allclose(X[0], _vec(" "), atol=1e-6)


def test_429_is_retried():
    fake = FakeTEI(fail_429=3)
    b = _backend(fake)
    b.embed(["a", "b"])
    assert b._http.n_retries_429 == 3


def test_server_error_raises_without_fallback():
    fake = FakeTEI(cap=2)
    b = _backend(fake, batch_size=2)
    b.batch_size = 5                                   # force an over-cap request
    with pytest.raises(RemoteError, match="422"):
        b.embed(["a", "b", "c", "d", "e"])


def test_dtype_mismatch_refused():
    with pytest.raises(IncompatibleModelError, match="float16"):
        _backend(FakeTEI(dtype="float16"), dtype="float32")
    assert _backend(FakeTEI(dtype="float32"), dtype="fp32").dtype == "float32"


def test_embedder_refuses_classification():
    with pytest.raises(IncompatibleModelError, match="non-classifier"):
        _backend(FakeTEI()).labels()


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def test_classify_raw_scores_softmax_in_label_index_order():
    fake = FakeTEI(model_type=FINBERT_TYPE)
    b = _backend(fake)
    assert b.labels() == ["positive", "negative", "neutral"]
    texts = ["up", "down big", "", "flat flat flat"]
    p = b.classify(texts)
    want = softmax64(np.stack([_logits(t or " ", 3) for t in texts]))
    np.testing.assert_allclose(p, want, atol=1e-12)
    req = next(r for r in fake.requests if r["path"] == "/predict")
    assert req["raw_scores"] is True and all(isinstance(x, list) for x in req["inputs"])


def test_classifier_refuses_embedding():
    with pytest.raises(IncompatibleModelError, match="non-embedding"):
        _backend(FakeTEI(model_type=FINBERT_TYPE)).embed(["x"])


# ---------------------------------------------------------------------------
# metadata / secrets / model switch
# ---------------------------------------------------------------------------

def test_key_never_in_info_repr_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeTEI(fail_429=1)
    b = _backend(fake)
    b.embed(["a"])
    blob = json.dumps(b.info(), default=str) + repr(b._http) + json.dumps(b.identity())
    assert KEY not in blob and KEY not in caplog.text
    assert b.info()["info"]["model_id"] == "/data/fake"


def test_model_switch_detected():
    fake = FakeTEI()
    b = _backend(fake)
    b.check_unchanged()
    fake.info = {**fake.info, "model_id": "/data/other"}
    with pytest.raises(IncompatibleModelError, match="changed mid-run"):
        b.check_unchanged()


def test_missing_env_raises(monkeypatch):
    monkeypatch.delenv("TEI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="TEI_BASE_URL"):
        TEIBackend()


# ---------------------------------------------------------------------------
# live smoke test (skipped unless a TEI server is configured and reachable)
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_tei_embedding_smoke():
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("TEI_BASE_URL"):
        pytest.skip("TEI_BASE_URL not set")
    try:
        probe = TEIBackend(dtype="float16")
    except (httpx.HTTPError, IncompatibleModelError) as exc:
        pytest.skip(f"TEI not usable as fp16 embedder here: {exc}")
    if "embedding" not in probe.model_type:
        pytest.skip("TEI currently serves a classifier")
    X = probe.embed(["Fed hikes rates", "", "Apple beats estimates"])
    assert X.shape[0] == 3
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
