"""Resume safety: rebuild an Embedder / Sentimeter from an artifact's card and refuse
a backend that no longer serves the recorded model (server queried first)."""
from __future__ import annotations

import shutil

import httpx
import pytest

from nlp.backends import IncompatibleModelError, assert_same_model, backend_from_identity
from nlp.backends.local import LocalBackend
from nlp.backends.tei import TEIBackend
from nlp.embedding import Embedder, embedder_from_card
from nlp.sentiment import FinbertSentimeter, sentimeter_from_card

from .test_backends_tei import DIM, KEY, FakeTEI


def test_assert_same_model_lists_every_difference(embedder_dir):
    b = LocalBackend(embedder_dir, task="embedding", dtype="float32", device="cpu")
    assert_same_model(b.identity(), b)
    recorded = {**b.identity(), "weights_sha256": "0" * 64, "dtype": "float16"}
    with pytest.raises(IncompatibleModelError, match="weights_sha256.*|dtype"):
        assert_same_model(recorded, b)


def test_artifacts_without_identity_cannot_be_resumed():
    with pytest.raises(ValueError, match="no backend identity"):
        backend_from_identity({}, task="embedding")


def test_tei_card_rebuilds_and_checks_the_live_server(monkeypatch):
    monkeypatch.setenv("TEI_BASE_URL", "http://tei.test")
    monkeypatch.setenv("TEI_API_KEY", KEY)
    fake = FakeTEI()
    transport = httpx.MockTransport(fake.handler)
    card = Embedder(TEIBackend(transport=transport), dim=DIM).model_card()
    assert card.serving["identity"]["model_id"] == "/data/fake"

    again = embedder_from_card(card, transport=transport)          # same server model
    assert again.backend.name == "tei" and again.dim == DIM

    fake.info = {**fake.info, "model_id": "/data/another-model"}      # switched on vertex
    with pytest.raises(IncompatibleModelError, match="model_id"):
        embedder_from_card(card, transport=transport)
    fake.info = {**fake.info, "model_id": "/data/fake", "model_dtype": "float32"}
    with pytest.raises(IncompatibleModelError):                       # dtype changed too
        embedder_from_card(card, transport=transport)


def test_local_card_requires_the_same_weights(embedder_dir, tmp_path):
    card = Embedder(LocalBackend(embedder_dir, task="embedding", dtype="float32", device="cpu"),
                    dim=16).model_card()
    assert embedder_from_card(card, model_path=str(embedder_dir), device="cpu").dim == 16
    copy = tmp_path / embedder_dir.name
    shutil.copytree(embedder_dir, copy)
    (copy / "README.md").write_text("tampered")
    with pytest.raises(IncompatibleModelError, match="weights_sha256"):
        embedder_from_card(card, model_path=str(copy), device="cpu")


def test_sentimeter_card_restores_settings(finbert_like_dir):
    s = FinbertSentimeter(LocalBackend(finbert_like_dir, task="classification",
                                       dtype="float32", device="cpu"), min_confidence=0.2)
    again = sentimeter_from_card(s.model_card(), model_path=str(finbert_like_dir), device="cpu")
    assert isinstance(again, FinbertSentimeter)
    assert again.min_confidence == 0.2 and again.score_rule == "band"
    assert again.backend.identity() == s.backend.identity()
