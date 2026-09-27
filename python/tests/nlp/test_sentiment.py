"""Sentimeters: score maps (hand-checked), confidence, NaN rule, shapes, checks.

The FinBERT band tests are the ones previously in
tests/ravenpack/headlines/test_sentiment_model.py, moved with the formula.
"""
from __future__ import annotations

import numpy as np
import pytest

from nlp.backends.base import Backend, IncompatibleModelError, softmax64
from nlp.backends.local import LocalBackend
from nlp.sentiment import FinbertSentimeter, RavenbertSentimeter, load_sentimeter
from nlp.sentiment.finbert import THIRD, finbert_band, top_margin
from nlp.sentiment.ravenbert import (
    CLASS_VALUES,
    ordinal_dispersion,
    ordinal_mean,
    ordinal_median,
)

from .conftest import TEXTS


class FixedBackend(Backend):
    """Returns preset probability rows (label order as given)."""

    name = "fixed"

    def __init__(self, labels, rows):
        super().__init__("float32")
        self._labels, self.rows = labels, np.asarray(rows, dtype=np.float64)

    def labels(self):
        return list(self._labels)

    def classify(self, texts):
        return self.rows[: len(texts)]

    def info(self):
        return {"engine": "fixed"}


RB_LABELS = [f"LABEL_{i}" for i in range(41)]
FB_LABELS = ["positive", "negative", "neutral"]          # not canonical order


def _band(neg, neu, pos) -> tuple[float, float]:
    s, lab = finbert_band(np.array([neg]), np.array([neu]), np.array([pos]))
    return float(s[0]), float(lab[0])


# ---------------------------------------------------------------------------
# FinBERT band (moved)
# ---------------------------------------------------------------------------

class TestFinbertBand:
    def test_hand_checked_values(self):
        s, lab = _band(0.05, 0.50, 0.45)          # argmax neutral: (0.40 / 0.45) / 3
        assert s == pytest.approx(0.4 / 0.45 / 3) and s == pytest.approx(0.2963, abs=1e-4)
        assert lab == 0.0
        s, lab = _band(0.10, 0.20, 0.70)          # positive: 1/3 + (2/3)(0.70 - 0.20)
        assert s == pytest.approx(2 / 3) and lab == 1.0
        s, lab = _band(0.70, 0.20, 0.10)
        assert s == pytest.approx(-2 / 3) and lab == -1.0
        assert _band(0.0, 0.0, 1.0) == (pytest.approx(1.0), 1.0)
        assert _band(1.0, 0.0, 0.0) == (pytest.approx(-1.0), -1.0)
        assert _band(0.0, 1.0, 0.0) == (0.0, 0.0)
        assert _band(0.30, 0.40, 0.30)[0] == 0.0

    def test_ties(self):
        assert _band(0.2, 0.4, 0.4) == (pytest.approx(THIRD), 1.0)
        assert _band(0.4, 0.4, 0.2) == (pytest.approx(-THIRD), -1.0)
        assert _band(0.45, 0.10, 0.45) == (0.0, 0.0)
        assert _band(1 / 3, 1 / 3, 1 / 3) == (0.0, 0.0)

    @pytest.mark.parametrize("p_neg", [0.0, 0.1, 0.25])
    def test_continuous_across_neutral_polar_boundaries(self, p_neg):
        rest = 1.0 - p_neg
        for sign, flip in ((1, False), (-1, True)):
            for e in (1e-7, 1e-9):
                a, b = (rest / 2 + e, rest / 2 - e), (rest / 2 - e, rest / 2 + e)
                for pole, neu in (a, b):
                    probs = (p_neg, neu, pole) if not flip else (pole, neu, p_neg)
                    assert _band(*probs)[0] == pytest.approx(sign * THIRD, abs=1e-6)

    def test_band_matches_argmax_on_random_simplex(self):
        p = np.random.default_rng(0).dirichlet([0.7, 0.7, 0.7], size=200_000)
        s, lab = finbert_band(p[:, 0], p[:, 1], p[:, 2])
        np.testing.assert_array_equal(lab, np.array([-1.0, 0.0, 1.0])[np.argmax(p, axis=1)])
        assert (s[lab == 1] >= THIRD).all() and (s[lab == -1] <= -THIRD).all()
        assert (np.abs(s[lab == 0]) < THIRD).all() and (np.abs(s) <= 1.0).all()


# ---------------------------------------------------------------------------
# RavenBERT ordinal maps
# ---------------------------------------------------------------------------

class TestRavenbertMaps:
    def test_mean_hand_checked(self):
        p = np.zeros((2, 41))
        p[0, [40, 39, 20, 0]] = [0.5, 0.2, 0.2, 0.1]        # s = 1, 0.95, 0, -1
        p[1, 20] = 1.0
        np.testing.assert_allclose(ordinal_mean(p), [0.5 + 0.2 * 0.95 - 0.1, 0.0])

    def test_median_point_masses_and_interpolation(self):
        p = np.zeros((5, 41))
        p[0, 30] = 1.0                                       # point mass at s = 0.5
        p[1, 0] = 1.0                                        # point mass at s = -1
        p[2, 40] = 1.0                                       # point mass at s = +1
        p[3, [20, 21]] = [0.25, 0.75]                        # bin of s=.05 is [.025, .075]
        p[4, [0, 40]] = [0.5, 0.5]                           # bimodal: top of the lower bin
        med = ordinal_median(p)
        np.testing.assert_allclose(med[:3], [0.5, -1.0, 1.0], atol=1e-12)
        # j = 21: s_21 - h + (0.5 - 0.25) / 0.75 * step
        assert med[3] == pytest.approx(0.05 - 0.025 + (0.25 / 0.75) * 0.05)
        assert med[4] == pytest.approx(-0.975)

    def test_dispersion_orders_adjacent_vs_opposite(self):
        adjacent = np.zeros(41)
        adjacent[[26, 27, 28]] = 1 / 3                       # s = 0.30, 0.35, 0.40
        opposite = np.zeros(41)
        opposite[[0, 40]] = 0.5                              # bimodal, EV = 0
        sig = ordinal_dispersion(np.stack([adjacent, opposite]))
        assert sig[0] < 0.05 and sig[1] == pytest.approx(1.0)
        # entropy would rank them the other way round
        ent = [-(q[q > 0] * np.log(q[q > 0])).sum() for q in (adjacent, opposite)]
        assert ent[0] > ent[1]

    def test_all_maps_in_range_on_random_simplex(self):
        q = np.random.default_rng(1).dirichlet(np.ones(41) * 0.3, size=5000)
        for f in (ordinal_mean, ordinal_median):
            assert (np.abs(f(q)) <= 1 + 1e-12).all()
        assert ((ordinal_dispersion(q) >= 0) & (ordinal_dispersion(q) <= 1)).all()
        np.testing.assert_allclose(ordinal_mean(q), q @ CLASS_VALUES)


def test_top_margin():
    np.testing.assert_allclose(top_margin(np.array([[0.2, 0.5, 0.3], [1 / 3] * 3])),
                               [0.2, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Sentimeter behaviour (fixed backend)
# ---------------------------------------------------------------------------

def test_finbert_canonical_order_read_from_labels():
    rows = [[0.7, 0.1, 0.2]]                                 # pos, neg, neu
    s = FinbertSentimeter(FixedBackend(FB_LABELS, rows))
    np.testing.assert_allclose(s.canonical("x"), [0.1, 0.2, 0.7])
    assert s.score("x") == pytest.approx(_band(0.1, 0.2, 0.7)[0])
    assert s.confidence("x") == pytest.approx(0.5)


def test_shapes_scalar_for_one_text_array_for_many():
    rows = np.full((3, 41), 1 / 41)
    s = RavenbertSentimeter(FixedBackend(RB_LABELS, rows))
    assert isinstance(s.score("one"), float)
    assert s.score(["a", "b", "c"]).shape == (3,)
    assert s.canonical("one").shape == (41,) and s.canonical(["a", "b"]).shape == (2, 41)


def test_min_confidence_sets_nan_not_zero():
    bimodal = np.zeros(41)
    bimodal[[0, 40]] = 0.5
    peaked = np.zeros(41)
    peaked[30] = 1.0
    s = RavenbertSentimeter(FixedBackend(RB_LABELS, [bimodal, peaked]), min_confidence=0.5)
    out = s.evaluate(["a", "b"])
    assert np.isnan(out["score"][0]) and out["score"][1] == pytest.approx(0.5)
    assert out["confidence"][0] == pytest.approx(0.0, abs=1e-6)
    assert out["score"].dtype == np.float32


def test_median_rule_selectable_and_unknown_rule_refused():
    p = np.zeros(41)
    p[[20, 21]] = [0.25, 0.75]
    s = RavenbertSentimeter(FixedBackend(RB_LABELS, [p]), score_rule="median")
    assert s.score("x") == pytest.approx(0.025 + 0.25 / 0.75 * 0.05, rel=1e-6)
    with pytest.raises(ValueError, match="score_rule"):
        FinbertSentimeter(FixedBackend(FB_LABELS, [[1, 0, 0]]), score_rule="mean")


@pytest.mark.parametrize("cls,labels", [
    (RavenbertSentimeter, ["negative", "neutral", "positive"]),
    (RavenbertSentimeter, [f"L{i}" for i in range(40)]),
    (FinbertSentimeter, RB_LABELS),
    (FinbertSentimeter, ["bearish", "neutral", "bullish"]),
])
def test_incompatible_heads_refused(cls, labels):
    with pytest.raises(IncompatibleModelError):
        cls(FixedBackend(labels, np.ones((1, len(labels))) / len(labels)))


def test_columns_and_card():
    s = FinbertSentimeter(FixedBackend(FB_LABELS, [[0.2, 0.3, 0.5]]), min_confidence=0.1)
    assert s.columns() == ["SENT_SCORE", "SENT_CONF"]
    assert s.columns(canonical=True)[-3:] == ["P_NEG", "P_NEU", "P_POS"]
    card = s.model_card()
    assert card.model_id == "finbert-sentiment" and card.version == "2.0"
    assert card.backend == "fixed" and card.serving["sentimeter"]["min_confidence"] == 0.1
    assert card.serving["checks"]["canonical_order"] == [1, 2, 0]


# ---------------------------------------------------------------------------
# Real classifiers (tiny random weights) through the local backend
# ---------------------------------------------------------------------------

def test_ravenbert_score_equals_sentiment_model_predict(ravenbert_like_dir):
    from ravenbert.sentiment.model import SentimentModel

    ref = SentimentModel.from_path(ravenbert_like_dir, device="cpu", precision="fp32")
    want = ref.predict(TEXTS, batch_size=64, optimized_inference=False)
    s = load_sentimeter("ravenbert", "local", "float32", model_path=str(ravenbert_like_dir),
                        device="cpu", batch_size=2)
    np.testing.assert_allclose(s.score(TEXTS), want, atol=1e-6)


def test_finbert_local_matches_manual_reorder(finbert_like_dir):
    import torch
    from ravenbert.sentiment.model import clean_text
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    s = FinbertSentimeter(LocalBackend(finbert_like_dir, task="classification",
                                       dtype="float32", device="cpu"))
    tok = AutoTokenizer.from_pretrained(str(finbert_like_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(finbert_like_dir)).eval()
    cleaned = [clean_text(t) for t in TEXTS]
    with torch.inference_mode():
        logits = model(**tok(cleaned, padding=True, return_tensors="pt")).logits.double().numpy()
    want = softmax64(logits)[:, [1, 2, 0]]                  # neg, neu, pos
    np.testing.assert_allclose(s.canonical(TEXTS), want, atol=1e-6)


def test_load_sentimeter_rejects_embedx_and_unknown_model():
    with pytest.raises(ValueError, match="embeddings only"):
        load_sentimeter("finbert", "embedx")
    with pytest.raises(ValueError, match="model must be one of"):
        load_sentimeter("vader")
