"""Sentimeters: the 41-grid remap, the load-time rules (SQL vs numpy), storage precision,
the empty-text rule, shapes and head checks."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nlp.backends.base import Backend, IncompatibleModelError, softmax64
from nlp.backends.local import LocalBackend
from nlp.sentiment import FinbertSentimeter, RavenbertSentimeter, load_sentimeter
from nlp.sentiment.base import GRID, GRID_COLUMNS, band_matrix, to_grid
from nlp.sentiment.finbert import THIRD
from nlp.sentiment.ordinal_sql import RULES, grid_select_sql, reference

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
FB_EDGES = FinbertSentimeter.band_edges


def _fb(neg, neu, pos) -> np.ndarray:
    return to_grid(np.array([[neg, neu, pos]]), FB_EDGES)


def _write_grid(path: Path, G: np.ndarray) -> Path:
    """A grid table as the sentiment job writes it: Float16, all-null rows kept null."""
    G16 = np.asarray(G, dtype=np.float16)
    empty = np.isnan(G16).all(axis=1)
    cols = {"RP_STORY_ID": pa.array([f"{i:07d}" for i in range(len(G16))])}
    for j, c in enumerate(GRID_COLUMNS):
        cols[c] = pa.array(G16[:, j], type=pa.float16(), mask=empty)
    pq.write_table(pa.table(cols), path)
    return path


def _sql(path: Path, rule: str) -> tuple[np.ndarray, np.ndarray]:
    rows = duckdb.sql(grid_select_sql(f"read_parquet('{path}')", rule)
                      + " ORDER BY RP_STORY_ID").fetchall()
    return (np.array([np.nan if r[1] is None else r[1] for r in rows]),
            np.array([np.nan if r[2] is None else r[2] for r in rows]))


def _distributions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Peaked, flat, bimodal, point masses and FinBERT-remapped rows (all valid)."""
    k = n // 5
    bimodal = np.zeros((k, 41))
    bimodal[:, 2], bimodal[:, 38] = 0.5, 0.5
    bimodal = 0.9 * bimodal + 0.1 * rng.dirichlet(np.ones(41), size=k)
    parts = [rng.dirichlet(np.full(41, 0.05), size=k),              # peaked
             rng.dirichlet(np.full(41, 20.0), size=k),              # flat
             bimodal,
             np.eye(41)[rng.integers(0, 41, k)],                    # point masses
             to_grid(rng.dirichlet(np.ones(3), size=n - 4 * k), FB_EDGES)]
    return np.vstack(parts)


# ---------------------------------------------------------------------------
# The remap (base.band_matrix / to_grid)
# ---------------------------------------------------------------------------

class TestRemap:
    def test_ravenbert_is_the_identity(self):
        assert np.array_equal(band_matrix(RavenbertSentimeter.band_edges), np.eye(41))

    def test_mass_conservation(self):
        rng = np.random.default_rng(0)
        for edges in (FB_EDGES, (-1.0, 0.0, 1.0), (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0)):
            P = rng.dirichlet(np.ones(len(edges) - 1), size=500)
            np.testing.assert_allclose(to_grid(P, edges).sum(axis=1), 1.0, atol=1e-12)

    def test_equal_density_within_each_band_with_split_edge_bins(self):
        M = band_matrix(FB_EDGES)
        lo = np.maximum(GRID - 0.025, -1.0)
        hi = np.minimum(GRID + 0.025, 1.0)
        for c, (a, b) in enumerate(zip(FB_EDGES[:-1], FB_EDGES[1:])):
            overlap = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
            np.testing.assert_allclose(M[c], overlap / (b - a), atol=1e-12)   # density 1/width
            full = (lo >= a - 1e-12) & (hi <= b + 1e-12) & (hi - lo > 0.04)
            assert len(set(M[c, full].tolist())) == 1                   # bit-identical
        straddle = np.flatnonzero((M > 0).sum(axis=0) == 2)              # bins across +-1/3
        assert GRID[straddle].round(2).tolist() == [-0.35, 0.35]
        np.testing.assert_allclose(M[:, straddle].sum(axis=0), [0.075, 0.075], atol=1e-12)

    def test_finbert_mean_argmax_and_confidence(self):
        rng = np.random.default_rng(1)
        P = rng.dirichlet(np.ones(3), size=2000)
        s_mean, _ = reference(to_grid(P, FB_EDGES), "mean")
        np.testing.assert_allclose(s_mean, (2 / 3) * (P[:, 2] - P[:, 0]), atol=1.5e-3)
        for p, want_s, want_c in (([0.8, 0.15, 0.05], -0.675, 0.90 * 0.8),
                                  ([0.1, 0.8, 0.1], 0.0, 0.975 * 0.8),
                                  ([0.05, 0.15, 0.8], 0.675, 0.90 * 0.8)):
            s, c = reference(_fb(*p), "argmax")
            assert s[0] == pytest.approx(want_s, abs=1e-12)
            assert c[0] == pytest.approx(want_c, abs=1e-12)

    @pytest.mark.parametrize("stored", [np.float64, np.float16])
    def test_polar_wins_when_most_probable(self, stored):
        # p_pos = 0.50 vs p_neu = 0.47: with point-count splitting neutral would win
        s, c = reference(_fb(0.03, 0.47, 0.50).astype(stored), "argmax")
        assert s[0] > 0 and s[0] == pytest.approx(0.675)
        assert c[0] == pytest.approx(0.90 * 0.50, abs=1e-3)

    @pytest.mark.parametrize("stored", [np.float64, np.float16])
    def test_neutral_wins_without_cross_band_tie(self, stored):
        G = _fb(0.0, 0.51, 0.49).astype(stored)
        s, c = reference(G, "argmax")
        assert s[0] == pytest.approx(0.0, abs=1e-12)          # mean of s in [-0.3, 0.3]
        assert c[0] == pytest.approx(0.975 * 0.51, abs=1e-3)
        q = G.astype(np.float64) / G.astype(np.float64).sum()
        top = np.flatnonzero(q[0] == q[0].max())
        assert np.all(np.abs(GRID[top]) <= THIRD)                        # all in the neutral band

    def test_two_and_five_class_heads(self):
        G2 = to_grid(np.array([[0.3, 0.7]]), (-1.0, 0.0, 1.0))
        assert G2[0, GRID < 0].sum() == pytest.approx(0.3 - 0.3 * 0.025)  # s=0 bin splits
        assert G2[0, 20] == pytest.approx(0.3 * 0.025 + 0.7 * 0.025)
        edges5 = (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0)
        G5 = to_grid(np.eye(5), edges5)
        for c, (a, b) in enumerate(zip(edges5[:-1], edges5[1:])):
            inside = (GRID > a) & (GRID < b)
            outside = (GRID < a - 0.03) | (GRID > b + 0.03)
            assert G5[c, inside].sum() > 0.8 and G5[c, outside].sum() == 0

    def test_bad_edges_refused(self):
        with pytest.raises(ValueError, match="band edges"):
            band_matrix((-1.0, 0.5, 0.2, 1.0))
        with pytest.raises(ValueError, match="band edges"):
            band_matrix((-0.9, 0.0, 1.0))


# ---------------------------------------------------------------------------
# The load-time rules: SQL (ordinal_sql) vs the numpy reference
# ---------------------------------------------------------------------------

class TestOrdinalSql:
    def test_sql_matches_numpy_on_the_same_stored_values(self, tmp_path):
        rng = np.random.default_rng(2)
        G16 = _distributions(rng, 10_000).astype(np.float16)
        path = _write_grid(tmp_path / "g.parquet", G16)
        for rule in RULES:
            s_sql, c_sql = _sql(path, rule)
            s_ref, c_ref = reference(G16, rule)
            np.testing.assert_allclose(s_sql, s_ref, rtol=0, atol=1e-6, err_msg=rule)
            np.testing.assert_allclose(c_sql, c_ref, rtol=0, atol=1e-6, err_msg=rule)
            assert np.all((s_sql >= -1) & (s_sql <= 1)) and np.all((c_sql >= 0) & (c_sql <= 1))

    def test_float16_storage_within_1e3_of_float64(self):
        """Float16 stores each P_i to 2^-11 relative. mean: within 1e-3 everywhere.
        median: a quantile moves by (error in the cumulative mass) / (density there), so it
        is within 1e-3 wherever the density at the three quartiles is >= 0.5 (per unit of
        s); below that (the valley of a bimodal output) it can move more. argmax: within
        1e-3 unless the two largest probabilities are closer than 2^-10 relative, where
        Float16 cannot separate them (they may tie, ties are averaged, or flip). The test
        also asserts that differences occur ONLY in those two zones."""
        rng = np.random.default_rng(3)
        G = _distributions(rng, 5_000)
        top2 = np.sort(G, axis=1)[:, -2:]
        clear_max = (top2[:, 1] - top2[:, 0]) >= 2.0 ** -10 * top2[:, 1]
        q = G / G.sum(axis=1, keepdims=True)
        c = np.cumsum(q, axis=1)
        width = np.minimum(GRID + 0.025, 1.0) - np.maximum(GRID - 0.025, -1.0)
        dense = np.ones(len(G), bool)
        for level in (0.25, 0.5, 0.75):
            j = np.argmax(c >= level, axis=1)
            dense &= q[np.arange(len(G)), j] / width[j] >= 0.5
        zone = {"mean": np.ones(len(G), bool), "median": dense, "argmax": clear_max}
        for rule in RULES:
            s64, c64 = reference(G, rule)
            s16, c16 = reference(G.astype(np.float16), rule)
            ok = zone[rule]
            np.testing.assert_allclose(s16[ok], s64[ok], atol=1e-3, err_msg=rule)
            np.testing.assert_allclose(c16[ok], c64[ok], atol=1e-3, err_msg=rule)
            differ = ~np.isclose(c16, c64, atol=1e-3) | ~np.isclose(s16, s64, atol=1e-3)
            assert not (differ & ok).any()

    def test_argmax_ties_are_averaged(self, tmp_path):
        tie = np.zeros((1, 41))
        tie[0, [4, 36]] = 0.5                                           # -0.8 and +0.8
        s, c = _sql(_write_grid(tmp_path / "t.parquet", np.vstack([tie, _fb(0.45, 0.1, 0.45)])),
                    "argmax")
        assert s[0] == pytest.approx(0.0) and c[0] == pytest.approx(1.0)
        assert s[1] == pytest.approx(0.0, abs=1e-12)                    # pos = neg tie -> 0

    def test_null_row_gives_null_sent_and_conf(self, tmp_path):
        G = np.vstack([np.full((1, 41), np.nan), np.eye(41)[[30]]])
        path = _write_grid(tmp_path / "n.parquet", G)
        for rule in RULES:
            s, c = _sql(path, rule)
            assert np.isnan(s[0]) and np.isnan(c[0])
            assert s[1] == pytest.approx(GRID[30]) or rule == "median"

    def test_unknown_rule_refused(self):
        with pytest.raises(ValueError, match="rule"):
            grid_select_sql("t", "mode")


# ---------------------------------------------------------------------------
# Sentimeter behaviour (fixed backend)
# ---------------------------------------------------------------------------

def test_finbert_canonical_order_and_grid():
    rows = [[0.7, 0.1, 0.2]]                                 # pos, neg, neu
    s = FinbertSentimeter(FixedBackend(FB_LABELS, rows))
    np.testing.assert_allclose(s.canonical("x"), [0.1, 0.2, 0.7])
    np.testing.assert_allclose(s.grid("x"), _fb(0.1, 0.2, 0.7)[0], atol=1e-15)


def test_shapes_scalar_for_one_text_array_for_many():
    rows = np.full((3, 41), 1 / 41)
    s = RavenbertSentimeter(FixedBackend(RB_LABELS, rows))
    assert s.grid("one").shape == (41,) and s.grid(["a", "b", "c"]).shape == (3, 41)
    assert s.canonical("one").shape == (41,) and s.canonical(["a", "b"]).shape == (2, 41)


def test_empty_cleaned_text_is_an_all_nan_row():
    s = RavenbertSentimeter(FixedBackend(RB_LABELS, np.eye(41)[[5, 6]]), clean=False)
    G = s.grid_batch(["stocks rally", "   ", "fed"])
    assert np.isnan(G[1]).all() and not np.isnan(G[[0, 2]]).any()
    np.testing.assert_allclose(G[[0, 2]], np.eye(41)[[5, 6]])            # rows stay aligned


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
    s = FinbertSentimeter(FixedBackend(FB_LABELS, [[0.2, 0.3, 0.5]]))
    assert s.columns() == list(GRID_COLUMNS)
    card = s.model_card()
    assert card.model_id == "finbert-sentiment" and card.version == "2.1"
    assert card.backend == "fixed"
    assert card.serving["sentimeter"]["band_edges"] == pytest.approx(list(FB_EDGES))
    assert card.serving["checks"]["canonical_order"] == [1, 2, 0]


# ---------------------------------------------------------------------------
# Real classifiers (tiny random weights) through the local backend
# ---------------------------------------------------------------------------

def test_ravenbert_grid_mean_equals_sentiment_model_predict(ravenbert_like_dir):
    from ravenbert.sentiment.model import SentimentModel

    ref = SentimentModel.from_path(ravenbert_like_dir, device="cpu", precision="fp32")
    want = ref.predict(TEXTS, batch_size=64, optimized_inference=False)
    s = load_sentimeter("ravenbert", "local", "float32", model_path=str(ravenbert_like_dir),
                        device="cpu", batch_size=2)
    got = reference(s.grid(TEXTS), "mean")[0]
    empty = np.array([not t.strip() for t in TEXTS])
    assert np.isnan(got[empty]).all()                        # no model output for ""
    np.testing.assert_allclose(got[~empty], np.asarray(want)[~empty], atol=1e-6)


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
    empty = np.array([not t.strip() for t in cleaned])
    G = s.grid(TEXTS)
    assert np.isnan(G[empty]).all()
    np.testing.assert_allclose(G[~empty], to_grid(want[~empty], FB_EDGES), atol=1e-6)


def test_load_sentimeter_rejects_embedx_and_unknown_model():
    with pytest.raises(ValueError, match="embeddings only"):
        load_sentimeter("finbert", "embedx")
    with pytest.raises(ValueError, match="model must be one of"):
        load_sentimeter("vader")
