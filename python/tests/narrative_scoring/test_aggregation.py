"""Headline -> narrative pooling and day accumulators (population statistics)."""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from narrative_scoring.aggregation import DayAccumulator, headline_narrative_scores
from narrative_scoring.config import AggRule
from narrative_scoring.validation import reference_day, reference_headline_narrative


def _dense_from_triplets(rows, cols, vals, shape):
    A = np.full(shape, np.nan, dtype=np.float32)
    A[rows, cols] = vals
    return A


class TestHeadlineNarrative:
    p2n = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)

    def test_mean_skips_absent_primitives(self):
        rows = np.array([0, 0, 0], dtype=np.int32)
        cols = np.array([0, 2, 3], dtype=np.int32)          # narrative 0: {0}; narrative 1: {2, 3}
        vals = np.array([0.5, 0.4, 0.8], dtype=np.float32)
        r, nid, score = headline_narrative_scores(rows, cols, vals, self.p2n, 3, AggRule.MEAN)
        assert nid.tolist() == [0, 1]
        np.testing.assert_allclose(score, [0.5, 0.6], atol=1e-7)

    def test_all_absent_narrative_is_absent_not_zero(self):
        rows = np.array([0], dtype=np.int32)
        cols = np.array([5], dtype=np.int32)                 # only narrative 2
        vals = np.array([0.7], dtype=np.float32)
        r, nid, score = headline_narrative_scores(rows, cols, vals, self.p2n, 3, AggRule.MEAN)
        assert nid.tolist() == [2]                          # narratives 0 and 1 do not appear

    def test_median_even_and_odd(self):
        rows = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        cols = np.array([2, 3, 4, 2, 3], dtype=np.int32)
        vals = np.array([0.9, 0.1, 0.5, 0.2, 0.8], dtype=np.float32)
        _, nid, score = headline_narrative_scores(rows, cols, vals, self.p2n, 3, AggRule.MEDIAN)
        np.testing.assert_allclose(score, [0.5, 0.5], atol=1e-7)

    def test_empty_input(self):
        e = np.empty(0, dtype=np.int32)
        r, nid, score = headline_narrative_scores(e, e, np.empty(0, np.float32), self.p2n, 3,
                                                  AggRule.MEAN)
        assert r.size == nid.size == score.size == 0

    @pytest.mark.parametrize("rule", [AggRule.MEAN, AggRule.MEDIAN])
    def test_matches_dense_reference(self, rule):
        rng = np.random.default_rng(7)
        n, p, n_narr = 40, 60, 12
        p2n = np.sort(rng.integers(0, n_narr, p)).astype(np.int32)
        A = np.where(rng.random((n, p)) < 0.1, rng.random((n, p)), np.nan).astype(np.float32)
        rows, cols = np.nonzero(~np.isnan(A))
        got = np.full((n, n_narr), np.nan)
        r, nid, s = headline_narrative_scores(rows.astype(np.int32), cols.astype(np.int32),
                                              A[rows, cols], p2n, n_narr, rule)
        got[r, nid] = s
        ref = reference_headline_narrative(A, p2n, n_narr, rule)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-7, equal_nan=True)


class TestDayAccumulator:
    def test_population_statistics(self):
        acc = DayAccumulator(3)
        acc.add_values(np.array([0, 0, 0, 1]), np.array([0.2, 0.4, 0.6, 0.5]))
        acc.n_headlines = 4
        intensity, std, total, peak = acc.stats()
        np.testing.assert_allclose(intensity[:2], [0.4, 0.5])
        np.testing.assert_allclose(total[:2], [1.2, 0.5])
        np.testing.assert_allclose(std[0], np.std([0.2, 0.4, 0.6], ddof=0))   # ddof=0
        assert std[0] != pytest.approx(np.std([0.2, 0.4, 0.6], ddof=1))
        assert std[1] == 0.0                                  # one observation -> 0, not NaN
        assert np.isnan(intensity[2]) and np.isnan(std[2]) and np.isnan(peak[2])
        np.testing.assert_allclose(peak[:2], [0.6, 0.5])

    def test_total_equals_support_times_intensity(self):
        rng = np.random.default_rng(2)
        acc = DayAccumulator(20)
        acc.add_values(rng.integers(0, 20, 500), rng.random(500))
        intensity, _, total, _ = acc.stats()
        has = acc.count > 0
        np.testing.assert_allclose(total[has], acc.count[has] * intensity[has], rtol=1e-12)

    def test_frame_nulls_where_no_support(self):
        acc = DayAccumulator(2)
        acc.add_values(np.array([0]), np.array([0.3]))
        acc.n_headlines = 5
        nodes = pl.DataFrame({"narrative_key": ["a", "b"]})
        f = acc.frame(nodes, date(2008, 9, 15))
        assert f["SUPPORT"].to_list() == [1, 0]
        assert f["INTENSITY"].to_list()[1] is None
        assert f["TOTAL_SCORE"].to_list()[1] is None
        assert f["PEAK"].to_list()[1] is None
        assert f["STD_SCORE"].to_list() == pytest.approx([0.0, None])
        assert f["N_HEADLINES"].to_list() == [5, 5]
        assert f["DATE"].to_list() == [date(2008, 9, 15)] * 2

    def test_chunk_order_invariance(self):
        rng = np.random.default_rng(5)
        ids = rng.integers(0, 15, 3000)
        vals = rng.random(3000)
        one = DayAccumulator(15)
        one.add_values(ids, vals)
        chunks = [(ids[i:i + 700], vals[i:i + 700]) for i in range(0, 3000, 700)]
        for perm in ([0, 1, 2, 3, 4], [4, 2, 0, 3, 1], [1, 3, 4, 0, 2]):
            acc = DayAccumulator(15)
            for k in perm:
                acc.add_values(*chunks[k])
            np.testing.assert_array_equal(acc.count, one.count)
            np.testing.assert_array_equal(acc.peak, one.peak)
            np.testing.assert_allclose(acc.total, one.total, rtol=1e-12)
            np.testing.assert_allclose(acc.sumsq, one.sumsq, rtol=1e-12)

    def test_matches_reference_day(self):
        rng = np.random.default_rng(6)
        N = np.where(rng.random((80, 9)) < 0.3, rng.random((80, 9)), np.nan)
        acc = DayAccumulator(9)
        r, c = np.nonzero(~np.isnan(N))
        acc.add_values(c, N[r, c])
        ref = reference_day(N)
        intensity, std, total, peak = acc.stats()
        np.testing.assert_array_equal(acc.count, ref["SUPPORT"])
        np.testing.assert_allclose(total, ref["TOTAL_SCORE"], rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(intensity, ref["INTENSITY"], rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(std, ref["STD_SCORE"], atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(peak, ref["PEAK"], rtol=0, equal_nan=True)

    def test_empty_accumulator_is_all_null(self):
        acc = DayAccumulator(4)
        intensity, std, total, peak = acc.stats()
        assert np.isnan(intensity).all() and np.isnan(std).all()
        assert np.isnan(total).all() and np.isnan(peak).all()
