"""The compiled select+aggregate kernel must reproduce the numpy path within tolerance.

Skipped (not failed) when the kernel is not built.
"""
from __future__ import annotations

import numpy as np
import pytest

from narrative_scoring._kernels import HAVE_SELECT, select_aggregate_rowwise
from narrative_scoring.aggregation import DayAccumulator, headline_narrative_scores
from narrative_scoring.config import AggRule
from narrative_scoring.selection import select

pytestmark = pytest.mark.skipif(not HAVE_SELECT, reason="select_aggregate kernel not built")


def numpy_path(S, tau, k, jump_min, p2n, n_narr, rule, want_prim):
    sel = select(S, tau, k, jump_cut=jump_min >= 0, jump_min_candidates=max(jump_min, 1))
    narr = DayAccumulator(n_narr)
    _, nid, nval = headline_narrative_scores(sel.rows, sel.cols, sel.vals, p2n, n_narr, rule)
    narr.add_values(nid, nval)
    prim = None
    if want_prim:
        prim = DayAccumulator(S.shape[1])
        prim.add_values(sel.cols.astype(np.int64), sel.vals)
    return sel, narr, prim


def check(S, tau, k, jump_min, p2n, n_narr, rule, want_prim, threads):
    tau32 = float(np.float32(tau))
    got = select_aggregate_rowwise(S, tau32, k, jump_min, p2n, n_narr, rule is AggRule.MEDIAN,
                                   want_prim, threads)
    sel, narr, prim = numpy_path(S, tau32, k, jump_min, p2n, n_narr, rule, want_prim)
    np.testing.assert_array_equal(got["narr_count"], narr.count)
    np.testing.assert_allclose(got["narr_total"], narr.total, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["narr_sumsq"], narr.sumsq, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["narr_peak"], narr.peak, rtol=1e-12)
    assert got["n_unassigned"] == sel.n_unassigned
    assert got["n_candidates"] == int(sel.n_candidates.sum())
    assert got["n_f0_survivors"] == int(sel.n_f0_survivors.sum())
    assert got["n_retained_pre_jump"] == sel.n_retained_pre_jump
    assert got["n_retained"] == int(sel.n_retained.sum())
    assert got["n_jump_trimmed"] == sel.n_jump_trimmed
    assert got["jump_gap_sum"] == pytest.approx(sel.jump_gap_sum, rel=1e-12, abs=1e-12)
    assert np.isinf(got["trim_threshold"]).all()          # n_trim not requested
    if want_prim:
        np.testing.assert_array_equal(got["prim_count"], prim.count)
        np.testing.assert_allclose(got["prim_total"], prim.total, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got["prim_sumsq"], prim.sumsq, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(got["prim_peak"], prim.peak)
    else:
        assert got["prim_count"] is None


@pytest.mark.parametrize("rule", [AggRule.MEAN, AggRule.MEDIAN])
@pytest.mark.parametrize("jump_min", [-1, 2, 10])
@pytest.mark.parametrize("n,p,n_narr,k,density", [
    (600, 60, 20, 3, 0.05),
    (400, 120, 30, 5, 0.02),
    (256, 1508, 419, 15, 0.005),      # real shape
])
def test_matches_numpy_path(n, p, n_narr, k, density, jump_min, rule):
    rng = np.random.default_rng(n * 7 + p)
    S = np.ascontiguousarray(rng.normal(0.2, 0.08, size=(n, p)).astype(np.float32))
    S[:, :5] = S[:, 5:10]                       # exact ties: '>=' keeps all of them
    tau = float(np.quantile(S, 1 - density))
    p2n = np.sort(rng.integers(0, n_narr, p)).astype(np.int32)
    check(S, tau, k, jump_min, p2n, n_narr, rule, True, 8)
    check(S, tau, k, jump_min, p2n, n_narr, rule, False, 3)


def test_thread_count_invariance():
    rng = np.random.default_rng(2)
    S = np.ascontiguousarray(rng.normal(0.2, 0.08, size=(2048, 1508)).astype(np.float32))
    p2n = np.sort(rng.integers(0, 419, 1508)).astype(np.int32)
    base = select_aggregate_rowwise(S, 0.3, 15, 5, p2n, 419, False, True, 1)
    for nt in (2, 4, 8):
        got = select_aggregate_rowwise(S, 0.3, 15, 5, p2n, 419, False, True, nt)
        for key in ("narr_count", "narr_peak", "prim_count", "prim_peak"):
            np.testing.assert_array_equal(base[key], got[key])
        for key in ("narr_total", "narr_sumsq", "prim_total", "prim_sumsq"):
            np.testing.assert_allclose(base[key], got[key], rtol=1e-12)
        for key in ("n_unassigned", "n_candidates", "n_retained", "n_jump_trimmed"):
            assert base[key] == got[key]


def test_q_before_tau_in_kernel():
    # k=2 candidates {0.9, 0.5}; tau=0.6 removes 0.5; 0.55 is above tau but not a candidate
    S = np.array([[0.9, 0.1, 0.5, 0.3]], dtype=np.float32)
    p2n = np.array([0, 1, 2, 3], dtype=np.int32)
    got = select_aggregate_rowwise(S, 0.6, 2, -1, p2n, 4, False, True, 1)
    assert got["prim_count"].tolist() == [1, 0, 0, 0]
    assert got["n_candidates"] == 2 and got["n_retained"] == 1
    S2 = np.array([[0.9, 0.7, 0.55, 0.3]], dtype=np.float32)
    got2 = select_aggregate_rowwise(S2, 0.5, 2, -1, p2n, 4, False, True, 1)
    assert got2["prim_count"].tolist() == [1, 1, 0, 0]


def test_all_below_tau_is_unassigned_not_zero():
    S = np.full((5, 300), 0.01, dtype=np.float32)
    p2n = np.sort(np.random.default_rng(1).integers(0, 40, 300)).astype(np.int32)
    got = select_aggregate_rowwise(S, 0.3, 3, -1, p2n, 40, False, True, 4)
    assert got["n_unassigned"] == 5 and got["n_retained"] == 0
    assert got["narr_count"].sum() == 0 and np.isinf(got["narr_peak"]).all()


def test_trim_threshold_matches_numpy():
    from narrative_scoring.f0 import n_trim, trim_threshold

    rng = np.random.default_rng(3)
    S = np.ascontiguousarray(rng.normal(0.2, 0.08, size=(300, 1508)).astype(np.float32))
    p2n = np.sort(rng.integers(0, 419, 1508)).astype(np.int32)
    got = select_aggregate_rowwise(S, 0.3, 16, -1, p2n, 419, False, False, 4, n_trim(1508, 0.1))
    np.testing.assert_array_equal(got["trim_threshold"], trim_threshold(S, 0.1))


def test_invalid_k_raises():
    S = np.zeros((2, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        select_aggregate_rowwise(S, 0.0, 6, -1, np.zeros(5, np.int32), 1, False, False, 1)
