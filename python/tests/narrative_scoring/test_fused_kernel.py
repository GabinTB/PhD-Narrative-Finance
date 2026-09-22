"""The compiled kernel must be indistinguishable from the numpy reference.

The kernel replaces three full-matrix numpy passes with one. Any divergence is a
silent change to the stored panel, so equivalence is asserted here rather than
assumed -- and two real bugs were caught this way before the kernel shipped:

  * a Hoare quickselect that was off by one on some duplicate-free inputs
    (replaced by std::nth_element);
  * -ffast-math reassociating ceil(n_surv * (100 - q) / 100) into
    n_surv * 0.05, which is not exactly representable, so 140 * 0.05 ceils to
    8 instead of 7 -- shifting the percentile cut by one position on every row
    whose survivor count divides exactly.

The tests are skipped, not failed, when the kernel is not built.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.skip(
    "Ray/GPU track suspended; canonical scorer is narrative_scoring.pipeline.score_dates",
    allow_module_level=True,
)
from narrative_scoring._kernels import HAVE_FUSED, gate_aggregate_rowwise  # noqa: E402

pytestmark = pytest.mark.skipif(not HAVE_FUSED, reason="fused kernel not built")


def dense_reference(S, tau, q, p2n, n_narr, use_median):
    """Steps 3-5 written the most naive way the spec allows."""
    A = np.where(S >= tau, S, np.nan)
    with np.errstate(invalid="ignore"):
        thr = np.nanpercentile(A, q, axis=1, keepdims=True)
    thr = np.where(np.isnan(thr), np.inf, thr)
    A = np.where(A >= thr, A, np.nan)
    cnt = np.zeros(n_narr, np.int64)
    tot = np.zeros(n_narr, np.float64)
    pk = np.full(n_narr, -np.inf)
    unassigned = 0
    for row in A:
        touched = False
        for j in range(n_narr):
            blk = row[p2n == j]
            blk = blk[np.isfinite(blk)]
            if blk.size:
                v = np.median(blk) if use_median else blk.mean()
                cnt[j] += 1
                tot[j] += v
                pk[j] = max(pk[j], v)
                touched = True
        if not touched:
            unassigned += 1
    return cnt, tot, pk, unassigned


def _check(S, tau, qs, p2n, n_narr, use_median, n_threads):
    got = gate_aggregate_rowwise(S, tau, qs, p2n, n_narr, use_median, n_threads)
    for (c, t, p, u, _kept, _fl), q in zip(got, qs):
        rc, rt, rp, ru = dense_reference(S, tau, q, p2n, n_narr, use_median)
        np.testing.assert_array_equal(c, rc)
        np.testing.assert_allclose(t, rt, rtol=0, atol=2e-4)
        m = np.isfinite(rp)
        np.testing.assert_allclose(p[m], rp[m], rtol=0, atol=2e-6)
        assert u == ru


@pytest.mark.parametrize("use_median", [False, True])
@pytest.mark.parametrize("n,n_prim,n_narr,density", [
    (600, 60, 20, 0.05),
    (400, 120, 30, 0.02),
    (256, 1508, 419, 0.018),   # real shape and real survivor density
])
def test_matches_dense_reference(n, n_prim, n_narr, density, use_median):
    rng = np.random.default_rng(n * 7 + n_prim)
    S = np.ascontiguousarray(rng.normal(0.2, 0.08, size=(n, n_prim)).astype(np.float32))
    S[:, :5] = S[:, 5:10]                       # force exact ties: `>=` keeps all of them
    tau = float(np.quantile(S, 1 - density))
    p2n = np.sort(rng.integers(0, n_narr, n_prim)).astype(np.int32)
    _check(S, tau, np.array([95.0, 99.0]), p2n, n_narr, use_median, 8)


def test_exact_integer_k_is_not_shifted():
    """n_surv * (100-q)/100 landing on an exact integer must ceil to itself.

    This is the -ffast-math regression: survivor counts divisible by 20 at
    q=95 were cut at k+1.
    """
    n_prim, n_narr = 1508, 419
    rng = np.random.default_rng(140)
    p2n = np.sort(rng.integers(0, n_narr, n_prim)).astype(np.int32)
    tau = 0.2705
    row = np.full(n_prim, 0.1, dtype=np.float32)
    row[:140] = np.linspace(0.5, 0.3, 140, dtype=np.float32)   # exactly 140 survivors
    S = np.ascontiguousarray(row[None, :])
    (_c, _t, _p, _u, kept, _fl), = gate_aggregate_rowwise(
        S, tau, np.array([95.0]), p2n, n_narr, False, 1)
    assert kept == 7, f"k=ceil(140*5/100)=7, kernel kept {kept}"


def test_floored_headline_never_resurrected():
    rng = np.random.default_rng(1)
    S = np.full((5, 300), 0.01, dtype=np.float32)
    p2n = np.sort(rng.integers(0, 40, 300)).astype(np.int32)
    for q in (95.0, 99.0):
        (c, _t, _p, u, kept, fl), = gate_aggregate_rowwise(S, 0.3, np.array([q]), p2n, 40, False, 4)
        assert kept == 0 and u == 5 and fl == S.size and c.sum() == 0


def test_thread_count_invariance():
    """Per-thread buffers must not race: any thread count gives the same answer."""
    rng = np.random.default_rng(2)
    S = np.ascontiguousarray(rng.normal(0.2, 0.08, size=(2048, 1508)).astype(np.float32))
    tau = float(np.quantile(S, 0.982))
    p2n = np.sort(rng.integers(0, 419, 1508)).astype(np.int32)
    qs = np.array([95.0, 99.0])
    base = gate_aggregate_rowwise(S, tau, qs, p2n, 419, False, 1)
    for nt in (2, 4, 8):
        got = gate_aggregate_rowwise(S, tau, qs, p2n, 419, False, nt)
        for (bc, bt, bp, bu, bk, bf), (c, t, p, u, k, f) in zip(base, got):
            np.testing.assert_array_equal(bc, c)
            np.testing.assert_allclose(bt, t, rtol=0, atol=1e-9)
            np.testing.assert_array_equal(bp, p)
            assert (bu, bk, bf) == (u, k, f)
