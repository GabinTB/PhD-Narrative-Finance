"""GpuScorer must reproduce the CPU steps 1-2 to floating-point tolerance.

Runs on the CPU device (no GPU on the test machine). On a real GPU the only
extra difference is cuBLAS's reduction order in the matmul; the tolerance
here is what the CPU-torch vs CPU-numpy comparison actually needs, and the
hybrid runner reports the measured GPU figure at run time.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from narrative_scoring import spec_pipeline as sp  # noqa: E402
from narrative_scoring.corrections import Correction  # noqa: E402
from narrative_scoring.gpu_scoring import GpuScorer, compare_with_cpu  # noqa: E402


class _Table:
    def __init__(self, n_texts, n_prim):
        self.n_texts, self.n_primitives = n_texts, n_prim


def _unit(rng, shape):
    x = rng.normal(size=shape).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.mark.parametrize("mode", [Correction.RAW, Correction.R1, Correction.R2])
@pytest.mark.parametrize("pooling", [sp.PoolRule.MAX, sp.PoolRule.MEAN])
def test_scores_match_cpu(mode, pooling):
    rng = np.random.default_rng(5)
    n, n_texts, n_prim = 512, 6, 200
    P = _unit(rng, (n_texts * n_prim, EMBEDDING_DIM := 384))
    mu = rng.normal(size=EMBEDDING_DIM).astype(np.float32) * 0.3
    mu_hat = (mu / np.linalg.norm(mu)).astype(np.float32)
    table = _Table(n_texts, n_prim)
    cfg = sp.ScoringConfig(mode=mode, paraphrase_pooling=pooling)
    P_scoring = sp.build_scoring_matrix(P, table, cfg, mu, mu_hat)

    scorer = GpuScorer(P_scoring, n_texts, n_prim, pooling, mode, mu, mu_hat, device="cpu")
    X = rng.normal(size=(n, EMBEDDING_DIM)).astype(np.float16)     # as stored on disk

    d = compare_with_cpu(scorer, X)
    assert d["max_abs"] < 5e-6, d


def test_degenerate_rows_are_zeroed_like_cpu():
    """A query that is (almost) exactly the mean direction collapses under R2 on both paths."""
    rng = np.random.default_rng(9)
    n_texts, n_prim = 6, 50
    P = _unit(rng, (n_texts * n_prim, 384))
    mu_hat = _unit(rng, (1, 384))[0]
    table = _Table(n_texts, n_prim)
    cfg = sp.ScoringConfig(mode=Correction.R2, paraphrase_pooling=sp.PoolRule.MAX)
    P_scoring = sp.build_scoring_matrix(P, table, cfg, None, mu_hat)
    scorer = GpuScorer(P_scoring, n_texts, n_prim, sp.PoolRule.MAX, Correction.R2, None, mu_hat, "cpu")
    X = np.stack([mu_hat, mu_hat * 3.0, _unit(rng, (1, 384))[0]]).astype(np.float32)
    S = scorer.score(X)
    assert np.all(S[:2] == 0.0), "mean-direction rows must be zeroed, never NaN"
    assert np.isfinite(S).all()
