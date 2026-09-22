"""Steps 1-2 of the spec on a GPU (torch): normalise + mode, matmul, paraphrase pooling.

Scope is deliberately narrow. Steps 3-5 (the F0 floor, the percentile cut and
the per-headline narrative aggregation) stay on the CPU in the verified fused
kernel, on the *same machine* as the GPU, so the compiled path and this one
differ only in how ``S`` is produced. That keeps every equivalence test for
the gate and the aggregation valid for the GPU worker unchanged, and confines
the CPU/GPU numerical difference to the matmul's reduction order (cuBLAS vs
OpenBLAS), which ``compare_with_cpu`` quantifies.

The same code runs on the CPU device, which is how it is tested here: this
machine has no GPU, the second one does.

Nothing in this module reads or prints where that machine is; the Ray layer
takes the host from the environment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from narrative_scoring.corrections import Correction
from narrative_scoring.schema import EMBEDDING_DIM
from narrative_scoring.spec_pipeline import PoolRule, _DEGENERATE_NORM_TOL


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is in the nlp group
        raise RuntimeError("gpu_scoring needs torch (uv sync --group nlp)") from exc
    return torch


def pick_device(prefer: str | None = None) -> str:
    """'cuda' when available (or a specific index like 'cuda:1'), else 'cpu'."""
    torch = _torch()
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GpuScorer:
    """Holds the mode-applied primitive-text matrix on the device.

    ``P_scoring`` is exactly what ``spec_pipeline.build_scoring_matrix`` returns
    (text-major; mean-vectors for MEAN pooling), so the two paths score against
    identical targets. ``mu``/``mu_hat`` are applied to the queries here with
    the same arithmetic as ``spec_pipeline.apply_mode``.
    """

    P_scoring: np.ndarray
    n_texts: int
    n_primitives: int
    pooling: PoolRule
    mode: Correction
    mu: np.ndarray | None = None
    mu_hat: np.ndarray | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        torch = _torch()
        self._P = torch.as_tensor(np.ascontiguousarray(self.P_scoring, dtype=np.float32),
                                  device=self.device)
        self._PT = self._P.T.contiguous()
        self._u = None
        self._mu = None
        if self.mode is Correction.R2:
            if self.mu_hat is None:
                raise ValueError("R2 requires mu_hat")
            self._u = torch.as_tensor(np.asarray(self.mu_hat, dtype=np.float32), device=self.device)
        elif self.mode is Correction.R1:
            if self.mu is None:
                raise ValueError("R1 requires mu")
            self._mu = torch.as_tensor(np.asarray(self.mu, dtype=np.float32), device=self.device)
        if self.pooling is PoolRule.MAX and self._P.shape[0] != self.n_texts * self.n_primitives:
            raise ValueError("MAX pooling expects n_texts * n_primitives text vectors")
        if self.pooling is PoolRule.MEAN and self._P.shape[0] != self.n_primitives:
            raise ValueError("MEAN pooling expects one mean-vector per primitive")

    # -- step 1 ----------------------------------------------------------------

    def _apply_mode(self, X):
        torch = _torch()
        X = X.float()
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if self.mode is Correction.RAW:
            return X
        if self.mode is Correction.R1:
            X = X - self._mu[None, :]
        else:
            X = X - (X @ self._u)[:, None] * self._u[None, :]
        norms = X.norm(dim=1, keepdim=True)
        degenerate = norms < _DEGENERATE_NORM_TOL
        X = X / norms.clamp_min(1e-12)
        if bool(degenerate.any()):
            X = torch.where(degenerate, torch.zeros_like(X), X)
        return X

    # -- step 2 ----------------------------------------------------------------

    def score(self, X_block: np.ndarray) -> np.ndarray:
        """(n, 384) fp16/fp32 embeddings -> (n, n_primitives) float32 pooled scores, on host."""
        torch = _torch()
        with torch.no_grad():
            X = torch.as_tensor(np.ascontiguousarray(X_block), device=self.device)
            H = self._apply_mode(X)
            R = H @ self._PT                                   # (n, n_texts*n_prim) or (n, n_prim)
            if self.pooling is PoolRule.MAX:
                # text-major: slab j is texts j*n_prim:(j+1)*n_prim, so a view
                # (n, n_texts, n_prim) and amax over dim 1 is the slab max.
                S = R.view(R.shape[0], self.n_texts, self.n_primitives).amax(dim=1)
            else:
                S = R                                          # mean-vector identity, exact
            return S.contiguous().cpu().numpy()


def compare_with_cpu(scorer: GpuScorer, X_block: np.ndarray) -> dict[str, float]:
    """Quantify the matmul-order difference against the CPU implementation.

    Returned as max abs / max rel difference on the pooled scores. The gate is
    a threshold on these values, so any headline whose score sits within this
    tolerance of tau can flip between machines; the hybrid assigns whole months
    to one machine so a panel is at least internally consistent.
    """
    from narrative_scoring import spec_pipeline as sp

    S_gpu = scorer.score(X_block)
    H = sp.apply_mode(X_block, scorer.mode, scorer.mu, scorer.mu_hat)

    class _T:  # duck-type the two fields primitive_scores reads
        n_texts = scorer.n_texts
        n_primitives = scorer.n_primitives

    S_cpu = sp.primitive_scores(H, scorer.P_scoring, _T, scorer.pooling)
    d = np.abs(S_gpu - S_cpu)
    return {
        "max_abs": float(d.max()),
        "max_rel": float((d / np.maximum(np.abs(S_cpu), 1e-6)).max()),
        "mean_abs": float(d.mean()),
    }
