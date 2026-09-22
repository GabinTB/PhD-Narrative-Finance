"""Warmup quantities: N_eff, effective rank, top-eigenvalue share, full spectrum.

Pure computation, no artifact of its own. The monthly tau_asof job calls
``compute_warmup`` at every cutoff with the most recent mu_asof row available
as of that cutoff, and pins the result on the tau row (N_EFF, EFFECTIVE_RANK,
LAMBDA1_SHARE, MU_DATE, MU_ARTIFACT_ID), so later tau drift decomposes into
null-pool drift versus N_eff (geometry) drift.

N_eff is the participation ratio (sum l)^2 / sum l^2 of the eigen-spectrum of
the Gram matrix of the corrected representative primitive vectors -- the
construction the F0 calibration has always used
(``primitives.representative_matrix`` -> ``f0.gram_spectrum``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from narrative_scoring.calibration import MuRecord
from narrative_scoring.config import ScoringConfig
from narrative_scoring.corrections import Correction
from narrative_scoring.f0 import Spectrum, gram_spectrum
from narrative_scoring.primitives import PrimitiveTable, embeddings_digest, representative_matrix


@dataclass(frozen=True)
class WarmupRecord:
    n_eff: float
    effective_rank: float
    lambda1_share: float
    n_eigenvalues: int
    mode: str
    paraphrase_style: str
    paraphrase_pooling: str
    include_master: bool
    warmup_digest: str
    taxonomy_name: str
    taxonomy_sha1: str
    paraphrase_sha1: str
    embeddings_digest: str
    n_primitives: int
    mu_date: str | None
    mu_asof_id: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def compute_warmup(
    table: PrimitiveTable, P: np.ndarray, config: ScoringConfig, mu: MuRecord | None, *,
    mu_asof_id: str | None = None,
) -> tuple[WarmupRecord, Spectrum]:
    """Spectrum of the corrected primitive Gram matrix under ``mu`` (None only for RAW)."""
    if config.mode is not Correction.RAW and mu is None:
        raise ValueError(f"mode {config.mode.value} needs a mu_asof row for the warmup")
    mu_v = mu.mu if mu is not None else None
    mu_hat_v = mu.mu_hat if mu is not None else None
    D = representative_matrix(P, table, config.mode, mu_v, mu_hat_v)
    spec = gram_spectrum(D)
    rec = WarmupRecord(
        n_eff=spec.n_eff, effective_rank=spec.effective_rank,
        lambda1_share=spec.lambda1_share, n_eigenvalues=int(spec.eigenvalues.size),
        mode=config.mode.value, paraphrase_style=config.paraphrase_style.value,
        paraphrase_pooling=config.paraphrase_pooling.value,
        include_master=config.include_master, warmup_digest=config.warmup_digest(),
        taxonomy_name=table.name, taxonomy_sha1=table.taxonomy_sha1,
        paraphrase_sha1=table.paraphrase_sha1, embeddings_digest=embeddings_digest(P),
        n_primitives=table.n_primitives,
        mu_date=mu.date.isoformat() if mu is not None else None,
        mu_asof_id=mu_asof_id,
    )
    return rec, spec
