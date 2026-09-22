"""Point-in-time calibration inputs: mu_asof rows and the F0 floor tau.

The pipeline never reads a calibration value directly. It asks a
``CalibrationProvider`` for the mu / mu_hat and the tau that were available
AS OF the day being scored, and the provider enforces the point-in-time
rule:

* ``mu_for(day)`` returns the mu_asof row for ``day`` itself (exact date)
  or, when that day has no row, the latest row strictly before it. A row
  dated after ``day`` is never used. The mu_asof artifact already embeds its
  own delay (its row for t is built from headlines strictly before
  t - delay), so "row dated <= day" is the whole guarantee needed here;
  how mu_asof is computed is not this module's business.
* ``tau_for(day)`` returns a ``TauRecord`` whose calibration window ended
  strictly before ``day``; otherwise ``LookaheadError``.

The production provider is ``tau_asof.TauSeriesProvider`` (monthly tau
series + mu_asof rows). There is no frozen or bootstrap tau: a day for which
no tau row is old enough is not scorable, full stop.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl


class LookaheadError(ValueError):
    """A calibration value dated at or after the day it would be used for."""


@dataclass(frozen=True)
class MuRecord:
    """The mu_asof row actually used for a scoring day."""

    date: date               # the row's DATE (<= the scoring day)
    mu: np.ndarray           # raw pooled mean (not unit norm)
    mu_hat: np.ndarray       # unit direction

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.mu.astype(np.float64)))


@dataclass(frozen=True)
class TauRecord:
    """A calibrated F0 floor and everything needed to reproduce it."""

    tau: float
    gaussian_tau: float      # diagnostic cross-check only, never the gate
    n_eff: float
    alpha: float
    trim_frac: float
    n_draws: int
    seed: int
    calibrated_from: date
    calibrated_through: date  # last day whose headlines fed the null pool
    mode: str
    paraphrase_pooling: str
    mu_date: date | None      # mu_asof row used to correct the calibration sample
    source_id: str = ""
    month_end: date | None = None   # set when the record comes from the tau_asof series

    def digest(self) -> str:
        d = asdict(self)
        for k in ("calibrated_from", "calibrated_through", "mu_date", "month_end"):
            d[k] = d[k].isoformat() if d[k] is not None else None
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("calibrated_from", "calibrated_through", "mu_date", "month_end"):
            d[k] = d[k].isoformat() if d[k] is not None else None
        d["id"] = self.digest()
        return d


class CalibrationProvider(Protocol):
    mu_asof_id: str | None
    mu_policy: str
    tau_source_id: str
    tau_policy: str

    def mu_for(self, day: date) -> MuRecord | None: ...

    def tau_for(self, day: date) -> TauRecord: ...


# ---------------------------------------------------------------------------
# mu_asof lookup
# ---------------------------------------------------------------------------

def load_mu_asof(path: Path) -> pl.DataFrame:
    """Read a mu_asof parquet (DATE, MU, MU_HAT, N), sorted by DATE."""
    return pl.read_parquet(path).sort("DATE")


def resolve_mu_asof(mu_df: pl.DataFrame, day: date) -> MuRecord:
    """Exact-date row for ``day``, else the latest row before it. Never a later row."""
    candidates = mu_df.filter(pl.col("DATE") <= day)
    if candidates.is_empty():
        raise LookaheadError(f"mu_asof has no row at or before {day}")
    row = candidates.sort("DATE").row(-1, named=True)
    return MuRecord(
        date=row["DATE"],
        mu=np.asarray(row["MU"], dtype=np.float32),
        mu_hat=np.asarray(row["MU_HAT"], dtype=np.float32),
    )
