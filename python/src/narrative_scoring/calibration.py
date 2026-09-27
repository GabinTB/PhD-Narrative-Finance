"""Point-in-time calibration inputs: mu_asof rows and the F0 floor tau.

The pipeline never reads a calibration value directly. It asks a
``CalibrationProvider`` for the mu / mu_hat and the tau that were available
AS OF the day being scored, and the provider enforces the point-in-time
rule:

* ``mu_for(day)`` returns the reference-vector row (``mu_asof`` series) for
  ``day`` itself (exact date) or, when that day has no row, the latest row
  strictly before it. A row dated after ``day`` is never used. The series
  already embeds its own delay (its row for t pools headlines up to
  t - delay), so "row dated <= day" is the whole guarantee needed here; how
  the reference is computed is ``nlp.reference_vector``'s business.
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
from typing import Protocol

from nlp.corrections import LookaheadError
from nlp.reference_vector import ReferenceValue

__all__ = ["CalibrationProvider", "LookaheadError", "ReferenceValue", "TauRecord"]


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

    def mu_for(self, day: date) -> ReferenceValue | None: ...

    def tau_for(self, day: date) -> TauRecord: ...
