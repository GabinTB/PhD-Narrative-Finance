"""Optional compiled kernels for the scoring hot path.

Built out-of-band (``python -m narrative_scoring._kernels.build``) because the
project builds with hatchling, which has no native-extension support. Import
is best-effort: when a ``.so`` is absent or was built for another interpreter
the corresponding ``HAVE_*`` flag is False and the numpy implementation is
used instead. The numpy path is the reference the kernels are asserted equal
to, so it must stay runnable everywhere. ``pipeline.score_dates`` logs which
path it took; rebuild after any Python-version or venv change.

* ``select_aggregate_rowwise`` -- the canonical scorer (selection.py +
  aggregation.py fused), flag ``HAVE_SELECT``.
* ``gate_aggregate_rowwise`` -- the older tau-then-percentile gate used only
  by the experimental Ray/GPU track (spec_pipeline.py, ray_hybrid.py), flag
  ``HAVE_FUSED``.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from narrative_scoring._kernels.select_aggregate import select_aggregate_rowwise  # noqa: F401

    HAVE_SELECT = True
except Exception as exc:  # noqa: BLE001 - any import failure must degrade, not crash
    select_aggregate_rowwise = None  # type: ignore[assignment]
    HAVE_SELECT = False
    log.debug("select_aggregate kernel unavailable (%s); using the numpy path", exc)

try:
    from narrative_scoring._kernels.fused_gate import gate_aggregate_rowwise  # noqa: F401

    HAVE_FUSED = True
except Exception as exc:  # noqa: BLE001
    gate_aggregate_rowwise = None  # type: ignore[assignment]
    HAVE_FUSED = False
    log.debug("fused_gate kernel unavailable (%s)", exc)

__all__ = ["select_aggregate_rowwise", "HAVE_SELECT", "gate_aggregate_rowwise", "HAVE_FUSED"]
