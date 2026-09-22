"""Optional compiled kernels for the scoring hot path.

The extension is built out-of-band (``python -m narrative_scoring._kernels.build``)
rather than by the wheel, because the project builds with hatchling, which has no
native-extension support. Import is therefore best-effort: if the ``.so`` is
absent or was built for another interpreter, ``HAVE_FUSED`` is False and
``spec_pipeline`` transparently uses its numpy implementation.

That fallback is not a nicety -- the numpy path is the reference the compiled
kernel is asserted equal to, so it must stay runnable everywhere.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from narrative_scoring._kernels.fused_gate import gate_aggregate_rowwise  # noqa: F401

    HAVE_FUSED = True
except Exception as exc:  # noqa: BLE001 - any import failure must degrade, not crash
    gate_aggregate_rowwise = None  # type: ignore[assignment]
    HAVE_FUSED = False
    log.debug("fused kernel unavailable (%s); using the numpy path", exc)

__all__ = ["gate_aggregate_rowwise", "HAVE_FUSED"]
