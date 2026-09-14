"""The garbage-catcher rejection layer.

A second, independent rejection layer.  F0 (null_model.py) rejects headlines
whose best score does not clear the noise floor; the garbage catcher rejects
headlines that score *well* against a topic outside the taxonomy.  These
catch different failures, so both are available and independently toggleable.

    taxonomy_max = S_taxonomy.max(axis=1)
    garbage_max  = S_garbage.max(axis=1)
    reject       = garbage_max > taxonomy_max     # strictly greater

Rejected headlines contribute nothing to any primitive.
"""
from __future__ import annotations

import numpy as np


def apply_garbage_filter(
    S_taxonomy: np.ndarray,
    S_garbage: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Returns (reject_mask, rejection_rate) so the marginal value of this
    layer over F0 alone can be measured.
    """
    taxonomy_max = np.asarray(S_taxonomy).max(axis=-1)
    garbage_max = np.asarray(S_garbage).max(axis=-1)
    reject = garbage_max > taxonomy_max
    rate = float(reject.mean()) if reject.size else 0.0
    return reject, rate
