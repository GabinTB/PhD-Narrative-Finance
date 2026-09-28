"""Per-headline selection: q-percentile candidates on the FULL row, then tau, then jump cut.

For one headline with primitive-score vector s (length P), in this order and
no other:

1. Candidate budget from the full raw row. k = n_candidates (ceil((1-q)*P),
   16 for q=0.99 and P=1508). A primitive is a candidate iff
   s[p] >= kth_largest(s), where the k-th largest is taken over ALL P scores,
   including those below tau. Ties at the boundary are all candidates (the
   cut is ">=", so the set can exceed k only on exact float32 ties).
2. Absolute floor. retained = candidates AND s[p] >= tau. A candidate below
   tau is absent; a score above tau outside the candidate set is absent.
   Equivalently: retained iff s[p] >= max(kth_largest(s), tau). tau never
   admits or rejects a whole headline; it only prunes the candidate set.
3. Optional jump cut (OFF by default). Only when the retained count EXCEEDS
   ``jump_min_candidates``: sort retained descending, gap_i = s_(i) - s_(i+1),
   keep the prefix above the largest gap (first occurrence on ties). No
   ratios, no gap-size threshold. If every gap is zero there is no boundary
   and the q + tau set is kept unchanged.

Optional bipolar pole mask (``config.mask_bipolar``), BEFORE step 1: for every
bipolar pair (``primitives.BipolarPairs``), pole score = mean of the pole's top-3
primitive scores in the row (all of them when it has fewer than 3), summed in
float64 in descending order; the pole with the higher score is kept, the other
pole's primitives are set to -inf, so they can be neither candidates nor
retained (and are not null draws, f0.sample_null_draws). Exact tie: pole a, the
first in the sorted table, never both. Unsigned and monopolar narratives are
untouched. ``select_with_pole_mask`` also selects the unmasked row, to count the
(headline, narrative) pairs whose retained set the mask changed.

Absent scores are never zero: everything downstream works on sparse
(row, col, value) triplets of the retained set only. All comparisons are in
float32 (tau is cast once), so the compiled kernel and this reference agree
bit for bit on which scores are kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from narrative_scoring.primitives import BipolarPairs

POLE_TOP = 3        # a pole's score is the mean of its top-POLE_TOP primitive scores


@dataclass
class Selection:
    """Retained (headline, primitive, score) triplets for one block, plus counts."""

    rows: np.ndarray            # int32 headline index within the block
    cols: np.ndarray            # int32 primitive index
    vals: np.ndarray            # float32 retained scores
    n_rows: int
    n_candidates: np.ndarray    # int64 per row: |{p : s[p] >= kth largest}| (>= k on ties)
    n_retained: np.ndarray      # int64 per row: after tau (and jump cut)
    n_f0_survivors: np.ndarray  # int64 per row: |{p : s[p] >= tau}| over the FULL row
    n_retained_pre_jump: int    # block total after tau, before the jump cut
    n_jump_trimmed: int         # rows whose retained set the jump cut shortened
    jump_gap_sum: float         # sum of the largest gap over rows where the cut applied

    @property
    def n_unassigned(self) -> int:
        return int((self.n_retained == 0).sum())


def kth_largest_rowwise(S: np.ndarray, k: int) -> np.ndarray:
    """k-th largest value of each row of ``S`` (1-indexed), O(P) per row via partition."""
    n_prim = S.shape[1]
    if not 1 <= k <= n_prim:
        raise ValueError(f"k must be in [1, {n_prim}], got {k}")
    return np.partition(S, n_prim - k, axis=1)[:, n_prim - k]


def select(
    S: np.ndarray,
    tau: float,
    n_candidates: int,
    *,
    jump_cut: bool = False,
    jump_min_candidates: int = 10,
) -> Selection:
    """Steps 1-3 above for a (n_head, n_prim) float32 score block."""
    S = np.asarray(S, dtype=np.float32)
    n_rows = S.shape[0]
    tau32 = np.float32(tau)
    kth = kth_largest_rowwise(S, n_candidates)
    n_cand = (S >= kth[:, None]).sum(axis=1)
    n_f0 = (S >= tau32).sum(axis=1)
    thr = np.maximum(kth, tau32)
    rows, cols = np.nonzero(S >= thr[:, None])
    vals = S[rows, cols]
    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)
    n_pre_jump = int(rows.size)

    n_trimmed, gap_sum = 0, 0.0
    if jump_cut and rows.size:
        rows, cols, vals, n_trimmed, gap_sum = apply_jump_cut(
            rows, cols, vals, n_rows, jump_min_candidates)

    return Selection(
        rows=rows, cols=cols, vals=vals, n_rows=n_rows,
        n_candidates=n_cand.astype(np.int64),
        n_retained=np.bincount(rows, minlength=n_rows).astype(np.int64),
        n_f0_survivors=n_f0.astype(np.int64),
        n_retained_pre_jump=n_pre_jump, n_jump_trimmed=n_trimmed, jump_gap_sum=gap_sum,
    )


def apply_jump_cut(
    rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_rows: int, min_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Largest-absolute-gap prefix cut on rows with more than ``min_candidates`` retained.

    Returns the trimmed triplets, how many rows were shortened and the sum of
    the cutting gaps. Rows at or below the minimum, and rows whose retained
    scores are all equal (no positive gap, hence no boundary), pass through.
    """
    counts = np.bincount(rows, minlength=n_rows)
    if not (counts > min_candidates).any():
        return rows, cols, vals, 0, 0.0

    order = np.lexsort((-vals, rows))            # by row, then score descending
    r, c, v = rows[order], cols[order], vals[order]
    same_row = r[1:] == r[:-1]
    gap = np.full(r.size, -np.inf, dtype=np.float64)
    gap[:-1][same_row] = (v[:-1].astype(np.float64) - v[1:].astype(np.float64))[same_row]

    starts = np.flatnonzero(np.r_[True, ~same_row])
    lens = np.diff(np.r_[starts, r.size])
    row_max_gap = np.maximum.reduceat(gap, starts)
    eligible = (lens > min_candidates) & (row_max_gap > 0.0)

    # first position attaining the row's max gap -> everything after it goes
    idx = np.arange(r.size)
    at_max = gap == np.repeat(row_max_gap, lens)
    first_max = np.minimum.reduceat(np.where(at_max, idx, r.size), starts)
    cut_rank = np.where(eligible, first_max - starts, lens - 1)
    rank = idx - np.repeat(starts, lens)
    keep = rank <= np.repeat(cut_rank, lens)
    return r[keep], c[keep], v[keep], int(eligible.sum()), float(row_max_gap[eligible].sum())


def pole_scores(S: np.ndarray, start: int, length: int) -> np.ndarray:
    """(n_head,) float64 mean of the top-``POLE_TOP`` scores of columns [start, start+length),
    summed in descending order (the compiled kernel's arithmetic, bit for bit)."""
    block = S[:, start:start + length]
    m = min(POLE_TOP, length)
    if length > m:
        block = np.partition(block, length - m, axis=1)[:, length - m:]
    top = -np.sort(-block, axis=1).astype(np.float64)
    acc = top[:, 0].copy()
    for j in range(1, m):
        acc += top[:, j]
    return acc / m


def apply_pole_mask(S: np.ndarray, pairs: BipolarPairs) -> np.ndarray:
    """Mask the losing pole of every pair to -inf, IN PLACE on the float32 block ``S``.
    Returns the per-row number of masked primitive scores (int64)."""
    n_masked = np.zeros(S.shape[0], dtype=np.int64)
    for a0, la, b0, lb in zip(pairs.a_start.tolist(), pairs.a_len.tolist(),
                              pairs.b_start.tolist(), pairs.b_len.tolist()):
        keep_a = pole_scores(S, a0, la) >= pole_scores(S, b0, lb)      # tie: pole a
        S[keep_a, b0:b0 + lb] = -np.inf
        S[~keep_a, a0:a0 + la] = -np.inf
        n_masked += np.where(keep_a, lb, la)
    return n_masked


@dataclass
class MaskedSelection:
    """``select`` on the masked block plus what the mask did to it."""

    selection: Selection
    n_masked: np.ndarray            # int64 per row: primitive scores set to -inf
    n_changed_retention: int        # (headline, narrative) pairs whose retained set changed

    @property
    def n_pole_masked(self) -> int:
        return int(self.n_masked.sum())


def select_with_pole_mask(
    S: np.ndarray,
    tau: float,
    n_candidates: int,
    pairs: BipolarPairs,
    prim_to_narr: np.ndarray,
    *,
    jump_cut: bool = False,
    jump_min_candidates: int = 10,
) -> MaskedSelection:
    """Mask ``S`` IN PLACE (``apply_pole_mask``), then ``select`` it. The unmasked selection
    of the same rows is computed too, only to count the (headline, narrative) pairs whose
    retained primitive set differs between the two."""
    if S.dtype != np.float32 or not S.flags.c_contiguous:
        raise ValueError("the pole mask works in place on a C-contiguous float32 block")
    kw = dict(jump_cut=jump_cut, jump_min_candidates=jump_min_candidates)
    unmasked = select(S, tau, n_candidates, **kw)
    n_masked = apply_pole_mask(S, pairs)
    sel = select(S, tau, n_candidates, **kw)
    n_prim = S.shape[1]
    before = unmasked.rows.astype(np.int64) * n_prim + unmasked.cols
    after = sel.rows.astype(np.int64) * n_prim + sel.cols
    diff = np.setxor1d(before, after)
    pair_keys = (diff // n_prim) * (int(prim_to_narr.max()) + 1) + prim_to_narr[diff % n_prim]
    return MaskedSelection(sel, n_masked, int(np.unique(pair_keys).size))
