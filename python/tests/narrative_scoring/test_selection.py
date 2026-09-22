"""Selection semantics: q-candidates on the full row, THEN tau, THEN optional jump cut."""
from __future__ import annotations

import numpy as np
import pytest

from narrative_scoring.config import n_candidates_for
from narrative_scoring.selection import apply_jump_cut, kth_largest_rowwise, select
from narrative_scoring.validation import reference_select


def dense(sel, n_prim):
    out = np.full((sel.n_rows, n_prim), np.nan, dtype=np.float32)
    out[sel.rows, sel.cols] = sel.vals
    return out


class TestCandidateBudget:
    def test_budget_from_percentile(self):
        assert n_candidates_for(0.99, 1508) == 16       # ceil((1-q) * P), owner-final
        assert n_candidates_for(0.95, 1508) == 76
        assert n_candidates_for(0.99, 50) == 1          # never below one
        assert n_candidates_for(0.5, 100) == 50         # exact binary fraction stays
        with pytest.raises(ValueError):
            n_candidates_for(99.0, 1508)                # legacy percent form rejected

    def test_kth_largest_is_over_the_full_row(self):
        S = np.array([[0.9, 0.1, 0.5, 0.3]], dtype=np.float32)
        assert kth_largest_rowwise(S, 1)[0] == np.float32(0.9)
        assert kth_largest_rowwise(S, 2)[0] == np.float32(0.5)
        assert kth_largest_rowwise(S, 4)[0] == np.float32(0.1)


class TestOrderOfOperations:
    def test_q_before_tau_high_ranked_candidate_below_tau_removed(self):
        # k=2 candidates: 0.9 and 0.5. tau=0.6 removes 0.5 even though it is a candidate.
        S = np.array([[0.9, 0.1, 0.5, 0.3]], dtype=np.float32)
        sel = select(S, tau=0.6, n_candidates=2)
        assert sel.cols.tolist() == [0]
        assert sel.n_candidates.tolist() == [2]
        assert sel.n_retained.tolist() == [1]
        assert sel.n_f0_survivors.tolist() == [1]        # 0.9 only, over the full row
        assert sel.n_retained_pre_jump == 1

    def test_low_ranked_score_above_tau_is_not_retained(self):
        # every score clears tau, but only the top-k=2 are candidates
        S = np.array([[0.9, 0.7, 0.8, 0.75]], dtype=np.float32)
        sel = select(S, tau=0.1, n_candidates=2)
        assert sorted(sel.cols.tolist()) == [0, 2]

    def test_not_top_q_among_tau_survivors(self):
        # Wrong order (tau first, then top-2 of survivors) would keep {0.9, 0.5}.
        # Right order keeps candidates {0.9, 0.8} then drops nothing below tau=0.45 ->
        # {0.9, 0.8}; 0.5 is outside the candidate budget regardless of tau.
        S = np.array([[0.9, 0.8, 0.5, 0.1]], dtype=np.float32)
        sel = select(S, tau=0.45, n_candidates=2)
        assert sorted(sel.cols.tolist()) == [0, 1]

    def test_all_absent_when_no_candidate_clears_tau(self):
        S = np.array([[0.2, 0.1, 0.15, 0.05]], dtype=np.float32)
        sel = select(S, tau=0.5, n_candidates=2)
        assert sel.rows.size == 0
        assert sel.n_unassigned == 1
        D = dense(sel, 4)
        assert np.isnan(D).all()

    def test_tau_is_not_a_headline_gate(self):
        # A row whose max is below tau is simply empty; another row is unaffected.
        S = np.array([[0.2, 0.1], [0.9, 0.1]], dtype=np.float32)
        sel = select(S, tau=0.5, n_candidates=1)
        assert sel.rows.tolist() == [1] and sel.cols.tolist() == [0]

    def test_absent_is_nan_never_zero(self):
        S = np.array([[0.9, 0.0, 0.5]], dtype=np.float32)   # a genuine 0.0 score
        sel = select(S, tau=-1.0, n_candidates=2)             # tau admits everything
        D = dense(sel, 3)
        assert D[0, 1] != 0.0 and np.isnan(D[0, 1])           # not a candidate -> NaN
        assert 0.0 not in sel.vals

    def test_retained_never_exceeds_candidates(self):
        rng = np.random.default_rng(0)
        S = rng.normal(0.2, 0.1, size=(200, 300)).astype(np.float32)
        sel = select(S, tau=0.25, n_candidates=3)
        assert (sel.n_retained <= sel.n_candidates).all()
        assert (sel.n_candidates >= 3).all()

    def test_ties_at_boundary_are_all_candidates(self):
        S = np.array([[0.5, 0.5, 0.5, 0.1]], dtype=np.float32)
        sel = select(S, tau=0.0, n_candidates=2)
        assert sorted(sel.cols.tolist()) == [0, 1, 2]

    def test_comparison_precision_float32(self):
        # tau rounds down in float32 (0.1697 -> 0.16969999); a score equal to
        # float32(tau) must be retained on every path, so tau is compared as float32.
        tau = 0.1697
        S = np.array([[np.float32(tau), 0.05]], dtype=np.float32)
        sel = select(S, tau=tau, n_candidates=1)
        assert sel.cols.tolist() == [0]

    def test_input_not_mutated(self):
        S = np.random.default_rng(1).normal(size=(5, 9)).astype(np.float32)
        before = S.copy()
        select(S, tau=0.0, n_candidates=2, jump_cut=True, jump_min_candidates=1)
        np.testing.assert_array_equal(S, before)


class TestJumpCut:
    def _row(self, vals):
        return np.asarray([vals], dtype=np.float32)

    def test_disabled_equals_q_tau_exactly(self):
        rng = np.random.default_rng(3)
        S = rng.normal(0.3, 0.1, size=(64, 120)).astype(np.float32)
        a = select(S, tau=0.3, n_candidates=12, jump_cut=False)
        b = select(S, tau=0.3, n_candidates=12, jump_cut=True, jump_min_candidates=10_000)
        np.testing.assert_array_equal(dense(a, 120), dense(b, 120))

    def test_enabled_never_retains_more(self):
        rng = np.random.default_rng(4)
        S = rng.normal(0.3, 0.1, size=(64, 120)).astype(np.float32)
        a = select(S, tau=0.2, n_candidates=12, jump_cut=False)
        b = select(S, tau=0.2, n_candidates=12, jump_cut=True, jump_min_candidates=3)
        assert (b.n_retained <= a.n_retained).all()
        Da, Db = dense(a, 120), dense(b, 120)
        kept_b = ~np.isnan(Db)
        np.testing.assert_array_equal(Db[kept_b], Da[kept_b])   # a subset, same values

    def test_one_match(self):
        S = self._row([0.9, 0.1, 0.1, 0.1, 0.1])
        sel = select(S, tau=0.5, n_candidates=3, jump_cut=True, jump_min_candidates=1)
        assert sel.cols.tolist() == [0] and sel.n_jump_trimmed == 0   # 1 retained, not > min

    def test_multi_match_cut_at_largest_gap(self):
        # retained (desc): 0.90, 0.88, 0.60, 0.58 -> largest gap 0.88-0.60 -> keep 2
        S = self._row([0.90, 0.60, 0.88, 0.58, 0.1, 0.05])
        sel = select(S, tau=0.5, n_candidates=4, jump_cut=True, jump_min_candidates=2)
        assert sorted(sel.cols.tolist()) == [0, 2]
        assert sel.n_jump_trimmed == 1
        assert sel.n_retained_pre_jump == 4
        assert sel.jump_gap_sum == pytest.approx(np.float32(0.88) - np.float32(0.60), abs=1e-7)

    def test_smooth_tail_cuts_at_first_largest_gap(self):
        # exactly equal gaps (dyadic values, exact in float32): the first occurrence
        # wins -> only the top score survives
        S = self._row([0.875, 0.75, 0.625, 0.5, 0.375, 0.1])
        sel = select(S, tau=0.3, n_candidates=5, jump_cut=True, jump_min_candidates=2)
        assert sel.cols.tolist() == [0]

    def test_all_equal_has_no_boundary(self):
        S = self._row([0.7, 0.7, 0.7, 0.7, 0.1])
        sel = select(S, tau=0.5, n_candidates=4, jump_cut=True, jump_min_candidates=2)
        assert sorted(sel.cols.tolist()) == [0, 1, 2, 3]
        assert sel.n_jump_trimmed == 0

    def test_all_below_tau(self):
        S = self._row([0.3, 0.2, 0.25, 0.1])
        sel = select(S, tau=0.5, n_candidates=3, jump_cut=True, jump_min_candidates=1)
        assert sel.rows.size == 0 and sel.n_unassigned == 1

    def test_min_candidates_is_strict(self):
        S = self._row([0.9, 0.5, 0.49, 0.1])
        keep_all = select(S, tau=0.4, n_candidates=3, jump_cut=True, jump_min_candidates=3)
        assert keep_all.n_retained.tolist() == [3]                # 3 is not > 3
        cut = select(S, tau=0.4, n_candidates=3, jump_cut=True, jump_min_candidates=2)
        assert cut.cols.tolist() == [0]

    def test_apply_jump_cut_passthrough_when_nothing_eligible(self):
        rows = np.array([0, 0, 1], dtype=np.int32)
        cols = np.array([1, 2, 0], dtype=np.int32)
        vals = np.array([0.9, 0.3, 0.5], dtype=np.float32)
        r, c, v, n, g = apply_jump_cut(rows, cols, vals, 2, min_candidates=5)
        assert n == 0 and g == 0.0 and r is rows and c is cols and v is vals


@pytest.mark.parametrize("jump", [False, True])
@pytest.mark.parametrize("seed,n,p,k,tau", [(0, 300, 200, 3, 0.35), (1, 128, 1508, 15, 0.3),
                                            (2, 50, 40, 1, 0.0), (3, 40, 30, 30, 0.45)])
def test_vectorised_matches_dense_reference(seed, n, p, k, tau, jump):
    rng = np.random.default_rng(seed)
    S = rng.normal(0.25, 0.1, size=(n, p)).astype(np.float32)
    S[:, :3] = S[:, 3:6]                       # exact ties
    sel = select(S, tau=tau, n_candidates=k, jump_cut=jump, jump_min_candidates=2)
    ref = reference_select(S, tau, k, jump_cut=jump, jump_min_candidates=2)
    np.testing.assert_array_equal(dense(sel, p), ref)
