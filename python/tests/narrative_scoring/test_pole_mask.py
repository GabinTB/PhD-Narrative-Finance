"""mask_bipolar: one pole per bipolar pair per headline (selection.apply_pole_mask, the
kernel's step 0, f0.sample_null_draws, validation.merge_poles). Synthetic data only."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from narrative_scoring._kernels import HAVE_SELECT, select_aggregate_rowwise
from narrative_scoring.aggregation import DayAccumulator, headline_narrative_scores
from narrative_scoring.calibration import TauRecord
from narrative_scoring.config import AggRule, ScoringConfig
from narrative_scoring.f0 import n_keep, n_trim, sample_null_draws, trim_threshold
from narrative_scoring.partitions import NullPartitionWriter, load_partitions
from narrative_scoring.pipeline import _accumulate_block, score_dates
from narrative_scoring.primitives import (
    BipolarPairs,
    PrimitiveTable,
    bipolar_pairs,
    load_primitive_table,
    primitive_scores,
    scoring_matrix,
)
from narrative_scoring.schema import EMBEDDING_DIM
from narrative_scoring.selection import apply_pole_mask, select_with_pole_mask
from narrative_scoring.streaming import InMemoryHeadlineSource
from narrative_scoring.validation import merge_poles
from nlp.corrections import Correction

from .conftest import TAX_NAME, FixedTauProvider, toy_rows, unit_rows, write_toy_taxonomy

D0 = date(2008, 9, 15)
KERNEL = pytest.param(True, marks=pytest.mark.skipif(not HAVE_SELECT, reason="kernel not built"))
PATHS = [False, KERNEL]


def _tau(tau: float, through: date = D0 - timedelta(days=1)) -> TauRecord:
    return TauRecord(tau=tau, gaussian_tau=tau, n_eff=10.0, alpha=0.01, trim_frac=0.1,
                     n_draws=1000, seed=0, calibrated_from=through - timedelta(days=30),
                     calibrated_through=through, mode="raw", paraphrase_pooling="max",
                     mu_date=through)


def _raw(q: float, **kw) -> ScoringConfig:
    return ScoringConfig(mode=Correction.RAW, q=q, **kw)


def _ids(table: PrimitiveTable, **where: str) -> list[int]:
    f = table.frame
    for col, v in where.items():
        f = f.filter(pl.col(col) == v)
    return f["primitive_id"].to_list()


def _along(h: np.ndarray, c: float, rng: np.random.Generator) -> np.ndarray:
    """A unit vector at cosine ``c`` from the unit vector ``h``."""
    u = rng.normal(size=h.shape[0])
    u -= (u @ h) * h
    u /= np.linalg.norm(u)
    return (c * h + np.sqrt(1.0 - c * c) * u).astype(np.float32)


def _embeddings(table: PrimitiveTable, vectors: dict[int, np.ndarray], rng) -> np.ndarray:
    """Primitive-major text embeddings: every text of primitive p is ``vectors[p]`` (so its
    max-pooled score is exactly h . vectors[p]), random unit vectors elsewhere."""
    P = unit_rows(rng, table.n_primitives * table.n_texts)
    for p, v in vectors.items():
        P[p * table.n_texts:(p + 1) * table.n_texts] = v
    return P


def _one_day(table, P, H, config, use_kernel, tau=0.3, **kw):
    return score_dates([D0], config, table=table, primitive_embeddings=P,
                       source=InMemoryHeadlineSource({D0: H}),
                       calibration=FixedTauProvider(_tau(tau)), use_kernel=use_kernel, **kw)


def _support(res, narrative: str, reservoir: str = "macro") -> int:
    row = res.narrative_daily.filter((pl.col("narrative") == narrative)
                                     & (pl.col("reservoir") == reservoir))
    assert row.height == 1
    return int(row["SUPPORT"][0])


# ---------------------------------------------------------------------------
# The pairs
# ---------------------------------------------------------------------------

def test_toy_table_has_one_pair_with_pole_a_first(toy_table):
    pairs = toy_table.bipolar_pairs
    easing = _ids(toy_table, narrative="liquidity-easing")
    stress = _ids(toy_table, narrative="liquidity-stress", reservoir="macro")
    assert pairs.n_pairs == 1
    assert (pairs.a_start[0], pairs.a_len[0]) == (easing[0], len(easing))
    assert (pairs.b_start[0], pairs.b_len[0]) == (stress[0], len(stress))
    assert stress == list(range(stress[0], stress[0] + len(stress)))       # contiguous
    # firm/balance-sheet/liquidity-stress is monopolar (one signed pole): not a pair
    assert toy_table.narrative_nodes["TYPE"].to_list().count("liquidity") == 3


def test_more_than_two_poles_or_a_split_pole_raises(toy_table):
    f = toy_table.frame
    third = f.with_columns(pl.when(pl.col("primitive") == "m-liq-stress-repo")
                           .then(pl.lit("panic")).otherwise(pl.col("pole")).alias("pole"))
    with pytest.raises(ValueError, match="more than two signed poles"):
        bipolar_pairs(third)
    split = f.with_columns(pl.when(pl.col("primitive") == "f-solv-lev")
                           .then(pl.lit("macro")).otherwise(pl.col("reservoir")).alias("reservoir"),
                           pl.when(pl.col("primitive") == "f-solv-lev")
                           .then(pl.lit("funding")).otherwise(pl.col("dimension"))
                           .alias("dimension"),
                           pl.when(pl.col("primitive") == "f-solv-lev")
                           .then(pl.lit("liquidity")).otherwise(pl.col("TYPE")).alias("TYPE"),
                           pl.when(pl.col("primitive") == "f-solv-lev")
                           .then(pl.lit("stress")).otherwise(pl.col("pole")).alias("pole"))
    with pytest.raises(ValueError, match="contiguous"):
        bipolar_pairs(split)


# ---------------------------------------------------------------------------
# The rule, end to end from embeddings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_kernel", PATHS)
def test_headline_hitting_both_poles_keeps_only_the_stronger(toy_table, use_kernel):
    rng = np.random.default_rng(1)
    h = unit_rows(rng, 1)[0]
    stress = _ids(toy_table, narrative="liquidity-stress", reservoir="macro")
    easing = _ids(toy_table, narrative="liquidity-easing")
    vec = {p: _along(h, 0.9, rng) for p in stress} | {easing[0]: _along(h, 0.8, rng)}
    P = _embeddings(toy_table, vec, rng)
    off = _one_day(toy_table, P, h[None], _raw(0.5), use_kernel)       # k = 4
    on = _one_day(toy_table, P, h[None], _raw(0.5, mask_bipolar=True), use_kernel)
    assert (_support(off, "liquidity-stress"), _support(off, "liquidity-easing")) == (1, 1)
    assert (_support(on, "liquidity-stress"), _support(on, "liquidity-easing")) == (1, 0)
    assert on.narrative_daily["SUPPORT"].sum() == 1                  # one narrative row
    d = on.day_diagnostics.row(0, named=True)
    assert d["N_POLE_MASKED"] == 1 and d["N_MASK_CHANGED_RETENTION"] == 1
    assert off.day_diagnostics["N_POLE_MASKED"].to_list() == [None]


@pytest.mark.parametrize("use_kernel", PATHS)
def test_exact_tie_keeps_the_first_pole_only(toy_table, use_kernel):
    """Bit-identical primitive vectors give bit-identical pooled scores: a true tie."""
    rng = np.random.default_rng(2)
    h = unit_rows(rng, 1)[0]
    v = _along(h, 0.9, rng)
    stress = _ids(toy_table, narrative="liquidity-stress", reservoir="macro")
    easing = _ids(toy_table, narrative="liquidity-easing")
    P = _embeddings(toy_table, {p: v for p in stress + easing}, rng)
    S = primitive_scores(h[None], scoring_matrix(P, toy_table, Correction.RAW,
                                                 _raw(0.5).paraphrase_pooling, None, None),
                         toy_table.n_primitives, toy_table.n_texts, _raw(0.5).paraphrase_pooling)
    assert len({S[0, p].tobytes() for p in stress + easing}) == 1      # the tie is exact
    on = _one_day(toy_table, P, h[None], _raw(0.5, mask_bipolar=True), use_kernel)
    # pole a = easing (first in the sorted table), even though stress has 3 primitives
    assert (_support(on, "liquidity-easing"), _support(on, "liquidity-stress")) == (1, 0)
    assert on.day_diagnostics["N_POLE_MASKED"].to_list() == [len(stress)]


def _two_vs_five_table(root: Path) -> PrimitiveTable:
    """Pole 'down' (5 primitives) sorts before pole 'up' (2), plus unsigned filler."""
    rows = []

    def add(typ, sub, role, name):
        rows.append({"TOPIC": "macro", "GROUP": "growth", "TYPE": typ, "SUB_TYPE": sub,
                     "CATEGORY": f"{typ}-{sub}" if sub else typ, "ROLE": role,
                     "OBSERVABILITY_CHANNEL": "", "DISPLAY_NAME": name, "DESCRIPTION": name,
                     "SCHEDULED": "", "VALID_ENTITY_TYPES": "", "TAGS": ""})

    for i in range(5):
        add("activity", "down", f"r{i}", f"down-{i}")
    for i in range(2):
        add("activity", "up", f"r{i}", f"up-{i}")
    for i in range(6):
        add(f"filler{i}", "", "r", f"filler-{i}")
    write_toy_taxonomy(root, rows=rows, name="Tie_v1")
    return load_primitive_table(root, "Tie_v1", "headline")


@pytest.mark.parametrize("use_kernel", PATHS)
def test_exact_tie_with_poles_of_different_length(tmp_path, use_kernel):
    table = _two_vs_five_table(tmp_path / "tie")
    pairs = table.bipolar_pairs
    down, up = _ids(table, pole="down"), _ids(table, pole="up")
    assert (pairs.a_start[0], pairs.a_len[0], pairs.b_len[0]) == (down[0], 5, 2)
    rng = np.random.default_rng(3)
    h = unit_rows(rng, 1)[0]
    v, w = _along(h, 0.9, rng), _along(h, 0.5, rng)
    # top-3 of 'down' = three copies of v (mean v), its 2 others lower; 'up' = two copies of v
    vec = {p: v for p in down[:3] + up} | {p: w for p in down[3:]}
    P = _embeddings(table, vec, rng)
    on = _one_day(table, P, h[None], _raw(0.6, mask_bipolar=True), use_kernel)    # k = 6
    nd = on.narrative_daily
    got = dict(zip(nd["narrative"].to_list(), nd["SUPPORT"].to_list()))
    assert got["activity-down"] == 1 and got["activity-up"] == 0
    assert on.day_diagnostics["N_POLE_MASKED"].to_list() == [2]


@pytest.mark.parametrize("use_kernel", PATHS)
def test_no_pairs_means_identical_output(tmp_path, use_kernel):
    """Unsigned and monopolar narratives only: the mask changes nothing but the config."""
    rows = [r for r in toy_rows() if r["DISPLAY_NAME"] != "m-liq-easing-ib"]
    table = load_primitive_table(write_toy_taxonomy(tmp_path / "t", rows=rows), TAX_NAME,
                                 "headline")
    assert table.bipolar_pairs.n_pairs == 0
    rng = np.random.default_rng(4)
    P = unit_rows(rng, table.n_primitives * table.n_texts)
    H = unit_rows(rng, 50)
    off = _one_day(table, P, H, _raw(0.75), use_kernel, tau=0.0)
    on = _one_day(table, P, H, _raw(0.75, mask_bipolar=True), use_kernel, tau=0.0)
    assert on.narrative_daily.equals(off.narrative_daily)
    assert on.day_diagnostics["N_POLE_MASKED"].to_list() == [0]
    assert on.day_diagnostics["N_MASK_CHANGED_RETENTION"].to_list() == [0]


def test_mask_touches_only_the_losing_pole(toy_table):
    rng = np.random.default_rng(5)
    S = rng.normal(0.2, 0.1, size=(300, toy_table.n_primitives)).astype(np.float32)
    before = S.copy()
    n_masked = apply_pole_mask(S, toy_table.bipolar_pairs)
    easing = _ids(toy_table, narrative="liquidity-easing")
    stress = _ids(toy_table, narrative="liquidity-stress", reservoir="macro")
    others = [p for p in range(toy_table.n_primitives) if p not in easing + stress]
    np.testing.assert_array_equal(S[:, others], before[:, others])
    lost_a = np.isneginf(S[:, easing]).all(axis=1)
    lost_b = np.isneginf(S[:, stress]).all(axis=1)
    assert (lost_a ^ lost_b).all()                           # exactly one pole per headline
    top3 = -np.sort(-before[:, stress], axis=1)[:, :3].astype(np.float64).sum(axis=1) / 3
    np.testing.assert_array_equal(lost_b, before[:, easing[0]].astype(np.float64) >= top3)
    np.testing.assert_array_equal(n_masked, np.where(lost_a, 1, 3))


# ---------------------------------------------------------------------------
# Kernel == numpy, diagnostics included
# ---------------------------------------------------------------------------

def _synthetic_pairs(p2n: np.ndarray, n_pairs: int, rng) -> BipolarPairs:
    """Pairs of adjacent narratives (both non-empty column blocks)."""
    starts = np.flatnonzero(np.r_[True, p2n[1:] != p2n[:-1]])
    lens = np.diff(np.r_[starts, p2n.size])
    cand = rng.choice(np.arange(0, starts.size - 1, 2), size=n_pairs, replace=False)
    cand.sort()
    i32 = lambda x: np.ascontiguousarray(x, dtype=np.int32)  # noqa: E731
    return BipolarPairs(i32(starts[cand]), i32(lens[cand]), i32(starts[cand + 1]),
                        i32(lens[cand + 1]))


def _numpy_masked(S, tau, k, jump_min, p2n, n_narr, rule, pairs, n_trim_rows):
    ms = select_with_pole_mask(S, tau, k, pairs, p2n, jump_cut=jump_min >= 0,
                               jump_min_candidates=max(jump_min, 1))
    sel = ms.selection
    narr = DayAccumulator(n_narr)
    _, nid, nval = headline_narrative_scores(sel.rows, sel.cols, sel.vals, p2n, n_narr, rule)
    narr.add_values(nid, nval)
    prim = DayAccumulator(S.shape[1])
    prim.add_values(sel.cols.astype(np.int64), sel.vals)
    thr = trim_threshold(S, 0.10) if n_trim_rows else None
    return ms, narr, prim, thr


@pytest.mark.skipif(not HAVE_SELECT, reason="kernel not built")
@pytest.mark.parametrize("rule", [AggRule.MEAN, AggRule.MEDIAN])
@pytest.mark.parametrize("jump_min", [-1, 2, 10])
@pytest.mark.parametrize("n,p,n_narr,k,n_pairs", [
    (500, 60, 20, 3, 6),
    (256, 1508, 419, 16, 51),      # real shape, 51 pairs
])
def test_kernel_matches_numpy_with_mask(n, p, n_narr, k, n_pairs, jump_min, rule):
    rng = np.random.default_rng(n + p)
    S = np.ascontiguousarray(rng.normal(0.2, 0.08, size=(n, p)).astype(np.float32))
    S[:, :5] = S[:, 5:10]                                   # float ties at the k boundary
    p2n = np.sort(np.r_[np.arange(n_narr), rng.integers(0, n_narr, p - n_narr)]).astype(np.int32)
    pairs = _synthetic_pairs(p2n, n_pairs, rng)
    for r in range(0, n, 7):                                # exact pole ties on some rows
        for a0, la, b0, lb in zip(pairs.a_start, pairs.a_len, pairs.b_start, pairs.b_len):
            S[r, a0:a0 + la] = S[r, a0]
            S[r, b0:b0 + lb] = S[r, a0]
    tau = float(np.float32(np.quantile(S, 0.99)))
    n_trim_rows = n_trim(p, 0.10)
    S_np, S_k = S.copy(), S.copy()
    ms, narr, prim, thr = _numpy_masked(S_np, tau, k, jump_min, p2n, n_narr, rule, pairs,
                                        n_trim_rows)
    for threads in (1, 4):
        S_k = S.copy()
        got = select_aggregate_rowwise(S_k, tau, k, jump_min, p2n, n_narr,
                                       rule is AggRule.MEDIAN, True, threads, n_trim_rows,
                                       pairs.a_start, pairs.a_len, pairs.b_start, pairs.b_len)
        np.testing.assert_array_equal(S_k, S_np)                # the kernel masks S in place
        np.testing.assert_array_equal(got["n_masked"], ms.n_masked)
        assert got["n_pole_masked"] == ms.n_pole_masked
        assert got["n_mask_changed_retention"] == ms.n_changed_retention
        np.testing.assert_array_equal(got["trim_threshold"], thr)
        np.testing.assert_array_equal(got["narr_count"], narr.count)
        np.testing.assert_allclose(got["narr_total"], narr.total, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got["narr_sumsq"], narr.sumsq, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got["narr_peak"], narr.peak, rtol=1e-12)
        np.testing.assert_array_equal(got["prim_count"], prim.count)
        np.testing.assert_array_equal(got["prim_peak"], prim.peak)
        sel = ms.selection
        assert got["n_unassigned"] == sel.n_unassigned
        assert got["n_candidates"] == int(sel.n_candidates.sum())
        assert got["n_f0_survivors"] == int(sel.n_f0_survivors.sum())
        assert got["n_retained_pre_jump"] == sel.n_retained_pre_jump
        assert got["n_retained"] == int(sel.n_retained.sum())
        assert got["n_jump_trimmed"] == sel.n_jump_trimmed
    assert ms.n_changed_retention > 0


def test_changed_retention_counts_narratives_not_primitives():
    # one pair: a = cols 0-1 (narrative 0), b = cols 2-4 (narrative 1); 2 more narratives
    p2n = np.array([0, 0, 1, 1, 1, 2, 3], dtype=np.int32)
    pairs = BipolarPairs(*(np.array([x], dtype=np.int32) for x in (0, 2, 2, 3)))
    S = np.array([[0.50, 0.49, 0.60, 0.59, 0.58, 0.40, 0.10]], dtype=np.float32)
    ms = select_with_pole_mask(S, 0.3, 3, pairs, p2n)
    # unmasked top-3 = cols 2,3,4 (narrative 1); masked: a loses, b kept -> same set
    assert ms.n_changed_retention == 0 and ms.n_pole_masked == 2
    S = np.array([[0.70, 0.69, 0.60, 0.59, 0.58, 0.40, 0.10]], dtype=np.float32)
    ms = select_with_pole_mask(S, 0.3, 4, pairs, p2n)
    # unmasked top-4 = 0,1 (n0) + 2,3 (n1); masked: b loses -> 0,1,5 + nothing else >= tau
    assert set(ms.selection.cols.tolist()) == {0, 1, 5}
    assert ms.n_changed_retention == 2                      # narrative 1 lost, narrative 2 won


# ---------------------------------------------------------------------------
# F0: trim threshold and null draws from the masked row
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_kernel", PATHS)
def test_trim_threshold_and_draws_come_from_the_masked_row(toy_table, use_kernel):
    rng = np.random.default_rng(6)
    S = np.ascontiguousarray(rng.normal(0.2, 0.1, size=(400, toy_table.n_primitives))
                             .astype(np.float32))
    pairs = toy_table.bipolar_pairs
    masked = S.copy()
    n_masked_ref = apply_pole_mask(masked, pairs)
    cfg = _raw(0.75, mask_bipolar=True)
    S_run = S.copy()
    thr, n_masked = _accumulate_block(
        S_run, 0.2, 2, cfg, toy_table.primitive_to_narrative, toy_table.n_narratives,
        DayAccumulator(toy_table.n_narratives), None, use_kernel, 2, None,
        n_trim(toy_table.n_primitives, cfg.trim_frac), pairs)
    np.testing.assert_array_equal(S_run, masked)
    np.testing.assert_array_equal(n_masked, n_masked_ref)
    np.testing.assert_array_equal(thr, trim_threshold(masked, cfg.trim_frac))
    draws = sample_null_draws(S_run, thr, 64, np.random.default_rng(0), n_masked=n_masked)
    assert np.isfinite(draws).all() and draws.size > 0
    loser = np.isneginf(masked)
    loser_values = set(S[loser].astype(np.float64).tolist())
    assert not loser_values & set(draws.tolist())


def test_null_draws_refuse_other_non_finite_values():
    rng = np.random.default_rng(7)
    S = rng.normal(size=(20, 30)).astype(np.float32)
    n_masked = np.zeros(20, dtype=np.int64)
    S[3, :4] = -np.inf
    n_masked[3] = 4
    thr = trim_threshold(S, 0.1)
    sample_null_draws(S, thr, 16, np.random.default_rng(0), n_masked=n_masked)
    with pytest.raises(ValueError, match="other than the pole mask"):
        sample_null_draws(S, thr, 16, np.random.default_rng(0))           # mask not declared
    bad = S.copy()
    bad[5, 0] = np.nan
    with pytest.raises(ValueError, match="other than the pole mask"):
        sample_null_draws(bad, thr, 16, np.random.default_rng(0), n_masked=n_masked)
    bad = S.copy()
    bad[3, :4] = np.inf                                   # right count, wrong sign
    with pytest.raises(ValueError, match="not the pole mask"):
        sample_null_draws(bad, thr, 16, np.random.default_rng(0), n_masked=n_masked)


@pytest.mark.parametrize("use_kernel", PATHS)
def test_null_partition_counts_exclude_masked_scores(toy_table, toy_embeddings, tmp_path,
                                                     use_kernel):
    rng = np.random.default_rng(8)
    cfg = _raw(0.75, mask_bipolar=True, null_draws_per_headline=5)
    days = [date(2008, 9, 1) + timedelta(days=i) for i in range(30)]
    X = {d: unit_rows(rng, 10) for d in days}
    sink = NullPartitionWriter(tmp_path / "parts", config=cfg, table=toy_table, seed=1)
    res = score_dates(days, cfg, table=toy_table, primitive_embeddings=toy_embeddings,
                      source=InMemoryHeadlineSource(X, chunk_size=4),
                      calibration=FixedTauProvider(_tau(0.2, through=date(2008, 8, 31))),
                      use_kernel=use_kernel, null_sink=sink)
    row = load_partitions(tmp_path / "parts").row(0, named=True)
    n_head = 30 * 10
    masked = int(res.day_diagnostics["N_POLE_MASKED"].sum())
    assert row["N_DRAWS_AVAILABLE"] == n_head * n_keep(toy_table.n_primitives, 0.1) - masked
    assert row["F0_CONFIG_ID"] == cfg.f0_digest() != _raw(0.75, null_draws_per_headline=5) \
        .f0_digest()


# ---------------------------------------------------------------------------
# merge_poles
# ---------------------------------------------------------------------------

def _merged_view(table: PrimitiveTable) -> tuple[pl.DataFrame, np.ndarray]:
    """Nodes and primitive -> node map with both poles of every pair mapped to one node."""
    pair_types = (table.frame.filter(pl.col("pole") != "")
                  .group_by("reservoir", "dimension", "TYPE")
                  .agg(pl.col("pole").n_unique().alias("n")).filter(pl.col("n") == 2).drop("n"))
    frame = table.frame.join(pair_types.with_columns(pl.lit(True).alias("_pair")),
                             on=["reservoir", "dimension", "TYPE"], how="left")
    frame = frame.with_columns(
        pl.when(pl.col("_pair")).then(pl.col("TYPE")).otherwise(pl.col("narrative"))
        .alias("narrative"),
        pl.when(pl.col("_pair")).then(pl.lit("")).otherwise(pl.col("pole")).alias("pole"),
    ).with_columns(pl.concat_str([pl.col("reservoir"), pl.col("dimension"), pl.col("narrative")],
                                 separator="|").alias("narrative_key")).sort("primitive_id")
    nodes = (frame.group_by("narrative_key", maintain_order=True)
             .agg(pl.col("reservoir").first(), pl.col("dimension").first(),
                  pl.col("TYPE").first(), pl.col("narrative").first(), pl.col("pole").first(),
                  pl.len().cast(pl.UInt32).alias("n_primitives"))
             .with_row_index("node"))
    node_of = dict(zip(nodes["narrative_key"].to_list(), nodes["node"].to_list()))
    p2n = np.asarray([node_of[k] for k in frame["narrative_key"].to_list()], dtype=np.int32)
    return nodes.drop("node").select(table.narrative_nodes.columns), p2n


def _day_frame(S, table, nodes, p2n, pairs, use_kernel, q=0.5, tau=0.1):
    cfg = _raw(q, mask_bipolar=pairs is not None)
    acc = DayAccumulator(nodes.height)
    k = max(1, int(np.ceil((1 - q) * table.n_primitives - 1e-9)))
    _accumulate_block(S.copy(), tau, k, cfg, p2n, nodes.height, acc, None, use_kernel, 2, None,
                      0, pairs)
    return acc.frame(nodes, D0)


def _both_poles_headlines(table: PrimitiveTable, P: np.ndarray, rng, n=300) -> np.ndarray:
    T = P.reshape(table.n_primitives, table.n_texts, EMBEDDING_DIM).mean(axis=1)
    stress = _ids(table, narrative="liquidity-stress", reservoir="macro")
    easing = _ids(table, narrative="liquidity-easing")
    H = T[stress].mean(axis=0) + T[easing].mean(axis=0) + 0.5 * unit_rows(rng, n)
    return (H / np.linalg.norm(H, axis=1, keepdims=True)).astype(np.float32)


@pytest.mark.parametrize("use_kernel", PATHS)
def test_merge_poles_of_a_masked_run_equals_scoring_the_merged_node(toy_table, toy_embeddings,
                                                                    use_kernel):
    rng = np.random.default_rng(9)
    H = _both_poles_headlines(toy_table, toy_embeddings, rng)
    S = primitive_scores(H, scoring_matrix(toy_embeddings, toy_table, Correction.RAW,
                                           _raw(0.5).paraphrase_pooling, None, None),
                         toy_table.n_primitives, toy_table.n_texts, _raw(0.5).paraphrase_pooling)
    pairs = toy_table.bipolar_pairs
    nodes, p2n = toy_table.narrative_nodes, toy_table.primitive_to_narrative
    m_nodes, m_p2n = _merged_view(toy_table)

    pole_level = _day_frame(S, toy_table, nodes, p2n, pairs, use_kernel)
    direct = _day_frame(S, toy_table, m_nodes, m_p2n, pairs, use_kernel)
    merged = merge_poles(pole_level)
    assert merged.height == direct.height == toy_table.n_narratives - 1
    a = merged.sort("narrative_key")
    b = direct.select(merged.columns).sort("narrative_key")
    assert a.drop("SUPPORT", "TOTAL_SCORE", "INTENSITY", "STD_SCORE", "PEAK").equals(
        b.drop("SUPPORT", "TOTAL_SCORE", "INTENSITY", "STD_SCORE", "PEAK"))
    np.testing.assert_array_equal(a["SUPPORT"].to_numpy(), b["SUPPORT"].to_numpy())
    # TOTAL is stored in float64; INTENSITY / STD in float32, and STD is recombined from the
    # stored float32 STD and INTENSITY (sqrt of a difference: float32 cancellation)
    for col, rtol, atol in (("TOTAL_SCORE", 1e-12, 1e-12), ("PEAK", 0.0, 0.0),
                            ("INTENSITY", 1e-6, 0.0), ("STD_SCORE", 0.0, 1e-6)):
        np.testing.assert_allclose(a[col].to_numpy().astype(np.float64),
                                   b[col].to_numpy().astype(np.float64), rtol=rtol, atol=atol)
    liq = a.filter(pl.col("narrative_key") == "macro|funding|liquidity")
    assert liq["SUPPORT"][0] > 0 and liq["pole"][0] == ""

    # unmasked: a headline retaining both poles is counted twice by the sum
    pole_level_u = _day_frame(S, toy_table, nodes, p2n, None, use_kernel)
    direct_u = _day_frame(S, toy_table, m_nodes, m_p2n, None, use_kernel)
    key = pl.col("narrative_key") == "macro|funding|liquidity"
    assert (merge_poles(pole_level_u).filter(key)["SUPPORT"][0]
            > direct_u.filter(key)["SUPPORT"][0])


def test_merge_poles_passes_monopolar_and_unsigned_rows_through(toy_table):
    rng = np.random.default_rng(10)
    nd = toy_table.narrative_nodes.with_columns(
        pl.lit(D0).alias("DATE"), pl.lit("all").alias("SENTIMENT"),
        pl.Series("SUPPORT", rng.integers(0, 5, toy_table.n_narratives), dtype=pl.Int32),
    ).with_columns(
        pl.when(pl.col("SUPPORT") > 0).then(pl.col("SUPPORT") * 0.4).alias("TOTAL_SCORE"),
        pl.when(pl.col("SUPPORT") > 0).then(pl.lit(0.4, dtype=pl.Float32)).alias("INTENSITY"),
        pl.when(pl.col("SUPPORT") > 0).then(pl.lit(0.0, dtype=pl.Float32)).alias("STD_SCORE"),
        pl.when(pl.col("SUPPORT") > 0).then(pl.lit(0.4, dtype=pl.Float32)).alias("PEAK"),
        pl.lit(10, dtype=pl.Int32).alias("N_HEADLINES"),
        pl.lit(10, dtype=pl.Int32).alias("N_LABELLED"),
    )
    out = merge_poles(nd)
    firm = out.filter(pl.col("narrative_key") == "firm|balance-sheet|liquidity-stress")
    assert firm.height == 1 and firm["pole"][0] == "stress"                # monopolar kept
    assert firm.equals(nd.filter(pl.col("narrative_key") == firm["narrative_key"][0])
                       .select(out.columns))
    solv = out.filter(pl.col("narrative") == "solvency")
    assert solv["pole"][0] == "" and solv["n_primitives"][0] == 2
    assert out.filter(pl.col("TYPE") == "liquidity").height == 2           # firm + merged macro
    with pytest.raises(ValueError, match="no TYPE column"):
        merge_poles(nd.drop("TYPE"))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_default_config_hashes_as_before_the_mask_existed():
    c = ScoringConfig()
    assert (c.digest(), c.f0_digest(), c.warmup_digest()) == (
        "9de874ad4aaa9d3e", "993efed4e4b98249", "002d567e415cea6c")   # v2.2.0 production ids
    m = ScoringConfig(mask_bipolar=True)
    assert m.digest() != c.digest()
    assert m.f0_digest() != c.f0_digest()
    assert m.warmup_digest() != c.warmup_digest()
    assert ScoringConfig.from_dict(m.to_dict()) == m
    legacy = {k: v for k, v in c.to_dict().items() if k != "mask_bipolar"}
    assert ScoringConfig.from_dict(legacy).digest() == c.digest()


def test_cli_flags(monkeypatch):
    import argparse

    from narrative_scoring.jobs import _config, add_scoring_args

    ap = argparse.ArgumentParser()
    add_scoring_args(ap)
    args = ap.parse_args(["--mask-bipolar"])
    assert _config(args).mask_bipolar and not args.no_primitive_daily
    args = ap.parse_args(["--no-primitive-daily"])
    assert not _config(args).mask_bipolar and args.no_primitive_daily
