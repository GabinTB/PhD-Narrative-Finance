"""End-to-end: streaming vs dense reference, live == historical, point-in-time guards,
metadata, sentiment split, day diagnostics, null-partition feed."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from narrative_scoring._kernels import HAVE_SELECT
from narrative_scoring.calibration import LookaheadError, TauRecord, resolve_mu_asof
from narrative_scoring.config import (
    AggRule,
    PoolRule,
    ScoringConfig,
    SentimentSplit,
    n_candidates_for,
)
from narrative_scoring.corrections import Correction, apply_mode
from narrative_scoring.partitions import MonthlyNullPartitionWriter, load_partitions
from narrative_scoring.pipeline import ParquetMonthWriter, date_range, score_dates
from narrative_scoring.primitives import primitive_scores, scoring_matrix
from narrative_scoring.schema import DAY_DIAGNOSTICS_SCHEMA, EMBEDDING_DIM, NARRATIVE_DAILY_SCHEMA
from narrative_scoring.streaming import InMemoryHeadlineSource, MemoryBudgetExceeded
from narrative_scoring.validation import (
    attention,
    reference_day,
    reference_headline_narrative,
    reference_select,
    summarize,
)

from .conftest import FixedTauProvider, unit_rows

D0 = date(2008, 9, 15)


def _mu_df(dates: list[date], seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    mus, hats = [], []
    for _ in dates:
        m = (rng.normal(size=EMBEDDING_DIM) * 0.2).astype(np.float32)
        mus.append(m)
        hats.append((m / np.linalg.norm(m)).astype(np.float32))
    return pl.DataFrame({"DATE": dates, "MU": mus, "MU_HAT": hats, "N": [10] * len(dates)},
                        schema={"DATE": pl.Date, "MU": pl.Array(pl.Float32, EMBEDDING_DIM),
                                "MU_HAT": pl.Array(pl.Float32, EMBEDDING_DIM), "N": pl.Int64})


def _tau(tau: float, through: date = D0 - timedelta(days=1)) -> TauRecord:
    return TauRecord(tau=tau, gaussian_tau=tau, n_eff=10.0, alpha=0.01, trim_frac=0.1,
                     n_draws=1000, seed=0, calibrated_from=through - timedelta(days=30),
                     calibrated_through=through, mode="r2", paraphrase_pooling="max",
                     mu_date=through)


def _headlines(rng, table, P, n):
    T = P.reshape(table.n_primitives, table.n_texts, EMBEDDING_DIM)
    centroid = T.mean(axis=1)                    # signal under max, mean AND median pooling
    base = centroid[rng.integers(0, table.n_primitives, n)]
    return (1.2 * base + 0.6 * unit_rows(rng, n)).astype(np.float32)   # NOT unit norm


@pytest.fixture
def world(toy_table, toy_embeddings):
    rng = np.random.default_rng(21)
    days = [D0 + timedelta(days=i) for i in range(3)]
    X = {d: _headlines(rng, toy_table, toy_embeddings, 40 + 25 * i) for i, d in enumerate(days)}
    mu_df = _mu_df([D0 - timedelta(days=40), D0 - timedelta(days=1), D0 + timedelta(days=1)])
    return {"days": days, "X": X, "mu_df": mu_df, "table": toy_table, "P": toy_embeddings}


def _dense_reference(world, config, tau, mu_rec, day, mask=None):
    t, P = world["table"], world["P"]
    mu, mu_hat = (mu_rec.mu, mu_rec.mu_hat) if mu_rec else (None, None)
    P_s = scoring_matrix(P, t, config.mode, config.paraphrase_pooling, mu, mu_hat)
    X = world["X"][day] if mask is None else world["X"][day][mask]
    H = apply_mode(X, config.mode, mu, mu_hat)
    S = primitive_scores(H, P_s, t.n_primitives, t.n_texts, config.paraphrase_pooling)
    k = n_candidates_for(config.q, t.n_primitives)
    A = reference_select(S, tau, k, jump_cut=config.jump_cut,
                         jump_min_candidates=config.jump_min_candidates)
    N = reference_headline_narrative(A, t.primitive_to_narrative, t.n_narratives,
                                     config.narrative_agg)
    return reference_day(N), reference_day(A), S.shape[0]


def _assert_day_matches(nd: pl.DataFrame, ref: dict) -> None:
    np.testing.assert_array_equal(nd["SUPPORT"].to_numpy(), ref["SUPPORT"])
    for col in ("TOTAL_SCORE", "INTENSITY", "STD_SCORE", "PEAK"):
        got = nd[col].to_numpy().astype(np.float64)
        np.testing.assert_allclose(got, ref[col], rtol=1e-5, atol=1e-6, equal_nan=True)


def _all(df: pl.DataFrame, day: date | None = None) -> pl.DataFrame:
    out = df.filter(pl.col("SENTIMENT") == "all")
    return out if day is None else out.filter(pl.col("DATE") == day)


KERNEL = pytest.param(True, marks=pytest.mark.skipif(not HAVE_SELECT, reason="kernel not built"))


@pytest.mark.parametrize("use_kernel", [False, KERNEL])
@pytest.mark.parametrize("config", [
    ScoringConfig(mode=Correction.R2, paraphrase_pooling=PoolRule.MAX, q=0.75),
    ScoringConfig(mode=Correction.R1, paraphrase_pooling=PoolRule.MEAN, q=0.75,
                  narrative_agg=AggRule.MEDIAN),
    ScoringConfig(mode=Correction.RAW, paraphrase_pooling=PoolRule.MEDIAN, q=0.75,
                  jump_cut=True, jump_min_candidates=1),
])
def test_streamed_chunks_match_dense_reference(world, config, use_kernel):
    tau = 0.15
    cal = FixedTauProvider(_tau(tau), world["mu_df"], mu_asof_id="mu-test")
    source = InMemoryHeadlineSource(world["X"], chunk_size=7)     # many small chunks
    res = score_dates(world["days"], config, table=world["table"], primitive_embeddings=world["P"],
                      source=source, calibration=cal, keep_primitive_daily=True,
                      use_kernel=use_kernel, threads=3)
    assert res.narrative_daily.schema == NARRATIVE_DAILY_SCHEMA
    assert res.day_diagnostics.schema == DAY_DIAGNOSTICS_SCHEMA
    assert res.n_days == 3 and not res.skipped_days
    assert set(res.narrative_daily["SENTIMENT"].unique()) == {"all"}
    for day in world["days"]:
        mu_rec = cal.mu_for(day) if config.mode is not Correction.RAW else None
        ref_narr, ref_prim, n_head = _dense_reference(world, config, tau, mu_rec, day)
        nd = _all(res.narrative_daily, day)
        _assert_day_matches(nd, ref_narr)
        assert (nd["N_HEADLINES"] == n_head).all() and (nd["N_LABELLED"] == n_head).all()
        _assert_day_matches(_all(res.primitive_daily, day), ref_prim)
        assert nd.height == world["table"].n_narratives
    assert res.day_diagnostics["N_RETAINED_POST_Q_TAU"].sum() > 0


def test_chunk_size_invariance(world):
    """Aggregation is chunk-order exact; the only chunk dependence left is BLAS rounding
    of S = H @ P.T (float32, ~1e-7 relative), hence the tolerance."""
    config = ScoringConfig(q=0.75)
    cal = FixedTauProvider(_tau(0.25), world["mu_df"])
    outs = []
    for cs in (1, 7, 1000):
        r = score_dates(world["days"], config, table=world["table"],
                        primitive_embeddings=world["P"],
                        source=InMemoryHeadlineSource(world["X"], chunk_size=cs), calibration=cal,
                        use_kernel=False)
        outs.append(r.narrative_daily)
    for other in outs[1:]:
        np.testing.assert_array_equal(outs[0]["SUPPORT"].to_numpy(), other["SUPPORT"].to_numpy())
        np.testing.assert_allclose(outs[0]["TOTAL_SCORE"].to_numpy(),
                                   other["TOTAL_SCORE"].to_numpy(), rtol=1e-5, equal_nan=True)


@pytest.mark.skipif(not HAVE_SELECT, reason="kernel not built")
def test_kernel_and_numpy_paths_agree(world):
    config = ScoringConfig(q=0.75, jump_cut=True, jump_min_candidates=1)
    cal = FixedTauProvider(_tau(0.25), world["mu_df"])
    src = InMemoryHeadlineSource(world["X"], chunk_size=13)
    a = score_dates(world["days"], config, table=world["table"], primitive_embeddings=world["P"],
                    source=src, calibration=cal, use_kernel=False, keep_primitive_daily=True)
    b = score_dates(world["days"], config, table=world["table"], primitive_embeddings=world["P"],
                    source=src, calibration=cal, use_kernel=True, keep_primitive_daily=True)
    for fa, fb in ((a.narrative_daily, b.narrative_daily), (a.primitive_daily, b.primitive_daily)):
        np.testing.assert_array_equal(fa["SUPPORT"].to_numpy(), fb["SUPPORT"].to_numpy())
        for col in ("TOTAL_SCORE", "INTENSITY", "STD_SCORE", "PEAK"):
            np.testing.assert_allclose(fa[col].to_numpy(), fb[col].to_numpy(), rtol=1e-6,
                                       equal_nan=True)
    for col in ("N_UNASSIGNED", "N_Q_CANDIDATES", "N_F0_SURVIVORS_PRE_Q", "N_RETAINED_PRE_JUMP",
                "N_RETAINED_POST_Q_TAU", "PCT_JUMP_APPLIED"):
        assert a.day_diagnostics[col].to_list() == b.day_diagnostics[col].to_list()
    np.testing.assert_allclose(a.day_diagnostics["MEAN_JUMP_GAP"].to_numpy(),
                               b.day_diagnostics["MEAN_JUMP_GAP"].to_numpy(), rtol=1e-9)


def test_live_single_day_equals_historical_range(world):
    config = ScoringConfig(q=0.75)
    cal = FixedTauProvider(_tau(0.25), world["mu_df"])
    src = InMemoryHeadlineSource(world["X"])
    hist = score_dates(world["days"], config, table=world["table"], primitive_embeddings=world["P"],
                       source=src, calibration=cal, use_kernel=False)
    for day in world["days"]:
        live = score_dates([day], config, table=world["table"], primitive_embeddings=world["P"],
                           source=src, calibration=cal, use_kernel=False)
        assert live.narrative_daily.equals(hist.narrative_daily.filter(pl.col("DATE") == day))
        assert live.metadata.tau_source_id == hist.metadata.tau_source_id
        assert live.metadata.config == hist.metadata.config


def test_mu_resolves_as_of_day_never_after(world):
    cal = FixedTauProvider(_tau(0.25), world["mu_df"])
    d = world["days"]
    assert cal.mu_for(d[0]).date == D0 - timedelta(days=1)          # latest row before
    assert cal.mu_for(d[1]).date == D0 + timedelta(days=1)          # exact row
    assert cal.mu_for(d[2]).date == D0 + timedelta(days=1)          # carried forward
    with pytest.raises(LookaheadError):
        resolve_mu_asof(world["mu_df"], D0 - timedelta(days=100))
    frozen = FixedTauProvider(_tau(0.25), world["mu_df"], freeze_mu_at=D0 - timedelta(days=1))
    assert frozen.mu_for(d[2]).date == D0 - timedelta(days=1)
    assert frozen.mu_policy.startswith("frozen@")
    later = FixedTauProvider(_tau(0.25), world["mu_df"], freeze_mu_at=D0 + timedelta(days=1))
    with pytest.raises(LookaheadError):
        later.mu_for(D0)


def test_tau_calibrated_through_scoring_day_is_refused(world):
    cal = FixedTauProvider(_tau(0.25, through=D0), world["mu_df"])
    with pytest.raises(LookaheadError, match="cannot score"):
        score_dates([D0], ScoringConfig(q=0.75), table=world["table"],
                    primitive_embeddings=world["P"], source=InMemoryHeadlineSource(world["X"]),
                    calibration=cal, use_kernel=False)


def test_per_day_mu_changes_the_scoring_matrix(world):
    config = ScoringConfig(mode=Correction.R2, q=0.75)
    cal = FixedTauProvider(_tau(0.25), world["mu_df"])
    res = score_dates(world["days"], config, table=world["table"], primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(world["X"]), calibration=cal, use_kernel=False)
    assert res.day_diagnostics["MU_DATE"].to_list() == [D0 - timedelta(days=1),
                                                        D0 + timedelta(days=1),
                                                        D0 + timedelta(days=1)]
    assert res.day_diagnostics["MU_NORM"].n_unique() == 2


def test_raw_mode_needs_no_mu(world):
    cal = FixedTauProvider(_tau(0.25), None)
    res = score_dates(world["days"], ScoringConfig(mode=Correction.RAW, q=0.75),
                      table=world["table"], primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(world["X"]), calibration=cal, use_kernel=False)
    assert res.day_diagnostics["MU_DATE"].null_count() == 3
    assert res.metadata.mu_policy == "none"


def test_missing_days_are_skipped_and_rss_logged(world):
    cal = FixedTauProvider(_tau(0.25), world["mu_df"])
    days = world["days"] + [D0 + timedelta(days=30)]
    res = score_dates(days, ScoringConfig(q=0.75), table=world["table"],
                      primitive_embeddings=world["P"], source=InMemoryHeadlineSource(world["X"]),
                      calibration=cal, use_kernel=False)
    assert res.skipped_days == [D0 + timedelta(days=30)]
    diag = res.day_diagnostics
    assert diag.height == 3
    assert (diag["RSS_GB_BEFORE"] > 0).all() and (diag["RSS_GB_AFTER"] > 0).all()
    assert res.peak_rss_gb >= diag["RSS_GB_AFTER"].max()
    assert (diag["SECONDS"] > 0).all()


def test_rss_guard_names_the_day(world):
    cal = FixedTauProvider(_tau(0.25), world["mu_df"])
    with pytest.raises(MemoryBudgetExceeded, match=str(D0)):
        score_dates([D0], ScoringConfig(q=0.75), table=world["table"],
                    primitive_embeddings=world["P"], source=InMemoryHeadlineSource(world["X"]),
                    calibration=cal, use_kernel=False, rss_budget_gb=1e-6)


def test_large_day_streams_in_bounded_chunks(world):
    rng = np.random.default_rng(9)
    big = {D0: (0.5 * unit_rows(rng, 5000) + 0.7 * world["X"][D0][rng.integers(0, 40, 5000)])}
    cal = FixedTauProvider(_tau(0.25), world["mu_df"])
    cfg = ScoringConfig(q=0.75)
    small = score_dates([D0], cfg, table=world["table"], primitive_embeddings=world["P"],
                        source=InMemoryHeadlineSource(big, chunk_size=256), calibration=cal,
                        use_kernel=False)
    whole = score_dates([D0], cfg, table=world["table"], primitive_embeddings=world["P"],
                        source=InMemoryHeadlineSource(big, chunk_size=10_000), calibration=cal,
                        use_kernel=False)
    assert small.day_diagnostics["N_HEADLINES"][0] == 5000
    np.testing.assert_array_equal(small.narrative_daily["SUPPORT"].to_numpy(),
                                  whole.narrative_daily["SUPPORT"].to_numpy())
    np.testing.assert_allclose(small.narrative_daily["TOTAL_SCORE"].to_numpy(),
                               whole.narrative_daily["TOTAL_SCORE"].to_numpy(), rtol=1e-5,
                               equal_nan=True)


def test_day_diagnostics_funnel(world):
    cfg = ScoringConfig(q=0.75, jump_cut=True, jump_min_candidates=1)
    # tau low enough that both q-candidates survive, so the jump cut has something to cut
    cal = FixedTauProvider(_tau(0.02), world["mu_df"], mu_asof_id="mu-x")
    res = score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(world["X"]), calibration=cal,
                      use_kernel=False)
    d = res.day_diagnostics
    n_head = d["N_HEADLINES"].to_numpy()
    k = n_candidates_for(0.75, world["table"].n_primitives)
    assert (d["N_Q_CANDIDATES"].to_numpy() >= k * n_head).all()
    assert (d["N_RETAINED_PRE_JUMP"] <= d["N_Q_CANDIDATES"]).all()
    assert (d["N_RETAINED_POST_Q_TAU"] <= d["N_RETAINED_PRE_JUMP"]).all()
    assert (d["N_F0_SURVIVORS_PRE_Q"] >= d["N_RETAINED_PRE_JUMP"]).all()
    np.testing.assert_allclose(d["MEAN_RETAINED_PER_HEADLINE"].to_numpy(),
                               d["N_RETAINED_POST_Q_TAU"].to_numpy() / n_head)
    np.testing.assert_allclose(
        d["PCT_TAU_PRUNED_WITHIN_Q"].to_numpy(),
        1 - d["N_RETAINED_PRE_JUMP"].to_numpy() / d["N_Q_CANDIDATES"].to_numpy())
    assert (d["PCT_JUMP_APPLIED"] > 0).any() and d["MEAN_JUMP_GAP"].drop_nulls().min() > 0
    assert d["CONFIG_ID"].unique().to_list() == [cfg.digest()]
    assert d["F0_CONFIG_ID"].unique().to_list() == [cfg.f0_digest()]
    assert d["TAU_SOURCE_ID"].unique().to_list() == [_tau(0.02).digest()]
    assert d["MU_ASOF_ID"].unique().to_list() == ["mu-x"]
    assert d["TAU_MONTH_END"].null_count() == 3           # static tau has no month_end
    assert d["N_WITH_SENTIMENT"].sum() == 0


def test_metadata_is_complete(world):
    cal = FixedTauProvider(_tau(0.25), world["mu_df"], mu_asof_id="mu_asof__abc")
    cfg = ScoringConfig(q=0.75, jump_cut=True, jump_min_candidates=4, label="x")
    res = score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(world["X"]), calibration=cal,
                      use_kernel=False, seed=7, code_version="deadbeef")
    m = res.metadata.to_dict()
    for key in ("mode", "paraphrase_style", "paraphrase_pooling", "q", "narrative_agg",
                "jump_cut", "jump_min_candidates", "alpha", "trim_frac", "sentiment_split",
                "neutral_eps", "null_draws_per_headline"):
        assert key in m["config"]
    assert m["percentile_axis"] == "row_wise"
    assert m["n_candidates"] == n_candidates_for(0.75, world["table"].n_primitives)
    assert m["taxonomy_sha1"] and m["paraphrase_sha1"] and m["primitive_embeddings_digest"]
    assert m["mu_asof_id"] == "mu_asof__abc" and m["mu_policy"] == "as_of_day"
    assert m["tau_source_id"] == _tau(0.25).digest() and m["tau_policy"] == "test-fixed"
    assert m["n_eff"] == 10.0
    assert m["config_id"] == cfg.digest() and m["f0_config_id"] == cfg.f0_digest()
    assert m["seed"] == 7 and m["code_version"] == "deadbeef"
    assert m["extra"]["first_tau_record"]["seed"] == 0
    assert m["extra"]["scoring_path"] == "numpy"
    row = summarize(res)
    assert row["mode"] == "r2" and row["n_days"] == 3 and "retained_per_headline" in row
    assert res.day_diagnostics["MU_NORM"].is_finite().all()


def test_parquet_writer_round_trip(world, tmp_path: Path):
    cal = FixedTauProvider(_tau(0.25), world["mu_df"])
    days = world["days"] + [date(2008, 10, 1)]
    X = dict(world["X"])
    X[date(2008, 10, 1)] = world["X"][D0]
    out, diag_dir = tmp_path / "out", tmp_path / "diag"
    res = score_dates(days, ScoringConfig(q=0.75), table=world["table"],
                      primitive_embeddings=world["P"], source=InMemoryHeadlineSource(X),
                      calibration=cal, writer=ParquetMonthWriter(out, diag_dir), use_kernel=False,
                      keep_primitive_daily=True, collect=False)
    assert res.narrative_daily.height == 0                      # collect=False
    assert sorted(p.name for p in out.glob("*.parquet")) == ["2008-09.parquet", "2008-10.parquet"]
    assert sorted(p.name for p in diag_dir.glob("*.parquet")) == ["2008-09.parquet",
                                                                   "2008-10.parquet"]
    sep = pl.read_parquet(out / "2008-09.parquet")
    assert sep["DATE"].n_unique() == 3 and sep.schema == NARRATIVE_DAILY_SCHEMA
    assert pl.read_parquet(diag_dir / "2008-09.parquet").schema == DAY_DIAGNOSTICS_SCHEMA
    assert (out / "primitive_daily" / "2008-10.parquet").exists()
    assert (out / "run_metadata.json").exists() and (diag_dir / "run_metadata.json").exists()


def test_attention_is_derived_not_stored(world):
    cal = FixedTauProvider(_tau(0.25), world["mu_df"])
    res = score_dates([D0], ScoringConfig(q=0.75), table=world["table"],
                      primitive_embeddings=world["P"], source=InMemoryHeadlineSource(world["X"]),
                      calibration=cal, use_kernel=False)
    assert "ATTENTION" not in res.narrative_daily.columns
    att = attention(res.narrative_daily)
    ok = att.filter(pl.col("SUPPORT") > 0)
    np.testing.assert_allclose(ok["ATTENTION"].to_numpy(),
                               ok["TOTAL_SCORE"].to_numpy() / ok["N_HEADLINES"].to_numpy())
    assert att.filter(pl.col("SUPPORT") == 0)["ATTENTION"].null_count() == \
        att.filter(pl.col("SUPPORT") == 0).height


def test_date_range_helper():
    assert list(date_range(D0, D0 + timedelta(days=2))) == [D0, D0 + timedelta(days=1),
                                                           D0 + timedelta(days=2)]
    assert list(date_range(D0, D0 - timedelta(days=1))) == []


# ---------------------------------------------------------------------------
# Sentiment split
# ---------------------------------------------------------------------------

def _sentiment_world(world, seed=5):
    rng = np.random.default_rng(seed)
    sent = {}
    for d, X in world["X"].items():
        s = rng.normal(size=X.shape[0]).astype(np.float32)
        s[::4] = np.nan                                  # missing every 4th
        s[1::9] = 0.0                                    # exact zeros
        sent[d] = s
    return sent


@pytest.mark.parametrize("use_kernel", [False, KERNEL])
@pytest.mark.parametrize("eps", [0.0, 0.3])
def test_sentiment_split_reconciles(world, eps, use_kernel):
    sent = _sentiment_world(world)
    cfg = ScoringConfig(q=0.75, sentiment_split=SentimentSplit.SIGN, neutral_eps=eps)
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    src = InMemoryHeadlineSource(world["X"], chunk_size=11, sentiment=sent)
    res = score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                      source=src, calibration=cal, use_kernel=use_kernel,
                      keep_primitive_daily=True)
    labels = {"all", "pos", "neg"} | ({"neu"} if eps > 0 else set())
    assert set(res.narrative_daily["SENTIMENT"].unique()) == labels
    for day in world["days"]:
        s = sent[day]
        nd = res.narrative_daily.filter(pl.col("DATE") == day)
        all_rows = nd.filter(pl.col("SENTIMENT") == "all").sort("narrative_key")
        n_head = s.shape[0]
        assert (all_rows["N_HEADLINES"] == n_head).all()
        assert (all_rows["N_LABELLED"] == n_head).all()
        # every label row matches the dense reference on its own headline subset
        masks = {"pos": np.isfinite(s) & (s > eps), "neg": np.isfinite(s) & (s < -eps)}
        if eps > 0:
            masks["neu"] = np.isfinite(s) & (np.abs(s) <= eps)
        support_sum = np.zeros(world["table"].n_narratives, dtype=np.int64)
        total_sum = np.zeros(world["table"].n_narratives)
        for lab, mask in masks.items():
            rows = nd.filter(pl.col("SENTIMENT") == lab).sort("narrative_key")
            ref_narr, _, n_lab = _dense_reference(world, cfg, 0.2, cal.mu_for(day), day, mask)
            _assert_day_matches(rows, ref_narr)
            assert (rows["N_HEADLINES"] == n_head).all() and (rows["N_LABELLED"] == n_lab).all()
            support_sum += rows["SUPPORT"].to_numpy()
            total_sum += rows["TOTAL_SCORE"].fill_null(0.0).to_numpy()
        # split rows reconcile with "all" MINUS the headlines that carry no label
        unl = ~np.isfinite(s) | (s == 0.0) if eps == 0 else ~np.isfinite(s)
        ref_unl, _, _ = _dense_reference(world, cfg, 0.2, cal.mu_for(day), day, unl)
        np.testing.assert_array_equal(support_sum + ref_unl["SUPPORT"],
                                      all_rows["SUPPORT"].to_numpy())
        np.testing.assert_allclose(total_sum + np.nan_to_num(ref_unl["TOTAL_SCORE"]),
                                   all_rows["TOTAL_SCORE"].fill_null(0.0).to_numpy(), rtol=1e-5)
        d = res.day_diagnostics.filter(pl.col("DATE") == day).row(0, named=True)
        assert d["N_WITH_SENTIMENT"] == int(sum(m.sum() for m in masks.values()))
    assert res.primitive_daily.filter(pl.col("SENTIMENT") == "pos").height > 0


def test_sentiment_all_labelled_reconciles_exactly(world):
    """With no missing sentiment and eps > 0, sum of split rows == the 'all' row."""
    rng = np.random.default_rng(6)
    sent = {d: rng.normal(size=X.shape[0]).astype(np.float32) for d, X in world["X"].items()}
    cfg = ScoringConfig(q=0.75, sentiment_split=SentimentSplit.SIGN, neutral_eps=0.2)
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    res = score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(world["X"], sentiment=sent), calibration=cal,
                      use_kernel=False)
    nd = res.narrative_daily
    split = (nd.filter(pl.col("SENTIMENT") != "all")
             .group_by(["DATE", "narrative_key"])
             .agg(pl.col("SUPPORT").sum(), pl.col("TOTAL_SCORE").sum(),
                  pl.col("N_LABELLED").sum())
             .sort(["DATE", "narrative_key"]))
    all_rows = nd.filter(pl.col("SENTIMENT") == "all").sort(["DATE", "narrative_key"])
    np.testing.assert_array_equal(split["SUPPORT"].to_numpy(), all_rows["SUPPORT"].to_numpy())
    np.testing.assert_allclose(split["TOTAL_SCORE"].fill_null(0.0).to_numpy(),
                               all_rows["TOTAL_SCORE"].fill_null(0.0).to_numpy(), rtol=1e-9)
    assert (split["N_LABELLED"] == all_rows["N_LABELLED"]).all()


def test_split_disabled_is_current_artifact_plus_all_label(world):
    sent = _sentiment_world(world)
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    off = score_dates(world["days"], ScoringConfig(q=0.75), table=world["table"],
                      primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(world["X"], sentiment=sent),
                      calibration=cal, use_kernel=False)
    on = score_dates(world["days"], ScoringConfig(q=0.75, sentiment_split=SentimentSplit.SIGN),
                     table=world["table"], primitive_embeddings=world["P"],
                     source=InMemoryHeadlineSource(world["X"], sentiment=sent),
                     calibration=cal, use_kernel=False)
    assert set(off.narrative_daily["SENTIMENT"].unique()) == {"all"}
    assert off.narrative_daily.equals(on.narrative_daily.filter(pl.col("SENTIMENT") == "all"))
    assert off.day_diagnostics["N_WITH_SENTIMENT"].sum() == 0


# ---------------------------------------------------------------------------
# Null-partition feed through the pipeline
# ---------------------------------------------------------------------------

def test_null_sink_receives_every_day_and_finalises_full_month(world, tmp_path: Path):
    rng = np.random.default_rng(3)
    month = list(date_range(date(2008, 9, 1), date(2008, 9, 30)))
    X = {d: _headlines(rng, world["table"], world["P"], 12) for d in month[::2]}   # quiet days too
    cfg = ScoringConfig(q=0.75, null_draws_per_headline=3)
    cal = FixedTauProvider(_tau(0.2, through=date(2008, 8, 31)), world["mu_df"])
    sink = MonthlyNullPartitionWriter(tmp_path / "parts", config=cfg, table=world["table"], seed=1)
    res = score_dates(month, cfg, table=world["table"], primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(X, chunk_size=5), calibration=cal,
                      use_kernel=False, null_sink=sink)
    assert len(res.skipped_days) == 15
    assert [p.name for p in sink.finalised] == ["2008-09.parquet"]
    parts = load_partitions(tmp_path / "parts")
    row = parts.row(0, named=True)
    assert row["N_HEADLINES"] == 15 * 12 and row["N_DAYS_CLOSED"] == 30
    assert row["N_DAYS_IN_MONTH"] == 30 and row["COVERAGE"] == 1.0
    assert row["N_DRAWS_AVAILABLE"] == 15 * 12 * (8 - 1)          # P=8, trim ceil(0.8)=1
    assert 0 < row["N_DRAWS_SAMPLED"] <= 15 * 12 * 3
    assert row["WELFORD_COUNT"] == row["N_DRAWS_SAMPLED"]
    assert not list((tmp_path / "parts" / "_open").glob("*.npz"))


def test_cold_start_null_only_days_feed_partitions(world, tmp_path: Path):
    """No tau row old enough: with tau_missing='null_only' the day feeds the partition and
    produces no rows; with the default it raises."""
    month = list(date_range(date(2008, 9, 1), date(2008, 9, 30)))
    X = {d: world["X"][D0] for d in month[::3]}
    cfg = ScoringConfig(q=0.75, null_draws_per_headline=3)
    late = FixedTauProvider(_tau(0.2, through=date(2008, 9, 15)), world["mu_df"])
    sink = MonthlyNullPartitionWriter(tmp_path / "p", config=cfg, table=world["table"])
    with pytest.raises(LookaheadError):
        score_dates(month[:3], cfg, table=world["table"], primitive_embeddings=world["P"],
                    source=InMemoryHeadlineSource(X), calibration=late, use_kernel=False)
    res = score_dates(month, cfg, table=world["table"], primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(X), calibration=late, use_kernel=False,
                      null_sink=sink, tau_missing="null_only")
    assert res.null_only_days == [d for d in month[::3] if d <= date(2008, 9, 15)]
    assert res.day_diagnostics["DATE"].to_list() == [d for d in month[::3] if d > date(2008, 9, 15)]
    assert res.narrative_daily["DATE"].n_unique() == res.n_days
    row = load_partitions(tmp_path / "p").row(0, named=True)
    assert row["N_HEADLINES"] == 10 * 40 and row["N_DAYS_CLOSED"] == 30   # every day closed


def test_cold_start_without_mu_skips_day_entirely(world, tmp_path: Path):
    early = FixedTauProvider(_tau(0.2, through=date(2000, 1, 1)),
                             _mu_df([D0 + timedelta(days=1)]))         # mu only from day 2
    cfg = ScoringConfig(q=0.75, null_draws_per_headline=3)
    sink = MonthlyNullPartitionWriter(tmp_path / "p", config=cfg, table=world["table"])
    res = score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(world["X"]), calibration=early,
                      use_kernel=False, null_sink=sink, tau_missing="null_only")
    assert res.n_days == 2 and D0 not in res.null_only_days
    sink.finalise_before(date(2008, 10, 1))
    row = load_partitions(tmp_path / "p").row(0, named=True)
    assert row["N_DAYS_CLOSED"] == 2 and row["N_HEADLINES"] == 65 + 90
    with pytest.raises(LookaheadError):
        score_dates([D0], cfg, table=world["table"], primitive_embeddings=world["P"],
                    source=InMemoryHeadlineSource(world["X"]), calibration=early,
                    use_kernel=False)


def test_combine_sentiment_reproduces_all_row(world):
    from narrative_scoring.validation import combine_sentiment

    rng = np.random.default_rng(6)
    sent = {d: rng.normal(size=X.shape[0]).astype(np.float32) for d, X in world["X"].items()}
    cfg = ScoringConfig(q=0.75, sentiment_split=SentimentSplit.SIGN, neutral_eps=0.2)
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    res = score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(world["X"], sentiment=sent), calibration=cal,
                      use_kernel=False, keep_primitive_daily=True)
    for df in (res.narrative_daily, res.primitive_daily):
        allc = combine_sentiment(df, ["pos", "neg", "neu"]).sort(["DATE", "narrative_key"])
        ref = df.filter(pl.col("SENTIMENT") == "all").sort(["DATE", "narrative_key"])
        if "primitive" in df.columns:
            allc, ref = allc.sort(["DATE", "primitive"]), ref.sort(["DATE", "primitive"])
        assert allc["SENTIMENT"].unique().to_list() == ["pos+neg+neu"]
        np.testing.assert_array_equal(allc["SUPPORT"].to_numpy(), ref["SUPPORT"].to_numpy())
        for col, tol in (("TOTAL_SCORE", 1e-9), ("INTENSITY", 1e-5), ("STD_SCORE", 2e-4),
                         ("PEAK", 0.0)):
            np.testing.assert_allclose(allc[col].to_numpy(), ref[col].to_numpy(), rtol=tol,
                                       atol=tol, equal_nan=True)
        assert (allc["N_HEADLINES"] == ref["N_HEADLINES"]).all()
        assert (allc["N_LABELLED"] == ref["N_LABELLED"]).all()
        assert allc.columns == ref.columns
    # an owner panel: neu + pos, checked against a direct rescore of that subset
    np_ = combine_sentiment(res.narrative_daily, ["neu", "pos"])
    s0 = sent[D0]
    mask = np.isfinite(s0) & (s0 > -0.2)
    ref_narr, _, n_lab = _dense_reference(world, cfg, 0.2, cal.mu_for(D0), D0, mask)
    day = np_.filter(pl.col("DATE") == D0).sort("narrative_key")
    _assert_day_matches(day, ref_narr)
    assert (day["N_LABELLED"] == n_lab).all()
    with pytest.raises(ValueError, match="not present"):
        combine_sentiment(res.narrative_daily, ["pos", "zzz"])
