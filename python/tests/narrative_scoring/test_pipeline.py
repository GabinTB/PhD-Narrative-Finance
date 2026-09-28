"""End-to-end: streaming vs dense reference, live == historical, point-in-time guards,
metadata, sentiment-bucket runs, day diagnostics, null-partition feed."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from narrative_scoring._kernels import HAVE_SELECT
from narrative_scoring.calibration import LookaheadError, TauRecord
from narrative_scoring.config import (
    SENTIMENT_BUCKETS,
    AggRule,
    PoolRule,
    ScoringConfig,
    SentimentFilter,
    n_candidates_for,
)
from narrative_scoring.partitions import MonthlyNullPartitionWriter, load_partitions
from narrative_scoring.pipeline import ParquetMonthWriter, date_range, score_dates
from narrative_scoring.primitives import primitive_scores, scoring_matrix
from narrative_scoring.schema import DAY_DIAGNOSTICS_SCHEMA, EMBEDDING_DIM, NARRATIVE_DAILY_SCHEMA
from narrative_scoring.streaming import InMemoryHeadlineSource, MemoryBudgetExceeded
from narrative_scoring.validation import (
    SentimentRun,
    attention,
    reference_day,
    reference_headline_narrative,
    reference_select,
    sum_sentiment_runs,
    summarize,
)
from nlp.corrections import Correction, apply_mode
from nlp.reference_vector import resolve_reference

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
        resolve_reference(world["mu_df"], D0 - timedelta(days=100))
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
                "jump_cut", "jump_min_candidates", "alpha", "trim_frac", "sentiment",
                "sentiment_rule", "neg_max", "pos_min", "min_conf", "null_draws_per_headline"):
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
# Sentiment-bucket runs (in memory: a keep mask per day plays the bucket filter)
# ---------------------------------------------------------------------------

def _bucket_labels(world, seed=5) -> dict[date, np.ndarray]:
    """One bucket per headline; the second day has no negative headline at all."""
    rng = np.random.default_rng(seed)
    names = np.array([b.value for b in SENTIMENT_BUCKETS])
    out = {}
    for i, (d, X) in enumerate(sorted(world["X"].items())):
        lab = names[rng.integers(0, 4, X.shape[0])]
        if i == 1:
            lab[lab == "negative"] = "neutral"
        out[d] = lab
    return out


def _bucket_cfg(bucket, **kw) -> ScoringConfig:
    return ScoringConfig(q=0.75, sentiment=bucket, sentiment_source="ravenbert", **kw)


def _bucket_runs(world, cal, *, use_kernel=False, keep_primitive_daily=False, labels=None,
                 tau_missing="raise"):
    labels = labels or _bucket_labels(world)
    runs = {}
    for b in SENTIMENT_BUCKETS:
        keep = {d: lab == b.value for d, lab in labels.items()}
        runs[b] = score_dates(world["days"], _bucket_cfg(b), table=world["table"],
                              primitive_embeddings=world["P"],
                              source=InMemoryHeadlineSource(world["X"], chunk_size=11, keep=keep),
                              calibration=cal, use_kernel=use_kernel,
                              keep_primitive_daily=keep_primitive_daily,
                              sentiment_artifact_id="hs-1", tau_missing=tau_missing)
    return runs, labels


@pytest.mark.parametrize("use_kernel", [False, KERNEL])
def test_bucket_runs_add_up_to_the_all_headlines_run(world, use_kernel):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    full = score_dates(world["days"], ScoringConfig(q=0.75), table=world["table"],
                       primitive_embeddings=world["P"],
                       source=InMemoryHeadlineSource(world["X"], chunk_size=11), calibration=cal,
                       use_kernel=use_kernel, keep_primitive_daily=True)
    runs, labels = _bucket_runs(world, cal, use_kernel=use_kernel, keep_primitive_daily=True)
    for primitive in (False, True):
        key = "primitive" if primitive else "narrative_key"
        total = sum_sentiment_runs([SentimentRun.from_result(r, primitive=primitive)
                                    for r in runs.values()]).sort(["DATE", key])
        ref = (full.primitive_daily if primitive else full.narrative_daily).sort(["DATE", key])
        assert total.columns == ref.columns and total.height == ref.height
        np.testing.assert_array_equal(total["SUPPORT"].to_numpy(), ref["SUPPORT"].to_numpy())
        # a headline's float32 scores depend slightly (~1e-7) on the rows batched with it
        # (BLAS blocking): the statistics agree to float rounding; SUPPORT is exact here
        for col, tol in (("TOTAL_SCORE", 1e-6), ("INTENSITY", 1e-5), ("STD_SCORE", 2e-4),
                         ("PEAK", 1e-6)):
            np.testing.assert_allclose(total[col].to_numpy(), ref[col].to_numpy(), rtol=tol,
                                       atol=tol, equal_nan=True)
        assert (total["N_HEADLINES"] == ref["N_HEADLINES"]).all()
        assert (total["N_LABELLED"] == ref["N_LABELLED"]).all()
        assert total["SENTIMENT"].unique().to_list() == ["all"]
    # attention adds up: each bucket run shares the day's total as denominator
    att = sum(attention(r.narrative_daily).sort(["DATE", "narrative_key"])["ATTENTION"]
              .fill_null(0.0).to_numpy() for r in runs.values())
    np.testing.assert_allclose(att, attention(full.narrative_daily).sort(
        ["DATE", "narrative_key"])["ATTENTION"].fill_null(0.0).to_numpy(), rtol=1e-6, atol=1e-12)
    # the same day set, every day with the day's total; the empty bucket day is written
    days = sorted(world["days"])
    for b, r in runs.items():
        d = r.day_diagnostics.sort("DATE")
        assert d["DATE"].to_list() == full.day_diagnostics["DATE"].sort().to_list()
        assert d["N_HEADLINES"].to_list() == [world["X"][x].shape[0] for x in days]
        assert d["N_SCORED"].to_list() == [int((labels[x] == b.value).sum()) for x in days]
        assert (d["N_WITH_SENTIMENT"] == d["N_SCORED"]).all()
        assert r.narrative_daily["SENTIMENT"].unique().to_list() == [b.value]
    neg = runs[SentimentFilter.NEGATIVE].narrative_daily.filter(pl.col("DATE") == days[1])
    assert neg.height == world["table"].n_narratives and (neg["SUPPORT"] == 0).all()
    assert neg["TOTAL_SCORE"].null_count() == neg.height and (neg["N_LABELLED"] == 0).all()


def test_a_day_without_mu_is_skipped_by_every_run(world):
    days = sorted(world["days"])
    late_mu = world["mu_df"].filter(pl.col("DATE") >= days[1])       # day 0 cannot be corrected
    cal = FixedTauProvider(_tau(0.2), late_mu)
    full = score_dates(days, ScoringConfig(q=0.75), table=world["table"],
                       primitive_embeddings=world["P"], source=InMemoryHeadlineSource(world["X"]),
                       calibration=cal, use_kernel=False, tau_missing="null_only")
    runs, _ = _bucket_runs(world, cal, tau_missing="null_only")
    want = full.day_diagnostics["DATE"].to_list()
    assert days[0] not in want and len(want) == len(days) - 1
    for r in runs.values():
        assert r.day_diagnostics["DATE"].to_list() == want


def test_a_bucket_run_never_feeds_null_partitions(world, tmp_path):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    sink = MonthlyNullPartitionWriter(tmp_path, config=_bucket_cfg(SentimentFilter.POSITIVE),
                                      table=world["table"], seed=0)
    with pytest.raises(ValueError, match="never feeds the null partitions"):
        score_dates(world["days"], _bucket_cfg(SentimentFilter.POSITIVE), table=world["table"],
                    primitive_embeddings=world["P"], source=InMemoryHeadlineSource(world["X"]),
                    calibration=cal, null_sink=sink, use_kernel=False)


def test_sentiment_config_validation_and_identity():
    with pytest.raises(ValueError, match="only meaningful"):
        ScoringConfig(sentiment_source="finbert")
    with pytest.raises(ValueError, match="sentiment_source must be one of"):
        ScoringConfig(sentiment="positive")
    with pytest.raises(ValueError, match="neg_max < pos_min"):
        _bucket_cfg("positive", neg_max=0.2, pos_min=0.2)
    with pytest.raises(ValueError, match="neg_max < pos_min"):
        _bucket_cfg("positive", neg_max=-1.5)
    with pytest.raises(ValueError, match="model sources only"):
        ScoringConfig(sentiment="positive", sentiment_source="ravenpack", sentiment_rule="css",
                      min_conf=0.2)
    with pytest.raises(ValueError, match="sentiment_rule"):
        ScoringConfig(sentiment="positive", sentiment_source="ravenpack")      # rule mean
    with pytest.raises(ValueError, match="sentiment_rule"):
        _bucket_cfg("positive", sentiment_rule="css")
    # a no-filter config hashes exactly as before the filter existed
    import hashlib
    import json

    legacy = {**ScoringConfig().to_dict(), "sentiment_split": "none", "neutral_eps": 0.0,
              "sentiment_source": "none", "sentiment_column": ""}
    for k in ("sentiment", "sentiment_rule", "neg_max", "pos_min", "min_conf", "label",
              "mask_bipolar"):
        legacy.pop(k)
    want = hashlib.sha1(json.dumps(legacy, sort_keys=True).encode()).hexdigest()[:16]
    assert ScoringConfig().digest() == want
    old_form = {**ScoringConfig().to_dict(), "sentiment_split": "none", "neutral_eps": 0.0,
                "sentiment_column": ""}
    for k in ("sentiment", "sentiment_rule", "neg_max", "pos_min", "min_conf", "mask_bipolar"):
        old_form.pop(k)
    assert ScoringConfig.from_dict(old_form).digest() == want
    with pytest.raises(ValueError, match="removed in-run sentiment split"):
        ScoringConfig.from_dict({**old_form, "sentiment_split": "sign"})
    # buckets differ in config_id, not in the filter-free digest nor in the null model
    cfgs = [_bucket_cfg(b) for b in SENTIMENT_BUCKETS]
    assert len({c.digest() for c in cfgs}) == 4
    assert len({c.digest_without_filter() for c in cfgs}) == 1
    assert {c.f0_digest() for c in cfgs} == {ScoringConfig().f0_digest()}
    assert _bucket_cfg("positive", min_conf=0.1).digest_without_filter() != \
        cfgs[0].digest_without_filter()


def test_bucket_identity_in_metadata_and_diagnostics(world):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    runs, _ = _bucket_runs(world, cal)
    r = runs[SentimentFilter.NEGATIVE]
    assert r.metadata.sentiment_artifact_id == "hs-1"
    assert r.metadata.config["sentiment"] == "negative"
    assert r.day_diagnostics["SENTIMENT_SOURCE_ID"].unique().to_list() == ["hs-1:mean:negative"]
    off = score_dates(world["days"], ScoringConfig(q=0.75), table=world["table"],
                      primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(world["X"]), calibration=cal,
                      use_kernel=False)
    assert off.day_diagnostics["SENTIMENT_SOURCE_ID"].null_count() == off.n_days
    assert (off.day_diagnostics["N_SCORED"] == off.day_diagnostics["N_HEADLINES"]).all()
    assert off.metadata.sentiment_artifact_id is None


def test_sum_sentiment_runs_checks_that_the_runs_belong_together(world):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    runs, _ = _bucket_runs(world, cal)
    good = [SentimentRun.from_result(r) for r in runs.values()]
    sum_sentiment_runs(good)

    def swap(i, **kw):
        out = list(good)
        out[i] = SentimentRun(**{**good[i].__dict__, **kw})
        return out

    with pytest.raises(ValueError, match="exactly once"):
        sum_sentiment_runs(good[:3])
    with pytest.raises(ValueError, match="exactly once"):
        sum_sentiment_runs(good[:3] + [good[0]])
    with pytest.raises(ValueError, match="sentiment artifact"):
        sum_sentiment_runs(swap(1, sentiment_artifact_id="hs-other"))
    other_rule = ScoringConfig(**{**good[2].config.to_dict(), "sentiment_rule": "median"})
    with pytest.raises(ValueError, match="sentiment_rule"):
        sum_sentiment_runs(swap(2, config=other_rule))
    other_q = ScoringConfig(**{**good[2].config.to_dict(), "q": 0.9})
    with pytest.raises(ValueError, match="bucket removed"):
        sum_sentiment_runs(swap(2, config=other_q))
    masked = ScoringConfig(**{**good[1].config.to_dict(), "mask_bipolar": True})
    with pytest.raises(ValueError, match="mask_bipolar"):
        sum_sentiment_runs(swap(1, config=masked))
    with pytest.raises(ValueError, match="tau_asof"):
        sum_sentiment_runs(swap(0, tau_asof_id="tau_other"))
    with pytest.raises(ValueError, match="mu_asof"):
        sum_sentiment_runs(swap(3, mu_asof_id="mu_other"))
    bumped = good[1].frame.with_columns(
        pl.when(pl.col("DATE") == D0).then(pl.col("N_HEADLINES") + 1)
        .otherwise(pl.col("N_HEADLINES")).alias("N_HEADLINES"))
    with pytest.raises(ValueError, match="N_HEADLINES differs"):
        sum_sentiment_runs(swap(1, frame=bumped))
    with pytest.raises(ValueError, match="day sets"):
        sum_sentiment_runs(swap(1, frame=good[1].frame.filter(pl.col("DATE") != D0)))
    # a missing (zero-support) row in one run is treated as zero: the sum is unchanged
    sparse = good[0].frame.filter(pl.col("SUPPORT") > 0)
    assert sparse.height < good[0].frame.height
    a = sum_sentiment_runs(good).sort(["DATE", "narrative_key"])
    b = sum_sentiment_runs(swap(0, frame=sparse)).sort(["DATE", "narrative_key"])
    assert a.equals(b)


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
