"""End-to-end: streaming vs dense reference, live == historical, point-in-time guards,
metadata, sentiment tags, asset layer, day diagnostics, null-partition feed."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from narrative_scoring._kernels import HAVE_SELECT
from narrative_scoring.assets import AssetDay, AssetUniverse
from narrative_scoring.calibration import LookaheadError, TauRecord
from narrative_scoring.config import (
    AggRule,
    PoolRule,
    ScoringConfig,
    n_candidates_for,
)
from narrative_scoring.partitions import MonthlyNullPartitionWriter, load_partitions
from narrative_scoring.pipeline import ParquetMonthWriter, date_range, score_dates
from narrative_scoring.primitives import primitive_scores, scoring_matrix
from narrative_scoring.schema import DAY_DIAGNOSTICS_SCHEMA, EMBEDDING_DIM, NARRATIVE_DAILY_SCHEMA
from narrative_scoring.streaming import InMemoryHeadlineSource, MemoryBudgetExceeded
from narrative_scoring.tags import SentimentTags
from narrative_scoring.validation import (
    attention,
    reference_day,
    reference_headline_narrative,
    reference_select,
    sum_tags,
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
    assert d["N_UNTAGGED"].sum() == 0 and d["SENTIMENT"].unique().to_list() == ["all"]


def test_metadata_is_complete(world):
    cal = FixedTauProvider(_tau(0.25), world["mu_df"], mu_asof_id="mu_asof__abc")
    cfg = ScoringConfig(q=0.75, jump_cut=True, jump_min_candidates=4, label="x")
    res = score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                      source=InMemoryHeadlineSource(world["X"]), calibration=cal,
                      use_kernel=False, seed=7, code_version="deadbeef")
    m = res.metadata.to_dict()
    for key in ("mode", "paraphrase_style", "paraphrase_pooling", "q", "narrative_agg",
                "jump_cut", "jump_min_candidates", "alpha", "trim_frac", "tags",
                "mask_bipolar", "null_draws_per_headline"):
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
# Sentiment tags: one pass, one label per tag (in memory: codes per day play the source)
# ---------------------------------------------------------------------------

TERNARY = [("neg", "x <= -1/3"), ("neu", "-1/3 < x < 1/3"), ("pos", "x >= 1/3")]


def _tag_cfg(tags=TERNARY, **kw) -> ScoringConfig:
    return ScoringConfig(q=0.75, tags=SentimentTags.build("ravenbert", "mean", tags), **kw)


def _codes(world, seed=5, n_tags=3, untagged=True) -> dict[date, np.ndarray]:
    """A tag code per headline (-1 sometimes); the second day has no tag-0 headline."""
    rng = np.random.default_rng(seed)
    out = {}
    for i, (d, X) in enumerate(sorted(world["X"].items())):
        c = rng.integers(-1 if untagged else 0, n_tags, X.shape[0]).astype(np.int8)
        if i == 1:
            c[c == 0] = 1
        out[d] = c
    return out


def _run(world, cfg, cal, codes=None, **kw):
    kw.setdefault("use_kernel", False)
    return score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                       source=InMemoryHeadlineSource(world["X"], chunk_size=11, tags=codes),
                       calibration=cal, **kw)


@pytest.mark.parametrize("use_kernel", [False, KERNEL])
def test_one_tag_covering_every_headline_is_the_untagged_run(world, use_kernel):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    plain = _run(world, ScoringConfig(q=0.75), cal, use_kernel=use_kernel,
                 keep_primitive_daily=True)
    zeros = {d: np.zeros(X.shape[0], dtype=np.int8) for d, X in world["X"].items()}
    one = _run(world, _tag_cfg([("every", "x >= -2")]), cal, zeros, use_kernel=use_kernel,
               keep_primitive_daily=True)
    for a, b in ((plain.narrative_daily, one.narrative_daily),
                 (plain.primitive_daily, one.primitive_daily)):
        assert b["SENTIMENT"].unique().to_list() == ["every"]
        assert a.drop("SENTIMENT").equals(b.drop("SENTIMENT"))           # bit for bit
    keep = [c for c in plain.day_diagnostics.columns
            if c not in ("SENTIMENT", "CONFIG_ID", "SENTIMENT_SOURCE_ID", "RSS_GB_BEFORE",
                         "RSS_GB_AFTER", "SECONDS")]
    assert plain.day_diagnostics.select(keep).equals(one.day_diagnostics.select(keep))


@pytest.mark.parametrize("use_kernel", [False, KERNEL])
def test_labels_add_up_to_the_untagged_run(world, use_kernel):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    full = _run(world, ScoringConfig(q=0.75), cal, use_kernel=use_kernel,
                keep_primitive_daily=True)
    codes = _codes(world, untagged=False)
    tagged = _run(world, _tag_cfg(), cal, codes, use_kernel=use_kernel,
                  keep_primitive_daily=True)
    for primitive in (False, True):
        key = "primitive" if primitive else "narrative_key"
        frame = tagged.primitive_daily if primitive else tagged.narrative_daily
        total = sum_tags(frame, tagged.day_diagnostics).sort(["DATE", key])
        ref = (full.primitive_daily if primitive else full.narrative_daily).sort(["DATE", key])
        assert total.columns == ref.columns and total.height == ref.height
        np.testing.assert_array_equal(total["SUPPORT"].to_numpy(), ref["SUPPORT"].to_numpy())
        # same S rows in both runs: only the summation order differs
        for col, tol in (("TOTAL_SCORE", 1e-12), ("INTENSITY", 1e-6), ("STD_SCORE", 1e-4),
                         ("PEAK", 0.0)):
            np.testing.assert_allclose(total[col].to_numpy(), ref[col].to_numpy(), rtol=tol,
                                       atol=tol, equal_nan=True)
        assert (total["N_HEADLINES"] == ref["N_HEADLINES"]).all()
        assert (total["N_LABELLED"] == ref["N_LABELLED"]).all()
    days = sorted(world["days"])
    d = tagged.day_diagnostics.sort("DATE", "SENTIMENT")
    assert d.height == 3 * len(days) and set(d["SENTIMENT"]) == {"neg", "neu", "pos"}
    for x in days:
        rows = d.filter(pl.col("DATE") == x)
        assert (rows["N_HEADLINES"] == world["X"][x].shape[0]).all()
        assert dict(zip(rows["SENTIMENT"], rows["N_SCORED"])) == {
            name: int((codes[x] == t).sum()) for t, (name, _) in enumerate(TERNARY)}
        assert (rows["N_UNTAGGED"] == 0).all()
    # a label with no headline that day is still written (SUPPORT 0, null statistics)
    neg = tagged.narrative_daily.filter((pl.col("DATE") == days[1])
                                        & (pl.col("SENTIMENT") == "neg"))
    assert neg.height == world["table"].n_narratives and (neg["SUPPORT"] == 0).all()
    assert neg["TOTAL_SCORE"].null_count() == neg.height and (neg["N_LABELLED"] == 0).all()


@pytest.mark.skipif(not HAVE_SELECT, reason="kernel not built")
@pytest.mark.parametrize("mask", [False, True])
def test_kernel_and_numpy_route_tags_alike(world, mask):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    cfg = _tag_cfg(mask_bipolar=mask, jump_cut=True, jump_min_candidates=1)
    codes = _codes(world)
    a = _run(world, cfg, cal, codes, use_kernel=False, keep_primitive_daily=True)
    b = _run(world, cfg, cal, codes, use_kernel=True, keep_primitive_daily=True)
    for fa, fb in ((a.narrative_daily, b.narrative_daily), (a.primitive_daily, b.primitive_daily)):
        np.testing.assert_array_equal(fa["SUPPORT"].to_numpy(), fb["SUPPORT"].to_numpy())
        for col in ("TOTAL_SCORE", "INTENSITY", "STD_SCORE", "PEAK"):
            np.testing.assert_allclose(fa[col].to_numpy(), fb[col].to_numpy(), rtol=1e-6,
                                       equal_nan=True)
    for col in ("N_SCORED", "N_UNTAGGED", "N_UNASSIGNED", "N_Q_CANDIDATES",
                "N_F0_SURVIVORS_PRE_Q", "N_RETAINED_PRE_JUMP", "N_RETAINED_POST_Q_TAU",
                "N_POLE_MASKED", "N_MASK_CHANGED_RETENTION"):
        assert a.day_diagnostics[col].to_list() == b.day_diagnostics[col].to_list(), col


@pytest.mark.parametrize("use_kernel", [False, KERNEL])
def test_untagged_headlines_feed_the_null_model_only(world, tmp_path, use_kernel):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    codes = _codes(world)
    sinks = {}
    for name, cfg, c in (("plain", ScoringConfig(q=0.75), None), ("tagged", _tag_cfg(), codes)):
        sinks[name] = MonthlyNullPartitionWriter(tmp_path / name, config=cfg,
                                                 table=world["table"], seed=3)
        r = _run(world, cfg, cal, c, use_kernel=use_kernel, null_sink=sinks[name])
        sinks[name].finalise_before(date(2008, 10, 1))
        if name == "tagged":
            tagged = r
    # the same draws from the same rows, whatever the tags (RSS is telemetry)
    assert load_partitions(tmp_path / "plain").drop("RSS_PEAK_GB").equals(
        load_partitions(tmp_path / "tagged").drop("RSS_PEAK_GB"))
    d = tagged.day_diagnostics
    for x in world["days"]:
        rows = d.filter(pl.col("DATE") == x)
        n_untagged = int((codes[x] < 0).sum())
        assert n_untagged > 0 and (rows["N_UNTAGGED"] == n_untagged).all()
        assert rows["N_SCORED"].sum() + n_untagged == world["X"][x].shape[0]
    total = sum_tags(tagged.narrative_daily, d)            # explicit, checked gap
    assert (total["N_LABELLED"] < total["N_HEADLINES"]).all()


def test_tags_in_metadata_and_diagnostics(world):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    r = _run(world, _tag_cfg(), cal, _codes(world), sentiment_artifact_id="hs-1")
    assert r.metadata.sentiment_artifact_id == "hs-1"
    assert r.metadata.config["tags"]["rule"] == "mean"
    assert r.day_diagnostics["SENTIMENT_SOURCE_ID"].unique().to_list() == ["hs-1:mean"]
    off = _run(world, ScoringConfig(q=0.75), cal)
    assert off.day_diagnostics["SENTIMENT_SOURCE_ID"].null_count() == off.n_days
    assert (off.day_diagnostics["N_SCORED"] == off.day_diagnostics["N_HEADLINES"]).all()
    assert off.day_diagnostics["SENTIMENT"].unique().to_list() == ["all"]
    assert off.metadata.sentiment_artifact_id is None


def test_a_tagged_config_needs_tag_codes(world):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    with pytest.raises(ValueError, match="gives none"):
        _run(world, _tag_cfg(), cal, None)


def test_tag_config_identity():
    import hashlib
    import json

    # a config without tags hashes exactly as before any sentiment layer existed
    legacy = {**ScoringConfig().to_dict(), "sentiment_split": "none", "neutral_eps": 0.0,
              "sentiment_source": "none", "sentiment_column": ""}
    for k in ("tags", "label", "mask_bipolar"):
        legacy.pop(k)
    want = hashlib.sha1(json.dumps(legacy, sort_keys=True).encode()).hexdigest()[:16]
    assert ScoringConfig().digest() == want
    split_form = {**{k: v for k, v in ScoringConfig().to_dict().items() if k != "tags"},
                  "sentiment_split": "none", "neutral_eps": 0.0, "sentiment_column": ""}
    split_form.pop("mask_bipolar")
    assert ScoringConfig.from_dict(split_form).digest() == want
    bucket_form = {**{k: v for k, v in ScoringConfig().to_dict().items() if k != "tags"},
                   "sentiment": "none", "sentiment_source": "none", "sentiment_rule": "mean",
                   "neg_max": -1 / 3, "pos_min": 1 / 3, "min_conf": 0.0}
    assert ScoringConfig.from_dict(bucket_form).digest() == want
    with pytest.raises(ValueError, match="removed in-run sentiment split"):
        ScoringConfig.from_dict({**split_form, "sentiment_split": "sign"})
    with pytest.raises(ValueError, match="single-bucket"):
        ScoringConfig.from_dict({**bucket_form, "sentiment": "negative"})
    # the tags enter config_id, never the null model
    tagged = _tag_cfg()
    assert tagged.digest() != ScoringConfig(q=0.75).digest()
    assert tagged.f0_digest() == ScoringConfig().f0_digest()
    assert tagged.warmup_digest() == ScoringConfig().warmup_digest()
    assert ScoringConfig.from_dict(tagged.to_dict()) == tagged
    assert _tag_cfg([("a", "x < 0"), ("b", "x >= 0")]).digest() != tagged.digest()
    with pytest.raises(ValueError, match="grid table"):
        ScoringConfig(tags=SentimentTags.build("ravenpack", "mean", TERNARY))


# ---------------------------------------------------------------------------
# Asset layer
# ---------------------------------------------------------------------------

def _universe(n_assets=4) -> AssetUniverse:
    frame = pl.DataFrame({
        "snapshot_date": ([date(2008, 9, 1)] * n_assets + [date(2008, 9, 16)] * 2
                          + [date(2008, 9, 1)]),
        "rp_entity_id": [f"E{i}" for i in range(n_assets)] + ["E0", "E2", None]})
    return AssetUniverse.from_frame(frame, artifact_id="u-1", min_relevance=0.75)


def _assets(world, n_assets=4, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    for d, X in world["X"].items():
        rows = []
        for _ in range(X.shape[0]):
            k = rng.integers(0, 3)
            a = sorted(rng.choice(n_assets, size=k, replace=False).tolist())
            rows.append([(int(i), int(rng.integers(0, 101))) for i in a])
        out[d] = rows
    return out


def _run_assets(world, cfg, cal, codes, assets, use_kernel=False):
    return score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                       source=InMemoryHeadlineSource(world["X"], chunk_size=11, tags=codes,
                                                     assets=assets),
                       calibration=cal, use_kernel=use_kernel, assets=_universe())


@pytest.mark.parametrize("use_kernel", [False, KERNEL])
def test_asset_attention_counts(world, use_kernel):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    codes, assets = _codes(world), _assets(world)
    r = _run_assets(world, _tag_cfg(), cal, codes, assets, use_kernel)
    att = r.asset_attention
    assert att is not None and att.height > 0
    for x in world["days"]:
        for t, (name, _) in enumerate(TERNARY):
            for a in range(4):
                rels = [w for row, c in zip(assets[x], codes[x]) if c == t
                        for i, w in row if i == a]
                got = att.filter((pl.col("DATE") == x) & (pl.col("SENTIMENT") == name)
                                 & (pl.col("RP_ENTITY_ID") == f"E{a}"))
                if not rels:
                    assert got.height == 0                      # sparse: no row
                    continue
                row = got.row(0, named=True)
                assert row["N_STORIES"] == len(rels)
                assert row["N_STORIES_REL"] == sum(w >= 75 for w in rels)   # 0.75
                assert row["REL_SUM"] == pytest.approx(sum(rels) / 100)
                assert row["N_HEADLINES"] == world["X"][x].shape[0]
    # membership as of the day: E1, E3 leave at the 2008-09-16 snapshot
    first, later = sorted(world["days"])[0], sorted(world["days"])[1]
    inu = att.filter(pl.col("RP_ENTITY_ID") == "E1").select("DATE", "IN_UNIVERSE").unique()
    assert dict(inu.iter_rows()).get(first) is True
    assert dict(inu.iter_rows()).get(later, False) is False
    unmapped = dict(r.day_diagnostics.select("DATE", "N_UNMAPPED").unique().iter_rows())
    assert unmapped[first] == 1 and unmapped[later] == 0      # the null row is in snapshot 1


@pytest.mark.parametrize("use_kernel", [False, KERNEL])
def test_an_asset_on_every_headline_reproduces_narrative_daily(world, use_kernel):
    """Relevance 100 on every headline: narrative x asset is narrative_daily, in all three
    column groups (plain, >= min_relevance, relevance-weighted with w = 1)."""
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    every = {d: [[(0, 100)] for _ in range(X.shape[0])] for d, X in world["X"].items()}
    r = _run_assets(world, _tag_cfg(), cal, _codes(world), every, use_kernel)
    nd = r.narrative_daily.filter(pl.col("SUPPORT") > 0).rename(
        {c: f"nd_{c}" for c in ("SUPPORT", "TOTAL_SCORE", "INTENSITY", "STD_SCORE", "PEAK")})
    cross = r.narrative_asset.filter(pl.col("RP_ENTITY_ID") == "E0")
    j = nd.join(cross, on=["DATE", "SENTIMENT", "narrative_key"], how="full", coalesce=True)
    assert j.height == nd.height == cross.height
    for col in ("SUPPORT", "SUPPORT_REL"):
        np.testing.assert_array_equal(j[col].to_numpy(), j["nd_SUPPORT"].to_numpy())
    for col in ("TOTAL_SCORE", "TOTAL_SCORE_REL", "TOTAL_SCORE_RELW"):
        np.testing.assert_allclose(j[col].to_numpy(), j["nd_TOTAL_SCORE"].to_numpy(), rtol=1e-12)
    for col in ("PEAK", "PEAK_REL"):
        np.testing.assert_array_equal(j[col].to_numpy(), j["nd_PEAK"].to_numpy())
    n = j["SUPPORT"].to_numpy().astype(np.float64)
    sumsq = n * (j["nd_STD_SCORE"].cast(pl.Float64).to_numpy() ** 2
                 + j["nd_INTENSITY"].cast(pl.Float64).to_numpy() ** 2)
    for col in ("SUMSQ", "SUMSQ_REL", "SUMSQ_RELW"):
        np.testing.assert_allclose(j[col].to_numpy(), sumsq, rtol=1e-5)


def test_relevance_columns_follow_the_threshold(world):
    """*_REL columns keep only headlines with RELEVANCE >= min_relevance; *_RELW weight
    by RELEVANCE / 100; plain columns keep every headline."""
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    rels = {d: [[(0, 60 if i % 2 else 59)] for i in range(X.shape[0])]
            for d, X in world["X"].items()}
    cfg = _tag_cfg([("every", "x >= -2")])
    zeros = {d: np.zeros(X.shape[0], dtype=np.int8) for d, X in world["X"].items()}
    frame = pl.DataFrame({"snapshot_date": [date(2008, 9, 1)], "rp_entity_id": ["E0"]})
    r = score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                    source=InMemoryHeadlineSource(world["X"], chunk_size=11, tags=zeros,
                                                  assets=rels),
                    calibration=cal, use_kernel=False,
                    assets=AssetUniverse.from_frame(frame, artifact_id="u", min_relevance=0.6))
    att = r.asset_attention
    for x in world["days"]:
        row = att.filter(pl.col("DATE") == x).row(0, named=True)
        n = world["X"][x].shape[0]
        assert row["N_STORIES"] == n and row["N_STORIES_REL"] == n // 2
        assert row["REL_SUM"] == pytest.approx((n // 2) * 0.60 + (n - n // 2) * 0.59)
    c = r.narrative_asset
    assert (c["SUPPORT_REL"] <= c["SUPPORT"]).all()
    assert (c["TOTAL_SCORE_REL"] <= c["TOTAL_SCORE"] + 1e-12).all()
    assert (c.filter(pl.col("SUPPORT_REL") == 0)["PEAK_REL"].null_count()
            == c.filter(pl.col("SUPPORT_REL") == 0).height)
    np.testing.assert_allclose(c["TOTAL_SCORE_RELW"].to_numpy(),
                               0.59 * c["TOTAL_SCORE"].to_numpy()
                               + 0.01 * c["TOTAL_SCORE_REL"].to_numpy(), rtol=1e-9)


@pytest.mark.skipif(not HAVE_SELECT, reason="kernel not built")
def test_kernel_and_numpy_asset_tables_agree(world):
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    codes, assets = _codes(world), _assets(world)
    a = _run_assets(world, _tag_cfg(mask_bipolar=True), cal, codes, assets, False)
    b = _run_assets(world, _tag_cfg(mask_bipolar=True), cal, codes, assets, True)
    assert a.asset_attention.equals(b.asset_attention)
    ka, kb = a.narrative_asset, b.narrative_asset
    assert ka.select("DATE", "SENTIMENT", "narrative_key", "RP_ENTITY_ID", "SUPPORT",
                     "SUPPORT_REL").equals(kb.select("DATE", "SENTIMENT", "narrative_key",
                                                     "RP_ENTITY_ID", "SUPPORT", "SUPPORT_REL"))
    for col in ("TOTAL_SCORE", "SUMSQ", "PEAK", "TOTAL_SCORE_REL", "SUMSQ_REL", "PEAK_REL",
                "TOTAL_SCORE_RELW", "SUMSQ_RELW"):
        np.testing.assert_allclose(ka[col].to_numpy(), kb[col].to_numpy(), rtol=1e-6)


def test_a_story_counts_once_per_asset_at_its_max_relevance():
    """Three AAPL events in one story: one (story, asset) entry, relevance = max."""
    from narrative_scoring.assets import headline_assets

    u = AssetUniverse.from_frame(pl.DataFrame({"snapshot_date": [date(2008, 1, 1)] * 2,
                                               "rp_entity_id": ["AAPL", "MSFT"]}),
                                 artifact_id="u")
    rows = pl.DataFrame({
        "_r": [0, 1],
        "RP_ENTITY_ID": [["AAPL", "AAPL", "USPL", "AAPL"], ["MSFT", "XXXX"]],
        "RELEVANCE": [[40, 100, 20, 70], [55, 100]],
    }, schema={"_r": pl.UInt32, "RP_ENTITY_ID": pl.List(pl.String),
               "RELEVANCE": pl.List(pl.UInt8)})
    indptr, asset, rel = headline_assets(rows, u.asset_map(), 2)
    assert indptr.tolist() == [0, 1, 2]
    assert asset.tolist() == [0, 1] and rel.tolist() == [100, 55]
    day = AssetDay(1, 3, 2, 0.75)
    day.add_attention(np.zeros(2, dtype=np.int8), indptr, asset, rel)
    assert day.n_stories.tolist() == [[1, 1]] and day.n_rel.tolist() == [[1, 0]]
    day.add_cross(np.zeros(2, dtype=np.int8), np.array([2, 0], dtype=np.int32),
                  np.array([[0, 2, -1], [-1, -1, -1]], dtype=np.int32),
                  np.array([[0.5, 0.25, np.nan], [np.nan] * 3]), indptr, asset, rel)
    keys, sums, peaks = day.cross()
    assert keys.size == 2 and sums[:, 0].tolist() == [1.0, 1.0]       # once per narrative
    assert peaks[:, 0].tolist() == [0.5, 0.25]


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


@pytest.mark.parametrize("use_kernel", [False, KERNEL])
def test_null_partitions_across_chunk_sizes(world, tmp_path, use_kernel):
    """The draw stream is one RNG per period and columns are drawn before rejection, so
    the sampled columns do not depend on chunk boundaries; only S's BLAS rounding does,
    which can move a kept/rejected draw exactly at the trim threshold (none here)."""
    cfg = ScoringConfig(q=0.75, null_draws_per_headline=5)
    cal = FixedTauProvider(_tau(0.2), world["mu_df"])
    parts = {}
    for cs in (7, 1000):
        sink = MonthlyNullPartitionWriter(tmp_path / str(cs), config=cfg, table=world["table"],
                                          seed=2)
        score_dates(world["days"], cfg, table=world["table"], primitive_embeddings=world["P"],
                    source=InMemoryHeadlineSource(world["X"], chunk_size=cs), calibration=cal,
                    use_kernel=use_kernel, null_sink=sink)
        sink.finalise_before(date(2008, 10, 1))
        parts[cs] = load_partitions(tmp_path / str(cs)).row(0, named=True)
    a, b = parts[7], parts[1000]
    for key in ("N_HEADLINES", "N_DRAWS_AVAILABLE", "N_DRAWS_SAMPLED", "WELFORD_COUNT"):
        assert a[key] == b[key], key
    assert a["WELFORD_MEAN"] == pytest.approx(b["WELFORD_MEAN"], rel=1e-6)
    assert a["WELFORD_M2"] == pytest.approx(b["WELFORD_M2"], rel=1e-5)
