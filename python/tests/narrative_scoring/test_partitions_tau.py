"""Monthly null partitions, the self-contained tau_asof build, its reader, the warmup step."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from narrative_scoring.calibration import LookaheadError, resolve_mu_asof
from narrative_scoring.config import ScoringConfig
from narrative_scoring.corrections import Correction
from narrative_scoring.f0 import TDigest, gaussian_tau, tail_probability
from narrative_scoring.partitions import (
    MonthlyNullPartitionWriter,
    load_partitions,
    month_end,
    month_range_days,
    partition_digest,
    partition_welford,
)
from narrative_scoring.schema import F0_PARTITION_SCHEMA, TAU_ASOF_SCHEMA
from narrative_scoring.tau_asof import (
    TauSeriesProvider,
    build_tau_rows,
    latest_cutoff,
    window_offset,
)
from narrative_scoring.warmup import compute_warmup

from .test_pipeline import _mu_df

MU_DF = _mu_df([date(2007, 12, 1), date(2008, 6, 1)])


def _fill_month(sink, y, m, rng, draws_per_day=2000, skip=()):
    for d in month_range_days(y, m):
        if d in skip:
            continue
        draws = rng.normal(0.25, 0.08, size=draws_per_day)
        sink.add_draws(d, draws, n_available=draws_per_day * 10, n_headlines=10)
        sink.close_day(d, 0.5)


@pytest.fixture
def cfg():
    return ScoringConfig(q=0.75, null_draws_per_headline=4, min_month_draws=1)


class TestPartitionWriter:
    def test_finalises_when_month_completes(self, toy_table, cfg, tmp_path: Path):
        sink = MonthlyNullPartitionWriter(tmp_path, config=cfg, table=toy_table, seed=1)
        rng = np.random.default_rng(0)
        _fill_month(sink, 2008, 9, rng, skip={date(2008, 9, 30)})
        assert sink.finalised == []
        assert (tmp_path / "_open" / "2008-09.npz").exists()      # persisted after every day
        sink.close_day(date(2008, 9, 30), 0.7)                    # the quiet last day closes it
        assert [p.name for p in sink.finalised] == ["2008-09.parquet"]
        assert not (tmp_path / "_open" / "2008-09.npz").exists()
        parts = load_partitions(tmp_path)
        assert parts.schema == F0_PARTITION_SCHEMA and parts.height == 1
        row = parts.row(0, named=True)
        assert row["MONTH_END"] == date(2008, 9, 30)
        assert row["N_DAYS_CLOSED"] == 30 and row["N_DAYS_IN_MONTH"] == 30
        assert row["COVERAGE"] == 1.0
        assert row["N_HEADLINES"] == 290 and row["N_DRAWS_SAMPLED"] == 29 * 2000
        assert row["N_DRAWS_AVAILABLE"] == 29 * 20000 and row["RSS_PEAK_GB"] == 0.7
        assert row["F0_CONFIG_ID"] == cfg.f0_digest() and row["SEED"] == 1
        d = partition_digest(row)
        assert d.count == 29 * 2000 and abs(d.quantile(0.5) - 0.25) < 0.01
        assert partition_welford(row).count == 29 * 2000

    def test_finalises_on_schedule_with_partial_coverage(self, toy_table, cfg, tmp_path: Path,
                                                         caplog):
        sink = MonthlyNullPartitionWriter(tmp_path, config=cfg, table=toy_table)
        rng = np.random.default_rng(3)
        days = month_range_days(2008, 9)
        for d in days[:10]:                                        # 10 of 30 days only
            sink.add_draws(d, rng.normal(size=100), 1000, 5)
            sink.close_day(d, 0.1)
        assert sink.finalised == []
        with caplog.at_level(logging.WARNING):
            sink.close_day(date(2008, 10, 1), 0.1)                 # a later month's day
        assert [p.name for p in sink.finalised] == ["2008-09.parquet"]
        row = load_partitions(tmp_path).row(0, named=True)
        assert row["N_DAYS_CLOSED"] == 10 and row["COVERAGE"] == pytest.approx(10 / 30)
        assert any("coverage 0.33" in r.message for r in caplog.records)
        # explicit on-schedule finalisation, and empty months are dropped not written
        sink.rng_for(date(2008, 11, 3))                            # opens November, no days
        assert sink.finalise_before(date(2009, 1, 1)) == [tmp_path / "2008-10.parquet"]
        assert sorted(p.name for p in tmp_path.glob("*.parquet")) == ["2008-09.parquet",
                                                                      "2008-10.parquet"]

    def test_open_state_resumes_identically(self, toy_table, cfg, tmp_path: Path):
        """A live process (one day per invocation) must equal one replay run."""
        days = month_range_days(2008, 9)

        def draws_for(d):
            return np.random.default_rng(d.toordinal()).normal(0.25, 0.08, size=500)

        one = MonthlyNullPartitionWriter(tmp_path / "one", config=cfg, table=toy_table, seed=2)
        for d in days:
            one.add_draws(d, draws_for(d), 5000, 10)
            one.close_day(d, 0.1)
        for d in days:                        # fresh writer per day, resuming from _open/
            w = MonthlyNullPartitionWriter(tmp_path / "many", config=cfg, table=toy_table, seed=2)
            w.add_draws(d, draws_for(d), 5000, 10)
            w.close_day(d, 0.1)
        assert (tmp_path / "one" / "2008-09.parquet").read_bytes() == \
            (tmp_path / "many" / "2008-09.parquet").read_bytes()

    def test_rng_is_per_month_and_persisted(self, toy_table, cfg, tmp_path: Path):
        w = MonthlyNullPartitionWriter(tmp_path, config=cfg, table=toy_table, seed=3)
        r1 = w.rng_for(date(2008, 9, 1)).random(3)
        w.close_day(date(2008, 9, 1), 0.0)
        w2 = MonthlyNullPartitionWriter(tmp_path, config=cfg, table=toy_table, seed=3)
        r2 = w2.rng_for(date(2008, 9, 2)).random(3)
        fresh = np.random.default_rng([3, 2008, 9])
        fresh.random(3)
        np.testing.assert_array_equal(r2, fresh.random(3))       # continues the same stream
        assert not np.array_equal(r1, r2)

    def test_immutability(self, toy_table, cfg, tmp_path: Path):
        sink = MonthlyNullPartitionWriter(tmp_path, config=cfg, table=toy_table)
        _fill_month(sink, 2008, 9, np.random.default_rng(1))
        again = MonthlyNullPartitionWriter(tmp_path, config=cfg, table=toy_table)
        with pytest.raises(RuntimeError, match="already final"):
            again.add_draws(date(2008, 9, 3), np.ones(5), 5, 1)
        sink2 = MonthlyNullPartitionWriter(tmp_path / "b", config=cfg, table=toy_table)
        sink2.add_draws(date(2008, 9, 3), np.ones(5), 5, 1)
        sink2.close_day(date(2008, 9, 3), 0.0)
        with pytest.raises(RuntimeError, match="already closed"):
            sink2.add_draws(date(2008, 9, 3), np.ones(5), 5, 1)


def _partitions(tmp_path, table, cfg, months, seed=0, loc=0.25, per_day=3000, sparse=()):
    sink = MonthlyNullPartitionWriter(tmp_path, config=cfg, table=table, seed=seed)
    rng = np.random.default_rng(seed)
    for (y, m) in months:
        n = 30 if (y, m) in sparse else per_day
        for d in month_range_days(y, m):
            sink.add_draws(d, rng.normal(loc, 0.08, size=n), n * 10, 10)
            sink.close_day(d, 0.2)
    return load_partitions(tmp_path)


class TestTauAsof:
    def test_rows_formulas_window_and_pinned_warmup(self, toy_table, toy_embeddings, cfg,
                                                     tmp_path: Path):
        months = [(2008, m) for m in range(1, 13)]
        parts = _partitions(tmp_path, toy_table, cfg, months)
        rows = build_tau_rows(parts, cfg, toy_table, toy_embeddings, MU_DF, mu_asof_id="mu-1",
                              window="3M", today=date(2009, 1, 15))
        assert rows.schema == TAU_ASOF_SCHEMA
        # today - 1M = 2008-12-15 -> cutoffs through 2008-11-30
        assert rows["MONTH_END"].to_list() == [month_end(2008, m) for m in range(1, 12)]
        assert rows["N_PARTITIONS"].to_list() == [1, 2, 3] + [3] * 8
        assert rows["WINDOW_MONTHS_USED"].to_list() == [1, 2, 3] + [3] * 8
        assert (rows["WINDOW_EXTENDED_BY"] == 0).all()
        r = rows.filter(pl.col("MONTH_END") == date(2008, 6, 30)).row(0, named=True)
        assert r["WINDOW_START"] == date(2008, 3, 31) and r["WINDOW"] == "3M"
        sel = [p for p in parts.to_dicts()
               if date(2008, 3, 31) < p["MONTH_END"] <= date(2008, 6, 30)]
        assert len(sel) == 3
        D = TDigest.merge([partition_digest(p) for p in sel])
        # warmup pinned per cutoff from the mu row available then
        mu = resolve_mu_asof(MU_DF, date(2008, 6, 30))
        warm, _ = compute_warmup(toy_table, toy_embeddings, cfg, mu)
        assert r["N_EFF"] == warm.n_eff and r["MU_DATE"] == date(2008, 6, 1)
        assert r["EFFECTIVE_RANK"] == warm.effective_rank
        assert r["LAMBDA1_SHARE"] == warm.lambda1_share and r["MU_ARTIFACT_ID"] == "mu-1"
        early = rows.filter(pl.col("MONTH_END") == date(2008, 3, 31)).row(0, named=True)
        assert early["MU_DATE"] == date(2007, 12, 1) and early["N_EFF"] != r["N_EFF"]
        p_tail = tail_probability(cfg.alpha, warm.n_eff)
        assert r["P_TAIL"] == p_tail and r["TAU_EMPIRICAL"] == D.quantile(p_tail)
        assert r["TAU_GAUSS"] == gaussian_tau(r["POOL_MEAN"], r["POOL_STD"], warm.n_eff,
                                              cfg.alpha)
        assert r["ABS_GAP"] == abs(r["TAU_GAUSS"] - r["TAU_EMPIRICAL"])
        assert r["POOL_COUNT"] == (30 + 31 + 30) * 3000 == D.count   # April, May, June
        assert r["POOL_Q99"] < r["POOL_Q999"] < r["POOL_Q9993"] < r["POOL_Q9999"]
        assert r["EMBEDDINGS_DIGEST"] == warm.embeddings_digest
        exp = build_tau_rows(parts, cfg, toy_table, toy_embeddings, MU_DF, window="expanding",
                             today=date(2009, 1, 15))
        assert exp["N_PARTITIONS"].to_list() == list(range(1, 12))
        assert exp["WINDOW_START"].to_list()[-1] == date(2007, 12, 31)
        assert exp["WINDOW_MONTHS_USED"].to_list() == list(range(1, 12))

    def test_short_window_cold_start_rows_are_valid(self, toy_table, toy_embeddings, cfg,
                                                     tmp_path: Path):
        parts = _partitions(tmp_path, toy_table, cfg, [(2008, 1), (2008, 2)])
        rows = build_tau_rows(parts, cfg, toy_table, toy_embeddings, MU_DF, window="5Y",
                              today=date(2008, 4, 1))
        assert rows["N_PARTITIONS"].to_list() == [1, 2]
        assert rows["WINDOW_MONTHS_USED"].to_list() == [1, 2]      # visibly short, still rows
        assert (rows["TAU_EMPIRICAL"] > 0).all()

    def test_min_month_draws_extends_window(self, toy_table, toy_embeddings, tmp_path: Path,
                                            caplog):
        cfg = ScoringConfig(q=0.75, null_draws_per_headline=4, min_month_draws=150_000)
        # months 1-4 normal (90k+ draws each), months 5-6 sparse (~900 draws each)
        parts = _partitions(tmp_path, toy_table, cfg, [(2008, m) for m in range(1, 7)],
                            sparse={(2008, 5), (2008, 6)})
        with caplog.at_level(logging.WARNING):
            rows = build_tau_rows(parts, cfg, toy_table, toy_embeddings, MU_DF, window="2M",
                                  today=date(2008, 8, 1))
        by = dict(zip(rows["MONTH_END"].to_list(), rows.to_dicts()))
        # Feb: 2 normal months >= 150k -> nominal window
        assert by[date(2008, 2, 29)]["WINDOW_EXTENDED_BY"] == 0
        assert by[date(2008, 2, 29)]["N_PARTITIONS"] == 2
        # Jun: May+Jun ~1.8k -> extend back: +Apr (~92k) then +Mar (~185k) -> +2 months
        jun = by[date(2008, 6, 30)]
        assert jun["WINDOW_EXTENDED_BY"] == 2 and jun["N_PARTITIONS"] == 4
        assert jun["WINDOW_MONTHS_USED"] == 4 and jun["WINDOW_START"] == date(2008, 2, 29)
        assert jun["POOL_COUNT"] >= 150_000 and jun["MIN_MONTH_DRAWS"] == 150_000
        assert any("window extended by 2 month(s)" in r.message for r in caplog.records)
        # Jan alone: 93k < 150k and no history -> logged, row still produced
        assert by[date(2008, 1, 31)]["WINDOW_EXTENDED_BY"] == 0
        assert any("history exhausted" in r.message for r in caplog.records)

    def test_deterministic_bytes(self, toy_table, toy_embeddings, cfg, tmp_path: Path):
        parts = _partitions(tmp_path, toy_table, cfg, [(2008, m) for m in range(1, 8)])
        a = build_tau_rows(parts, cfg, toy_table, toy_embeddings, MU_DF, today=date(2008, 9, 1),
                           code_version="c")
        b = build_tau_rows(parts, cfg, toy_table, toy_embeddings, MU_DF, today=date(2008, 9, 1),
                           code_version="c")
        assert a.equals(b)
        pa_, pb_ = tmp_path / "a.parquet", tmp_path / "b.parquet"
        a.write_parquet(pa_)
        b.write_parquet(pb_)
        assert pa_.read_bytes() == pb_.read_bytes()

    def test_rejects_mismatched_inputs(self, toy_table, toy_embeddings, cfg, tmp_path: Path):
        parts = _partitions(tmp_path, toy_table, cfg, [(2008, 1)])
        with pytest.raises(ValueError, match="another F0 config"):
            build_tau_rows(parts, ScoringConfig(trim_frac=0.2), toy_table, toy_embeddings, MU_DF,
                           today=date(2008, 4, 1))
        with pytest.raises(ValueError, match="needs mu_asof"):
            build_tau_rows(parts, cfg, toy_table, toy_embeddings, None, today=date(2008, 4, 1))
        with pytest.raises(ValueError):
            window_offset("5X")
        assert latest_cutoff(parts, date(2008, 2, 15)) is None
        assert latest_cutoff(parts, date(2008, 3, 1)) == date(2008, 1, 31)
        raw = ScoringConfig(mode=Correction.RAW, q=0.75, null_draws_per_headline=4,
                            min_month_draws=1)
        raw_parts = _partitions(tmp_path / "raw", toy_table, raw, [(2008, 1)])
        rows = build_tau_rows(raw_parts, raw, toy_table, toy_embeddings, None,
                              today=date(2008, 4, 1))
        assert rows["MU_DATE"].null_count() == 1

    def test_gap_is_a_warning_only(self, toy_table, toy_embeddings, tmp_path: Path, caplog):
        cfg = ScoringConfig(q=0.75, null_draws_per_headline=4, min_month_draws=1,
                            gap_alert_threshold=0.02)
        sink = MonthlyNullPartitionWriter(tmp_path, config=cfg, table=toy_table)
        rng = np.random.default_rng(4)
        for m in (1, 2, 3, 4):
            for d in month_range_days(2008, m):
                # fat right tail: the Gaussian extrapolation misses the empirical quantile
                sink.add_draws(d, rng.lognormal(-1.5, 0.9, size=3000), 30000, 10)
                sink.close_day(d, 0.0)
        with caplog.at_level(logging.WARNING):
            rows = build_tau_rows(load_partitions(tmp_path), cfg, toy_table, toy_embeddings,
                                  MU_DF, today=date(2008, 6, 1))
        assert (rows["ABS_GAP"] > 0.02).all()
        msgs = [r for r in caplog.records if "tau_gauss - tau_empirical" in r.message]
        assert len(msgs) == 4 and all(r.levelno == logging.WARNING for r in msgs)
        assert all("informational" in r.message for r in msgs)


class TestTauSeriesProvider:
    def test_lookahead_and_lookup(self, toy_table, toy_embeddings, cfg, tmp_path: Path):
        parts = _partitions(tmp_path, toy_table, cfg, [(2008, m) for m in range(1, 5)])
        rows = build_tau_rows(parts, cfg, toy_table, toy_embeddings, MU_DF, today=date(2008, 6, 1))
        prov = TauSeriesProvider(rows, MU_DF, tau_source_id="tau-1", mu_asof_id="mu-1")
        assert prov.tau_policy == "tau_asof" and prov.mu_policy == "as_of_day"
        with pytest.raises(LookaheadError):
            prov.tau_for(date(2008, 1, 31))                      # snapshot == day
        with pytest.raises(LookaheadError):
            prov.tau_for(date(2008, 1, 15))
        rec = prov.tau_for(date(2008, 2, 1))
        assert rec.month_end == date(2008, 1, 31) and rec.source_id == "tau-1"
        assert rec.tau == rows.row(0, named=True)["TAU_EMPIRICAL"]
        assert rec.mu_date == date(2007, 12, 1)
        assert prov.tau_for(date(2008, 3, 31)).month_end == date(2008, 2, 29)
        assert prov.tau_for(date(2008, 9, 1)).month_end == date(2008, 4, 30)   # latest known
        assert prov.mu_for(date(2008, 2, 1)).date == date(2007, 12, 1)
        empty = TauSeriesProvider(pl.DataFrame(schema=TAU_ASOF_SCHEMA), MU_DF, tau_source_id="")
        with pytest.raises(LookaheadError):
            empty.tau_for(date(2008, 2, 1))                      # cold start: nothing yet


class TestWarmup:
    def test_compute(self, toy_table, toy_embeddings):
        cfg = ScoringConfig()
        mu = resolve_mu_asof(MU_DF, date(2008, 6, 1))
        rec, spec = compute_warmup(toy_table, toy_embeddings, cfg, mu, mu_asof_id="mu-1")
        assert 1.0 <= rec.n_eff <= rec.effective_rank <= toy_table.n_primitives
        assert 0 < rec.lambda1_share <= 1 and rec.n_eigenvalues == toy_table.n_primitives
        assert rec.mu_date == "2008-06-01" and rec.mu_asof_id == "mu-1"
        assert spec.eigenvalues[0] >= spec.eigenvalues[-1]
        with pytest.raises(ValueError, match="needs a mu_asof"):
            compute_warmup(toy_table, toy_embeddings, cfg, None)
        raw, _ = compute_warmup(toy_table, toy_embeddings, ScoringConfig(mode=Correction.RAW), None)
        assert raw.mu_date is None and raw.n_eff > 0

    def test_n_eff_matches_calibration_construction(self, toy_table, toy_embeddings):
        from narrative_scoring.f0 import compute_n_eff
        from narrative_scoring.primitives import representative_matrix

        cfg = ScoringConfig()
        mu = resolve_mu_asof(MU_DF, date(2008, 6, 1))
        rec, _ = compute_warmup(toy_table, toy_embeddings, cfg, mu)
        D = representative_matrix(toy_embeddings, toy_table, cfg.mode, mu.mu, mu.mu_hat)
        assert rec.n_eff == pytest.approx(compute_n_eff(D), rel=1e-12)


def test_month_helpers():
    assert month_end(2008, 2) == date(2008, 2, 29)
    assert (month_end(2008, 12) + timedelta(days=1)) == date(2009, 1, 1)
