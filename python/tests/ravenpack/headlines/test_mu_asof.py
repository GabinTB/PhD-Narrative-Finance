"""Tests for ravenpack.headlines.mu_asof.

No real GDrive/production data needed: ``compute_daily_sums`` (pass 1) is
exercised with DuckDB monkeypatched out (a fake connection returning
synthetic per-month polars results) -- everything downstream of it
(``build_cumulative``, ``compute_mu_asof``) is driven from synthetic
``(days, sums, counts)`` triples built by hand.

Covers:
  - compute_daily_sums: datetime.date dict keys vs. the pd.Timestamp days
    index returned (regression for a KeyError on lookup)
  - build_cumulative: zero-filling, cumsum shape, the prefixed zero row
  - compute_mu_asof: expanding mode, rolling mode, delay skipping days with no
    data, MU_HAT unit-norm for every output row, N matches expected counts
  - verify_artifact: unit-norm check, gap detection, N > 0 check
  - Period parsing: valid/invalid specs, rolling > delay validation
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ravenpack.headlines.mu_asof import (
    build_cumulative,
    compute_daily_sums,
    compute_mu_asof,
    offset_of,
    parse_delay,
    parse_mode,
    parse_period,
    validate_window_after_delay,
    verify_artifact,
)
from ravenpack.headlines.schema import EMBEDDING_DIM, MU_ASOF_SCHEMA

# ---------------------------------------------------------------------------
# compute_daily_sums
# ---------------------------------------------------------------------------


class _FakeDuckDBConn:
    """Stand-in for duckdb.connect(): returns one canned polars result per
    call to .sql(...).pl(), regardless of the query text, and ignores
    PRAGMA execute() calls."""

    def __init__(self, result: pl.DataFrame):
        self._result = result

    def execute(self, *args, **kwargs):
        return None

    def sql(self, _query: str):
        return self

    def pl(self) -> pl.DataFrame:
        return self._result

    def close(self):
        return None


class TestComputeDailySums:
    def test_date_key_type_does_not_raise(self, tmp_path: Path, monkeypatch):
        """Regression: DuckDB's CAST(... AS DATE) comes back as datetime.date
        via polars, so day_sum/day_cnt are keyed by datetime.date. Building
        the returned index as pd.DatetimeIndex(sorted(day_sum.keys())) yields
        pd.Timestamp entries -- indexing day_sum[d] with those Timestamps
        directly raised KeyError (Timestamp != date, even for the same day).
        """
        headlines_dir = tmp_path / "headlines"
        embeddings_dir = tmp_path / "embeddings"
        headlines_dir.mkdir()
        embeddings_dir.mkdir()
        (headlines_dir / "2000-01.parquet").touch()
        (embeddings_dir / "2000-01.parquet").touch()

        result = pl.DataFrame({
            "day": [date(2000, 1, 1), date(2000, 1, 2)],
            "day_sum": [[1.0] * EMBEDDING_DIM, [2.0] * EMBEDDING_DIM],
            "n": [3, 5],
        })
        monkeypatch.setattr(
            "ravenpack.headlines.mu_asof.duckdb.connect",
            lambda: _FakeDuckDBConn(result),
        )

        days, sums, counts = compute_daily_sums(headlines_dir, embeddings_dir, threads=1)

        assert isinstance(days, pd.DatetimeIndex)
        assert list(days.date) == [date(2000, 1, 1), date(2000, 1, 2)]
        assert counts.tolist() == [3, 5]
        assert sums.shape == (2, EMBEDDING_DIM)
        np.testing.assert_array_equal(sums[0], np.full(EMBEDDING_DIM, 1.0))
        np.testing.assert_array_equal(sums[1], np.full(EMBEDDING_DIM, 2.0))

    def test_merges_across_months(self, tmp_path: Path, monkeypatch):
        """Two monthly files, each contributing distinct days -- output
        should cover both months' days without collapsing or dropping any."""
        headlines_dir = tmp_path / "headlines"
        embeddings_dir = tmp_path / "embeddings"
        headlines_dir.mkdir()
        embeddings_dir.mkdir()
        for name in ("2000-01.parquet", "2000-02.parquet"):
            (headlines_dir / name).touch()
            (embeddings_dir / name).touch()

        results = {
            "2000-01.parquet": pl.DataFrame({
                "day": [date(2000, 1, 15)],
                "day_sum": [[1.0] * EMBEDDING_DIM],
                "n": [4],
            }),
            "2000-02.parquet": pl.DataFrame({
                "day": [date(2000, 2, 15)],
                "day_sum": [[2.0] * EMBEDDING_DIM],
                "n": [6],
            }),
        }

        # duckdb.connect() carries no info about which month is being
        # queried, so route by inspecting which files currently exist to
        # read -- simplest correct stand-in is a counter over the sorted
        # (deterministic) glob order compute_daily_sums iterates in.
        call_order = iter(sorted(results))
        monkeypatch.setattr(
            "ravenpack.headlines.mu_asof.duckdb.connect",
            lambda: _FakeDuckDBConn(results[next(call_order)]),
        )

        days, sums, counts = compute_daily_sums(headlines_dir, embeddings_dir, threads=1)

        assert list(days.date) == [date(2000, 1, 15), date(2000, 2, 15)]
        assert counts.tolist() == [4, 6]


# ---------------------------------------------------------------------------
# build_cumulative
# ---------------------------------------------------------------------------


class TestBuildCumulative:
    def test_no_gaps_shape_and_prefix_row(self):
        days = pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"])
        sums = np.stack([np.full(EMBEDDING_DIM, float(i + 1)) for i in range(3)])
        counts = np.array([2, 3, 5])

        grid, cumsum_ext, cumcount_ext = build_cumulative(days, sums, counts)

        assert len(grid) == 3
        assert cumsum_ext.shape == (4, EMBEDDING_DIM)
        assert cumcount_ext.shape == (4,)
        # Prefix row/entry is always zero -- "no history yet".
        np.testing.assert_array_equal(cumsum_ext[0], np.zeros(EMBEDDING_DIM))
        assert cumcount_ext[0] == 0
        # Cumulative sum/count at the end covers everything.
        np.testing.assert_allclose(cumsum_ext[-1], sums.sum(axis=0))
        assert cumcount_ext[-1] == counts.sum()

    def test_zero_fills_missing_calendar_days(self):
        # Day 2 is missing (a Sunday with no headlines) -- the grid must still
        # include it, zero-filled, so week/month offsets land on exact dates.
        days = pd.DatetimeIndex(["2000-01-01", "2000-01-03"])
        sums = np.stack([np.ones(EMBEDDING_DIM), np.full(EMBEDDING_DIM, 3.0)])
        counts = np.array([1, 1])

        grid, cumsum_ext, cumcount_ext = build_cumulative(days, sums, counts)

        assert len(grid) == 3
        assert grid[1] == pd.Timestamp("2000-01-02")
        # cumcount after day 2 (index 2, zero-filled) equals cumcount after day 1.
        assert cumcount_ext[2] == cumcount_ext[1]
        np.testing.assert_array_equal(cumsum_ext[2], cumsum_ext[1])
        # cumcount after day 3 picks up both observed days.
        assert cumcount_ext[3] == 2


# ---------------------------------------------------------------------------
# compute_mu_asof
# ---------------------------------------------------------------------------


def _synthetic_grid(n_days: int, seed: int = 0):
    """A grid of n_days consecutive days, each with a distinct random daily
    sum vector and a fixed count of 2 headlines/day, plus its cumulative
    extension -- shared setup for compute_mu_asof tests."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2000-01-01", periods=n_days, freq="D")
    sums = rng.standard_normal((n_days, EMBEDDING_DIM))
    counts = np.full(n_days, 2, dtype=np.int64)
    grid, cumsum_ext, cumcount_ext = build_cumulative(days, sums, counts)
    return days, grid, cumsum_ext, cumcount_ext


class TestComputeMuAsof:
    def test_output_schema(self):
        days, grid, cumsum_ext, cumcount_ext = _synthetic_grid(10)
        result = compute_mu_asof(
            days, grid[0], cumsum_ext, cumcount_ext,
            delay=pd.DateOffset(days=1), mode="expanding",
        )
        assert result.schema == MU_ASOF_SCHEMA

    def test_expanding_mode_accumulates_all_prior_history(self):
        days, grid, cumsum_ext, cumcount_ext = _synthetic_grid(10)
        delay = pd.DateOffset(days=1)
        result = compute_mu_asof(
            days, grid[0], cumsum_ext, cumcount_ext, delay=delay, mode="expanding",
        )

        # asof(day i) should pool every day strictly before day i (delay=1d
        # excludes day i itself): N grows as 2, 4, 6, ... starting at day 1.
        n_values = result["N"].to_list()
        assert n_values == [2 * i for i in range(1, len(n_values) + 1)]
        # day 0 has no prior history (delay excludes it) -> skipped entirely.
        assert result["DATE"].to_list()[0] == days[1].date()

    def test_rolling_mode_uses_fixed_window(self):
        days, grid, cumsum_ext, cumcount_ext = _synthetic_grid(20)
        delay = pd.DateOffset(days=1)
        window = pd.DateOffset(weeks=1)  # 7 days
        result = compute_mu_asof(
            days, grid[0], cumsum_ext, cumcount_ext, delay=delay, mode=window,
        )

        # Once the window is fully "inside" the history (far enough from day
        # 0), N should plateau at 2/day * 7 days = 14, not keep growing.
        n_values = result["N"].to_list()
        assert n_values[-1] == 14
        assert max(n_values) == 14

    def test_delay_skips_dates_with_no_prior_data(self):
        days, grid, cumsum_ext, cumcount_ext = _synthetic_grid(5)
        # A delay longer than the whole observed history: every asof date's
        # cutoff falls before grid_start, so every window is empty.
        delay = pd.DateOffset(days=30)
        result = compute_mu_asof(
            days, grid[0], cumsum_ext, cumcount_ext, delay=delay, mode="expanding",
        )
        assert result.is_empty()

    def test_mu_hat_is_unit_norm(self):
        days, grid, cumsum_ext, cumcount_ext = _synthetic_grid(15)
        result = compute_mu_asof(
            days, grid[0], cumsum_ext, cumcount_ext,
            delay=pd.DateOffset(days=1), mode=pd.DateOffset(weeks=1),
        )
        mu_hat = np.asarray(result["MU_HAT"].to_list(), dtype=np.float64)
        norms = np.linalg.norm(mu_hat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_mu_and_mu_hat_same_direction(self):
        days, grid, cumsum_ext, cumcount_ext = _synthetic_grid(15)
        result = compute_mu_asof(
            days, grid[0], cumsum_ext, cumcount_ext,
            delay=pd.DateOffset(days=1), mode="expanding",
        )
        mu = np.asarray(result["MU"].to_list(), dtype=np.float64)
        mu_hat = np.asarray(result["MU_HAT"].to_list(), dtype=np.float64)
        expected = mu / np.linalg.norm(mu, axis=1, keepdims=True)
        np.testing.assert_allclose(mu_hat, expected, atol=1e-6)

    def test_n_matches_expected_counts(self):
        days, grid, cumsum_ext, cumcount_ext = _synthetic_grid(10)
        result = compute_mu_asof(
            days, grid[0], cumsum_ext, cumcount_ext,
            delay=pd.DateOffset(days=1), mode="expanding",
        )
        assert all(n > 0 for n in result["N"].to_list())
        assert result["N"].to_list() == [2 * i for i in range(1, len(result) + 1)]


# ---------------------------------------------------------------------------
# Period parsing
# ---------------------------------------------------------------------------


class TestPeriodParsing:
    @pytest.mark.parametrize("spec,unit,n", [("5d", "d", 5), ("2W", "W", 2), ("3M", "M", 3)])
    def test_parse_period_valid(self, spec, unit, n):
        assert parse_period(spec, allowed_units="dWM") == (n, unit)

    @pytest.mark.parametrize("spec", ["0d", "-1W", "5x", "abc", ""])
    def test_parse_period_invalid(self, spec):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_period(spec, allowed_units="dWM")

    def test_parse_period_rejects_disallowed_unit(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_period("3d", allowed_units="WM")

    def test_offset_of(self):
        assert offset_of(5, "d") == pd.DateOffset(days=5)
        assert offset_of(2, "W") == pd.DateOffset(weeks=2)
        assert offset_of(3, "M") == pd.DateOffset(months=3)

    def test_parse_delay_allows_all_units(self):
        assert parse_delay("1d") == pd.DateOffset(days=1)
        assert parse_delay("2W") == pd.DateOffset(weeks=2)
        assert parse_delay("3M") == pd.DateOffset(months=3)

    def test_parse_mode_expanding(self):
        assert parse_mode("expanding") == "expanding"

    def test_parse_mode_rolling(self):
        assert parse_mode("6M") == pd.DateOffset(months=6)
        assert parse_mode("4W") == pd.DateOffset(weeks=4)

    def test_parse_mode_rejects_day_granularity(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_mode("5d")

    def test_validate_window_after_delay_ok(self):
        # window (6M) is longer than delay (1M) -- no error.
        validate_window_after_delay(parse_delay("1M"), parse_mode("6M"))

    def test_validate_window_after_delay_noop_for_expanding(self):
        validate_window_after_delay(parse_delay("6M"), parse_mode("expanding"))

    def test_validate_window_shorter_than_delay_raises(self):
        with pytest.raises(ValueError, match="must be longer than the delay"):
            validate_window_after_delay(parse_delay("2M"), parse_mode("1M"))

    def test_validate_window_equal_to_delay_raises(self):
        with pytest.raises(ValueError):
            validate_window_after_delay(parse_delay("4W"), parse_mode("4W"))


# ---------------------------------------------------------------------------
# verify_artifact
# ---------------------------------------------------------------------------


def _write_mu_asof(
    path: Path,
    n_rows: int = 10,
    *,
    start: str = "2000-01-01",
    gap_after: int | None = None,
    bad_norm_at: int | None = None,
    bad_n_at: int | None = None,
) -> None:
    """Write a schema-valid mu_asof parquet, optionally with an injected flaw."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n_rows, freq="D")
    if gap_after is not None:
        # Push every date after `gap_after` forward by 10 days -- opens a gap.
        dates = pd.DatetimeIndex(
            list(dates[: gap_after + 1]) + list(dates[gap_after + 1 :] + pd.Timedelta(days=10))
        )

    mu = rng.standard_normal((n_rows, EMBEDDING_DIM)).astype(np.float32)
    mu_hat = (mu / np.linalg.norm(mu, axis=1, keepdims=True)).astype(np.float32)
    n = np.full(n_rows, 5, dtype=np.int64)

    if bad_norm_at is not None:
        mu_hat[bad_norm_at] = mu_hat[bad_norm_at] * 2.0  # norm 2, not 1
    if bad_n_at is not None:
        n[bad_n_at] = 0

    pl.DataFrame(
        {
            "DATE": list(dates.date),
            "MU": list(mu),
            "MU_HAT": list(mu_hat),
            "N": n.tolist(),
        },
        schema=MU_ASOF_SCHEMA,
    ).write_parquet(path / "mu_asof_delay-1M_mode-expanding.parquet")


def _fake_artifact(path: Path, *, sources: bool = True):
    hp = {"delay": "1M", "mode": "expanding", "dim": EMBEDDING_DIM}
    if sources:
        hp.update({"source_headlines": "hl__1", "source_embeddings": "emb__1"})
    return SimpleNamespace(
        artifact_id="mu_asof__v0.1.0__20260101",
        path=path,
        meta=SimpleNamespace(hyperparams=hp),
    )


class TestVerifyArtifact:
    def test_clean_artifact_has_no_findings(self, tmp_path: Path):
        _write_mu_asof(tmp_path)
        assert verify_artifact(_fake_artifact(tmp_path)) == []

    def test_missing_source_hyperparams_errors(self, tmp_path: Path):
        _write_mu_asof(tmp_path)
        findings = verify_artifact(_fake_artifact(tmp_path, sources=False))
        assert any(
            f.severity.value == "error" and "source_headlines" in f.message
            for f in findings
        )

    def test_n_nonpositive_errors(self, tmp_path: Path):
        _write_mu_asof(tmp_path, bad_n_at=3)
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert any(
            f.severity.value == "error" and "N <= 0" in f.message for f in findings
        )

    def test_non_unit_mu_hat_errors(self, tmp_path: Path):
        _write_mu_asof(tmp_path, bad_norm_at=2)
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert any(
            f.severity.value == "error" and "unit-norm" in f.message for f in findings
        )

    def test_large_gap_warns(self, tmp_path: Path):
        _write_mu_asof(tmp_path, n_rows=10, gap_after=4)
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert any(
            f.severity.value == "warning" and "gap" in f.message for f in findings
        )

    def test_small_gap_within_tolerance_ok(self, tmp_path: Path):
        # Consecutive daily dates -> max gap is 1 day, well under the 5-day
        # tolerance for weekends/holidays.
        _write_mu_asof(tmp_path, n_rows=10)
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert not any("gap" in f.message for f in findings)

    def test_no_parquet_errors(self, tmp_path: Path):
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert len(findings) >= 1
        assert any(f.severity.value == "error" and "no parquet" in f.message for f in findings)

    def test_empty_file_errors(self, tmp_path: Path):
        pl.DataFrame(schema=MU_ASOF_SCHEMA).write_parquet(
            tmp_path / "mu_asof_delay-1M_mode-expanding.parquet"
        )
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert any(f.severity.value == "error" and "empty" in f.message for f in findings)
