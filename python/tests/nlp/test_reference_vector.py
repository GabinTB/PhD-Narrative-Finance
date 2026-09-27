"""Tests for nlp.reference_vector.

The dated mean path is checked against the ORIGINAL mu_asof algorithm
(``build_cumulative`` + ``compute_mu_asof``, copied below verbatim from the
pre-refactor ravenpack/headlines/mu_asof.py) on random daily grids with gaps,
for day / week / month delays and expanding / rolling windows. Min / max are
checked against a brute-force window scan, and every pooling's online updates
against its offline fit.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nlp.corrections import Correction, LookaheadError, apply_mode
from nlp.reference_vector import (
    REFERENCE_SCHEMA,
    ReferenceValue,
    ReferenceVector,
    offset_of,
    parse_delay,
    parse_period,
    parse_window,
    resolve_reference,
    validate_window_after_delay,
)

DIM = 16


# ---------------------------------------------------------------------------
# The original mu_asof algorithm (reference oracle, pre-refactor code)
# ---------------------------------------------------------------------------

def _orig_build_cumulative(days, sums, counts):
    full_grid = pd.date_range(days.min(), days.max(), freq="D")
    n = len(full_grid)
    sum_full = np.zeros((n, sums.shape[1]), dtype=np.float64)
    count_full = np.zeros(n, dtype=np.int64)
    idx = full_grid.get_indexer(days)
    sum_full[idx] = sums
    count_full[idx] = counts
    cumsum_ext = np.vstack(
        [np.zeros((1, sums.shape[1]), dtype=np.float64), np.cumsum(sum_full, axis=0)])
    cumcount_ext = np.concatenate([[0], np.cumsum(count_full)])
    return full_grid, cumsum_ext, cumcount_ext


def _orig_asof_index(d, grid_start, n):
    offset = (pd.Timestamp(d).normalize() - grid_start).days
    return int(np.clip(offset + 1, 0, n))


def _orig_compute_mu_asof(asof_dates, grid_start, cumsum_ext, cumcount_ext, delay, mode):
    n = cumcount_ext.shape[0] - 1
    rows = {"DATE": [], "MU": [], "MU_HAT": [], "N": []}
    for t in asof_dates:
        cutoff = t - delay
        end = _orig_asof_index(cutoff, grid_start, n)
        start = 0 if mode == "expanding" else _orig_asof_index(cutoff - mode, grid_start, n)
        n_t = int(cumcount_ext[end] - cumcount_ext[start])
        if n_t == 0:
            continue
        mu_t = cumsum_ext[end] - cumsum_ext[start]
        norm = np.linalg.norm(mu_t)
        if norm == 0.0:
            continue
        rows["DATE"].append(pd.Timestamp(t).date())
        rows["MU"].append((mu_t / n_t).astype(np.float32))
        rows["MU_HAT"].append((mu_t / norm).astype(np.float32))
        rows["N"].append(n_t)
    return rows


def _random_daily(n_days: int = 400, seed: int = 0, gap_rate: float = 0.3):
    """Random per-day (sum, count) with missing days, and the raw rows behind them."""
    rng = np.random.default_rng(seed)
    all_days = pd.date_range("2008-01-01", periods=n_days, freq="D")
    keep = rng.random(n_days) > gap_rate
    days = all_days[keep]
    counts = rng.integers(1, 6, size=len(days))
    rows = rng.normal(size=(int(counts.sum()), DIM)).astype(np.float32)
    row_days = np.repeat(days.values.astype("datetime64[D]"), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    sums = np.add.reduceat(rows.astype(np.float64), starts, axis=0)
    return days, sums, counts, rows, row_days


# ---------------------------------------------------------------------------
# Mean pooling == original mu_asof
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delay,window", [
    ("1d", "expanding"), ("1M", "expanding"), ("2W", "expanding"),
    ("1d", "8W"), ("1M", "3M"), ("1d", "1M"),
])
def test_dated_mean_series_equals_original_mu_asof(delay, window):
    days, sums, counts, _, _ = _random_daily(seed=hash((delay, window)) % 1000)
    grid, cs, cc = _orig_build_cumulative(days, sums, counts)
    want = _orig_compute_mu_asof(days, grid[0], cs, cc, parse_delay(delay), parse_window(window))

    ref = ReferenceVector.from_daily(days, counts, sums=sums, pooling="mean",
                                     delay=delay, window=window)
    got = ref.series(days)

    assert got["DATE"].to_list() == want["DATE"]
    assert got["N"].to_list() == want["N"]
    np.testing.assert_allclose(np.stack(got["MU"].to_list()), np.stack(want["MU"]),
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.stack(got["MU_HAT"].to_list()), np.stack(want["MU_HAT"]),
                               rtol=0, atol=1e-7)


def test_series_schema_and_unit_directions():
    days, sums, counts, _, _ = _random_daily(n_days=60)
    got = ReferenceVector.from_daily(days, counts, sums=sums, as_of=True, delay="1d").series()
    assert got.schema == pl.Schema({"DATE": pl.Date, "MU": pl.Array(pl.Float32, DIM),
                                    "MU_HAT": pl.Array(pl.Float32, DIM), "N": pl.Int64})
    mu_hat = np.stack(got["MU_HAT"].to_list()).astype(np.float64)
    np.testing.assert_allclose(np.linalg.norm(mu_hat, axis=1), 1.0, atol=1e-6)
    assert REFERENCE_SCHEMA["MU"] == pl.Array(pl.Float32, 384)


def test_value_for_day_includes_cutoff_day_never_day_itself():
    """Value for t pools (cutoff - window, cutoff], cutoff = t - delay: t itself never."""
    X = np.eye(DIM, dtype=np.float32)[:3]
    days = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    ref = ReferenceVector.fit(X, days, pooling="mean", as_of=True, delay="1d")
    v = ref.value(date(2020, 1, 3))           # cutoff 2020-01-02: rows 0 and 1
    assert v.n == 2 and v.date == date(2020, 1, 3)
    np.testing.assert_allclose(v.mu[:3], [0.5, 0.5, 0.0], atol=1e-7)


# ---------------------------------------------------------------------------
# Min / max pooling == brute force; online == offline
# ---------------------------------------------------------------------------

def _brute_extreme(rows, row_days, t, delay, window, op):
    cutoff = np.datetime64(pd.Timestamp(t) - parse_delay(delay), "D")
    lo = (np.datetime64("0001-01-01") if window == "expanding"
          else np.datetime64(pd.Timestamp(cutoff) - parse_window(window), "D"))
    sel = (row_days <= cutoff) & (row_days > lo)
    return None if not sel.any() else op(rows[sel], axis=0)


@pytest.mark.parametrize("pooling,op", [("min", np.min), ("max", np.max)])
@pytest.mark.parametrize("window", ["expanding", "8W", "3M"])
def test_min_max_series_equal_brute_force(pooling, op, window):
    days, _, counts, rows, row_days = _random_daily(n_days=300, seed=3)
    ref = ReferenceVector.fit(rows, row_days, pooling=pooling, as_of=True,
                              delay="1d", window=window)
    got = ref.series(days)
    got_by_day = dict(zip(got["DATE"].to_list(), got["MU"].to_list()))
    for t in days:
        want = _brute_extreme(rows, row_days, t, "1d", window, op)
        if want is None:
            assert t.date() not in got_by_day
        else:
            np.testing.assert_array_equal(np.asarray(got_by_day[t.date()]), want)


@pytest.mark.parametrize("pooling", ["mean", "min", "max"])
def test_online_updates_equal_offline_fit(pooling):
    _, _, _, rows, row_days = _random_daily(n_days=120, seed=5)
    offline = ReferenceVector.fit(rows, row_days, pooling=pooling, as_of=True,
                                  delay="1d", window="8W")
    online = ReferenceVector(pooling, as_of=True, delay="1d", window="8W")
    # arrive day by day, each day split into two batches
    for d in np.unique(row_days):
        X = rows[row_days == d]
        online.update(X[: len(X) // 2 + 1], day=d.item())
        if len(X) > 1:
            online.update(X[len(X) // 2 + 1:], day=d.item())
    a, b = offline.series(), online.series()
    assert a["DATE"].to_list() == b["DATE"].to_list() and a["N"].to_list() == b["N"].to_list()
    np.testing.assert_allclose(np.stack(a["MU"].to_list()), np.stack(b["MU"].to_list()),
                               rtol=0, atol=1e-6)


@pytest.mark.parametrize("pooling,want", [
    ("mean", lambda X: X.mean(0)), ("min", lambda X: X.min(0)), ("max", lambda X: X.max(0)),
])
def test_undated_value_and_batches(pooling, want):
    X = np.random.default_rng(7).normal(size=(50, DIM)).astype(np.float32)
    ref = ReferenceVector(pooling).update(X[:20]).update(X[20:])
    v = ref.value()
    assert v.date is None and v.n == 50
    np.testing.assert_allclose(v.mu, want(X), atol=1e-6)
    np.testing.assert_allclose(v.mu_hat, want(X) / np.linalg.norm(want(X)), atol=1e-6)
    fitted = ReferenceVector.fit(X, pooling=pooling).value()
    np.testing.assert_allclose(fitted.mu, v.mu, atol=1e-6)


def test_update_from_texts_with_embedder():
    class FakeEmbedder:
        def encode(self, texts):
            return np.stack([np.full(DIM, float(len(t)), dtype=np.float32) for t in texts])

    ref = ReferenceVector("mean").update(texts=["ab", "abcd"], embedder=FakeEmbedder())
    np.testing.assert_allclose(ref.value().mu, np.full(DIM, 3.0))


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_empty_window_raises_lookahead():
    ref = ReferenceVector.fit(np.ones((1, DIM), np.float32), [date(2020, 1, 10)],
                              as_of=True, delay="1d")
    ref.update(np.empty((0, DIM), np.float32), day=date(2020, 1, 20))
    with pytest.raises(LookaheadError):
        ref.value(date(2020, 1, 10))         # cutoff 01-09: nothing yet


def test_value_beyond_updated_state_raises():
    ref = ReferenceVector.fit(np.ones((2, DIM), np.float32),
                              [date(2020, 1, 1), date(2020, 1, 2)], as_of=True, delay="1d")
    ref.value(date(2020, 1, 3))              # cutoff 01-02 == last seen day: fine
    with pytest.raises(ValueError, match="update it first"):
        ref.value(date(2020, 1, 5))          # cutoff 01-04: days 01-03/04 never seen
    ref.update(np.empty((0, DIM), np.float32), day=date(2020, 1, 4))   # a seen, empty day
    assert ref.value(date(2020, 1, 5)).n == 2


def test_dated_and_undated_day_rules():
    with pytest.raises(ValueError, match="needs a day"):
        ReferenceVector(as_of=True).update(np.ones((1, DIM)))
    with pytest.raises(ValueError, match="takes no day"):
        ReferenceVector().update(np.ones((1, DIM)), day=date(2020, 1, 1))


def test_rejects_nan_and_dim_change_and_bad_pooling():
    ref = ReferenceVector()
    with pytest.raises(ValueError, match="NaN"):
        ref.update(np.full((1, DIM), np.nan))
    ref.update(np.ones((1, DIM)))
    with pytest.raises(ValueError, match="dim"):
        ref.update(np.ones((1, DIM + 1)))
    with pytest.raises(ValueError, match="pooling"):
        ReferenceVector("median")


def test_rolling_window_must_clear_delay():
    with pytest.raises(ValueError, match="longer than the delay"):
        ReferenceVector(as_of=True, delay="2M", window="1M")


# ---------------------------------------------------------------------------
# Persisted series + correction interplay
# ---------------------------------------------------------------------------

def test_resolve_reference_never_uses_a_later_row():
    series = pl.DataFrame({
        "DATE": [date(2020, 1, 1), date(2020, 1, 5)],
        "MU": [np.ones(DIM, np.float32), 2 * np.ones(DIM, np.float32)],
        "MU_HAT": [np.ones(DIM, np.float32) / 4, np.ones(DIM, np.float32) / 4],
        "N": [3, 4],
    })
    assert resolve_reference(series, date(2020, 1, 4)).date == date(2020, 1, 1)
    assert resolve_reference(series, date(2020, 1, 5)).n == 4
    with pytest.raises(LookaheadError):
        resolve_reference(series, date(2019, 12, 31))


def test_value_feeds_correction_with_point_in_time_guard():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(30, DIM)).astype(np.float32)
    ref = ReferenceVector.fit(X, [date(2020, 1, 1) + timedelta(days=i % 5) for i in range(30)],
                              as_of=True, delay="1d")
    v = ref.value(date(2020, 1, 4))
    H = rng.normal(size=(5, DIM)).astype(np.float32)
    rows, day = Correction.R2.correct(H, v, as_of=date(2020, 1, 4))
    np.testing.assert_array_equal(rows, apply_mode(H, Correction.R2, v.mu, v.mu_hat))
    with pytest.raises(LookaheadError):
        Correction.R2.correct(H, v, as_of=date(2020, 1, 3))
    assert isinstance(v, ReferenceValue)


# ---------------------------------------------------------------------------
# Period parsing (moved from ravenpack/headlines/mu_asof.py)
# ---------------------------------------------------------------------------

class TestPeriodParsing:
    @pytest.mark.parametrize("spec,unit,n", [("5d", "d", 5), ("2W", "W", 2), ("3M", "M", 3)])
    def test_parse_period_valid(self, spec, unit, n):
        assert parse_period(spec, allowed_units="dWM") == (n, unit)

    @pytest.mark.parametrize("spec", ["0d", "-1W", "5x", "abc", ""])
    def test_parse_period_invalid(self, spec):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_period(spec, allowed_units="dWM")

    def test_parse_period_rejects_disallowed_unit(self):
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

    def test_parse_window(self):
        assert parse_window("expanding") == "expanding"
        assert parse_window("6M") == pd.DateOffset(months=6)
        assert parse_window("4W") == pd.DateOffset(weeks=4)
        with pytest.raises(argparse.ArgumentTypeError):
            parse_window("5d")

    def test_validate_window_after_delay(self):
        validate_window_after_delay(parse_delay("1M"), parse_window("6M"))
        validate_window_after_delay(parse_delay("6M"), parse_window("expanding"))
        with pytest.raises(ValueError, match="must be longer than the delay"):
            validate_window_after_delay(parse_delay("2M"), parse_window("1M"))
        with pytest.raises(ValueError):
            validate_window_after_delay(parse_delay("4W"), parse_window("4W"))
