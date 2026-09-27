"""datalake.periods: the single calendar (periods, keys, spans, RNG seeds)."""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from datalake.periods import (
    FREQS,
    PeriodError,
    parse_key,
    parse_span,
    partition_file,
    period_key,
    period_of,
    period_rng_seed,
    periods,
)


@pytest.mark.parametrize("freq", FREQS)
def test_periods_cover_the_range_contiguously_without_overlap(freq):
    rng = np.random.default_rng(0)
    for _ in range(50):
        start = date(1999, 1, 1) + timedelta(days=int(rng.integers(0, 9000)))
        end = start + timedelta(days=int(rng.integers(0, 800)))
        ps = periods(start, end, freq)
        assert ps[0].contains(start) and ps[-1].contains(end)
        for a, b in zip(ps, ps[1:]):
            assert b.first == a.last + timedelta(days=1)          # contiguous, no overlap
        assert all(p.freq == freq and parse_key(p.key) == p for p in ps)


def test_month_keys_are_todays_file_names():
    for y in (2000, 2008, 2024):
        for m in range(1, 13):
            p = period_of(date(y, m, 17), "M")
            assert p.key == f"{y}-{m:02d}" and partition_file(p.key) == f"{y}-{m:02d}.parquet"
            assert p.first == date(y, m, 1) and p.n_days == pd.Timestamp(y, m, 1).days_in_month


@pytest.mark.parametrize("day,freq,key,first,last", [
    (date(2008, 2, 29), "D", "2008-02-29", date(2008, 2, 29), date(2008, 2, 29)),
    (date(2008, 1, 1), "W", "2008-W01", date(2007, 12, 31), date(2008, 1, 6)),   # ISO year
    (date(2010, 1, 3), "W", "2009-W53", date(2009, 12, 28), date(2010, 1, 3)),
    (date(2008, 2, 29), "M", "2008-02", date(2008, 2, 1), date(2008, 2, 29)),
    (date(2008, 5, 5), "Q", "2008Q2", date(2008, 4, 1), date(2008, 6, 30)),
    (date(2008, 12, 31), "Y", "2008", date(2008, 1, 1), date(2008, 12, 31)),
])
def test_period_of_hand_checked(day, freq, key, first, last):
    p = period_of(day, freq)
    assert (p.key, p.first, p.last) == (key, first, last)
    assert period_key(day, freq) == key and parse_key(key) == p


def test_next_previous_and_days():
    p = period_of(date(2008, 12, 15), "M")
    assert p.next().key == "2009-01" and p.previous().key == "2008-11"
    q1 = period_of(date(2008, 3, 1), "Q").days()
    assert q1[0] == date(2008, 1, 1) and q1[-1] == date(2008, 3, 31) and len(q1) == 91


@pytest.mark.parametrize("bad", ["2008-13", "2008Q5", "2008-W60", "08-01", "", "2008-02-30"])
def test_bad_keys_rejected(bad):
    with pytest.raises(PeriodError):
        parse_key(bad)


def test_bad_freq_and_reversed_range():
    with pytest.raises(PeriodError):
        period_of(date(2008, 1, 1), "H")
    with pytest.raises(PeriodError):
        periods(date(2008, 2, 1), date(2008, 1, 1), "M")


def test_month_seed_is_historical_and_frequencies_never_collide():
    assert period_rng_seed(7, period_of(date(2008, 3, 9), "M")) == [7, 2008, 3]
    seeds = {tuple(period_rng_seed(0, period_of(date(2008, 1, 1), f))) for f in FREQS}
    assert len(seeds) == len(FREQS)


@pytest.mark.parametrize("spec,offset", [
    ("10d", pd.DateOffset(days=10)), ("2W", pd.DateOffset(weeks=2)),
    ("3M", pd.DateOffset(months=3)), ("1Q", pd.DateOffset(months=3)),
    ("5Y", pd.DateOffset(years=5)),
])
def test_parse_span(spec, offset):
    s = parse_span(spec)
    assert s.offset() == offset and s.spec == spec and not s.expanding


def test_parse_span_expanding_units_and_errors():
    assert parse_span("expanding", allow_expanding=True).offset() is None
    with pytest.raises(PeriodError):
        parse_span("expanding")
    with pytest.raises(PeriodError, match="allowed units"):
        parse_span("3d", units="WM")
    for bad in ("0d", "-1W", "5x", "abc", ""):
        with pytest.raises(PeriodError):
            parse_span(bad)
    # usable where the replaced parsers raised argparse / ValueError
    with pytest.raises(argparse.ArgumentTypeError):
        parse_span("5x")
    with pytest.raises(ValueError):
        parse_span("5x")
