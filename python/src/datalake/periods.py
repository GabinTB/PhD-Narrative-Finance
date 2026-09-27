"""The one calendar: period frequencies, period keys, spans.

Every time granularity in the lake -- storage partitions, job work units,
calibration periods -- is a ``Freq``:

    D  day              key 2008-01-15     (partition file 2008-01-15.parquet)
    W  ISO week Mon-Sun key 2008-W03       (ISO year + ISO week)
    M  calendar month   key 2008-01        (today's YYYY-MM.parquet layout)
    Q  calendar quarter key 2008Q1
    Y  calendar year    key 2008

``periods(start, end, freq)`` lists the full periods that intersect
[start, end], in order, contiguous and non-overlapping. ``parse_key`` inverts
``period_key`` (the frequency is read from the key's shape), so a reader can
map a partition file back to its dates without knowing how it was written.

``parse_span`` is the single parser for every user-written delay or window
('10d', '2W', '3M', '1Q', '5Y', 'expanding'); ``Span.offset`` is the
matching ``pd.DateOffset``.

``period_rng_seed`` seeds per-period random streams; for M it is exactly the
historical ``[seed, year, month]``, so monthly outputs are unchanged.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import pandas as pd

Freq = Literal["D", "W", "M", "Q", "Y"]
FREQS: tuple[str, ...] = ("D", "W", "M", "Q", "Y")
DEFAULT_FREQ: Freq = "M"


class PeriodError(ValueError, argparse.ArgumentTypeError):
    """An invalid frequency, period key or span (usable as a ValueError and as an
    argparse type error, as the parsers it replaces raised one or the other)."""


def check_freq(freq: str) -> Freq:
    if freq not in FREQS:
        raise PeriodError(f"frequency must be one of {FREQS}, got {freq!r}")
    return freq  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Period:
    """One calendar period: ``first`` and ``last`` are inclusive days."""

    first: date
    last: date
    freq: str
    key: str

    def contains(self, day: date) -> bool:
        return self.first <= day <= self.last

    @property
    def n_days(self) -> int:
        return (self.last - self.first).days + 1

    def days(self) -> list[date]:
        return [self.first + timedelta(days=i) for i in range(self.n_days)]

    def next(self) -> Period:
        return period_of(self.last + timedelta(days=1), self.freq)

    def previous(self) -> Period:
        return period_of(self.first - timedelta(days=1), self.freq)


def _month_last(y: int, m: int) -> date:
    return (date(y + (m == 12), 1 if m == 12 else m + 1, 1) - timedelta(days=1))


def period_of(day: date, freq: str) -> Period:
    """The period of ``freq`` that contains ``day``."""
    freq = check_freq(freq)
    if isinstance(day, pd.Timestamp):
        day = day.date()
    if freq == "D":
        return Period(day, day, "D", day.isoformat())
    if freq == "W":
        iso_year, iso_week, iso_weekday = day.isocalendar()
        first = day - timedelta(days=iso_weekday - 1)
        return Period(first, first + timedelta(days=6), "W", f"{iso_year}-W{iso_week:02d}")
    if freq == "M":
        return Period(day.replace(day=1), _month_last(day.year, day.month), "M",
                      f"{day.year}-{day.month:02d}")
    if freq == "Q":
        q = (day.month - 1) // 3 + 1
        first = date(day.year, 3 * q - 2, 1)
        return Period(first, _month_last(day.year, 3 * q), "Q", f"{day.year}Q{q}")
    return Period(date(day.year, 1, 1), date(day.year, 12, 31), "Y", f"{day.year}")


def period_key(day: date, freq: str) -> str:
    return period_of(day, freq).key


def periods(start: date, end: date, freq: str) -> list[Period]:
    """The full periods of ``freq`` intersecting [start, end], in order."""
    if end < start:
        raise PeriodError(f"end {end} before start {start}")
    out = [period_of(start, freq)]
    while out[-1].last < end:
        out.append(out[-1].next())
    return out


_KEY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("D", re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")),
    ("W", re.compile(r"^(\d{4})-W(\d{2})$")),
    ("M", re.compile(r"^(\d{4})-(\d{2})$")),
    ("Q", re.compile(r"^(\d{4})Q([1-4])$")),
    ("Y", re.compile(r"^(\d{4})$")),
)


def parse_key(key: str) -> Period:
    """The period a key names; the frequency is inferred from the key's shape."""
    for freq, pattern in _KEY_PATTERNS:
        m = pattern.match(key)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        try:
            if freq == "D":
                return period_of(date(g[0], g[1], g[2]), "D")
            if freq == "W":
                return period_of(date.fromisocalendar(g[0], g[1], 1), "W")
            if freq == "M":
                return period_of(date(g[0], g[1], 1), "M")
            if freq == "Q":
                return period_of(date(g[0], 3 * g[1] - 2, 1), "Q")
            return period_of(date(g[0], 1, 1), "Y")
        except ValueError as exc:
            raise PeriodError(f"invalid period key {key!r}: {exc}") from None
    raise PeriodError(f"not a period key: {key!r}")


def partition_file(key: str, suffix: str = ".parquet") -> str:
    return f"{key}{suffix}"


_SEED_CODE = {"D": 1, "W": 2, "Q": 4, "Y": 5}


def period_rng_seed(seed: int, period: Period) -> list[int]:
    """Entropy for a per-period RNG. M keeps the historical ``[seed, year, month]``;
    other frequencies carry a frequency code so their streams never coincide."""
    f = period.first
    if period.freq == "M":
        return [seed, f.year, f.month]
    if period.freq == "D":
        return [seed, _SEED_CODE["D"], f.year, f.month, f.day]
    if period.freq == "W":
        iso_year, iso_week, _ = f.isocalendar()
        return [seed, _SEED_CODE["W"], iso_year, iso_week]
    if period.freq == "Q":
        return [seed, _SEED_CODE["Q"], f.year, (f.month - 1) // 3 + 1]
    return [seed, _SEED_CODE["Y"], f.year]


# ---------------------------------------------------------------------------
# Spans ('10d', '2W', '3M', '1Q', '5Y', 'expanding')
# ---------------------------------------------------------------------------

EXPANDING = "expanding"
_SPAN_RE = re.compile(r"^(\d+)([dWMQY])$")


@dataclass(frozen=True)
class Span:
    """A length of calendar time: ``n`` units of d / W / M / Q / Y, or expanding."""

    n: int
    unit: str
    spec: str

    @property
    def expanding(self) -> bool:
        return self.unit == EXPANDING

    def offset(self) -> pd.DateOffset | None:
        if self.expanding:
            return None
        return {"d": pd.DateOffset(days=self.n), "W": pd.DateOffset(weeks=self.n),
                "M": pd.DateOffset(months=self.n), "Q": pd.DateOffset(months=3 * self.n),
                "Y": pd.DateOffset(years=self.n)}[self.unit]


def parse_span(spec: str, *, units: str = "dWMQY", allow_expanding: bool = False) -> Span:
    """Parse '10d' / '2W' / '3M' / '1Q' / '5Y' (n > 0), or 'expanding' when allowed.

    ``units`` restricts the accepted units (e.g. a window may refuse days).
    """
    if spec == EXPANDING:
        if not allow_expanding:
            raise PeriodError("'expanding' is not allowed here")
        return Span(0, EXPANDING, spec)
    m = _SPAN_RE.match(spec or "")
    if not m or int(m.group(1)) <= 0:
        raise PeriodError(f"invalid span {spec!r}, expected e.g. '5d', '2W', '3M', '1Q', '5Y' "
                          "(n > 0)")
    n, unit = int(m.group(1)), m.group(2)
    if unit not in units:
        raise PeriodError(f"span {spec!r} uses unit {unit!r}, allowed units here are "
                          f"{list(units)}")
    return Span(n, unit, spec)
