"""Reference vector: the pooled embedding the R1 / R2 corrections are taken against.

A ``ReferenceVector`` pools embeddings into one vector ``mu`` and its unit
direction ``mu_hat = mu / ||mu||``, with pooling ``mean`` (the classic mu),
``min`` or ``max`` (elementwise). It works offline and online from the same
state:

  * online   ``update(X, day=...)`` folds a new batch into the state;
  * offline  ``fit(X, days)`` / ``from_daily(...)`` build the state in one
             vectorised pass (``np.add.reduceat`` / ``np.minimum.reduceat``).

State (per day when dated, one bucket otherwise), never the raw rows:

    mean       running sum (float64, (d,)) and count n
    min / max  running elementwise min / max (float32, (d,)) and count n

so an update is O(n d) and a query never re-reads data.

Dated references (``as_of=True``)
---------------------------------
The value for day t pools the days in the window ``(cutoff - window, cutoff]``,
``cutoff = t - delay``, INCLUDING the cutoff day itself (expanding: every day
up to and including the cutoff). ``delay`` is at least one day, so day t is
never used: the value for t is available at the start of t. This is exactly
the rule of the original ``mu_asof`` pipeline, whose docstring said "strictly
before t - delay" while its index arithmetic included the cutoff day; the
code, which produced the stored artifacts, is what is reproduced here.

Days with no data inside the window's calendar span are zero-count days
(identity elements: 0 for sums, +inf / -inf for min / max). A value whose
window holds no data, or whose pooled vector is exactly zero (no direction),
does not exist: ``value`` raises ``LookaheadError`` and ``series`` skips it.

Persisted form: the ``mu_asof`` daily series (``REFERENCE_SCHEMA``: DATE, MU,
MU_HAT, N), read back point-in-time by ``resolve_reference``.
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from nlp.corrections import LookaheadError

log = logging.getLogger(__name__)

Pooling = Literal["mean", "min", "max"]
POOLINGS: tuple[str, ...] = ("mean", "min", "max")

DEFAULT_DIM = 384


def reference_schema(dim: int = DEFAULT_DIM) -> pl.Schema:
    """The persisted daily series: DATE, MU (raw pooled), MU_HAT (unit), N (count)."""
    return pl.Schema({
        "DATE": pl.Date,
        "MU": pl.Array(pl.Float32, dim),
        "MU_HAT": pl.Array(pl.Float32, dim),
        "N": pl.Int64,
    })


REFERENCE_SCHEMA: pl.Schema = reference_schema(DEFAULT_DIM)


# ---------------------------------------------------------------------------
# Delay / window specs ('5d', '2W', '3M', 'expanding')
# ---------------------------------------------------------------------------

_PERIOD_RE = re.compile(r"^(\d+)([dWM])$")


def parse_period(spec: str, allowed_units: str) -> tuple[int, str]:
    m = _PERIOD_RE.match(spec)
    if not m or int(m.group(1)) <= 0:
        raise argparse.ArgumentTypeError(
            f"invalid period '{spec}', expected e.g. '5d', '2W', '3M' (n > 0)"
        )
    n, unit = int(m.group(1)), m.group(2)
    if unit not in allowed_units:
        raise argparse.ArgumentTypeError(
            f"period '{spec}' uses unit '{unit}', allowed units here are {list(allowed_units)}"
        )
    return n, unit


def offset_of(n: int, unit: str) -> pd.DateOffset:
    return {
        "d": pd.DateOffset(days=n),
        "W": pd.DateOffset(weeks=n),
        "M": pd.DateOffset(months=n),
    }[unit]


def parse_delay(spec: str) -> pd.DateOffset:
    """Minimum granularity is 1 day: 'Xd', 'XW', 'XM' all allowed."""
    n, unit = parse_period(spec, allowed_units="dWM")
    return offset_of(n, unit)


def parse_window(spec: str) -> str | pd.DateOffset:
    """'expanding', or a rolling window >= 1 week: 'XW', 'XM' (no days)."""
    if spec == "expanding":
        return "expanding"
    n, unit = parse_period(spec, allowed_units="WM")
    return offset_of(n, unit)


def _offset_days(offset: pd.DateOffset, anchor: pd.Timestamp) -> int:
    """Calendar-day length of a DateOffset measured back from a fixed anchor."""
    return (anchor - (anchor - offset)).days


def validate_window_after_delay(delay: pd.DateOffset, window: str | pd.DateOffset) -> None:
    """Raise if a rolling window would not fully clear the delay period."""
    if window == "expanding":
        return
    anchor = pd.Timestamp("2000-01-01")
    delay_days = _offset_days(delay, anchor)
    window_days = _offset_days(window, anchor)
    if window_days <= delay_days:
        raise ValueError(
            f"rolling window ({window_days}d) must be longer than the delay "
            f"({delay_days}d); a window this short would look inside the delay period"
        )


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceValue:
    """One pooled reference: what a correction is taken against.

    ``date`` is the day the value is FOR (None when undated); ``mu`` is the
    raw pooled vector (not unit norm), ``mu_hat`` its unit direction, ``n``
    the number of embeddings pooled (None when read from an old series).
    """

    date: date | None
    mu: np.ndarray
    mu_hat: np.ndarray
    n: int | None = None

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.mu.astype(np.float64)))


def load_reference_series(path: Path) -> pl.DataFrame:
    """Read a persisted series (DATE, MU, MU_HAT, N), sorted by DATE."""
    return pl.read_parquet(path).sort("DATE")


def resolve_reference(series: pl.DataFrame, day: date) -> ReferenceValue:
    """Exact-date row for ``day``, else the latest row before it. Never a later row.

    The series already embeds its own delay (its row for t pools data up to
    t - delay), so "row dated <= day" is the whole point-in-time guarantee.
    """
    candidates = series.filter(pl.col("DATE") <= day)
    if candidates.is_empty():
        raise LookaheadError(f"reference series has no row at or before {day}")
    row = candidates.sort("DATE").row(-1, named=True)
    return ReferenceValue(
        date=row["DATE"],
        mu=np.asarray(row["MU"], dtype=np.float32),
        mu_hat=np.asarray(row["MU_HAT"], dtype=np.float32),
        n=row.get("N"),
    )


# ---------------------------------------------------------------------------
# ReferenceVector
# ---------------------------------------------------------------------------

_UNDATED = None    # the single state key of an undated reference


class ReferenceVector:
    """Pooled reference vector, online (``update``) and offline (``fit``).

    Args:
        pooling: ``"mean"`` (sum / n), ``"min"`` or ``"max"`` (elementwise).
        as_of:   False: one pooled vector over everything seen. True: a dated
                 reference keeping one state per day; ``value(day)`` pools the
                 window ``(cutoff - window, cutoff]``, ``cutoff = day - delay``.
        delay:   dated only; '1d', '2W', '1M' (>= 1 day).
        window:  dated only; 'expanding' or a rolling period ('8W', '3M').

    Embeddings are pooled as given (no renormalisation), like the original
    mu_asof over stored unit embeddings.
    """

    def __init__(self, pooling: Pooling = "mean", *, as_of: bool = False,
                 delay: str = "1d", window: str = "expanding") -> None:
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        self.pooling: str = pooling
        self.as_of = bool(as_of)
        self.delay_spec, self.window_spec = delay, window
        self._delay = parse_delay(delay)
        self._window = parse_window(window)
        validate_window_after_delay(self._delay, self._window)
        self.dim: int | None = None
        # key (np.datetime64[D] when dated, None otherwise) -> [stat (d,), n]
        self._state: dict[Any, list] = {}
        self._through: np.datetime64 | None = None
        self._built: dict[str, Any] | None = None

    # -- construction -----------------------------------------------------

    def _identity(self) -> np.ndarray:
        assert self.dim is not None
        if self.pooling == "mean":
            return np.zeros(self.dim, dtype=np.float64)
        fill = np.inf if self.pooling == "min" else -np.inf
        return np.full(self.dim, fill, dtype=np.float32)

    def _check_rows(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError(f"expected (n, d) embeddings, got shape {X.shape}")
        if self.dim is None:
            self.dim = int(X.shape[1])
        elif X.shape[1] != self.dim:
            raise ValueError(f"embedding dim {X.shape[1]} != reference dim {self.dim}")
        if X.size and not np.isfinite(X).all():
            raise ValueError("embeddings contain NaN or inf; refusing to pool them")
        return X

    def _key(self, day: date | None) -> Any:
        if self.as_of:
            if day is None:
                raise ValueError("a dated reference (as_of=True) needs a day for every update")
            return np.datetime64(day, "D")
        if day is not None:
            raise ValueError("an undated reference (as_of=False) takes no day")
        return _UNDATED

    def _fold(self, key: Any, stat: np.ndarray, n: int) -> None:
        """Merge one batch statistic (sum / min / max over its rows) into a key."""
        cur = self._state.get(key)
        if cur is None:
            self._state[key] = [stat, n]
            return
        if self.pooling == "mean":
            cur[0] = cur[0] + stat
        elif self.pooling == "min":
            cur[0] = np.minimum(cur[0], stat)
        else:
            cur[0] = np.maximum(cur[0], stat)
        cur[1] += n

    def update(self, X: np.ndarray | None = None, *, day: date | None = None,
               texts: list[str] | None = None, embedder: Any = None) -> ReferenceVector:
        """Fold a new batch into the state (online). O(n d), vectorised.

        Give either ``X`` ((n, d) or (d,) embeddings) or ``texts`` with an
        ``embedder`` exposing ``encode(texts) -> (n, d)``. An empty batch is
        allowed: for a dated reference it records ``day`` as seen (a day with
        no data), which ``value`` needs to know the state is complete through it.
        """
        if texts is not None:
            if embedder is None or X is not None:
                raise ValueError("pass either X, or texts together with an embedder")
            X = np.atleast_2d(embedder.encode(texts))
        if X is None:
            raise ValueError("nothing to update with: pass X or texts + embedder")
        X = self._check_rows(X)
        key = self._key(day)
        if len(X):
            if self.pooling == "mean":
                stat = X.sum(axis=0, dtype=np.float64)
            elif self.pooling == "min":
                stat = X.min(axis=0).astype(np.float32)
            else:
                stat = X.max(axis=0).astype(np.float32)
            self._fold(key, stat, len(X))
        elif key not in self._state and self.dim is not None:
            self._state[key] = [self._identity(), 0]
        if self.as_of and (self._through is None or key > self._through):
            self._through = key
        self._built = None
        return self

    @classmethod
    def fit(cls, X: np.ndarray, days: Any = None, **kwargs: Any) -> ReferenceVector:
        """Offline: build the state from all rows at once (one vectorised group-by)."""
        ref = cls(**kwargs)
        X = ref._check_rows(X)
        if not ref.as_of:
            if days is not None:
                raise ValueError("an undated reference takes no days")
            return ref.update(X)
        keys = np.asarray(days, dtype="datetime64[D]")
        if keys.shape != (len(X),):
            raise ValueError(f"need one day per row: {keys.shape} days for {len(X)} rows")
        order = np.argsort(keys, kind="stable")
        keys, X = keys[order], X[order]
        uniq, starts = np.unique(keys, return_index=True)
        counts = np.diff(np.append(starts, len(keys)))
        stats = {
            "mean": lambda: np.add.reduceat(X.astype(np.float64), starts, axis=0),
            "min": lambda: np.minimum.reduceat(X, starts, axis=0).astype(np.float32),
            "max": lambda: np.maximum.reduceat(X, starts, axis=0).astype(np.float32),
        }[ref.pooling]()
        return ref._load_daily(uniq, stats, counts)

    @classmethod
    def from_daily(cls, days: Any, counts: np.ndarray, *, sums: np.ndarray | None = None,
                   mins: np.ndarray | None = None, maxs: np.ndarray | None = None,
                   **kwargs: Any) -> ReferenceVector:
        """Offline: build a dated state from per-day aggregates (e.g. computed in SQL)."""
        if not kwargs.pop("as_of", True):
            raise ValueError("per-day aggregates build a dated reference (as_of=True)")
        ref = cls(as_of=True, **kwargs)
        stats = {"mean": sums, "min": mins, "max": maxs}[ref.pooling]
        if stats is None:
            raise ValueError(f"pooling={ref.pooling!r} needs its per-day aggregate")
        dtype = np.float64 if ref.pooling == "mean" else np.float32
        return ref._load_daily(np.asarray(days, dtype="datetime64[D]"),
                               np.asarray(stats, dtype=dtype), np.asarray(counts))

    def _load_daily(self, keys: np.ndarray, stats: np.ndarray,
                    counts: np.ndarray) -> ReferenceVector:
        if stats.ndim != 2 or len(stats) != len(keys) or len(counts) != len(keys):
            raise ValueError("days, per-day aggregates and counts must align")
        self.dim = int(stats.shape[1])
        for k, s, n in zip(keys, stats, counts):
            self._fold(k, s.copy(), int(n))
        if len(keys):
            self._through = keys.max()
        self._built = None
        return self

    # -- queries ----------------------------------------------------------

    def _build(self) -> dict[str, Any]:
        """Dense daily grid + prefix structures, rebuilt lazily after updates."""
        if self._built is not None:
            return self._built
        if not self._state:
            raise LookaheadError("reference vector has no data")
        keys = np.array(sorted(self._state), dtype="datetime64[D]")
        start = keys[0]
        n_grid = int((self._through - start).astype(int)) + 1
        idx = (keys - start).astype(int)
        counts = np.zeros(n_grid, dtype=np.int64)
        counts[idx] = [self._state[k][1] for k in keys]
        stat = np.tile(self._identity(), (n_grid, 1))
        stat[idx] = np.stack([self._state[k][0] for k in keys])
        built: dict[str, Any] = {
            "start": start, "n": n_grid,
            "cumcount": np.concatenate([[0], np.cumsum(counts)]),
        }
        if self.pooling == "mean":
            built["cumsum"] = np.vstack([np.zeros((1, self.dim)), np.cumsum(stat, axis=0)])
        else:
            built["stat"] = stat
        self._built = built
        return built

    def _window_bounds(self, days: pd.DatetimeIndex,
                       b: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Prefix-array bounds (start, end]: index i means "through grid day i - 1"."""
        grid_start = pd.Timestamp(b["start"])
        cutoff = days - self._delay
        end = np.clip((cutoff - grid_start).days.to_numpy() + 1, 0, b["n"])
        if self._window == "expanding":
            begin = np.zeros_like(end)
        else:
            begin = np.clip(((cutoff - self._window) - grid_start).days.to_numpy() + 1, 0, b["n"])
        return begin, end

    def _range_extreme(self, begin: np.ndarray, end: np.ndarray, b: dict[str, Any]) -> np.ndarray:
        """Elementwise min / max of grid rows [begin, end) per query (sparse table)."""
        op = np.minimum if self.pooling == "min" else np.maximum
        stat = b["stat"]
        out = np.tile(self._identity(), (len(begin), 1))
        ok = end > begin
        if self._window == "expanding":
            if "prefix" not in b:
                b["prefix"] = op.accumulate(stat, axis=0)
            out[ok] = b["prefix"][end[ok] - 1]
            return out
        if "table" not in b:                      # levels[k][i] = op over rows [i, i + 2^k)
            levels = [stat]
            while 2 ** len(levels) <= len(stat):
                half = 2 ** (len(levels) - 1)
                prev = levels[-1]
                levels.append(op(prev[:-half], prev[half:]))
            b["table"] = levels
        levels = b["table"]
        length = end - begin
        k = np.zeros_like(length)
        k[ok] = np.floor(np.log2(length[ok])).astype(int)
        for level in np.unique(k[ok]):
            sel = ok & (k == level)
            lo, hi = begin[sel], end[sel] - 2 ** level
            out[sel] = op(levels[level][lo], levels[level][hi])
        return out

    def _pooled(self, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pooled (k, d) float64, norms (k,), counts (k,)) for each day."""
        b = self._build()
        begin, end = self._window_bounds(days, b)
        n = b["cumcount"][end] - b["cumcount"][begin]
        if self.pooling == "mean":
            total = b["cumsum"][end] - b["cumsum"][begin]
            norms = np.linalg.norm(total, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = total / n[:, None]
            # mu_hat from the SUM (same direction as the mean, as in mu_asof)
            return np.concatenate([mu, total], axis=1), norms, n
        pooled = self._range_extreme(begin, end, b).astype(np.float64)
        pooled[n == 0] = 0.0
        norms = np.linalg.norm(pooled, axis=1)
        return np.concatenate([pooled, pooled], axis=1), norms, n

    def series(self, days: Any = None) -> pl.DataFrame:
        """The persisted daily form (REFERENCE_SCHEMA) for each requested day.

        Defaults to every day with data. Days whose window holds no data, or
        whose pooled vector is exactly zero, are skipped (logged), matching the
        original mu_asof series.
        """
        if not self.as_of:
            raise ValueError("series() needs a dated reference (as_of=True)")
        if days is None:
            keys = np.array(sorted(k for k, v in self._state.items() if v[1] > 0),
                            dtype="datetime64[D]")
        else:
            keys = np.asarray(days, dtype="datetime64[D]")
        idx = pd.DatetimeIndex(keys)
        both, norms, n = self._pooled(idx)
        d = self.dim
        keep = (n > 0) & (norms > 0)
        if (~keep).any():
            log.warning("%d/%d day(s) skipped: empty window or zero pooled vector",
                        int((~keep).sum()), len(keep))
        mu = both[keep, :d].astype(np.float32)
        mu_hat = (both[keep, d:] / norms[keep, None]).astype(np.float32)
        return pl.DataFrame(
            {"DATE": keys[keep], "MU": list(mu), "MU_HAT": list(mu_hat), "N": n[keep]},
            schema=reference_schema(d),
        )

    def value(self, day: date | None = None) -> ReferenceValue:
        """The reference for ``day`` (dated) or over everything seen (undated).

        Raises ``LookaheadError`` when the window holds no data (or pools to
        an exactly-zero vector), and ``ValueError`` when the state has not been
        updated through the cutoff yet: pooling then would silently drop days
        that simply have not been seen.
        """
        if not self.as_of:
            if day is not None:
                raise ValueError("an undated reference takes no day")
            state = self._state.get(_UNDATED)
            if state is None or state[1] == 0:
                raise LookaheadError("reference vector has no data")
            stat, n = state
            mu = stat / n if self.pooling == "mean" else stat.astype(np.float64)
            norm = float(np.linalg.norm(stat if self.pooling == "mean" else mu))
            if norm == 0.0:
                raise LookaheadError("pooled reference vector is exactly zero")
            direction = (stat if self.pooling == "mean" else mu) / norm
            return ReferenceValue(None, mu.astype(np.float32), direction.astype(np.float32), int(n))
        if day is None:
            raise ValueError("a dated reference needs the day to evaluate")
        cutoff = np.datetime64(pd.Timestamp(day) - self._delay, "D")
        if self._through is None or cutoff > self._through:
            raise ValueError(f"state covers through {self._through}, not the cutoff {cutoff} "
                             f"of {day}; update it first")
        frame = self.series([day])
        if frame.is_empty():
            raise LookaheadError(f"no reference data in the window for {day}")
        row = frame.row(0, named=True)
        return ReferenceValue(day, np.asarray(row["MU"], dtype=np.float32),
                              np.asarray(row["MU_HAT"], dtype=np.float32), int(row["N"]))

    def describe(self) -> dict[str, Any]:
        """Identity of the reference, for artifact hyperparams / metadata."""
        d: dict[str, Any] = {"pooling": self.pooling, "as_of": self.as_of, "dim": self.dim}
        if self.as_of:
            d.update(delay=self.delay_spec, window=self.window_spec)
        return d
