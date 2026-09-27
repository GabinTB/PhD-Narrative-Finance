"""Null partitions (kind ``f0_monthly_partitions``): the production F0 pool.

One partition per CALIBRATION PERIOD of frequency ``freq`` (``datalake.periods``;
default M, a calendar month -- the kind and column names keep their historical
"month" wording: MONTH_END is the period's last day, N_DAYS_IN_MONTH its day
count). While the scorer streams the days of period P, every headline's
primitive score vector (post-correction, post-pooling) is trimmed with the
existing F0 rule (top ``trim_frac`` dropped) and a uniform subsample of the
kept draws (``config.null_draws_per_headline`` per headline) is folded into
P's t-digest and Welford moments. Raw draws are never retained.

A period is FINALISED on schedule: when its last calendar day is closed, or
when the scorer closes any day of a LATER period (whatever days were closed
by then are the period's coverage; days never seen simply do not contribute).
One immutable ``{period key}.parquet`` row (schema.F0_PARTITION_SCHEMA)
records N_DAYS_CLOSED / N_DAYS_IN_MONTH; coverage below ``COVERAGE_WARN`` is
logged as a warning, never blocks. Until finalisation the open state (digest,
moments, days closed, RNG state) is persisted under ``_open/`` after every
closed day, so a live process closing one day per invocation and a replay
over a range produce the same partition through the same code path.

Determinism: the per-period RNG is seeded by ``period_rng_seed`` (for M the
historical (seed, year, month)), the headline source delivers rows in a fixed
order (ParquetHeadlineSource orders by RP_STORY_ID), and the digest is
order-deterministic, so the same inputs and seed give byte-identical partitions.
"""
from __future__ import annotations

import calendar
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl

from datalake.periods import Period, parse_key, period_of, period_rng_seed
from datalake.periods import partition_file as period_file
from narrative_scoring.config import ScoringConfig
from narrative_scoring.f0 import TDigest, Welford
from narrative_scoring.primitives import PrimitiveTable
from narrative_scoring.schema import F0_PARTITION_SCHEMA

log = logging.getLogger(__name__)

OPEN_DIR = "_open"
DEFAULT_COMPRESSION = 2000.0
COVERAGE_WARN = 0.80


class NullDrawSink(Protocol):
    def add_draws(self, day: date, draws: np.ndarray, n_available: int,
                  n_headlines: int) -> None: ...

    def close_day(self, day: date, rss_gb: float) -> None: ...

    def rng_for(self, day: date) -> np.random.Generator: ...


def month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def month_days(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def partition_file(y: int, m: int) -> str:
    return f"{y}-{m:02d}.parquet"


@dataclass
class PeriodState:
    """The open accumulation of one calibration period."""

    period: Period
    seed: int
    compression: float
    digest: TDigest = field(init=False)
    welford: Welford = field(default_factory=Welford)
    rng: np.random.Generator = field(init=False)
    days_covered: set[date] = field(default_factory=set)
    n_headlines: int = 0
    n_available: int = 0
    n_sampled: int = 0
    rss_peak: float = 0.0

    def __post_init__(self) -> None:
        self.digest = TDigest(self.compression)
        self.rng = np.random.default_rng(period_rng_seed(self.seed, self.period))

    @property
    def key(self) -> str:
        return self.period.key

    @property
    def complete(self) -> bool:
        return len(self.days_covered) >= self.period.n_days

    @property
    def coverage(self) -> float:
        return len(self.days_covered) / self.period.n_days

    # -- persistence of the open state ------------------------------------

    def save(self, path: Path) -> None:
        self.digest.flush()
        payload = {
            "key": self.period.key, "seed": self.seed,
            "compression": self.compression,
            "welford": [self.welford.count, self.welford.mean, self.welford.m2],
            "days_covered": sorted(d.isoformat() for d in self.days_covered),
            "n_headlines": self.n_headlines, "n_available": self.n_available,
            "n_sampled": self.n_sampled, "rss_peak": self.rss_peak,
            "digest": [self.digest.count, self.digest.min, self.digest.max],
            "rng_state": self.rng.bit_generator.state,
        }
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, means=self.digest.means, weights=self.digest.weights,
                     meta=np.frombuffer(json.dumps(payload).encode(), dtype=np.uint8))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "PeriodState":
        with np.load(path) as z:
            meta = json.loads(z["meta"].tobytes().decode())
            means, weights = z["means"], z["weights"]
        period = (parse_key(meta["key"]) if "key" in meta          # legacy open state: a month
                  else period_of(date(meta["year"], meta["month"], 1), "M"))
        st = cls(period, meta["seed"], meta["compression"])
        st.digest = TDigest.from_arrays(means, weights, *meta["digest"], meta["compression"])
        st.welford = Welford(*meta["welford"])
        st.days_covered = {date.fromisoformat(d) for d in meta["days_covered"]}
        st.n_headlines, st.n_available = meta["n_headlines"], meta["n_available"]
        st.n_sampled, st.rss_peak = meta["n_sampled"], meta["rss_peak"]
        st.rng.bit_generator.state = meta["rng_state"]
        return st


class NullPartitionWriter:
    """The ``NullDrawSink`` that builds and persists one partition per calibration
    period (``freq``, default M) under ``out_dir``."""

    def __init__(
        self, out_dir: Path, *, config: ScoringConfig, table: PrimitiveTable, seed: int = 0,
        compression: float = DEFAULT_COMPRESSION, input_ids: dict[str, Any] | None = None,
        code_version: str | None = None, freq: str = "M",
    ):
        self.freq = freq
        self.out_dir = Path(out_dir)
        self.open_dir = self.out_dir / OPEN_DIR
        self.open_dir.mkdir(parents=True, exist_ok=True)
        self.config, self.table, self.seed = config, table, seed
        self.compression = float(compression)
        self.input_ids = dict(input_ids or {})
        self.code_version = code_version
        self.states: dict[str, PeriodState] = {}
        for p in sorted(self.open_dir.glob("*.npz")):
            st = PeriodState.load(p)
            if st.period.freq != freq:
                raise RuntimeError(f"open partition {p.name} is {st.period.freq}, "
                                   f"the writer is {freq}")
            self.states[st.key] = st
        self.finalised: list[Path] = []

    # -- sink protocol ----------------------------------------------------

    def period_of(self, day: date) -> Period:
        return period_of(day, self.freq)

    def is_final(self, period: Period) -> bool:
        return (self.out_dir / period_file(period.key)).exists()

    def _state(self, day: date) -> PeriodState:
        period = self.period_of(day)
        st = self.states.get(period.key)
        if st is None:
            if self.is_final(period):
                raise RuntimeError(f"partition {period_file(period.key)} is already final; "
                                   "it cannot be re-accumulated")
            st = PeriodState(period, self.seed, self.compression)
            self.states[period.key] = st
        return st

    def rng_for(self, day: date) -> np.random.Generator:
        return self._state(day).rng

    def add_draws(self, day: date, draws: np.ndarray, n_available: int, n_headlines: int) -> None:
        st = self._state(day)
        if day in st.days_covered:
            raise RuntimeError(f"{day} was already closed in partition "
                               f"{period_file(st.key)}")
        st.digest.add(draws)
        st.welford.add(draws)
        st.n_headlines += int(n_headlines)
        st.n_available += int(n_available)
        st.n_sampled += int(draws.size)

    def close_day(self, day: date, rss_gb: float) -> None:
        """Mark ``day`` closed; finalise its period if complete, and any earlier open one."""
        self.finalise_before(day)
        st = self._state(day)
        st.days_covered.add(day)
        st.rss_peak = max(st.rss_peak, float(rss_gb) if np.isfinite(rss_gb) else 0.0)
        if st.complete:
            self._finalise_state(st)
        else:
            st.save(self.open_dir / f"{st.key}.npz")

    def finalise_before(self, day: date) -> list[Path]:
        """Finalise every open period strictly earlier than ``day``'s period (on schedule).

        A period with no closed day at all (e.g. before the first mu_asof row)
        has nothing to finalise and is dropped instead of written empty.
        """
        out = []
        current = self.period_of(day).first
        for st in sorted(self.states.values(), key=lambda s: s.period.first):
            if st.period.last < current:
                if st.days_covered:
                    out.append(self._finalise_state(st))
                else:
                    (self.open_dir / f"{st.key}.npz").unlink(missing_ok=True)
                    del self.states[st.key]
        return out

    def _finalise_state(self, st: PeriodState) -> Path:
        path = self._finalise(st)
        self.finalised.append(path)
        (self.open_dir / f"{st.key}.npz").unlink(missing_ok=True)
        del self.states[st.key]
        return path

    def flush_open(self) -> None:
        for st in self.states.values():
            st.save(self.open_dir / f"{st.key}.npz")

    # -- finalisation -----------------------------------------------------

    def _finalise(self, st: PeriodState) -> Path:
        st.digest.flush()
        if st.coverage < COVERAGE_WARN:
            log.warning("null partition %s finalised with coverage %.2f (%d/%d days closed)",
                        st.key, st.coverage, len(st.days_covered), st.period.n_days)
        row = {
            "MONTH_END": st.period.last,
            "F0_CONFIG_ID": self.config.f0_digest(),
            "TAXONOMY_SHA1": self.table.taxonomy_sha1,
            "PARAPHRASE_SHA1": self.table.paraphrase_sha1,
            "SEED": st.seed, "COMPRESSION": st.compression,
            "N_HEADLINES": st.n_headlines,
            "N_DAYS_CLOSED": len(st.days_covered),
            "N_DAYS_IN_MONTH": st.period.n_days,
            "COVERAGE": st.coverage,
            "N_DRAWS_AVAILABLE": st.n_available, "N_DRAWS_SAMPLED": st.n_sampled,
            "WELFORD_COUNT": st.welford.count, "WELFORD_MEAN": st.welford.mean,
            "WELFORD_M2": st.welford.m2,
            "DIGEST_MEANS": st.digest.means.tolist(),
            "DIGEST_WEIGHTS": st.digest.weights.tolist(),
            "DIGEST_MIN": st.digest.min if st.digest.count else float("nan"),
            "DIGEST_MAX": st.digest.max if st.digest.count else float("nan"),
            "RSS_PEAK_GB": st.rss_peak,
            "INPUT_IDS": json.dumps(self.input_ids, sort_keys=True),
            "CODE_VERSION": self.code_version or "",
        }
        path = self.out_dir / period_file(st.key)
        tmp = path.with_suffix(".parquet.tmp")
        pl.DataFrame([row], schema=F0_PARTITION_SCHEMA).write_parquet(tmp, compression="zstd")
        tmp.replace(path)
        log.info("finalised null partition %s: %d headlines, %d/%d draws sampled, %d centroids",
                 path.name, st.n_headlines, st.n_sampled, st.n_available, st.digest.n_centroids)
        return path


# ---------------------------------------------------------------------------
# Reading partitions back
# ---------------------------------------------------------------------------

def load_partitions(directory: Path) -> pl.DataFrame:
    """All finalised partitions under ``directory``, sorted by MONTH_END."""
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema=F0_PARTITION_SCHEMA)
    return pl.concat([pl.read_parquet(f) for f in files]).sort("MONTH_END")


def partition_digest(row: dict[str, Any]) -> TDigest:
    return TDigest.from_arrays(
        np.asarray(row["DIGEST_MEANS"], dtype=np.float64),
        np.asarray(row["DIGEST_WEIGHTS"], dtype=np.float64),
        row["WELFORD_COUNT"] if row["DIGEST_MEANS"] else 0,
        row["DIGEST_MIN"], row["DIGEST_MAX"], row["COMPRESSION"],
    )


def partition_welford(row: dict[str, Any]) -> Welford:
    return Welford(int(row["WELFORD_COUNT"]), float(row["WELFORD_MEAN"]), float(row["WELFORD_M2"]))


def months_between(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m, y = (1, y + 1) if m == 12 else (m + 1, y)
    return out


def month_range_days(y: int, m: int) -> list[date]:
    first = date(y, m, 1)
    return [first + timedelta(days=i) for i in range(month_days(y, m))]


MonthlyNullPartitionWriter = NullPartitionWriter      # historical name
MonthState = PeriodState                              # historical name


class SkipClosedDays:
    """Null sink for a RESUMED run: never feeds a day twice.

    The partition writer checkpoints its open month at every closed day (draw
    summaries, closed days, RNG state) and reloads it on start. A resumed run
    re-scores the unfinished month for narrative_daily; its days already closed
    in the checkpoint -- or in an already final partition -- get a scratch RNG and
    their draws are dropped (draws only feed partitions, never the day's scores).
    The remaining days continue the saved RNG stream, so the partition equals the
    one an uninterrupted run would have written.
    """

    def __init__(self, writer: NullPartitionWriter) -> None:
        self.writer = writer
        self._scratch = np.random.default_rng(0)

    def _closed(self, day: date) -> bool:
        period = self.writer.period_of(day)
        if self.writer.is_final(period):
            return True
        st = self.writer.states.get(period.key)
        return st is not None and day in st.days_covered

    def rng_for(self, day: date) -> np.random.Generator:
        return self._scratch if self._closed(day) else self.writer.rng_for(day)

    def add_draws(self, day: date, draws: np.ndarray, n_available: int, n_headlines: int) -> None:
        if not self._closed(day):
            self.writer.add_draws(day, draws, n_available, n_headlines)

    def close_day(self, day: date, rss_gb: float) -> None:
        if not self._closed(day):
            self.writer.close_day(day, rss_gb)

    def finalise_before(self, day: date) -> list[Path]:
        return self.writer.finalise_before(day)

    def flush_open(self) -> None:
        self.writer.flush_open()

    @property
    def finalised(self) -> list[Path]:
        return self.writer.finalised
