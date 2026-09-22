"""Monthly null partitions (``f0_monthly_partitions``): the production F0 pool.

While the scorer streams the days of month M, every headline's primitive
score vector (post-correction, post-pooling) is trimmed with the existing F0
rule (top ``trim_frac`` dropped) and a uniform subsample of the kept draws
(``config.null_draws_per_headline`` per headline) is folded into M's
t-digest and Welford moments. Raw draws are never retained.

A month is FINALISED on schedule: when its last calendar day is closed, or
when the scorer closes any day of a LATER month (whatever days were closed
by then are the month's coverage; days never seen simply do not contribute).
One immutable ``YYYY-MM.parquet`` row (schema.F0_PARTITION_SCHEMA) records
N_DAYS_CLOSED / N_DAYS_IN_MONTH; coverage below ``COVERAGE_WARN`` is logged
as a warning, never blocks. Until finalisation the open state (digest,
moments, days closed, RNG state) is persisted under ``_open/`` after every
closed day, so a live process closing one day per invocation and a replay
over a range produce the same partition through the same code path.

Determinism: the per-month RNG is seeded from (seed, year, month), the
headline source delivers rows in a fixed order (ParquetHeadlineSource
orders by RP_STORY_ID), and the digest is order-deterministic, so the same
inputs and seed give byte-identical partitions.
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
class MonthState:
    year: int
    month: int
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
        self.rng = np.random.default_rng([self.seed, self.year, self.month])

    @property
    def complete(self) -> bool:
        return len(self.days_covered) >= month_days(self.year, self.month)

    @property
    def coverage(self) -> float:
        return len(self.days_covered) / month_days(self.year, self.month)

    # -- persistence of the open state ------------------------------------

    def save(self, path: Path) -> None:
        self.digest.flush()
        payload = {
            "year": self.year, "month": self.month, "seed": self.seed,
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
    def load(cls, path: Path) -> "MonthState":
        with np.load(path) as z:
            meta = json.loads(z["meta"].tobytes().decode())
            means, weights = z["means"], z["weights"]
        st = cls(meta["year"], meta["month"], meta["seed"], meta["compression"])
        st.digest = TDigest.from_arrays(means, weights, *meta["digest"], meta["compression"])
        st.welford = Welford(*meta["welford"])
        st.days_covered = {date.fromisoformat(d) for d in meta["days_covered"]}
        st.n_headlines, st.n_available = meta["n_headlines"], meta["n_available"]
        st.n_sampled, st.rss_peak = meta["n_sampled"], meta["rss_peak"]
        st.rng.bit_generator.state = meta["rng_state"]
        return st


class MonthlyNullPartitionWriter:
    """The ``NullDrawSink`` that builds and persists monthly partitions under ``out_dir``."""

    def __init__(
        self, out_dir: Path, *, config: ScoringConfig, table: PrimitiveTable, seed: int = 0,
        compression: float = DEFAULT_COMPRESSION, input_ids: dict[str, Any] | None = None,
        code_version: str | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.open_dir = self.out_dir / OPEN_DIR
        self.open_dir.mkdir(parents=True, exist_ok=True)
        self.config, self.table, self.seed = config, table, seed
        self.compression = float(compression)
        self.input_ids = dict(input_ids or {})
        self.code_version = code_version
        self.states: dict[tuple[int, int], MonthState] = {}
        for p in sorted(self.open_dir.glob("*.npz")):
            st = MonthState.load(p)
            self.states[(st.year, st.month)] = st
        self.finalised: list[Path] = []

    # -- sink protocol ----------------------------------------------------

    def _state(self, day: date) -> MonthState:
        key = (day.year, day.month)
        st = self.states.get(key)
        if st is None:
            if (self.out_dir / partition_file(*key)).exists():
                raise RuntimeError(f"partition {partition_file(*key)} is already final; "
                                   "it cannot be re-accumulated")
            st = MonthState(day.year, day.month, self.seed, self.compression)
            self.states[key] = st
        return st

    def rng_for(self, day: date) -> np.random.Generator:
        return self._state(day).rng

    def add_draws(self, day: date, draws: np.ndarray, n_available: int, n_headlines: int) -> None:
        st = self._state(day)
        if day in st.days_covered:
            raise RuntimeError(f"{day} was already closed in partition "
                               f"{partition_file(st.year, st.month)}")
        st.digest.add(draws)
        st.welford.add(draws)
        st.n_headlines += int(n_headlines)
        st.n_available += int(n_available)
        st.n_sampled += int(draws.size)

    def close_day(self, day: date, rss_gb: float) -> None:
        """Mark ``day`` closed; finalise its month if complete, and any earlier open month."""
        self.finalise_before(day)
        st = self._state(day)
        st.days_covered.add(day)
        st.rss_peak = max(st.rss_peak, float(rss_gb) if np.isfinite(rss_gb) else 0.0)
        if st.complete:
            self._finalise_state(st)
        else:
            st.save(self.open_dir / f"{st.year}-{st.month:02d}.npz")

    def finalise_before(self, day: date) -> list[Path]:
        """Finalise every open month strictly earlier than ``day``'s month (on schedule).

        A month with no closed day at all (e.g. before the first mu_asof row)
        has nothing to finalise and is dropped instead of written empty.
        """
        out = []
        for key in sorted(self.states):
            if key < (day.year, day.month):
                st = self.states[key]
                if st.days_covered:
                    out.append(self._finalise_state(st))
                else:
                    (self.open_dir / f"{key[0]}-{key[1]:02d}.npz").unlink(missing_ok=True)
                    del self.states[key]
        return out

    def _finalise_state(self, st: MonthState) -> Path:
        key = (st.year, st.month)
        path = self._finalise(st)
        self.finalised.append(path)
        (self.open_dir / f"{key[0]}-{key[1]:02d}.npz").unlink(missing_ok=True)
        del self.states[key]
        return path

    def flush_open(self) -> None:
        for (y, m), st in self.states.items():
            st.save(self.open_dir / f"{y}-{m:02d}.npz")

    # -- finalisation -----------------------------------------------------

    def _finalise(self, st: MonthState) -> Path:
        st.digest.flush()
        if st.coverage < COVERAGE_WARN:
            log.warning("null partition %d-%02d finalised with coverage %.2f (%d/%d days closed)",
                        st.year, st.month, st.coverage, len(st.days_covered),
                        month_days(st.year, st.month))
        row = {
            "MONTH_END": month_end(st.year, st.month),
            "F0_CONFIG_ID": self.config.f0_digest(),
            "TAXONOMY_SHA1": self.table.taxonomy_sha1,
            "PARAPHRASE_SHA1": self.table.paraphrase_sha1,
            "SEED": st.seed, "COMPRESSION": st.compression,
            "N_HEADLINES": st.n_headlines,
            "N_DAYS_CLOSED": len(st.days_covered),
            "N_DAYS_IN_MONTH": month_days(st.year, st.month),
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
        path = self.out_dir / partition_file(st.year, st.month)
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
