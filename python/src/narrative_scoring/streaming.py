"""Bounded-memory, point-in-time access to headline embeddings, one day at a time.

A ``HeadlineSource`` yields, for one calendar day, ``Chunk`` objects: a
contiguous float32 (n, EMBEDDING_DIM) embedding block of at most
``chunk_size`` rows plus, optionally, one sentiment score per row (NaN where
unknown). That is the only contract the pipeline relies on, so a live
micro-batch, a historical range and an in-memory test fixture all go
through the same code.

``ParquetHeadlineSource`` is the datalake implementation. The day filter is
pushed into the DuckDB query (one query per day, ~0.8 s on a 1.1M-row
month measured locally), the embedding is cast to a fixed-size FLOAT array
in SQL, and each Arrow batch is viewed as numpy without going through
polars or a per-row list conversion: exactly one copy, made by DuckDB.
DuckDB's own memory limit bounds the join, and ``prefetch`` overlaps the
next chunk's I/O with the current chunk's scoring. Rows are ordered by
RP_STORY_ID so the chunk stream, and with it the seeded null-draw sample,
is deterministic across runs (a hash join's output order is not).

A filtered run (``config.sentiment`` = a bucket, sentiment_filter.py) scores only the
headlines of its bucket: the day query LEFT JOINs the SENT / CONF of the day's stories,
computed from the ``headline_sentiment`` partition (ravenpack/headlines/sentiment.py)
restricted to those stories, and keeps the rows of the bucket, so nothing else is read
into Python or scored. Strict coverage: a headline without a sentiment ROW raises; a row
with null P_* (no model output) is "unscored", not an error. ``day_total(day)`` counts
the day's headlines x embeddings rows with the SAME query builder and no filter: the
denominator a filtered run shares with the all-headlines run, so attention adds up.
The score is a function of the story as published, hence available at the headline's
own timestamp (point-in-time).
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator, Protocol

import duckdb
import numpy as np
import pyarrow as pa

from datalake.layout import Layout
from narrative_scoring.schema import EMBEDDING_DIM
from narrative_scoring.sentiment_filter import SentimentBucket

log = logging.getLogger(__name__)


class MemoryBudgetExceeded(RuntimeError):
    """Raised when process RSS crosses the budget; the message names the day."""


def rss_gb() -> float:
    """Resident set size of this process in GB (Linux /proc; NaN elsewhere)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


@dataclass
class Chunk:
    """One bounded block of a day's headlines."""

    embeddings: np.ndarray                 # float32 (n, EMBEDDING_DIM), C-contiguous

    @property
    def n(self) -> int:
        return int(self.embeddings.shape[0])


class HeadlineSource(Protocol):
    """Point-in-time day access: the day's (possibly bucket-filtered) headlines, and the
    day's total headline count before any filter."""

    def iter_day(self, day: date) -> Iterator[Chunk]: ...

    def day_total(self, day: date) -> int: ...

    def describe(self) -> str: ...


@dataclass
class ParquetHeadlineSource:
    """Partitioned headline + embedding artifacts, joined on RP_STORY_ID.

    Both artifacts share one ``layout`` (an embedding partition has the key of the
    headlines partition it was computed from), so a day is read from the one pair
    of files that holds it, whatever the partition frequency. With ``sentiment`` (a
    ``SentimentBucket``), only the headlines of that bucket are yielded.
    """

    headlines_dir: Path
    embeddings_dir: Path
    chunk_size: int = 8_192
    threads: int = 8
    duckdb_memory_limit: str = "6GB"
    temp_directory: str | None = "/tmp/duckdb_spill"
    source_id: str = ""
    sentiment: SentimentBucket | None = None
    layout: Layout = field(default_factory=Layout)

    def __post_init__(self) -> None:
        self.headlines_dir = Path(self.headlines_dir)
        self.embeddings_dir = Path(self.embeddings_dir)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

    def describe(self) -> str:
        base = self.source_id or f"parquet:{self.headlines_dir.name}+{self.embeddings_dir.name}"
        return base if self.sentiment is None else f"{base}+sentiment:{self.sentiment.describe()}"

    def has_day(self, day: date) -> bool:
        return (self.layout.file_for(self.headlines_dir, day).exists()
                and self.layout.file_for(self.embeddings_dir, day).exists())

    def _connect(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        conn.execute("SET enable_progress_bar=false")
        conn.execute(f"SET threads={int(self.threads)}")
        conn.execute(f"SET memory_limit='{self.duckdb_memory_limit}'")
        if self.temp_directory:
            Path(self.temp_directory).mkdir(parents=True, exist_ok=True)
            conn.execute(f"SET temp_directory='{self.temp_directory}'")
        return conn

    def _day_sql(self, day: date, *, count: bool = False) -> str:
        """THE day query: headlines x embeddings on RP_STORY_ID within the day. ``count``
        gives its row count with no sentiment filter (``day_total``); otherwise the
        embeddings (+ the strict-coverage flag, filtered to the bucket, when ``sentiment``
        is set) ordered by RP_STORY_ID."""
        hl = self.layout.file_for(self.headlines_dir, day)
        emb = self.layout.file_for(self.embeddings_dir, day)
        within = (f"h.TIMESTAMP_UTC >= '{day.isoformat()}' "
                  f"AND h.TIMESTAMP_UTC < '{(day + timedelta(days=1)).isoformat()}'")
        joined = f"read_parquet('{hl}') h JOIN read_parquet('{emb}') e USING (RP_STORY_ID)"
        if count:
            return f"SELECT COUNT(*) FROM {joined} WHERE {within}"
        if self.sentiment is None:
            return (f"SELECT CAST(e.EMBEDDING AS FLOAT[{EMBEDDING_DIM}]) AS EMBEDDING "
                    f"FROM {joined} WHERE {within} ORDER BY h.RP_STORY_ID")
        day_ids = (f"SELECT h.RP_STORY_ID FROM read_parquet('{hl}') h WHERE {within}")
        sent = self.sentiment.select_sql(self._sentiment_file(day), day_ids)
        return (f"SELECT CAST(e.EMBEDDING AS FLOAT[{EMBEDDING_DIM}]) AS EMBEDDING, "
                f"s.RP_STORY_ID IS NULL AS NO_ROW "
                f"FROM {joined} LEFT JOIN ({sent}) s USING (RP_STORY_ID) "
                f"WHERE {within} AND ({self.sentiment.predicate()} OR s.RP_STORY_ID IS NULL) "
                f"ORDER BY h.RP_STORY_ID")

    def day_total(self, day: date) -> int:
        """The day's headlines x embeddings rows, unfiltered (= a ``none`` run's N_HEADLINES)."""
        if not self.has_day(day):
            return 0
        conn = self._connect()
        try:
            return int(conn.sql(self._day_sql(day, count=True)).fetchone()[0])
        finally:
            conn.close()

    def iter_day(self, day: date) -> Iterator[Chunk]:
        if not self.has_day(day):
            return
        conn = self._connect()
        try:
            for batch in conn.sql(self._day_sql(day)).to_arrow_reader(self.chunk_size):
                if not batch.num_rows:
                    continue
                if self.sentiment is not None:
                    no_row = batch.column(1).to_numpy(zero_copy_only=False)
                    if no_row.any():
                        raise LookupError(
                            f"{day}: {int(no_row.sum())} headline(s) have no row in "
                            f"{self.sentiment.describe()} (strict coverage)")
                yield Chunk(arrow_embeddings_to_numpy(batch.column(0)))
        finally:
            conn.close()

    def _sentiment_file(self, day: date) -> Path:
        """The day's sentiment partition file (must exist)."""
        assert self.sentiment is not None
        path = self.layout.file_for(self.sentiment.sentiment_dir, day)
        if not path.exists():
            raise FileNotFoundError(f"{day}: sentiment partition file missing: {path} "
                                    f"({self.sentiment.describe()})")
        return path


def arrow_embeddings_to_numpy(column: pa.Array) -> np.ndarray:
    """FixedSizeList<float32>[dim] -> contiguous float32 (n, dim), zero-copy when possible."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if column.null_count:
        raise ValueError(f"{column.null_count} null embedding(s) in batch")
    flat = column.flatten() if hasattr(column, "flatten") else column.values
    X = flat.to_numpy(zero_copy_only=False).reshape(len(column), EMBEDDING_DIM)
    return np.ascontiguousarray(X, dtype=np.float32)


@dataclass
class InMemoryHeadlineSource:
    """Embeddings already in memory, keyed by day. Tests/notebooks.

    ``keep`` (optional, per day, a boolean mask over that day's rows) plays the role of a
    sentiment bucket: only kept rows are yielded, while ``day_total`` still counts all.
    """

    days: dict[date, np.ndarray] = field(default_factory=dict)
    chunk_size: int = 8_192
    source_id: str = "in-memory"
    keep: dict[date, np.ndarray] | None = None

    def describe(self) -> str:
        return self.source_id

    def day_total(self, day: date) -> int:
        X = self.days.get(day)
        return 0 if X is None else int(X.shape[0])

    def iter_day(self, day: date) -> Iterator[Chunk]:
        X = self.days.get(day)
        if X is None:
            return
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self.keep is not None:
            mask = np.asarray(self.keep.get(day, np.zeros(X.shape[0], bool)), dtype=bool)
            if mask.shape != (X.shape[0],):
                raise ValueError(f"keep mask for {day} has shape {mask.shape}, expected "
                                 f"({X.shape[0]},)")
            X = X[mask]
        for i in range(0, X.shape[0], self.chunk_size):
            yield Chunk(X[i:i + self.chunk_size])


def prefetch(iterator: Iterator[Any], depth: int = 2) -> Iterator[Any]:
    """Run ``iterator`` on a background thread with a bounded queue.

    DuckDB releases the GIL while producing batches, so the next chunk's
    join + decode overlaps with scoring the current one. Exceptions on the
    producer side are re-raised on the consumer side; the queue depth
    bounds how many chunks are in flight.
    """
    q: queue.Queue = queue.Queue(maxsize=max(1, depth))
    done = object()

    def run() -> None:
        try:
            for item in iterator:
                q.put(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer side
            q.put(exc)
        finally:
            q.put(done)

    threading.Thread(target=run, daemon=True, name="headline-prefetch").start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item
