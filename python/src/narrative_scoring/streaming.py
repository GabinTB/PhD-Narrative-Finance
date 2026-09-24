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

Sentiment is an optional side channel (``SentimentSource``): one SENT_* column
of a ``headline_sentiment`` artifact (ravenpack/headlines/sentiment.py, the
score database contract), LEFT JOINed on RP_STORY_ID in the same day query,
so the score arrives in the same Arrow batch as the embedding, already
aligned, with no per-row Python lookup. Strict: a missing month file or
column, a scored headline without a sentiment row, or a value outside
[-1, 1] raises; a NaN value is a legitimate "no score" and passes through.
The score is a function of the story as published, hence available at the
headline's own timestamp (point-in-time).
"""
from __future__ import annotations

import logging
import queue
import re
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator, Protocol

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from narrative_scoring.schema import EMBEDDING_DIM

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
    sentiment: np.ndarray | None = None    # float32 (n,), NaN = missing; None = no provider

    @property
    def n(self) -> int:
        return int(self.embeddings.shape[0])


class SentimentSource(Protocol):
    """Where a day's sentiment scores live: one column of monthly parquet files keyed by
    RP_STORY_ID (the headline_sentiment contract)."""

    column: str
    artifact_id: str

    def month_path(self, day: date) -> Path: ...

    def describe(self) -> str: ...


SCORE_COLUMN_RE = re.compile(r"^SENT_[A-Z0-9_]+$")


@dataclass
class ParquetSentimentSource:
    """One ``SENT_*`` column of a registered ``headline_sentiment`` artifact."""

    sentiment_dir: Path
    column: str
    artifact_id: str = ""

    def __post_init__(self) -> None:
        self.sentiment_dir = Path(self.sentiment_dir)
        if not SCORE_COLUMN_RE.match(self.column):
            raise ValueError(f"sentiment column must match {SCORE_COLUMN_RE.pattern}, "
                             f"got {self.column!r}")

    def month_path(self, day: date) -> Path:
        return self.sentiment_dir / month_file(day)

    def describe(self) -> str:
        return f"{self.artifact_id or self.sentiment_dir.name}:{self.column}"


class HeadlineSource(Protocol):
    """Point-in-time day access."""

    def iter_day(self, day: date) -> Iterator[Chunk]: ...

    def describe(self) -> str: ...


def month_file(day: date) -> str:
    return f"{day.year}-{day.month:02d}.parquet"


@dataclass
class ParquetHeadlineSource:
    """Monthly ``YYYY-MM.parquet`` headline + embedding artifacts, joined on RP_STORY_ID."""

    headlines_dir: Path
    embeddings_dir: Path
    chunk_size: int = 8_192
    threads: int = 8
    duckdb_memory_limit: str = "6GB"
    temp_directory: str | None = "/tmp/duckdb_spill"
    source_id: str = ""
    sentiment: SentimentSource | None = None

    def __post_init__(self) -> None:
        self.headlines_dir = Path(self.headlines_dir)
        self.embeddings_dir = Path(self.embeddings_dir)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

    def describe(self) -> str:
        base = self.source_id or f"parquet:{self.headlines_dir.name}+{self.embeddings_dir.name}"
        return base if self.sentiment is None else f"{base}+sentiment:{self.sentiment.describe()}"

    def has_day(self, day: date) -> bool:
        name = month_file(day)
        return (self.headlines_dir / name).exists() and (self.embeddings_dir / name).exists()

    def iter_day(self, day: date) -> Iterator[Chunk]:
        if not self.has_day(day):
            return
        name = month_file(day)
        hl, emb = self.headlines_dir / name, self.embeddings_dir / name
        day_after = (day + timedelta(days=1)).isoformat()
        sent_cols, sent_join = "", ""
        if self.sentiment is not None:
            sent_path = self._sentiment_file(day)
            col = self.sentiment.column
            sent_cols = f", CAST(s.{col} AS FLOAT) AS SENT, s.RP_STORY_ID IS NULL AS NO_ROW"
            sent_join = f"LEFT JOIN read_parquet('{sent_path}') s USING (RP_STORY_ID)"
        conn = duckdb.connect()
        try:
            conn.execute("SET enable_progress_bar=false")
            conn.execute(f"SET threads={int(self.threads)}")
            conn.execute(f"SET memory_limit='{self.duckdb_memory_limit}'")
            if self.temp_directory:
                Path(self.temp_directory).mkdir(parents=True, exist_ok=True)
                conn.execute(f"SET temp_directory='{self.temp_directory}'")
            query = f"""
                SELECT CAST(e.EMBEDDING AS FLOAT[{EMBEDDING_DIM}]) AS EMBEDDING{sent_cols}
                FROM read_parquet('{hl}') h
                JOIN read_parquet('{emb}') e USING (RP_STORY_ID)
                {sent_join}
                WHERE h.TIMESTAMP_UTC >= '{day.isoformat()}'
                  AND h.TIMESTAMP_UTC < '{day_after}'
                ORDER BY h.RP_STORY_ID
            """
            for batch in conn.sql(query).to_arrow_reader(self.chunk_size):
                if not batch.num_rows:
                    continue
                X = arrow_embeddings_to_numpy(batch.column(0))
                sent = (self._checked_sentiment(batch, day) if self.sentiment is not None
                        else None)
                yield Chunk(X, sent)
        finally:
            conn.close()

    def _sentiment_file(self, day: date) -> Path:
        """The day's sentiment month file, after the file and column checks."""
        assert self.sentiment is not None
        path = self.sentiment.month_path(day)
        if not path.exists():
            raise FileNotFoundError(f"{day}: sentiment month file missing: {path} "
                                    f"({self.sentiment.describe()})")
        schema = pq.read_schema(path)
        col = self.sentiment.column
        if col not in schema.names:
            raise KeyError(f"{day}: sentiment column {col!r} not in {path.name} "
                           f"(has {[n for n in schema.names if n.startswith('SENT_')]})")
        if schema.field(col).type != pa.float32():
            raise TypeError(f"{day}: sentiment column {col!r} is {schema.field(col).type}, "
                            f"expected float32")
        return path

    def _checked_sentiment(self, batch: pa.RecordBatch, day: date) -> np.ndarray:
        """float32 scores of the batch; raises on unmatched stories or out-of-range values."""
        no_row = batch.column(2).to_numpy(zero_copy_only=False)
        if no_row.any():
            raise LookupError(f"{day}: {int(no_row.sum())} scored headline(s) have no row in "
                              f"{self.sentiment.describe()} (strict coverage)")
        col = batch.column(1)
        if col.null_count:
            raise ValueError(f"{day}: {col.null_count} null sentiment value(s) in "
                             f"{self.sentiment.describe()}; the contract uses NaN")
        sent = col.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        bad = ~np.isnan(sent) & ~((sent >= -1.0) & (sent <= 1.0))
        if bad.any():
            raise ValueError(f"{day}: {int(bad.sum())} sentiment value(s) outside [-1, 1] in "
                             f"{self.sentiment.describe()}")
        return sent


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
    """Embeddings (and optional sentiment) already in memory, keyed by day. Tests/notebooks."""

    days: dict[date, np.ndarray] = field(default_factory=dict)
    chunk_size: int = 8_192
    source_id: str = "in-memory"
    sentiment: dict[date, np.ndarray] | None = None

    def describe(self) -> str:
        return self.source_id

    def iter_day(self, day: date) -> Iterator[Chunk]:
        X = self.days.get(day)
        if X is None:
            return
        X = np.ascontiguousarray(X, dtype=np.float32)
        sent = None
        if self.sentiment is not None:
            s = self.sentiment.get(day)
            sent = (np.asarray(s, dtype=np.float32) if s is not None
                    else np.full(X.shape[0], np.nan, dtype=np.float32))
            if sent.shape != (X.shape[0],):
                raise ValueError(f"sentiment for {day} has shape {sent.shape}, expected "
                                 f"({X.shape[0]},)")
        for i in range(0, X.shape[0], self.chunk_size):
            yield Chunk(X[i:i + self.chunk_size],
                        None if sent is None else sent[i:i + self.chunk_size])


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
