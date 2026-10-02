"""Bounded-memory, point-in-time access to headline embeddings, one day at a time.

A ``HeadlineSource`` yields, for one calendar day, ``Chunk`` objects: a contiguous
float32 (n, EMBEDDING_DIM) embedding block of at most ``chunk_size`` rows, plus, when
the pass has those layers, each row's sentiment tag code (``tags.SentimentTags``) and
its assets (CSR over ``assets.AssetUniverse``). That is the only contract the pipeline
relies on, so a live micro-batch, a historical range and an in-memory test fixture
all go through the same code. Every row of a day is yielded (the null model sees every
headline); the tags only route a row's scores.

``PartitionHeadlineSource`` is the datalake implementation: each storage partition is
read ONCE (headlines, embeddings, and the sentiment grid / entity lists when needed),
joined once on RP_STORY_ID with strict coverage, sorted by (day, RP_STORY_ID) and served
day by day as slices. ``ParquetHeadlineSource`` is the previous per-day DuckDB query,
kept as the reference the partition source is tested against (same rows, same order,
same float32 values). Rows are ordered by RP_STORY_ID within a day so the chunk stream,
and with it the seeded null-draw sample, is deterministic across runs.

Point-in-time: a sentiment score and an entity list are functions of the story as
published, available at the headline's own timestamp.
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
import polars as pl
import pyarrow as pa

from datalake.layout import Layout
from narrative_scoring.assets import ENTITY_COL, AssetUniverse, headline_assets
from narrative_scoring.schema import EMBEDDING_DIM
from narrative_scoring.tags import UNTAGGED, SentimentTags
from nlp.sentiment import GRID_COLUMNS
from nlp.sentiment.ordinal_sql import reference

log = logging.getLogger(__name__)

ID_COL = "RP_STORY_ID"
_SCORE_BLOCK = 1_000_000          # rows per block when computing grid scores


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
    """One bounded block of a day's headlines.

    ``tags``: int8 tag code per row (-1 untagged), None without a sentiment layer.
    ``asset_indptr`` / ``asset_idx`` / ``asset_rel``: each row's assets (CSR, one entry
    per (row, asset), RELEVANCE = max), None without an asset layer."""

    embeddings: np.ndarray                 # float32 (n, EMBEDDING_DIM), C-contiguous
    tags: np.ndarray | None = None
    asset_indptr: np.ndarray | None = None
    asset_idx: np.ndarray | None = None
    asset_rel: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.embeddings.shape[0])


class HeadlineSource(Protocol):
    """Point-in-time day access: every headline of the day, in a deterministic order."""

    def iter_day(self, day: date) -> Iterator[Chunk]: ...

    def describe(self) -> str: ...


@dataclass
class ParquetHeadlineSource:
    """The per-day DuckDB query (headlines x embeddings within the day, ordered by
    RP_STORY_ID): the reference ``PartitionHeadlineSource`` is tested against. It reads
    a whole embeddings partition for every day; production uses the partition source."""

    headlines_dir: Path
    embeddings_dir: Path
    chunk_size: int = 8_192
    threads: int = 8
    duckdb_memory_limit: str = "6GB"
    temp_directory: str | None = "/tmp/duckdb_spill"
    source_id: str = ""
    layout: Layout = field(default_factory=Layout)

    def __post_init__(self) -> None:
        self.headlines_dir = Path(self.headlines_dir)
        self.embeddings_dir = Path(self.embeddings_dir)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

    def describe(self) -> str:
        return self.source_id or f"parquet:{self.headlines_dir.name}+{self.embeddings_dir.name}"

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
        hl = self.layout.file_for(self.headlines_dir, day)
        emb = self.layout.file_for(self.embeddings_dir, day)
        within = (f"h.TIMESTAMP_UTC >= '{day.isoformat()}' "
                  f"AND h.TIMESTAMP_UTC < '{(day + timedelta(days=1)).isoformat()}'")
        joined = f"read_parquet('{hl}') h JOIN read_parquet('{emb}') e USING (RP_STORY_ID)"
        if count:
            return f"SELECT COUNT(*) FROM {joined} WHERE {within}"
        return (f"SELECT CAST(e.EMBEDDING AS FLOAT[{EMBEDDING_DIM}]) AS EMBEDDING "
                f"FROM {joined} WHERE {within} ORDER BY h.RP_STORY_ID")

    def day_total(self, day: date) -> int:
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
                if batch.num_rows:
                    yield Chunk(arrow_embeddings_to_numpy(batch.column(0)))
        finally:
            conn.close()


def day_expr(dtype: pl.DataType) -> pl.Expr:
    """The calendar day of TIMESTAMP_UTC, stored as an ISO string (the ingest) or a
    datetime; the string form compares exactly as the day query's string bounds do."""
    ts = pl.col("TIMESTAMP_UTC")
    if dtype == pl.String:
        return ts.str.slice(0, 10).str.to_date("%Y-%m-%d")
    return ts.dt.date()


def grid_tag_codes(P: np.ndarray, tags: SentimentTags) -> np.ndarray:
    """Tag codes of the grid rows ``P`` (n, 41): the ordinal rule's score
    (``ordinal_sql.reference``, the reference the SQL rules are tested against), then
    the tag intervals; an all-null row (no model output) is untagged."""
    out = np.full(P.shape[0], UNTAGGED, dtype=np.int8)
    for i in range(0, P.shape[0], _SCORE_BLOCK):
        sent, _ = reference(P[i:i + _SCORE_BLOCK], tags.rule)
        out[i:i + _SCORE_BLOCK] = tags.codes(sent)
    return out


@dataclass
class PartitionHeadlineSource:
    """Every input of a pass read ONCE per storage partition, served day by day.

    A partition's needed columns are read once and joined once on RP_STORY_ID, with
    strict coverage: every headline has exactly one embedding row (and one sentiment
    row when ``sentiment_dir`` is set) and no row lacks a headline; a mismatch raises,
    naming the partition. The rows are sorted by (day, RP_STORY_ID) and kept, the
    embeddings in float16 (converted to float32 exactly, chunk by chunk), the tag codes
    in int8 and the assets as CSR, until a day of another partition is asked for.
    Same rows, order and float32 values as ``ParquetHeadlineSource``.

    One partition is held at a time (2022-03: ~9.5M rows x 384 x 2 B = 7.3 GB of
    embeddings); the pipeline's ``prefetch`` thread loads the next one while the last
    chunks of the previous are scored.
    """

    headlines_dir: Path
    embeddings_dir: Path
    chunk_size: int = 8_192
    source_id: str = ""
    layout: Layout = field(default_factory=Layout)
    sentiment_dir: Path | None = None
    tags: SentimentTags | None = None
    assets: AssetUniverse | None = None
    _path: Path | None = field(default=None, init=False, repr=False)
    _emb: np.ndarray | None = field(default=None, init=False, repr=False)
    _codes: np.ndarray | None = field(default=None, init=False, repr=False)
    _csr: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(default=None, init=False,
                                                                   repr=False)
    _days: dict[date, tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.headlines_dir = Path(self.headlines_dir)
        self.embeddings_dir = Path(self.embeddings_dir)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if (self.sentiment_dir is None) != (self.tags is None):
            raise ValueError("sentiment tags need the sentiment directory, and vice versa")
        if self.sentiment_dir is not None:
            self.sentiment_dir = Path(self.sentiment_dir)

    def describe(self) -> str:
        return self.source_id or f"partition:{self.headlines_dir.name}+{self.embeddings_dir.name}"

    def has_day(self, day: date) -> bool:
        return (self.layout.file_for(self.headlines_dir, day).exists()
                and self.layout.file_for(self.embeddings_dir, day).exists())

    def _read_strict(self, path: Path, columns: list[str], what: str, n: int,
                     ids: pl.DataFrame) -> pl.DataFrame:
        """``columns`` of ``path`` in the order of ``ids`` (one row per story id)."""
        if not path.exists():
            raise FileNotFoundError(f"{what} partition missing: {path}")
        df = pl.read_parquet(path, columns=[ID_COL, *columns])
        if df[ID_COL].is_duplicated().any():
            raise ValueError(f"{path.name}: duplicate {ID_COL} in the {what} partition")
        out = ids.join(df, on=ID_COL, how="inner", maintain_order="left")
        if out.height != n or df.height != n:
            raise ValueError(f"{path.name}: {n} headlines, {df.height} {what} rows, "
                             f"{out.height} matched (strict coverage)")
        return out

    def _load(self, day: date) -> bool:
        """Hold the partition of ``day`` (reading it if another one is held). False when
        the partition does not exist."""
        hl_path = self.layout.file_for(self.headlines_dir, day)
        if hl_path == self._path:
            return True
        self._path, self._emb, self._codes, self._csr, self._days = None, None, None, None, {}
        if not self.has_day(day):
            return False
        name = hl_path.name
        cols = [ID_COL, "TIMESTAMP_UTC"]
        if self.assets is not None:
            cols += [ENTITY_COL, "RELEVANCE"]
        try:
            hl = pl.read_parquet(hl_path, columns=cols)
        except pl.exceptions.ColumnNotFoundError as exc:
            raise ValueError(f"{name}: the headlines lack {cols[2:]} (the asset layer needs "
                             "headlines with RELEVANCE)") from exc
        if hl[ID_COL].is_duplicated().any():
            raise ValueError(f"{name}: duplicate {ID_COL} in the headlines partition")
        rows = (hl.with_columns(day_expr(hl.schema["TIMESTAMP_UTC"]).alias("_day"))
                .drop("TIMESTAMP_UTC").sort(["_day", ID_COL])
                .with_row_index("_r"))
        n = rows.height
        ids = rows.select(ID_COL)
        em = self._read_strict(self.layout.file_for(self.embeddings_dir, day), ["EMBEDDING"],
                               "embeddings", n, ids)
        if em["EMBEDDING"].null_count():
            raise ValueError(f"{name}: {em['EMBEDDING'].null_count()} null embedding(s)")
        emb = em["EMBEDDING"].to_numpy()
        if emb.shape != (n, EMBEDDING_DIM):
            raise ValueError(f"{name}: embeddings of shape {emb.shape}")
        del em
        codes = None
        if self.tags is not None:
            grid = self._read_strict(self.layout.file_for(self.sentiment_dir, day),
                                     list(GRID_COLUMNS), "sentiment", n, ids)
            P = grid.select(GRID_COLUMNS).to_numpy().astype(np.float64)
            del grid
            codes = grid_tag_codes(P, self.tags)
            del P
        csr = None
        if self.assets is not None:
            csr = headline_assets(rows, self.assets.asset_map(), n)
        bounds = (rows.group_by("_day")
                  .agg(pl.col("_r").min().alias("lo"), pl.col("_r").max().alias("hi")))
        self._days = {d: (int(lo), int(hi) + 1) for d, lo, hi in bounds.iter_rows()}
        self._emb, self._codes, self._csr, self._path = emb, codes, csr, hl_path
        log.info("partition %s: %d headlines over %d day(s) held", name, n, len(self._days))
        return True

    def day_total(self, day: date) -> int:
        with self._lock:
            if not self._load(day):
                return 0
            lo, hi = self._days.get(day, (0, 0))
            return hi - lo

    def iter_day(self, day: date) -> Iterator[Chunk]:
        with self._lock:
            if not self._load(day) or day not in self._days:
                return
            lo, hi = self._days[day]
            emb, codes, csr = self._emb, self._codes, self._csr
        for i in range(lo, hi, self.chunk_size):
            j = min(i + self.chunk_size, hi)
            block = emb[i:j]
            if not np.isfinite(block).all():
                raise ValueError(f"{day}: non-finite embedding value(s)")
            chunk = Chunk(np.ascontiguousarray(block, dtype=np.float32))
            if codes is not None:
                chunk.tags = codes[i:j]
            if csr is not None:
                indptr, aidx, arel = csr
                a0, a1 = indptr[i], indptr[j]
                chunk.asset_indptr = indptr[i:j + 1] - a0
                chunk.asset_idx, chunk.asset_rel = aidx[a0:a1], arel[a0:a1]
            yield chunk


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

    ``tags`` (optional, per day, int8 codes over that day's rows) plays the sentiment
    layer; ``assets`` (optional, per day, a list per row of (asset index, relevance))
    the asset layer."""

    days: dict[date, np.ndarray] = field(default_factory=dict)
    chunk_size: int = 8_192
    source_id: str = "in-memory"
    tags: dict[date, np.ndarray] | None = None
    assets: dict[date, list[list[tuple[int, int]]]] | None = None

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
        codes = None
        if self.tags is not None:
            codes = np.asarray(self.tags.get(day, np.full(X.shape[0], UNTAGGED)), dtype=np.int8)
            if codes.shape != (X.shape[0],):
                raise ValueError(f"tags for {day} have shape {codes.shape}, expected "
                                 f"({X.shape[0]},)")
        for i in range(0, X.shape[0], self.chunk_size):
            j = min(i + self.chunk_size, X.shape[0])
            chunk = Chunk(X[i:j], tags=codes[i:j] if codes is not None else None)
            if self.assets is not None:
                per_row = self.assets.get(day, [[] for _ in range(X.shape[0])])[i:j]
                counts = [len(r) for r in per_row]
                chunk.asset_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
                chunk.asset_idx = np.array([a for r in per_row for a, _ in r], dtype=np.int32)
                chunk.asset_rel = np.array([w for r in per_row for _, w in r], dtype=np.uint8)
            yield chunk


def prefetch(iterator: Iterator[Any], depth: int = 2) -> Iterator[Any]:
    """Run ``iterator`` on a background thread with a bounded queue.

    The next chunk's (or next partition's) read overlaps with scoring the current one.
    Exceptions on the producer side are re-raised on the consumer side; the queue depth
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
