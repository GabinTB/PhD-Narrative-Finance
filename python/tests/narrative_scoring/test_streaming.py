"""Parquet day access: SQL day filter, single-copy conversion, prefetch error propagation."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from narrative_scoring.schema import EMBEDDING_DIM
from narrative_scoring.streaming import (
    InMemoryHeadlineSource,
    ParquetHeadlineSource,
    ParquetSentimentSource,
    arrow_embeddings_to_numpy,
    month_file,
    prefetch,
    rss_gb,
)


def _write_month(root: Path, per_day: dict[date, np.ndarray]) -> tuple[Path, Path]:
    hl_dir, em_dir = root / "headlines", root / "embeddings"
    hl_dir.mkdir(parents=True), em_dir.mkdir(parents=True)
    ids, ts, embs = [], [], []
    for d, X in per_day.items():
        for i, row in enumerate(X):
            ids.append(f"{d.isoformat()}-{i}")
            ts.append(f"{d.isoformat()} {i % 24:02d}:30:00.000")
            embs.append(row.astype(np.float16))
    hl = pl.DataFrame({"RP_STORY_ID": ids, "TIMESTAMP_UTC": ts, "SOURCE_NAME": ["s"] * len(ids)})
    em = pl.DataFrame({"RP_STORY_ID": ids[::-1], "EMBEDDING": embs[::-1]},
                      schema={"RP_STORY_ID": pl.String,
                              "EMBEDDING": pl.Array(pl.Float16, EMBEDDING_DIM)})
    name = month_file(next(iter(per_day)))
    hl.write_parquet(hl_dir / name)
    em.write_parquet(em_dir / name)
    return hl_dir, em_dir


@pytest.fixture
def month(tmp_path: Path):
    rng = np.random.default_rng(0)
    days = {date(2008, 9, 15): rng.normal(size=(50, EMBEDDING_DIM)).astype(np.float32),
            date(2008, 9, 16): rng.normal(size=(23, EMBEDDING_DIM)).astype(np.float32)}
    hl, em = _write_month(tmp_path, days)
    return days, ParquetHeadlineSource(hl, em, chunk_size=16, threads=2, temp_directory=None)


def test_day_filter_and_single_copy(month):
    days, src = month
    chunks = [c.embeddings for c in src.iter_day(date(2008, 9, 15))]
    assert [c.shape[0] for c in chunks] == [16, 16, 16, 2]
    for c in chunks:
        assert c.dtype == np.float32 and c.flags.c_contiguous and c.shape[1] == EMBEDDING_DIM
    got = np.concatenate(chunks)
    want = days[date(2008, 9, 15)].astype(np.float16).astype(np.float32)
    # rows come back ordered by RP_STORY_ID ("<day>-<i>" sorts lexicographically)
    order = np.argsort([f"{date(2008, 9, 15).isoformat()}-{i}" for i in range(50)])
    np.testing.assert_allclose(got, want[order], atol=1e-6)
    again = np.concatenate([c.embeddings for c in src.iter_day(date(2008, 9, 15))])
    np.testing.assert_array_equal(got, again)                     # deterministic order
    assert all(c.sentiment is None for c in src.iter_day(date(2008, 9, 15)))
    assert sum(c.n for c in src.iter_day(date(2008, 9, 16))) == 23
    assert list(src.iter_day(date(2008, 9, 17))) == []            # a day with no headlines
    assert list(src.iter_day(date(2008, 10, 1))) == []            # a month with no file
    assert src.has_day(date(2008, 9, 1)) and not src.has_day(date(2009, 1, 1))


def _write_sentiment(root: Path, day: date, n: int, *, drop: int | None = None,
                     value=None, column: str = "SENT_X", dtype=pl.Float32) -> Path:
    """A headline_sentiment month for the fixture day: SENT_X = i % 5 - 2 scaled to [-1, 1],
    NaN for every third story, plus the other fixture day's stories."""
    d = root / "sentiment"
    d.mkdir(exist_ok=True)
    ids, vals = [], []
    for dd, count in ((day, n), (date(2008, 9, 16), 23)):
        for i in range(count):
            if dd == day and i == drop:
                continue
            ids.append(f"{dd.isoformat()}-{i}")
            vals.append(float("nan") if i % 3 == 0 else (i % 5 - 2) / 2)
    if value is not None:
        vals[1] = value
    pl.DataFrame({"RP_STORY_ID": ids, column: pl.Series(vals, dtype=dtype),
                  "P_POS": pl.Series([0.5] * len(ids), dtype=pl.Float32)}) \
        .write_parquet(d / month_file(day))
    return d


D = date(2008, 9, 15)


def test_sentiment_joined_and_aligned_with_rows(month, tmp_path):
    days, src = month
    src.sentiment = ParquetSentimentSource(_write_sentiment(tmp_path, D, 50), "SENT_X", "art-1")
    chunks = list(src.iter_day(D))
    assert [c.n for c in chunks] == [16, 16, 16, 2]
    sent = np.concatenate([c.sentiment for c in chunks])
    # rows come back ordered by RP_STORY_ID, the sentiment must follow the same order
    order = sorted(range(50), key=lambda i: f"{D.isoformat()}-{i}")
    want = np.array([np.nan if i % 3 == 0 else (i % 5 - 2) / 2 for i in order], np.float32)
    np.testing.assert_array_equal(sent, want)                  # NaN passes through as NaN
    assert sent.dtype == np.float32
    assert src.describe().endswith("+sentiment:art-1:SENT_X")


@pytest.mark.parametrize("kwargs, column, err, match", [
    ({"drop": 7}, "SENT_X", LookupError, "no row"),
    ({"value": 1.5}, "SENT_X", ValueError, "outside"),
    ({"value": float("inf")}, "SENT_X", ValueError, "outside"),
    ({}, "SENT_Y", KeyError, "not in"),
    ({"dtype": pl.Float64}, "SENT_X", TypeError, "float32"),
])
def test_sentiment_strict_failures(month, tmp_path, kwargs, column, err, match):
    _, src = month
    d = _write_sentiment(tmp_path, D, 50, **kwargs)
    src.sentiment = ParquetSentimentSource(d, column, "art-1")
    with pytest.raises(err, match=match):
        list(src.iter_day(D))


def test_sentiment_missing_month_file_and_bad_column_name(month, tmp_path):
    _, src = month
    (tmp_path / "empty").mkdir()
    src.sentiment = ParquetSentimentSource(tmp_path / "empty", "SENT_X")
    with pytest.raises(FileNotFoundError, match="sentiment month file missing"):
        list(src.iter_day(D))
    with pytest.raises(ValueError, match="SENT_"):
        ParquetSentimentSource(tmp_path, "P_POS")
    with pytest.raises(ValueError, match="SENT_"):
        ParquetSentimentSource(tmp_path, "SENT_X; DROP TABLE x")


def test_arrow_conversion_rejects_nulls():
    arr = pa.array([[1.0] * EMBEDDING_DIM, None], type=pa.list_(pa.float32(), EMBEDDING_DIM))
    with pytest.raises(ValueError, match="null"):
        arrow_embeddings_to_numpy(arr)
    ok = pa.array([[1.0] * EMBEDDING_DIM, [2.0] * EMBEDDING_DIM],
                  type=pa.list_(pa.float32(), EMBEDDING_DIM))
    X = arrow_embeddings_to_numpy(ok)
    assert X.shape == (2, EMBEDDING_DIM) and X[1, 0] == 2.0 and X.flags.c_contiguous


def test_in_memory_source_chunks():
    X = np.arange(10 * EMBEDDING_DIM, dtype=np.float64).reshape(10, EMBEDDING_DIM)
    src = InMemoryHeadlineSource({date(2008, 1, 1): X}, chunk_size=4)
    chunks = list(src.iter_day(date(2008, 1, 1)))
    assert [c.n for c in chunks] == [4, 4, 2]
    assert all(c.embeddings.dtype == np.float32 and c.sentiment is None for c in chunks)
    assert list(src.iter_day(date(2008, 1, 2))) == []
    with_s = InMemoryHeadlineSource({date(2008, 1, 1): X}, chunk_size=4,
                                    sentiment={date(2008, 1, 1): np.arange(10) - 5.0})
    chunks = list(with_s.iter_day(date(2008, 1, 1)))
    np.testing.assert_array_equal(np.concatenate([c.sentiment for c in chunks]),
                                  np.arange(10) - 5.0)
    missing = InMemoryHeadlineSource({date(2008, 1, 1): X}, chunk_size=4, sentiment={})
    assert np.isnan(next(missing.iter_day(date(2008, 1, 1))).sentiment).all()


def test_prefetch_preserves_order_and_propagates_errors():
    def gen():
        for i in range(5):
            yield np.full((1, 1), i, dtype=np.float32)
        raise RuntimeError("producer failed")

    out = []
    with pytest.raises(RuntimeError, match="producer failed"):
        for x in prefetch(gen(), depth=2):
            out.append(int(x[0, 0]))
    assert out == [0, 1, 2, 3, 4]
    assert list(prefetch(iter([]), depth=1)) == []


def test_rss_is_reported():
    assert rss_gb() > 0.0
