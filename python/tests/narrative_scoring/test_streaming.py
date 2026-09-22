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


class _MockSentiment:
    def scores(self, story_ids):
        # sign from the story index, NaN for every third headline
        out = np.array([float(int(s.rsplit("-", 1)[1]) % 5 - 2) for s in story_ids])
        out[[i for i, s in enumerate(story_ids) if int(s.rsplit("-", 1)[1]) % 3 == 0]] = np.nan
        return out

    def describe(self):
        return "mock"


def test_sentiment_provider_aligned_with_rows(month):
    days, src = month
    src.sentiment = _MockSentiment()
    chunks = list(src.iter_day(date(2008, 9, 15)))
    sent = np.concatenate([c.sentiment for c in chunks])
    assert sent.dtype == np.float32 and sent.shape == (50,)
    assert np.isnan(sent).sum() == len([i for i in range(50) if i % 3 == 0])
    assert "sentiment:mock" in src.describe()


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
