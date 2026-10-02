"""Parquet day access: SQL day filter, single-copy conversion, sentiment-bucket filter,
day totals, prefetch error propagation."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from datalake.layout import Layout
from narrative_scoring.config import SENTIMENT_BUCKETS, ScoringConfig
from narrative_scoring.schema import EMBEDDING_DIM
from narrative_scoring.sentiment_filter import SentimentBucket, bucket_of
from narrative_scoring.streaming import (
    InMemoryHeadlineSource,
    ParquetHeadlineSource,
    arrow_embeddings_to_numpy,
    prefetch,
    rss_gb,
)
from nlp.sentiment import GRID_COLUMNS
from nlp.sentiment.ordinal_sql import reference


def _file(day):
    return Layout().file_for(Path('.'), day).name


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
    name = _file(next(iter(per_day)))
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
    assert sum(c.n for c in src.iter_day(date(2008, 9, 16))) == 23
    assert src.day_total(date(2008, 9, 15)) == 50 and src.day_total(date(2008, 9, 16)) == 23
    assert src.day_total(date(2008, 9, 17)) == 0 and src.day_total(date(2008, 10, 1)) == 0
    assert list(src.iter_day(date(2008, 9, 17))) == []            # a day with no headlines
    assert list(src.iter_day(date(2008, 10, 1))) == []            # a month with no file
    assert src.has_day(date(2008, 9, 1)) and not src.has_day(date(2009, 1, 1))


D = date(2008, 9, 15)


def _story_ids(day: date, n: int) -> list[str]:
    return [f"{day.isoformat()}-{i}" for i in range(n)]


def _write_grid_sentiment(root: Path, n: int, *, drop: int | None = None,
                          seed: int = 1) -> tuple[Path, list[str], np.ndarray]:
    """A model (grid) sentiment partition for the fixture month: random distributions,
    one point mass per pole, and story 0 with no model output (all null)."""
    import pyarrow.parquet as pq

    d = root / "sentiment_grid"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = _story_ids(D, n) + _story_ids(date(2008, 9, 16), 23)
    G = rng.dirichlet(np.full(41, 0.3), size=len(ids))
    G[1], G[2] = np.eye(41)[0], np.eye(41)[40]
    G[0] = np.nan
    G16 = G.astype(np.float16)
    keep = [i for i, sid in enumerate(ids) if not (drop is not None and sid == ids[drop])]
    empty = np.isnan(G16).all(axis=1)[keep]
    cols = {"RP_STORY_ID": pa.array([ids[i] for i in keep])}
    for j, c in enumerate(GRID_COLUMNS):
        cols[c] = pa.array(G16[keep, j], type=pa.float16(), mask=empty)
    pq.write_table(pa.table(cols), d / _file(D))
    return d, ids, G16


def _write_css_sentiment(root: Path, n: int) -> tuple[Path, list[str], np.ndarray]:
    d = root / "sentiment_css"
    d.mkdir(exist_ok=True)
    ids = _story_ids(D, n) + _story_ids(date(2008, 9, 16), 23)
    vals = np.array([np.nan if i % 4 == 0 else ((i * 37) % 21 - 10) / 10 for i in range(len(ids))],
                    np.float32)
    pl.DataFrame({"RP_STORY_ID": ids, "SENT_CSS": pl.Series(vals, dtype=pl.Float32),
                  "SENT_ESS_MEAN": pl.Series(vals, dtype=pl.Float32)}).write_parquet(d / _file(D))
    return d, ids, vals


def _cfg(bucket, source="ravenbert", rule="mean", neg_max=-1 / 3, pos_min=1 / 3, min_conf=0.0):
    return ScoringConfig(sentiment=bucket, sentiment_source=source, sentiment_rule=rule,
                         neg_max=neg_max, pos_min=pos_min, min_conf=min_conf)


SETUPS = [
    ("ravenbert", "mean", -1 / 3, 1 / 3, 0.0),
    ("ravenbert", "median", -0.5, 0.1, 0.6),             # asymmetric + a confidence floor
    ("finbert", "argmax", -0.2, 0.2, 0.3),
    ("ravenpack", "css", -1 / 3, 1 / 3, 0.0),
    ("ravenpack", "css", -0.05, 0.5, 0.0),
]


@pytest.mark.parametrize("source,rule,neg_max,pos_min,min_conf", SETUPS)
def test_buckets_match_numpy_and_partition_the_day(month, tmp_path, source, rule, neg_max,
                                                   pos_min, min_conf):
    days, src = month
    n = 50
    if source == "ravenpack":
        sdir, ids, vals = _write_css_sentiment(tmp_path, n)
        sent = np.where(np.isnan(vals), np.nan, vals).astype(np.float64)
        conf = np.full(len(ids), np.nan)
    else:
        sdir, ids, G16 = _write_grid_sentiment(tmp_path, n)
        sent, conf = reference(G16, rule)
    order = sorted(range(n), key=lambda i: ids[i])          # the source's RP_STORY_ID order
    X = days[D].astype(np.float16).astype(np.float32)
    total = 0
    for bucket in SENTIMENT_BUCKETS:
        cfg = _cfg(bucket, source, rule, neg_max, pos_min, min_conf)
        src.sentiment = SentimentBucket(cfg, sdir, "art-1")
        got = [c.embeddings for c in src.iter_day(D)]
        got = np.concatenate(got) if got else np.zeros((0, EMBEDDING_DIM), np.float32)
        want_rows = [i for i in order if bucket_of(sent[[i]], conf[[i]], cfg)[0] == bucket.value]
        np.testing.assert_allclose(got, X[want_rows], atol=1e-6)
        total += got.shape[0]
        assert src.day_total(D) == n                          # the filter never changes it
    assert total == n                                          # four buckets = the day
    if source != "ravenpack":                                  # story 0 has no model output
        cfg = _cfg(SENTIMENT_BUCKETS[-1], source, rule, neg_max, pos_min, min_conf)
        assert bucket_of(sent[[0]], conf[[0]], cfg)[0] == "unscored"


def test_bucket_filter_strict_coverage_and_missing_file(month, tmp_path):
    _, src = month
    sdir, _, _ = _write_grid_sentiment(tmp_path, 50, drop=7)
    for bucket in SENTIMENT_BUCKETS:                           # whatever the bucket
        src.sentiment = SentimentBucket(_cfg(bucket), sdir, "art-1")
        with pytest.raises(LookupError, match="no row"):
            list(src.iter_day(D))
    (tmp_path / "empty").mkdir()
    src.sentiment = SentimentBucket(_cfg(SENTIMENT_BUCKETS[0]), tmp_path / "empty", "art-1")
    with pytest.raises(FileNotFoundError, match="sentiment partition file missing"):
        list(src.iter_day(D))
    assert "art-1:mean:positive" in src.describe()


def test_bucket_needs_a_filtered_config(tmp_path):
    with pytest.raises(ValueError, match="sentiment filter"):
        SentimentBucket(ScoringConfig(), tmp_path)


def test_arrow_conversion_rejects_nulls():
    arr = pa.array([[1.0] * EMBEDDING_DIM, None], type=pa.list_(pa.float32(), EMBEDDING_DIM))
    with pytest.raises(ValueError, match="null"):
        arrow_embeddings_to_numpy(arr)
    ok = pa.array([[1.0] * EMBEDDING_DIM, [2.0] * EMBEDDING_DIM],
                  type=pa.list_(pa.float32(), EMBEDDING_DIM))
    X = arrow_embeddings_to_numpy(ok)
    assert X.shape == (2, EMBEDDING_DIM) and X[1, 0] == 2.0 and X.flags.c_contiguous


def test_in_memory_source_chunks_keep_mask_and_day_total():
    X = np.arange(10 * EMBEDDING_DIM, dtype=np.float64).reshape(10, EMBEDDING_DIM)
    src = InMemoryHeadlineSource({date(2008, 1, 1): X}, chunk_size=4)
    chunks = list(src.iter_day(date(2008, 1, 1)))
    assert [c.n for c in chunks] == [4, 4, 2]
    assert all(c.embeddings.dtype == np.float32 for c in chunks)
    assert list(src.iter_day(date(2008, 1, 2))) == []
    keep = np.arange(10) % 3 == 0
    kept = InMemoryHeadlineSource({date(2008, 1, 1): X}, chunk_size=4,
                                  keep={date(2008, 1, 1): keep})
    got = np.concatenate([c.embeddings for c in kept.iter_day(date(2008, 1, 1))])
    np.testing.assert_array_equal(got, X[keep].astype(np.float32))
    assert kept.day_total(date(2008, 1, 1)) == 10 and kept.day_total(date(2008, 1, 2)) == 0
    none_kept = InMemoryHeadlineSource({date(2008, 1, 1): X}, keep={})
    assert list(none_kept.iter_day(date(2008, 1, 1))) == []


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


# ---------------------------------------------------------------------------
# PartitionHeadlineSource: one read per partition, the same chunks as the day query
# ---------------------------------------------------------------------------

def _partition_source(src: ParquetHeadlineSource, chunk_size: int = 16):
    from narrative_scoring.streaming import PartitionHeadlineSource

    return PartitionHeadlineSource(src.headlines_dir, src.embeddings_dir, chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [1, 7, 16, 1000])
def test_partition_source_yields_the_day_query_chunks(month, chunk_size):
    days, sql = month
    sql.chunk_size = chunk_size
    part = _partition_source(sql, chunk_size)
    for d in (date(2008, 9, 15), date(2008, 9, 16), date(2008, 9, 17), date(2008, 10, 1)):
        want = [c.embeddings for c in sql.iter_day(d)]
        got = [c.embeddings for c in part.iter_day(d)]
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert g.dtype == np.float32 and g.flags.c_contiguous
            np.testing.assert_array_equal(g, w)                    # bit for bit
        assert part.day_total(d) == sql.day_total(d)


def test_partition_source_reads_a_partition_once(month, monkeypatch):
    _, sql = month
    part = _partition_source(sql)
    reads = []
    real = pl.read_parquet
    monkeypatch.setattr(pl, "read_parquet", lambda *a, **k: reads.append(a[0]) or real(*a, **k))
    for d in (date(2008, 9, 15), date(2008, 9, 16), date(2008, 9, 15)):
        list(part.iter_day(d))
        part.day_total(d)
    assert len(reads) == 2                                        # headlines + embeddings


def _rewrite(path: Path, fn) -> None:
    fn(pl.read_parquet(path)).write_parquet(path)


@pytest.mark.parametrize("what", ["missing", "extra", "duplicate"])
def test_partition_source_strict_coverage(month, what):
    _, sql = month
    name = _file(date(2008, 9, 15))
    if what == "missing":
        _rewrite(sql.embeddings_dir / name, lambda df: df.slice(1))
    elif what == "extra":
        _rewrite(sql.headlines_dir / name, lambda df: df.slice(1))
    else:
        _rewrite(sql.embeddings_dir / name, lambda df: pl.concat([df, df.slice(0, 1)]))
    with pytest.raises(ValueError, match="strict coverage|duplicate"):
        list(_partition_source(sql).iter_day(date(2008, 9, 15)))


def test_partition_source_accepts_datetime_timestamps(month):
    _, sql = month
    want = [c.embeddings for c in sql.iter_day(date(2008, 9, 16))]
    _rewrite(sql.headlines_dir / _file(date(2008, 9, 16)),
             lambda df: df.with_columns(pl.col("TIMESTAMP_UTC").str.to_datetime()))
    got = [c.embeddings for c in _partition_source(sql).iter_day(date(2008, 9, 16))]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)
