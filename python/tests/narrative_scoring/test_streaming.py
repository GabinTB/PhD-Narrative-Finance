"""Day access: the per-day SQL reference, the one-read partition source (embeddings, tag
codes, assets from the entities table), UTC days, single-copy conversion, day totals,
prefetch error propagation. Headlines are written in the rp_headlines shape (UTC
datetimes); entities in the rp_headline_entities shape (one row per story x entity)."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from datalake.layout import Layout
from narrative_scoring.schema import EMBEDDING_DIM
from narrative_scoring.streaming import (
    InMemoryHeadlineSource,
    ParquetHeadlineSource,
    arrow_embeddings_to_numpy,
    prefetch,
    rss_gb,
)
from narrative_scoring.tags import SentimentTags
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
            ts.append(datetime(d.year, d.month, d.day, i % 24, 30, tzinfo=timezone.utc))
            embs.append(row.astype(np.float16))
    hl = pl.DataFrame({"RP_STORY_ID": ids, "TIMESTAMP_UTC": ts, "SOURCE_NAME": ["s"] * len(ids)},
                      schema_overrides={"TIMESTAMP_UTC": pl.Datetime("us", "UTC")})
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


TERNARY = [("neg", "x <= -1/3"), ("neu", "-1/3 < x < 1/3"), ("pos", "x >= 1/3")]


def _tagged_source(src: ParquetHeadlineSource, sdir: Path, rule: str, chunk_size: int = 16,
                   tags=TERNARY, assets=None):
    from narrative_scoring.streaming import PartitionHeadlineSource

    return PartitionHeadlineSource(src.headlines_dir, src.embeddings_dir, chunk_size=chunk_size,
                                   sentiment_dir=sdir,
                                   tags=SentimentTags.build("ravenbert", rule, tags),
                                   assets=assets)


@pytest.mark.parametrize("rule", ["mean", "median", "argmax"])
def test_partition_source_gives_each_row_its_grid_tag(month, tmp_path, rule):
    days, sql = month
    n = 50
    sdir, ids, G16 = _write_grid_sentiment(tmp_path, n)
    part = _tagged_source(sql, sdir, rule)
    chunks = list(part.iter_day(D))
    codes = np.concatenate([c.tags for c in chunks])
    order = sorted(range(n), key=lambda i: ids[i])          # the source's RP_STORY_ID order
    sent, _ = reference(G16[order], rule)
    want = SentimentTags.build("ravenbert", rule, TERNARY).codes(sent)
    np.testing.assert_array_equal(codes, want)
    assert codes[order.index(0)] == -1                       # no model output: untagged
    assert set(codes.tolist()) >= {0, 2}                     # the two point masses
    # the embeddings are those of the untagged source, every row yielded
    got = np.concatenate([c.embeddings for c in chunks])
    np.testing.assert_array_equal(got, np.concatenate([c.embeddings for c in sql.iter_day(D)]))


def test_partition_source_sentiment_strict_coverage(month, tmp_path):
    _, sql = month
    sdir, _, _ = _write_grid_sentiment(tmp_path, 50, drop=7)
    with pytest.raises(ValueError, match="strict coverage"):
        list(_tagged_source(sql, sdir, "mean").iter_day(D))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="sentiment partition missing"):
        list(_tagged_source(sql, tmp_path / "empty", "mean").iter_day(D))
    from narrative_scoring.streaming import PartitionHeadlineSource

    with pytest.raises(ValueError, match="sentiment directory"):
        PartitionHeadlineSource(sql.headlines_dir, sql.embeddings_dir, sentiment_dir=sdir)


def _with_entities(sql: ParquetHeadlineSource, extra: pl.DataFrame | None = None,
                   ) -> tuple[Path, dict[str, list]]:
    """An entities partition next to the fixture headlines (rp_headline_entities shape:
    one row per story x entity): row 0 has AAPL and USPL, odd rows MSFT and OTHER, even
    rows none."""
    hl = pl.read_parquet(sql.headlines_dir / _file(D))
    rows = []
    for i, sid in enumerate(hl["RP_STORY_ID"].to_list()):
        if i == 0:
            rows += [(sid, "AAPL", 100), (sid, "USPL", 20)]
        elif i % 2:
            rows += [(sid, "MSFT", i % 101), (sid, "OTHER", 100)]
    ent = pl.DataFrame(rows, schema={"RP_STORY_ID": pl.String, "RP_ENTITY_ID": pl.String,
                                     "RELEVANCE": pl.UInt8}, orient="row")
    if extra is not None:
        ent = pl.concat([ent, extra])
    edir = sql.headlines_dir.parent / "entities"
    edir.mkdir(exist_ok=True)
    ent.write_parquet(edir / _file(D))
    by_id: dict[str, list] = {}
    for sid, e, w in rows:
        by_id.setdefault(sid, []).append((e, w))
    return edir, by_id


def _universe(*ids: str):
    from narrative_scoring.assets import AssetUniverse

    return AssetUniverse.from_frame(pl.DataFrame({"snapshot_date": [date(2008, 1, 1)] * len(ids),
                                                  "rp_entity_id": list(ids)}), artifact_id="u")


def test_partition_source_gives_each_row_its_assets_once(month):
    from narrative_scoring.streaming import PartitionHeadlineSource

    _, sql = month
    edir, by_id = _with_entities(sql)
    part = PartitionHeadlineSource(sql.headlines_dir, sql.embeddings_dir, chunk_size=7,
                                   assets=_universe("AAPL", "MSFT"), entities_dir=edir)
    rows = []
    for c in part.iter_day(D):
        for r in range(c.n):
            lo, hi = c.asset_indptr[r], c.asset_indptr[r + 1]
            rows.append(list(zip(c.asset_idx[lo:hi].tolist(), c.asset_rel[lo:hi].tolist())))
    ids = sorted(i for i in pl.read_parquet(sql.headlines_dir / _file(D))["RP_STORY_ID"]
                 if i.startswith(D.isoformat()))
    assert len(rows) == len(ids) == 50
    for sid, got in zip(ids, rows):
        want = sorted((0 if e == "AAPL" else 1, w) for e, w in by_id.get(sid, [])
                      if e in ("AAPL", "MSFT"))
        assert got == want, sid
    assert rows[ids.index(f"{D.isoformat()}-0")] == [(0, 100)]
    assert rows[ids.index(f"{D.isoformat()}-2")] == []                # no entity at all


def test_the_asset_layer_needs_the_entities_table(month):
    from narrative_scoring.streaming import PartitionHeadlineSource

    _, sql = month
    with pytest.raises(ValueError, match="entities directory"):
        PartitionHeadlineSource(sql.headlines_dir, sql.embeddings_dir, assets=_universe("AAPL"))
    part = PartitionHeadlineSource(sql.headlines_dir, sql.embeddings_dir,
                                   assets=_universe("AAPL"),
                                   entities_dir=sql.headlines_dir.parent / "nowhere")
    with pytest.raises(FileNotFoundError, match="entities partition"):
        list(part.iter_day(D))


def test_entities_of_absent_stories_are_refused(month):
    from narrative_scoring.streaming import PartitionHeadlineSource

    _, sql = month
    orphan = pl.DataFrame({"RP_STORY_ID": ["nope"], "RP_ENTITY_ID": ["AAPL"],
                           "RELEVANCE": [90]}, schema_overrides={"RELEVANCE": pl.UInt8})
    edir, _ = _with_entities(sql, extra=orphan)
    part = PartitionHeadlineSource(sql.headlines_dir, sql.embeddings_dir,
                                   assets=_universe("AAPL"), entities_dir=edir)
    with pytest.raises(ValueError, match="absent from the headlines"):
        list(part.iter_day(D))
    other = pl.DataFrame({"RP_STORY_ID": ["nope"], "RP_ENTITY_ID": ["ZZZ"],
                          "RELEVANCE": [90]}, schema_overrides={"RELEVANCE": pl.UInt8})
    edir, _ = _with_entities(sql, extra=other)                    # not a universe entity
    assert sum(c.n for c in PartitionHeadlineSource(
        sql.headlines_dir, sql.embeddings_dir, assets=_universe("AAPL"),
        entities_dir=edir).iter_day(D)) == 50


def test_arrow_conversion_rejects_nulls():
    arr = pa.array([[1.0] * EMBEDDING_DIM, None], type=pa.list_(pa.float32(), EMBEDDING_DIM))
    with pytest.raises(ValueError, match="null"):
        arrow_embeddings_to_numpy(arr)
    ok = pa.array([[1.0] * EMBEDDING_DIM, [2.0] * EMBEDDING_DIM],
                  type=pa.list_(pa.float32(), EMBEDDING_DIM))
    X = arrow_embeddings_to_numpy(ok)
    assert X.shape == (2, EMBEDDING_DIM) and X[1, 0] == 2.0 and X.flags.c_contiguous


def test_in_memory_source_chunks_tags_assets_and_day_total():
    X = np.arange(10 * EMBEDDING_DIM, dtype=np.float64).reshape(10, EMBEDDING_DIM)
    src = InMemoryHeadlineSource({date(2008, 1, 1): X}, chunk_size=4)
    chunks = list(src.iter_day(date(2008, 1, 1)))
    assert [c.n for c in chunks] == [4, 4, 2]
    assert all(c.embeddings.dtype == np.float32 for c in chunks)
    assert list(src.iter_day(date(2008, 1, 2))) == []
    codes = (np.arange(10) % 3 - 1).astype(np.int8)
    assets = [[(0, 90)] if i % 2 else [] for i in range(10)]
    tagged = InMemoryHeadlineSource({date(2008, 1, 1): X}, chunk_size=4,
                                    tags={date(2008, 1, 1): codes},
                                    assets={date(2008, 1, 1): assets})
    chunks = list(tagged.iter_day(date(2008, 1, 1)))
    np.testing.assert_array_equal(np.concatenate([c.tags for c in chunks]), codes)
    assert chunks[0].asset_indptr.tolist() == [0, 0, 1, 1, 2]
    assert chunks[0].asset_idx.tolist() == [0, 0] and chunks[0].asset_rel.tolist() == [90, 90]
    assert tagged.day_total(date(2008, 1, 1)) == 10 and tagged.day_total(date(2008, 1, 2)) == 0


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


def test_entities_are_scanned_once_per_partition(month, monkeypatch):
    from narrative_scoring.streaming import PartitionHeadlineSource

    _, sql = month
    edir, _ = _with_entities(sql)
    part = PartitionHeadlineSource(sql.headlines_dir, sql.embeddings_dir,
                                   assets=_universe("AAPL"), entities_dir=edir)
    scans = []
    real = pl.scan_parquet
    monkeypatch.setattr(pl, "scan_parquet", lambda *a, **k: scans.append(a[0]) or real(*a, **k))
    for d in (date(2008, 9, 15), date(2008, 9, 16), date(2008, 9, 15)):
        list(part.iter_day(d))
        part.day_total(d)
    assert len(scans) == 1


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


def test_days_are_utc_days_whatever_the_machine_zone(month, monkeypatch):
    """A story at 23:30 UTC belongs to its UTC day in both sources, even when the process
    runs in another time zone (DuckDB pinned to UTC; polars converts nothing)."""
    import time

    days, sql = month
    want = {d: sql.day_total(d) for d in (date(2008, 9, 15), date(2008, 9, 16))}
    assert want == {date(2008, 9, 15): 50, date(2008, 9, 16): 23}
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    try:
        part = _partition_source(sql)
        for d, n in want.items():
            assert sql.day_total(d) == n and part.day_total(d) == n
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
