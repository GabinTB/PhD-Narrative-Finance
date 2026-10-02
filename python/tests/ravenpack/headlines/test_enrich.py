"""In-place enrichment of a ravenpack_headlines artifact with RELEVANCE and RP_SOURCE_ID.

The stored ("old") partitions are produced by the real ingest path on synthetic raw zips,
then stripped of the two new columns, so they hold exactly what an artifact written
before the columns existed holds.
"""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from datalake import DatalakeIndex
from datalake.jobs import PAUSE_FILE, JobError, JobRunner, JobState
from datalake.layout import Layout
from datalake.meta import hash_directory
from ravenpack.headlines.enrich import (
    NEW_COLUMNS,
    EnrichJob,
    enrich_frame,
    is_enriched,
    next_patch,
    raw_story_lists,
)
from ravenpack.headlines.ingest import KIND, ingest_range, verify_artifact
from ravenpack.headlines.schema import STRUCTURED_SCHEMA

from .test_ingest import _make_zip

START, END = date(2010, 1, 1), date(2010, 3, 31)


def _month_rows(month: int) -> pd.DataFrame:
    """Four stories per month; story 0 holds three events on one company (AAPL) plus a
    place, the others one or two detections with and without event sentiment."""
    rows = []
    for i in range(4):
        day = 3 + 7 * i
        ts = f"2010-{month:02d}-{day:02d} 1{i}:00:00"
        sid = f"S{month:02d}{i}"
        base = {"TIMESTAMP_UTC": ts, "RP_STORY_ID": sid, "COUNTRY_CODE": "US",
                "NEWS_TYPE": "FULL-ARTICLE", "SOURCE_NAME": "Reuters",
                "RP_SOURCE_ID": f"SRC{i % 2}", "HEADLINE": f"headline {sid}"}
        if i == 0:
            dets = [("AAPL1", "COMP", "Apple Inc.", 0.31, 100),
                    ("AAPL1", "COMP", "Apple Inc.", None, 100),
                    ("USPL", "PLCE", "United States", None, 20),
                    ("AAPL1", "COMP", "Apple Inc.", -0.57, 100)]
        else:
            dets = [(f"E{i}", "COMP", f"Company {i}", 0.1 * i if i % 2 else None, 60 + i),
                    ("USPL", "PLCE", "United States", None, 0)][: 1 + i % 2]
        for eid, etype, name, ess, rel in dets:
            rows.append({**base, "RP_ENTITY_ID": eid, "ENTITY_TYPE": etype,
                         "ENTITY_NAME": name, "EVENT_SENTIMENT_SCORE": ess, "RELEVANCE": rel})
    return pd.DataFrame(rows)


def _raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for m in (1, 2, 3):
        _make_zip(raw, 2010, m, _month_rows(m))
    return raw


def _old_lake(tmp_path: Path, raw: Path, freq: str = "M") -> tuple[DatalakeIndex, str]:
    """A headlines artifact as written before RELEVANCE / RP_SOURCE_ID existed."""
    layout = Layout(freq, START, END)
    fresh = tmp_path / "fresh"
    ingest_range(raw, fresh, layout=layout)
    index = DatalakeIndex(tmp_path / "lake")
    with index.run(kind=KIND, pipeline="t", pipeline_version="v0.1.0",
                   hyperparams=layout.hyperparams(), verifier=KIND,
                   hash_pattern="*.parquet") as run:
        for p in sorted(fresh.glob("*.parquet")):
            pl.read_parquet(p).drop(list(NEW_COLUMNS)).write_parquet(run.out_dir / p.name)
    return index, run.artifact_id


def _job(index: DatalakeIndex, aid: str, raw: Path, root: Path) -> EnrichJob:
    return EnrichJob(index.get(aid), raw, backup_dir=root / "_backup" / aid,
                     staging_dir=root / "_staging" / aid)


def _runner(index: DatalakeIndex) -> JobRunner:
    return JobRunner(index, allow_dirty=True, handle_signals=False, heartbeat_s=0.05)


def _lists(month: int, *, batch_rows: int | None = None) -> pl.DataFrame:
    df = pl.from_pandas(_month_rows(month)).select(
        "RP_STORY_ID", "RP_ENTITY_ID", "ENTITY_TYPE", "ENTITY_NAME",
        pl.col("EVENT_SENTIMENT_SCORE").cast(pl.Float64), pl.col("RELEVANCE").cast(pl.Float64),
        "RP_SOURCE_ID")
    if batch_rows is None:
        return raw_story_lists([df])
    return raw_story_lists([df.slice(i, batch_rows) for i in range(0, df.height, batch_rows)])


# ---------------------------------------------------------------------------
# raw_story_lists / enrich_frame
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_rows", [1, 2, 3, 5])
def test_story_lists_do_not_depend_on_batch_boundaries(batch_rows):
    assert _lists(1, batch_rows=batch_rows).equals(_lists(1))


def test_a_story_seen_in_two_places_raises():
    df = pl.from_pandas(_month_rows(1)).select(
        "RP_STORY_ID", "RP_ENTITY_ID", "ENTITY_TYPE", "ENTITY_NAME",
        pl.col("EVENT_SENTIMENT_SCORE").cast(pl.Float64), pl.col("RELEVANCE").cast(pl.Float64),
        "RP_SOURCE_ID")
    shuffled = pl.concat([df.slice(0, 1), df.slice(5), df.slice(1, 4)])
    with pytest.raises(ValueError, match="not contiguous"):
        raw_story_lists([shuffled])


def test_source_varying_within_a_story_raises():
    df = pl.from_pandas(_month_rows(1)).with_columns(
        pl.when(pl.int_range(pl.len()) == 1).then(pl.lit("OTHER"))
        .otherwise(pl.col("RP_SOURCE_ID")).alias("RP_SOURCE_ID"))
    with pytest.raises(ValueError, match="RP_SOURCE_ID varies"):
        raw_story_lists([df.with_columns(pl.col("RELEVANCE").cast(pl.Float64))])


@pytest.mark.parametrize("bad", [None, 50.5, 101.0, -1.0])
def test_relevance_must_be_an_integer_in_0_100(bad):
    df = pl.from_pandas(_month_rows(1)).with_columns(pl.col("RELEVANCE").cast(pl.Float64))
    df = df.with_columns(pl.when(pl.int_range(pl.len()) == 2).then(pl.lit(bad))
                         .otherwise(pl.col("RELEVANCE")).alias("RELEVANCE"))
    with pytest.raises(ValueError, match="RELEVANCE"):
        raw_story_lists([df])


def _stored(tmp_path: Path, month: int = 1) -> pl.DataFrame:
    raw = _raw(tmp_path)
    fresh = tmp_path / "fresh"
    ingest_range(raw, fresh, layout=Layout("M", START, END))
    return pl.read_parquet(fresh / f"2010-{month:02d}.parquet").drop(list(NEW_COLUMNS))


def test_enriched_partition_is_the_stored_one_plus_aligned_columns(tmp_path):
    stored = _stored(tmp_path)
    out = enrich_frame(stored, _lists(1), exact=True, where="2010-01")
    assert out.columns == STRUCTURED_SCHEMA.names
    for c in stored.columns:
        assert out[c].equals(stored[c]), c
    s0 = out.filter(pl.col("RP_STORY_ID") == "S010").row(0, named=True)
    # one entry per detection, aligned: three AAPL events and the place
    assert s0["RP_ENTITY_ID"] == ["AAPL1", "AAPL1", "USPL", "AAPL1"]
    assert s0["RELEVANCE"] == [100, 100, 20, 100]
    assert s0["RP_SOURCE_ID"] == "SRC0"
    assert out["RELEVANCE"].list.len().equals(out["RP_ENTITY_ID"].list.len())


def test_a_stored_list_that_differs_from_the_raw_rows_raises(tmp_path):
    stored = _stored(tmp_path).with_columns(
        pl.when(pl.col("RP_STORY_ID") == "S011")
        .then(pl.col("ENTITY_NAME").list.eval(pl.element() + "x"))
        .otherwise(pl.col("ENTITY_NAME")).alias("ENTITY_NAME"))
    with pytest.raises(ValueError, match="ENTITY_NAME rebuilt .* e.g. S011"):
        enrich_frame(stored, _lists(1), exact=True)


def test_reordered_detections_raise(tmp_path):
    stored = _stored(tmp_path).with_columns(
        pl.when(pl.col("RP_STORY_ID") == "S010")
        .then(pl.col("RP_ENTITY_ID").list.reverse())
        .otherwise(pl.col("RP_ENTITY_ID")).alias("RP_ENTITY_ID"))
    with pytest.raises(ValueError, match="RP_ENTITY_ID rebuilt"):
        enrich_frame(stored, _lists(1), exact=True)


def test_story_sets_must_match_for_whole_months(tmp_path):
    stored = _stored(tmp_path)
    with pytest.raises(ValueError, match="no raw rows"):
        enrich_frame(stored, _lists(1).filter(pl.col("RP_STORY_ID") != "S012"), exact=True)
    with pytest.raises(ValueError, match="absent from the partition"):
        enrich_frame(stored.filter(pl.col("RP_STORY_ID") != "S012"), _lists(1), exact=True)
    # a day / week partition is a subset of its raw month
    sub = stored.filter(pl.col("RP_STORY_ID") != "S012")
    assert enrich_frame(sub, _lists(1), exact=False).height == sub.height


def test_an_already_enriched_partition_is_refused(tmp_path):
    out = enrich_frame(_stored(tmp_path), _lists(1), exact=True)
    with pytest.raises(ValueError, match="already enriched"):
        enrich_frame(out, _lists(1), exact=True)


def test_next_patch_keeps_major_minor():
    assert next_patch("v0.1.0") == "v0.1.1"
    assert next_patch("v2.10.9") == "v2.10.10"


# ---------------------------------------------------------------------------
# The update job on a registered artifact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq", ["D", "M", "Q"])
def test_enrichment_rewrites_the_artifact_in_place(tmp_path, freq):
    raw = _raw(tmp_path)
    index, aid = _old_lake(tmp_path, raw, freq)
    before = index.get(aid)
    originals = {p.name: p.read_bytes() for p in sorted(before.path.glob("*.parquet"))}
    old_frames = {n: pl.read_parquet(before.path / n) for n in originals}

    done = _runner(index).update_job(aid, _job(index, aid, raw, tmp_path))

    assert done.artifact_id == aid and not done.partial
    assert done.meta.hyperparams == before.meta.hyperparams
    assert [r.pipeline_version for r in done.meta.runs] == ["v0.1.0", "v0.1.1"]
    assert "RELEVANCE" in done.meta.runs[-1].notes
    backup = tmp_path / "_backup" / aid
    assert {p.name: p.read_bytes() for p in sorted(backup.glob("*.parquet"))} == originals
    assert not list((tmp_path / "_staging" / aid).glob("*"))
    for name, old in old_frames.items():
        new = pl.read_parquet(done.path / name)
        assert new.columns == STRUCTURED_SCHEMA.names
        assert new.select(old.columns).equals(old), name
    # the index and the sidecar hold the hashes of the rewritten files
    assert done.file_hashes == {k: v["digest"] for k, v in
                                hash_directory(done.path, pattern="*.parquet").items()}
    assert not [f for f in verify_artifact(done) if f.severity.name == "ERROR"]


def test_killed_enrichment_resumes_to_the_uninterrupted_result(tmp_path):
    raw = _raw(tmp_path / "ref")
    ref_index, ref_id = _old_lake(tmp_path / "ref", raw)
    ref = _runner(ref_index).update_job(ref_id, _job(ref_index, ref_id, raw, tmp_path / "ref"))

    index, aid = _old_lake(tmp_path / "run", raw)
    job = _job(index, aid, raw, tmp_path / "run")
    real, calls = job.run_unit, [0]

    def run_unit(unit, ctx):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("killed")
        real(unit, ctx)

    job.run_unit = run_unit
    with pytest.raises(RuntimeError, match="killed"):
        _runner(index).update_job(aid, job)
    assert index.get(aid).partial
    assert JobState.read(index.get(aid).path).status == "failed"
    done = _runner(index).resume_job(aid, _job(index, aid, raw, tmp_path / "run"))
    assert not done.partial
    got = {p.name: pl.read_parquet(p) for p in sorted(done.path.glob("*.parquet"))}
    want = {p.name: pl.read_parquet(p) for p in sorted(ref.path.glob("*.parquet"))}
    assert sorted(got) == sorted(want)
    assert all(got[n].equals(want[n]) for n in want)


def test_pause_then_resume(tmp_path):
    raw = _raw(tmp_path)
    index, aid = _old_lake(tmp_path, raw)
    job = _job(index, aid, raw, tmp_path)
    real = job.run_unit

    def run_unit(unit, ctx):
        real(unit, ctx)
        (ctx.out_dir / PAUSE_FILE).write_text("stop")

    job.run_unit = run_unit
    runner = _runner(index)
    art = runner.update_job(aid, job)
    assert art.partial and runner.status(aid).status == "paused"
    done = runner.resume_job(aid, _job(index, aid, raw, tmp_path))
    assert not done.partial
    assert all(is_enriched(p) for p in done.path.glob("*.parquet"))


def test_a_crash_between_the_two_moves_resumes_from_the_backup(tmp_path):
    raw = _raw(tmp_path)
    index, aid = _old_lake(tmp_path, raw)
    art = index.get(aid)
    backup = tmp_path / "_backup" / aid
    backup.mkdir(parents=True)
    shutil.move(art.path / "2010-02.parquet", backup / "2010-02.parquet")  # original moved,
    done = _runner(index).update_job(aid, _job(index, aid, raw, tmp_path))  # enriched not in
    assert is_enriched(done.path / "2010-02.parquet")
    assert (backup / "2010-02.parquet").exists()


def test_an_unenriched_file_next_to_a_backup_is_refused(tmp_path):
    raw = _raw(tmp_path)
    index, aid = _old_lake(tmp_path, raw)
    backup = tmp_path / "_backup" / aid
    backup.mkdir(parents=True)
    shutil.copy(index.get(aid).path / "2010-01.parquet", backup / "2010-01.parquet")
    with pytest.raises(JobError, match="inspect both"):
        _runner(index).update_job(aid, _job(index, aid, raw, tmp_path))


def test_jobs_resume_cannot_finish_an_enrichment(tmp_path):
    """The generic resume rebuilds an IngestJob (v0.3) for a v0.1 artifact and refuses,
    so a half-enriched artifact is never marked complete by it."""
    raw = _raw(tmp_path)
    index, aid = _old_lake(tmp_path, raw)
    job = _job(index, aid, raw, tmp_path)
    job.run_unit = lambda unit, ctx: (_ for _ in ()).throw(RuntimeError("killed"))
    with pytest.raises(RuntimeError):
        _runner(index).update_job(aid, job)
    with pytest.raises(JobError, match="major.minor"):
        _runner(index).resume(aid, raw_dir=raw)


def test_written_schema_is_the_ingest_schema(tmp_path):
    raw = _raw(tmp_path)
    index, aid = _old_lake(tmp_path, raw)
    done = _runner(index).update_job(aid, _job(index, aid, raw, tmp_path))
    import pyarrow.parquet as pq

    schema = pq.read_schema(done.path / "2010-01.parquet")
    assert schema.remove_metadata().equals(STRUCTURED_SCHEMA)
    assert schema.field("RELEVANCE").type == pa.list_(pa.uint8())
