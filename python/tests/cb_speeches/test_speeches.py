"""bis_gingado.cb_speeches.speeches: base load, updates, point-in-time reader, verifier."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pyarrow.parquet as pq
import pytest

from bis_gingado.cb_speeches.bis import SpeechDataError
from bis_gingado.cb_speeches.speeches import (
    SpeechesJob,
    data_files,
    file_sources,
    plan_files,
    read_speeches,
    verify_artifact,
)
from datalake import DatalakeIndex
from datalake.jobs import JobError, JobRunner
from datalake.layout import Layout
from tests.cb_speeches.fakes import BULK_URL, FakeBIS, every_third_day, speech

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


class Clock:
    """Deterministic clock: every call advances one minute."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(minutes=1)
        return self.now


def _fake() -> FakeBIS:
    return FakeBIS({2007: every_third_day(date(2007, 11, 1), date(2007, 12, 31)),
                    2008: every_third_day(date(2008, 1, 1), date(2008, 6, 30))})


def _runner(index):
    return JobRunner(index, allow_dirty=True, handle_signals=False, heartbeat_s=0.05)


def _job(fake, freq="M", start=date(2008, 1, 1), end=date(2008, 6, 30), clock=None):
    return SpeechesJob.new(Layout(freq, start, end), url=BULK_URL, temp=True,
                           transport=fake.transport(), clock=clock or Clock(), retry_wait_s=0)


@pytest.fixture
def lake(tmp_path):
    with DatalakeIndex(tmp_path / "lake") as index:
        yield index


def _load(lake, fake, **kw):
    return _runner(lake).start(_job(fake, **kw))


def test_base_load_partitions_by_speech_date(lake):
    fake = _fake()
    art = _load(lake, fake)
    names = sorted(p.name for p in art.path.glob("*.parquet"))
    assert names == [f"2008-{m:02d}.parquet" for m in range(1, 7)]
    df = read_speeches(art)
    assert df.height == len(fake.rows[2008])
    assert df["change"].unique().to_list() == ["base"]
    assert fake.downloads() == 1                       # one bulk zip for six partitions
    assert art.meta.hyperparams == {"source": "bis_cbspeeches", "partition_freq": "M",
                                    "start": "2008-01-01", "end": "2008-06-30"}
    assert verify_artifact(art) == []


def test_raw_zip_plan_and_manifest_are_recorded(lake):
    fake = _fake()
    art = _load(lake, fake)
    raws = list(art.path.glob("raw-speeches-*.zip"))
    assert len(raws) == 1 and raws[0].read_bytes() == fake.zip()
    plan = json.loads(plan_files(art.path)[0].read_text())
    assert plan["mode"] == "base" and plan["url"] == BULK_URL
    src = file_sources(art.path / "2008-01.parquet")["sources"][0]
    assert src["url"] == BULK_URL and src["etag"] and src["sha256"]
    assert raws[0].name in art.file_hashes and plan_files(art.path)[0].name in art.file_hashes


def test_empty_partition_is_an_empty_file(lake):
    fake = FakeBIS({2008: [speech(date(2008, 1, 2), 0)]})
    art = _load(lake, fake, freq="Q", end=date(2008, 12, 31))
    assert pq.read_metadata(art.path / "2008Q2.parquet").num_rows == 0
    assert read_speeches(art).height == 1


def test_speeches_outside_the_range_are_not_written(lake):
    fake = _fake()
    art = _load(lake, fake, start=date(2008, 2, 1))
    assert read_speeches(art)["date"].min() >= date(2008, 2, 1)
    assert "outside the range (not written): " in art.meta.runs[-1].notes


def test_update_with_nothing_new_writes_empty_deltas_without_downloading(lake):
    fake = _fake()
    art = _load(lake, fake)
    before = fake.downloads()
    done = _runner(lake).update(art.artifact_id, transport=fake.transport(),
                                clock=Clock(T0 + timedelta(days=1)))
    assert done.artifact_id == art.artifact_id and not done.partial
    assert len(done.meta.runs) == 2
    deltas = sorted(p.name for p in done.path.glob("update-*.parquet"))
    assert len(deltas) == 1                            # one delta per update
    assert pq.read_metadata(done.path / deltas[0]).num_rows == 0
    assert fake.downloads() == before                  # ETag unchanged: HEAD only
    assert read_speeches(done).equals(read_speeches(art))
    assert verify_artifact(done) == []


def test_update_records_new_revised_and_removed(lake):
    fake = _fake()
    art = _load(lake, fake)
    base = read_speeches(art)
    base_files = {p.name: p.read_bytes() for p in art.path.glob("2008-*.parquet")}
    rows = fake.rows[2008]
    revised_id, removed_id = "r080101a", "r080104b"
    rows[0] = dict(rows[0], text="revised text")
    rows[:] = [r for r in rows if removed_id not in r["url"]]
    rows.append(speech(date(2008, 6, 30), 5))
    fake.rows[2009] = [speech(date(2009, 1, 5), 0)]
    t1 = T0 + timedelta(days=1)
    done = _runner(lake).update(art.artifact_id, transport=fake.transport(), clock=Clock(t1))
    assert {p.name: p.read_bytes() for p in done.path.glob("2008-*.parquet")} == base_files
    delta = pl.concat([pl.read_parquet(p) for p in done.path.glob("update-*.parquet")])
    changes = dict(zip(delta["speech_id"].to_list(), delta["change"].to_list()))
    assert changes == {revised_id: "revised", removed_id: "removed", "r080630f": "new",
                       "r090105a": "new"}
    now = read_speeches(done)
    assert removed_id not in now["speech_id"].to_list()
    assert now.filter(pl.col("speech_id") == revised_id)["text"].item() == "revised text"
    assert "r090105a" in now["speech_id"].to_list()             # after the declared end
    assert read_speeches(done, as_of=t1).equals(base)            # before the update's vintage
    assert "update" in done.meta.runs[-1].notes
    assert verify_artifact(done) == []


def test_killed_update_resumes_to_the_uninterrupted_result(tmp_path):
    def run(root, kill: bool):
        fake = _fake()
        with DatalakeIndex(root) as index:
            art = _runner(index).start(_job(fake, freq="Y", start=date(2007, 1, 1),
                                            end=date(2008, 12, 31)))
            fake.rows[2007].append(speech(date(2007, 12, 30), 7))
            fake.rows[2008].append(speech(date(2008, 12, 30), 7))
            job = SpeechesJob.for_update(index.get(art.artifact_id), index,
                                         transport=fake.transport(),
                                         clock=Clock(T0 + timedelta(days=1)))
            if kill:
                real, calls = job.run_unit, [0]

                def run_unit(unit, ctx):             # killed inside the update's unit
                    calls[0] += 1
                    if calls[0] == 1:
                        raise RuntimeError("killed")
                    real(unit, ctx)
                job.run_unit = run_unit
                with pytest.raises(RuntimeError, match="killed"):
                    _runner(index).update_job(art.artifact_id, job)
                assert index.get(art.artifact_id).partial
                with pytest.raises(JobError, match="partial"):
                    _runner(index).update(art.artifact_id, transport=fake.transport())
                done = _runner(index).resume(art.artifact_id, transport=fake.transport())
            else:
                done = _runner(index).update_job(art.artifact_id, job)
            # data, not bytes: the manifests carry the real fetch time of each download
            return {p.name: pl.read_parquet(p) for p in sorted(done.path.glob("*.parquet"))}

    got, want = run(tmp_path / "a", kill=True), run(tmp_path / "b", kill=False)
    assert sorted(got) == sorted(want)
    assert all(got[name].equals(want[name]) for name in want)


def test_update_vintage_must_move_forward(lake):
    fake = _fake()
    art = _load(lake, fake)
    with pytest.raises(SpeechDataError, match="later"):
        SpeechesJob.for_update(art, lake, transport=fake.transport(), clock=Clock(T0))


def test_one_id_with_two_contents_fails_the_base_load(lake):
    row = speech(date(2008, 1, 2), 0)
    fake = FakeBIS({2008: [row], 2009: [dict(row, date="2009-01-02 00:00:00")]})
    with pytest.raises(SpeechDataError, match="different content"):
        _load(lake, fake, freq="Y", end=date(2009, 12, 31))


def test_base_load_survives_cut_downloads(lake):
    fake = _fake()
    fake.cut = 2
    art = _load(lake, fake)
    assert read_speeches(art).height == len(fake.rows[2008]) and not art.partial


def test_verifier_flags_a_tampered_raw_zip(lake):
    art = _load(lake, _fake())
    raw = next(art.path.glob("raw-*.zip"))
    raw.write_bytes(b"tampered")
    assert any("does not match" in f.message for f in verify_artifact(art))


def test_data_files_order_is_base_then_deltas(lake):
    fake = _fake()
    art = _load(lake, fake)
    _runner(lake).update(art.artifact_id, transport=fake.transport(),
                         clock=Clock(T0 + timedelta(days=1)))
    names = [p.name for p in data_files(art.path)]
    assert names[-1].startswith("update-") and names[0] == "2008-01.parquet"
