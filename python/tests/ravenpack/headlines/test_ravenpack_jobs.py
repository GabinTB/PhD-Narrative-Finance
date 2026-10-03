"""Ingest and headline sentiment as datalake Jobs: units, resume, the jobs CLI."""
from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

import ravenpack.headlines.ingest as ingest_mod
import ravenpack.headlines.sentiment_vendor as vendor_mod
from datalake import DatalakeIndex
from datalake.jobs import JobRunner, JobState
from datalake.jobs_cli import main as jobs_main
from datalake.layout import Layout
from ravenpack.headlines.ingest import KIND, IngestJob, verify_artifact
from ravenpack.headlines.sentiment import KIND as SENT_KIND
from ravenpack.headlines.sentiment import verify_artifact as verify_sentiment

from .test_ingest import _make_zip, _month_df
from .test_sentiment_vendor import _csv


def _files(path: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(path.glob("*.parquet"))}


@pytest.fixture
def raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for m in (1, 2):
        _make_zip(raw, 2010, m, _month_df(2010, m))
    return raw


def _runner(index) -> JobRunner:
    return JobRunner(index, allow_dirty=True, handle_signals=False)


def test_daily_ingest_reads_each_raw_month_once_and_resumes_identically(tmp_path, raw,
                                                                         monkeypatch):
    real = ingest_mod._iter_month_deduped
    reads: list[str] = []

    def counting(zf, member, *a, **k):
        reads.append(member)
        return real(zf, member, *a, **k)

    monkeypatch.setattr(ingest_mod, "_iter_month_deduped", counting)
    layout = Layout("D", date(2010, 1, 1), date(2010, 2, 28))
    with DatalakeIndex(tmp_path / "ref") as index:
        ref = _runner(index).start(IngestJob(raw, layout))
        state = JobState.read(ref.path)
        assert (state.units_done, state.units_total) == (59, 59)
        assert [m.rsplit("/", 1)[1] for m in reads] == ["2010-01.csv", "2010-02.csv"]
        assert len(ref.file_hashes) == 12                          # days with stories
        assert [f for f in verify_artifact(ref) if f.severity.name == "ERROR"] == []

    def killed_in_february(zf, member, *a, **k):
        if member.endswith("2010-02.csv"):
            raise RuntimeError("killed")
        return real(zf, member, *a, **k)

    with DatalakeIndex(tmp_path / "lake") as index:
        monkeypatch.setattr(ingest_mod, "_iter_month_deduped", killed_in_february)
        with pytest.raises(RuntimeError, match="killed"):
            _runner(index).start(IngestJob(raw, layout))
        part = index.list(KIND, include_partial=True)[0]
        assert JobState.read(part.path).units_done == 31
        assert len(_files(part.path)) == 6                            # January only
        monkeypatch.setattr(ingest_mod, "_iter_month_deduped", real)
        done = _runner(index).resume(part.artifact_id, raw_dir=raw)
        assert not done.partial and _files(done.path) == _files(ref.path)


def test_legacy_year_range_artifact_resumes_with_its_recorded_hyperparams(tmp_path, raw):
    with DatalakeIndex(tmp_path / "lake") as index:
        hp = {"start_year": 2010, "end_year": 2010}                 # a pre-layout artifact
        with pytest.raises(RuntimeError):
            with index.run(kind=KIND, pipeline="PhD-Narrative-Finance",
                           pipeline_version=IngestJob.pipeline_version, hyperparams=hp):
                raise RuntimeError("crashed before writing anything")
        part = index.list(KIND, include_partial=True)[0]
        done = _runner(index).resume(part.artifact_id, raw_dir=raw)
        assert not done.partial and done.meta.hyperparams == hp
        assert sorted(done.file_hashes) == ["2010-01.parquet", "2010-02.parquet"]


def _vendor_lake(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    stem = "RavenPackAnalytics_AllEntities_1.0_2008"
    with zipfile.ZipFile(raw / f"{stem}.zip", "w") as zf:
        for month, sid in ((1, "J"), (2, "F")):
            zf.writestr(f"{stem}/2008-{month:02d}.csv", _csv([(sid, "0.10", "0.50", "100")]))
    lake = tmp_path / "lake"
    with DatalakeIndex(lake) as dl:
        with dl.run(kind="rp_headlines", pipeline="t", pipeline_version="v0",
                    hyperparams={"start_year": 2008, "end_year": 2008}) as r:
            for key, ids in (("2008-01", ["J", "x"]), ("2008-02", ["F"])):
                pl.DataFrame({"RP_STORY_ID": ids}).write_parquet(r.out_dir / f"{key}.parquet")
    return raw, lake


def test_vendor_sentiment_through_the_jobs_cli(tmp_path, monkeypatch):
    raw, lake = _vendor_lake(tmp_path)
    argv = ["--datalake-root", str(lake), "start", "headline_sentiment", "--source",
            "ravenpack", "--raw-dir", str(raw), "--temp"]
    real = vendor_mod.raw_month_batches

    def killed_in_february(raw_dir, year, month, *a):
        if month == 2:
            raise RuntimeError("killed")
        return real(raw_dir, year, month, *a)

    monkeypatch.setattr(vendor_mod, "raw_month_batches", killed_in_february)
    with pytest.raises(RuntimeError, match="killed"):
        jobs_main(argv)
    monkeypatch.setattr(vendor_mod, "raw_month_batches", real)
    with DatalakeIndex(lake) as dl:
        part = dl.list(SENT_KIND, include_partial=True)[0]
        # 10 of the 12 declared months have no headlines partition: not units
        assert (JobState.read(part.path).units_done, JobState.read(part.path).units_total) \
            == (1, 2)
    assert jobs_main(["--datalake-root", str(lake), "resume", part.artifact_id,
                      "--opt", f"raw_dir={raw}"]) == 0
    with DatalakeIndex(lake) as dl:
        art = dl.get(part.artifact_id)
        assert not art.partial and verify_sentiment(art) == []
        assert art.meta.hyperparams["rule"] == "css+ess_mean+ess_wmean"
        jan = pl.read_parquet(art.path / "2008-01.parquet")
        assert jan["RP_STORY_ID"].to_list() == ["J", "x"]


def test_sentiment_refuses_a_partition_freq_other_than_its_headlines(tmp_path):
    raw, lake = _vendor_lake(tmp_path)
    with pytest.raises(ValueError, match="follows its headlines' partitioning"):
        jobs_main(["--datalake-root", str(lake), "start", "headline_sentiment", "--source",
                   "ravenpack", "--raw-dir", str(raw), "--partition-freq", "Q", "--temp"])
