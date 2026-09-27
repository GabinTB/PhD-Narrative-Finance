"""datalake.jobs: the shared lifecycle, on a toy job (no pipeline code involved)."""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from datalake import DatalakeIndex
from datalake.jobs import (
    LOCK_FILE,
    LOG_FILE,
    PAUSE_FILE,
    STATE_FILE,
    Job,
    JobError,
    JobLocked,
    JobRunner,
    JobState,
    Unit,
    check_resume_version,
    list_jobs,
    register_job,
)
from datalake.layout import Layout, layout_from_hyperparams
from datalake.periods import parse_key


class FakeModel:
    """A 'remote model' whose identity can be switched under a running job."""

    def __init__(self, served: str = "model-a") -> None:
        self.served, self.recorded = served, served

    def identity(self) -> dict[str, Any]:
        return {"backend": "fake", "served": self.served}

    def check_unchanged(self) -> None:
        if self.served != self.recorded:
            raise JobError(f"served model changed: {self.recorded} -> {self.served}")


@register_job
class ToyJob(Job):
    kind = "toy_series"
    pipeline_version = "v1.2.0"
    die_at: str | None = None                    # unit key that raises (a kill)
    pause_after: str | None = None               # unit key after which a PAUSE file appears
    switch_after: str | None = None              # unit key after which the model changes

    def __init__(self, layout: Layout, seed: int = 0, temp: bool = False,
                 model: FakeModel | None = None) -> None:
        self.layout, self.seed, self.temp = layout, seed, temp
        self.model = model or FakeModel()

    def params(self) -> dict[str, Any]:
        return {**self.layout.hyperparams(), "seed": self.seed}

    def backends(self) -> list[Any]:
        return [self.model]

    def units(self) -> list[Unit]:
        return [Unit(p.key) for p in self.layout.expected()]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return (out_dir / f"{unit.key}.parquet").exists()

    def run_unit(self, unit: Unit, ctx: Any) -> None:
        import polars as pl

        if unit.key == self.die_at:
            raise RuntimeError(f"killed during {unit.key}")
        period = parse_key(unit.key)
        pl.DataFrame({"day": period.days(), "value": [hash((self.seed, d.toordinal())) % 997
                                                       for d in period.days()]}
                     ).write_parquet(ctx.out_dir / f"{unit.key}.parquet")
        ctx.log.info("wrote %s", unit.key)
        if unit.key == self.pause_after:
            (ctx.out_dir / PAUSE_FILE).write_text("please pause")
        if unit.key == self.switch_after:
            self.model.served = "model-b"

    def finalize(self, ctx: Any) -> None:
        (ctx.out_dir / "summary.txt").write_text("done")

    @classmethod
    def from_artifact(cls, artifact, index, **kwargs) -> ToyJob:
        hp = artifact.meta.hyperparams
        return cls(layout_from_hyperparams(hp), seed=hp["seed"],
                   temp=artifact.meta.pipeline_version.endswith("__TEMP"), **kwargs)


def _clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"]):
        subprocess.run(cmd, cwd=repo, check=True)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


@pytest.fixture
def env(tmp_path):
    index = DatalakeIndex(tmp_path / "lake")
    runner = JobRunner(index, repo_dir=_clean_repo(tmp_path), handle_signals=False,
                       heartbeat_s=0.05)
    yield index, runner
    ToyJob.die_at = ToyJob.pause_after = ToyJob.switch_after = None
    index.close()


LAYOUT = Layout("M", date(2008, 1, 1), date(2008, 6, 30))


def _files(art) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(art.path.glob("*.parquet"))}


def test_start_runs_to_completion_with_state_log_and_no_lock(env):
    index, runner = env
    art = runner.start(ToyJob(LAYOUT))
    assert not art.partial and sorted(_files(art)) == [f"2008-0{m}.parquet" for m in range(1, 7)]
    state = JobState.read(art.path)
    assert (state.status, state.units_done, state.units_total) == ("complete", 6, 6)
    assert state.backends == [{"backend": "fake", "served": "model-a"}]
    assert not (art.path / LOCK_FILE).exists() and (art.path / "summary.txt").exists()
    text = (art.path / LOG_FILE).read_text()
    assert "wrote 2008-03" in text and "job complete" in text
    assert STATE_FILE not in art.file_hashes and LOG_FILE not in art.file_hashes
    assert runner.status(art.artifact_id).status == "complete"


@pytest.mark.parametrize("freq", ["D", "W", "M", "Q"])
def test_kill_then_resume_is_byte_identical(env, tmp_path, freq):
    index, runner = env
    layout = Layout(freq, date(2008, 1, 1), date(2008, 12, 31) if freq == "Q"
                    else date(2008, 2, 20))
    ref_index = DatalakeIndex(tmp_path / "ref")
    ref = JobRunner(ref_index, repo_dir=runner.repo_dir, handle_signals=False).start(
        ToyJob(layout))
    units = [u.key for u in ToyJob(layout).units()]
    ToyJob.die_at = units[len(units) // 2]
    with pytest.raises(RuntimeError, match="killed"):
        runner.start(ToyJob(layout))
    part = index.list("toy_series", include_partial=True)[0]
    assert part.partial and JobState.read(part.path).status == "failed"
    assert "killed" in JobState.read(part.path).last_error
    assert runner.status(part.artifact_id).status == "failed"
    ToyJob.die_at = None
    done = runner.resume(part.artifact_id)
    assert not done.partial and done.artifact_id == part.artifact_id
    assert _files(done) == _files(ref)
    assert len(done.meta.runs) == 2


def test_live_lock_blocks_a_second_process_and_stale_lock_is_taken_over(env):
    index, runner = env
    ToyJob.die_at = "2008-03"
    with pytest.raises(RuntimeError):
        runner.start(ToyJob(LAYOUT))
    part = index.list("toy_series", include_partial=True)[0]
    ToyJob.die_at = None
    lock = part.path / LOCK_FILE
    from datalake.artifact import utc_now_iso

    lock.write_text(json.dumps({"host": "other", "pid": 1, "heartbeat_at": utc_now_iso()}))
    assert runner.status(part.artifact_id).status == "running"
    with pytest.raises(JobLocked, match="held by other"):
        runner.resume(part.artifact_id)
    lock.write_text(json.dumps({"host": "other", "pid": 1,
                                "heartbeat_at": "2000-01-01T00:00:00+00:00"}))
    assert runner.status(part.artifact_id).status == "stale"
    assert not runner.resume(part.artifact_id).partial          # stale lock taken over


def test_pause_stops_at_a_unit_boundary_then_resume_completes(env):
    index, runner = env
    ToyJob.pause_after = "2008-02"
    art = runner.start(ToyJob(LAYOUT))
    assert art.partial and sorted(_files(art)) == ["2008-01.parquet", "2008-02.parquet"]
    st = runner.status(art.artifact_id)
    assert st.status == "paused" and (st.units_done, st.units_total) == (2, 6)
    assert [s.artifact_id for s in list_jobs(index, runner, kinds=["toy_series"])] == [
        art.artifact_id]
    ToyJob.pause_after = None
    done = runner.resume(art.artifact_id)
    assert not done.partial and len(_files(done)) == 6
    assert not (done.path / PAUSE_FILE).exists()


def test_runner_pause_request_is_the_same_path(env):
    index, runner = env
    ToyJob.die_at = "2008-04"
    with pytest.raises(RuntimeError):
        runner.start(ToyJob(LAYOUT))
    ToyJob.die_at = None
    part = index.list("toy_series", include_partial=True)[0]
    runner.pause(part.artifact_id)                    # e.g. from another machine
    assert (part.path / PAUSE_FILE).exists()
    again = runner.resume(part.artifact_id)           # resume clears a stale PAUSE request
    assert not again.partial


def test_model_switch_mid_run_fails_the_job_with_its_reason(env):
    index, runner = env
    ToyJob.switch_after = "2008-02"
    with pytest.raises(JobError, match="served model changed"):
        runner.start(ToyJob(LAYOUT))
    part = index.list("toy_series", include_partial=True)[0]
    state = JobState.read(part.path)
    assert state.status == "failed" and "model-b" in state.last_error
    assert state.units_done == 2


def test_check_reports_a_model_that_is_no_longer_served(env):
    index, runner = env
    ToyJob.die_at = "2008-02"
    with pytest.raises(RuntimeError):
        runner.start(ToyJob(LAYOUT))
    ToyJob.die_at = None
    part = index.list("toy_series", include_partial=True)[0]
    assert runner.check(part.artifact_id) == []
    switched = FakeModel("model-b")
    switched.recorded = "model-a"
    problems = runner.check(part.artifact_id, model=switched)
    assert problems and "served model changed" in problems[0]


def test_version_guard():
    check_resume_version("v1.2.0", "v1.2.7")
    check_resume_version("v1.2.0__TEMP", "v1.2.1")
    with pytest.raises(JobError, match="major.minor"):
        check_resume_version("v1.2.0", "v1.3.0")


def test_resume_refuses_another_major_minor(env):
    index, runner = env
    ToyJob.die_at = "2008-02"
    with pytest.raises(RuntimeError):
        runner.start(ToyJob(LAYOUT))
    ToyJob.die_at = None
    part = index.list("toy_series", include_partial=True)[0]
    ToyJob.pipeline_version = "v1.3.0"
    try:
        with pytest.raises(JobError, match="major.minor"):
            runner.resume(part.artifact_id)
    finally:
        ToyJob.pipeline_version = "v1.2.0"


def test_dirty_tree_refused_for_production_runs(env):
    index, runner = env
    (runner.repo_dir / "f.txt").write_text("changed")               # dirty tree
    with pytest.raises(JobError, match="dirty git tree"):
        runner.start(ToyJob(LAYOUT))
    assert not runner.start(ToyJob(LAYOUT, temp=True)).partial      # TEMP runs allowed
    runner.allow_dirty = True
    art = runner.start(ToyJob(Layout("M", date(2009, 1, 1), date(2009, 1, 31))))
    assert "allow_dirty" in art.meta.runs[-1].notes


def test_json_log_lines_and_log_file_off(env, tmp_path):
    index, runner = env
    runner.log_json = True
    art = runner.start(ToyJob(Layout("M", date(2008, 1, 1), date(2008, 1, 31))))
    lines = [json.loads(line) for line in (art.path / LOG_FILE).read_text().splitlines()]
    tagged = [x for x in lines if x.get("unit") == "2008-01"]
    assert tagged and tagged[0]["kind"] == "toy_series" and tagged[0]["artifact"]
    runner.log_file = False
    art2 = runner.start(ToyJob(Layout("M", date(2010, 1, 1), date(2010, 1, 31))))
    assert not (art2.path / LOG_FILE).exists()


def test_update_is_refused_for_a_job_without_updates(env):
    index, runner = env
    art = runner.start(ToyJob(LAYOUT))
    with pytest.raises(JobError, match="does not support updates"):
        runner.update(art.artifact_id)
    assert len(index.get(art.artifact_id).meta.runs) == 1
