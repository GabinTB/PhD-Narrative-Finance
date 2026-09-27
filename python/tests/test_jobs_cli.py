"""The ``jobs`` command line, driven on a toy job in a temporary lake."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from datalake import DatalakeIndex
from datalake.jobs import LOCK_FILE, PAUSE_FILE, Job, Unit, register_job
from datalake.jobs_cli import _opts, main
from datalake.layout import Layout, layout_from_hyperparams
from datalake.periods import parse_key


@register_job
class CliToyJob(Job):
    """Toy job for the CLI tests."""

    kind = "toy_cli"
    pipeline_version = "v0.1.0"
    die_at: str | None = None

    def __init__(self, layout: Layout, temp: bool = True, scale: int = 1,
                 update: str | None = None) -> None:
        self.layout, self.temp, self.scale, self.update = layout, temp, scale, update

    def params(self) -> dict[str, Any]:
        return self.layout.hyperparams()

    def units(self) -> list[Unit]:
        if self.update:
            return [Unit(self.update)]
        return [Unit(p.key) for p in self.layout.expected()]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return (out_dir / f"{unit.key}.parquet").exists()

    def run_unit(self, unit: Unit, ctx: Any) -> None:
        import polars as pl

        if unit.key == self.die_at:
            raise RuntimeError(f"killed during {unit.key}")
        days = [] if self.update else parse_key(unit.key).days()
        pl.DataFrame({"day": days}, schema={"day": pl.Date}).write_parquet(
            ctx.out_dir / f"{unit.key}.parquet")

    @classmethod
    def add_cli_args(cls, parser) -> None:
        parser.add_argument("--freq", default="M")
        parser.add_argument("--start", type=date.fromisoformat, required=True)
        parser.add_argument("--end", type=date.fromisoformat, required=True)

    @classmethod
    def from_args(cls, args, index) -> CliToyJob:
        return cls(Layout(args.freq, args.start, args.end))

    @classmethod
    def from_artifact(cls, artifact, index, **kwargs) -> CliToyJob:
        return cls(layout_from_hyperparams(artifact.meta.hyperparams), **kwargs)

    @classmethod
    def for_update(cls, artifact, index, **kwargs) -> CliToyJob:
        return cls(layout_from_hyperparams(artifact.meta.hyperparams),
                   update=f"update-{len(artifact.meta.runs)}", **kwargs)


@pytest.fixture
def lake(tmp_path):
    yield tmp_path / "lake"
    CliToyJob.die_at = None


def _run(lake: Path, *argv: str) -> int:
    return main(["--datalake-root", str(lake), *argv])


def _partial(lake: Path):
    with DatalakeIndex(lake) as index:
        return index.list("toy_cli", include_partial=True)[0]


def test_start_then_status_reports_complete(lake, capsys):
    assert _run(lake, "start", "toy_cli", "--freq", "W",
                "--start", "2008-01-01", "--end", "2008-01-31") == 0
    out = capsys.readouterr().out
    assert "complete" in out and "5/5" in out
    art = _partial(lake)
    assert not art.partial
    assert _run(lake, "status", art.artifact_id) == 0
    assert "complete" in capsys.readouterr().out


def test_failed_run_is_listed_then_resumed(lake, capsys):
    CliToyJob.die_at = "2008-03"
    with pytest.raises(RuntimeError, match="killed"):
        _run(lake, "start", "toy_cli", "--start", "2008-01-01", "--end", "2008-06-30")
    CliToyJob.die_at = None
    art = _partial(lake)
    capsys.readouterr()
    assert _run(lake, "list", "--state", "failed") == 0
    listed = capsys.readouterr().out
    assert art.artifact_id in listed and "2/6" in listed and "killed" in listed
    assert _run(lake, "list", "--state", "running") == 0
    assert art.artifact_id not in capsys.readouterr().out
    assert _run(lake, "check", art.artifact_id) == 0
    assert capsys.readouterr().out.strip().endswith("ok")
    assert _run(lake, "resume", art.artifact_id) == 0
    assert "complete" in capsys.readouterr().out
    assert not _partial(lake).partial


def test_pause_writes_the_request_and_status_shows_a_live_lock(lake, capsys):
    CliToyJob.die_at = "2008-02"
    with pytest.raises(RuntimeError):
        _run(lake, "start", "toy_cli", "--start", "2008-01-01", "--end", "2008-03-31")
    CliToyJob.die_at = None
    art = _partial(lake)
    assert _run(lake, "pause", art.artifact_id) == 0
    assert (art.path / PAUSE_FILE).exists()
    from datalake.artifact import utc_now_iso

    (art.path / LOCK_FILE).write_text(json.dumps(
        {"host": "elsewhere", "pid": 7, "heartbeat_at": utc_now_iso()}))
    capsys.readouterr()
    assert _run(lake, "status") == 0
    assert "running" in capsys.readouterr().out


def test_opts_are_typed():
    assert _opts(["batch-size=64", "device=cuda", "fp=0.5", "strict=true"]) == {
        "batch_size": 64, "device": "cuda", "fp": 0.5, "strict": True}
    with pytest.raises(SystemExit):
        _opts(["nokey"])


def test_missing_root_is_an_error(monkeypatch, capsys):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("DATALAKE_ROOT", raising=False)
    assert main(["list"]) == 2


def test_update_appends_an_execution_to_a_complete_artifact(lake, capsys):
    assert _run(lake, "start", "toy_cli", "--start", "2008-01-01", "--end", "2008-02-29") == 0
    art = _partial(lake)
    capsys.readouterr()
    assert _run(lake, "update", art.artifact_id) == 0
    out = capsys.readouterr().out
    assert "complete" in out and "1/1" in out
    done = _partial(lake)
    assert len(done.meta.runs) == 2 and not done.partial
    assert (done.path / "update-1.parquet").exists()
    assert done.meta.runs[-1].produced == ["update-1.parquet"]


def test_update_of_a_partial_artifact_is_refused(lake):
    from datalake.jobs import JobError

    CliToyJob.die_at = "2008-02"
    with pytest.raises(RuntimeError):
        _run(lake, "start", "toy_cli", "--start", "2008-01-01", "--end", "2008-03-31")
    CliToyJob.die_at = None
    with pytest.raises(JobError, match="partial"):
        _run(lake, "update", _partial(lake).artifact_id)
