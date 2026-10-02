"""datalake folder groups ({layer}/{group}/{kind}/{id}) and move()."""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

import datalake.index as index_mod
from datalake.index import DatalakeError, DatalakeIndex, _same_device
from datalake.jobs import JobRunner
from datalake.layout import Layout
from datalake.meta import META_FILENAME, README_FILENAME, read_meta
from datalake.verify import verify
from tests.test_jobs import ToyJob, _clean_repo

PIPELINE, VERSION = "PhD-Narrative-Finance", "v0.1.0"


@pytest.fixture
def index(tmp_path: Path):
    with DatalakeIndex(tmp_path / "lake") as idx:
        yield idx


def _make(index: DatalakeIndex, kind: str, *, group: str | None = None,
          sources: list[str] | None = None, n: int = 2, **hp):
    with index.run(kind=kind, pipeline=PIPELINE, pipeline_version=VERSION,
                   hyperparams=hp, sources=sources, group=group) as run:
        for i in range(n):
            (run.out_dir / f"2010-{i + 1:02d}.parquet").write_bytes(b"x" * (i + 1))
    return index.get(run.artifact_id)


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

class TestGroups:
    def test_grouped_run_is_filed_under_its_group(self, index):
        art = _make(index, "toy_series", group="Vendor", a=1)
        assert art.path == index.root / "derived" / "Vendor" / "toy_series" / art.artifact_id
        assert art.group == "Vendor" and read_meta(art.path)[0].group == "Vendor"
        assert "- **Group**: `Vendor`" in (art.path / README_FILENAME).read_text()
        root = _make(index, "other_kind", a=1)
        assert root.path == index.root / "derived" / "other_kind" / root.artifact_id
        assert root.group is None

    def test_group_is_not_part_of_the_identity(self, index):
        a = _make(index, "toy_series", group="Vendor", a=1)
        assert "Vendor" not in a.artifact_id

    def test_default_group_scope(self, index):
        with index.default_group("Vendor"):
            inside = _make(index, "toy_series", a=1)
            explicit = _make(index, "toy_other", group="Other", a=1)
        outside = _make(index, "toy_series", a=2)
        assert (inside.group, explicit.group, outside.group) == ("Vendor", "Other", None)

    def test_job_group_applies_to_the_artifact_and_its_siblings(self, tmp_path):
        class GroupedJob(ToyJob):
            group = "Toy"

            def session(self, ctx):
                return ctx.index.run(kind="toy_sibling", pipeline=PIPELINE,
                                     pipeline_version=VERSION, hyperparams={"of": "x"})

        index = DatalakeIndex(tmp_path / "lake")
        runner = JobRunner(index, repo_dir=_clean_repo(tmp_path), handle_signals=False)
        art = runner.start(GroupedJob(Layout("M", date(2008, 1, 1), date(2008, 2, 29))))
        sibling = index.list("toy_sibling", include_partial=True)[0]
        assert art.group == "Toy" and sibling.group == "Toy"
        assert art.path.parent.parent.name == "Toy"
        index.close()

    def test_old_sidecar_without_group_loads_as_root(self, index):
        art = _make(index, "toy_series", a=1)
        meta_path = art.path / META_FILENAME
        payload = json.loads(meta_path.read_text())
        payload.pop("group"), payload.pop("relink")
        meta_path.write_text(json.dumps(payload))
        assert read_meta(art.path)[0].group is None
        assert index.reindex() == 1 and index.get(art.artifact_id).group is None

    def test_reindex_finds_root_and_grouped_and_skips_nested(self, index):
        a = _make(index, "toy_series", group="Vendor", a=1)
        b = _make(index, "toy_series_root", a=1)
        nested = a.path / "sub"
        nested.mkdir()
        (nested / META_FILENAME).write_text("{not an artifact")     # content, never parsed
        assert index.reindex() == 2
        assert index.get(a.artifact_id).path == a.path
        assert index.get(b.artifact_id).group is None

    def test_reindex_folder_wins_over_sidecar(self, index):
        art = _make(index, "toy_series", a=1)
        target = index.root / "derived" / "Moved" / "toy_series" / art.artifact_id
        target.parent.mkdir(parents=True)
        art.path.rename(target)                                       # moved by hand
        assert index.reindex() == 1
        got = index.get(art.artifact_id)
        assert (got.group, got.path) == ("Moved", target)

    def test_name_collisions_are_refused(self, index):
        _make(index, "toy_series", a=1)                               # root-level kind folder
        with pytest.raises(DatalakeError, match="name of a kind folder"):
            _make(index, "other", group="toy_series", a=1)
        _make(index, "other", group="Vendor", a=1)
        with pytest.raises(DatalakeError, match="name of a group folder"):
            _make(index, "Vendor", a=1)
        with pytest.raises(DatalakeError, match="invalid group name"):
            _make(index, "other", group="a/b", a=2)

    def test_list_and_latest_filter_on_group(self, index):
        g = _make(index, "toy_series", group="Vendor", a=1)
        r = _make(index, "toy_series", a=2)
        assert [a.artifact_id for a in index.list("toy_series", group="Vendor")] == [g.artifact_id]
        assert [a.artifact_id for a in index.list("toy_series", group="")] == [r.artifact_id]
        assert len(index.list("toy_series")) == 2
        assert index.latest("toy_series", group="Vendor").artifact_id == g.artifact_id

    def test_old_index_gets_the_group_column(self, tmp_path):
        root = tmp_path / "lake"
        root.mkdir()
        schema = (Path(index_mod.__file__).parent / "schema.sql").read_text()
        old_schema = schema.replace(",\n\n    -- Folder group", "\n    -- Folder group")
        old_schema = "\n".join(line for line in old_schema.splitlines()
                               if not line.strip().startswith("grp "))
        conn = sqlite3.connect(root / "index.db")
        conn.executescript(old_schema)
        assert "grp" not in {r[1] for r in conn.execute("PRAGMA table_info(artifacts)")}
        conn.close()
        with DatalakeIndex(root) as idx:
            cols = {r["name"] for r in idx._conn.execute("PRAGMA table_info(artifacts)")}
            assert "grp" in cols
            assert _make(idx, "toy_series", group="Vendor", a=1).group == "Vendor"
        with DatalakeIndex(root) as idx:                              # idempotent
            assert len(idx.list()) == 1


# ---------------------------------------------------------------------------
# move
# ---------------------------------------------------------------------------

class TestMove:
    def test_move_and_back_keeps_content_identity_and_lineage(self, index):
        parent = _make(index, "toy_parent", a=1)
        child = _make(index, "toy_child", sources=[parent.artifact_id], a=1)
        before = {p.name: p.read_bytes() for p in child.path.glob("*.parquet")}
        old, new = index.move(child.artifact_id, "Vendor")
        assert not old.exists() and new.is_dir()
        assert not (index.root / "derived" / "toy_child").exists()   # empty kind folder removed
        got = index.get(child.artifact_id)
        assert (got.path, got.group) == (new, "Vendor")
        assert got.file_hashes == child.file_hashes
        assert {p.name: p.read_bytes() for p in new.glob("*.parquet")} == before
        assert index.parents(child.artifact_id) == [parent.artifact_id]
        assert index.children(parent.artifact_id) == [child.artifact_id]
        assert read_meta(new)[0].group == "Vendor"
        assert verify(index, check_content=False).ok
        index.move(child.artifact_id, None)
        assert index.get(child.artifact_id).path == child.path
        assert not (index.root / "derived" / "Vendor").exists()       # empty group folder removed

    def test_dry_run_moves_nothing(self, index):
        art = _make(index, "toy_series", a=1)
        _, target = index.move(art.artifact_id, "Vendor", dry_run=True)
        assert art.path.is_dir() and not target.exists()
        assert index.get(art.artifact_id).group is None

    def test_refuses_a_live_lock(self, index):
        art = _make(index, "toy_series", a=1)
        from datalake.artifact import utc_now_iso
        (art.path / "job.lock").write_text(json.dumps({"heartbeat_at": utc_now_iso()}))
        with pytest.raises(DatalakeError, match="live job"):
            index.move(art.artifact_id, "Vendor")

    def test_refuses_partial_unless_allowed_and_resume_works_after(self, tmp_path):
        index = DatalakeIndex(tmp_path / "lake")
        runner = JobRunner(index, repo_dir=_clean_repo(tmp_path), handle_signals=False)
        job = ToyJob(Layout("M", date(2008, 1, 1), date(2008, 4, 30)))
        ToyJob.die_at = "2008-03"
        try:
            with pytest.raises(RuntimeError):
                runner.start(job)
        finally:
            ToyJob.die_at = None
        art = index.list("toy_series", include_partial=True)[0]
        with pytest.raises(DatalakeError, match="partial"):
            index.move(art.artifact_id, "Vendor")
        _, new = index.move(art.artifact_id, "Vendor", allow_partial=True)
        done = runner.resume(art.artifact_id)
        assert not done.partial and done.path == new
        assert sorted(p.stem for p in new.glob("*.parquet")) == ["2008-01", "2008-02",
                                                                 "2008-03", "2008-04"]
        index.close()

    def test_refuses_an_existing_target(self, index):
        art = _make(index, "toy_series", a=1)
        (index.root / "derived" / "Vendor" / "toy_series" / art.artifact_id).mkdir(parents=True)
        with pytest.raises(DatalakeError, match="already exists"):
            index.move(art.artifact_id, "Vendor")

    def test_refuses_another_device(self, tmp_path, monkeypatch):
        src = tmp_path / "a"
        src.mkdir()
        real_stat = Path.stat

        def fake_stat(self, *args, **kwargs):
            st = real_stat(self, *args, **kwargs)
            if self == src:
                import os
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev + 1, *tuple(st)[3:]))
            return st

        monkeypatch.setattr(Path, "stat", fake_stat)
        with pytest.raises(DatalakeError, match="different filesystems"):
            _same_device(src, tmp_path / "b" / "c")

    def test_crash_after_rename_is_repaired_by_reindex(self, index, monkeypatch):
        art = _make(index, "toy_series", a=1)

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(index_mod, "write_sidecars", boom)
        with pytest.raises(OSError):
            index.move(art.artifact_id, "Vendor")
        monkeypatch.undo()
        index.reindex()
        got = index.get(art.artifact_id)
        assert got.group == "Vendor" and got.path.is_dir()
        assert got.file_hashes == art.file_hashes
