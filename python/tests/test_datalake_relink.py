"""datalake.relink and datalake.equivalence: same files, new inputs or code, checked."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import datalake.jobs as jobs_mod
from datalake.equivalence import (
    FilesIdentical,
    KeysCover,
    ProjectionEqual,
    RerunSample,
    parse_check,
)
from datalake.index import DatalakeIndex
from datalake.jobs import Job, Unit
from datalake.meta import README_FILENAME, read_meta
from datalake.relink import RelinkError, relink_order, substitute
from datalake.verify import verify

PIPELINE, VERSION = "PhD-Narrative-Finance", "v0.1.0"
MONTHS = ("2010-01", "2010-02")


@pytest.fixture
def index(tmp_path: Path):
    with DatalakeIndex(tmp_path / "lake") as idx:
        yield idx


def _parent_rows(month: str) -> pl.DataFrame:
    n = 5 if month == "2010-01" else 4
    return pl.DataFrame({"ID": [f"{month}-{i}" for i in range(n)],
                         "TEXT": [f"headline {month} {i}" for i in range(n)],
                         "VALUE": [float(i) for i in range(n)]})


def _make_parent(index: DatalakeIndex, kind: str = "toy_parent", *, shuffle: bool = False,
                 rename: bool = False, extra: bool = False, change: str | None = None,
                 **hp) -> Any:
    with index.run(kind=kind, pipeline=PIPELINE, pipeline_version=VERSION,
                   hyperparams=hp) as run:
        for m in MONTHS:
            f = _parent_rows(m)
            if change == m:
                f = f.with_columns(pl.when(pl.col("ID") == f"{m}-0").then(pl.lit("edited"))
                                   .otherwise(pl.col("TEXT")).alias("TEXT"))
            if shuffle:
                f = f.reverse()
            if extra:
                f = f.with_columns(pl.lit("x").alias("NEW_FIELD"))
            if rename:
                f = f.rename({"TEXT": "HEADLINE"})
            f.write_parquet(run.out_dir / f"{m}.parquet")
    return index.get(run.artifact_id)


class ChildJob(Job):
    """Child of a toy_parent: per month, VALUE * factor (+ bias: a 'code change')."""

    kind = "toy_child"
    pipeline_version = "v0.1.0"
    hash_pattern = "*.parquet"
    bias: float = 0.0

    def __init__(self, parent_id: str, index: DatalakeIndex, factor: float = 2.0) -> None:
        self.parent_id, self.index, self.factor = parent_id, index, factor

    def params(self) -> dict[str, Any]:
        return {"parent_id": self.parent_id, "factor": self.factor}

    def sources(self) -> list[Any]:
        return [self.parent_id]

    def units(self) -> list[Unit]:
        return [Unit(m) for m in MONTHS]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return (out_dir / f"{unit.key}.parquet").exists()

    def run_unit(self, unit: Unit, ctx: Any) -> None:
        parent = self.index.get(self.parent_id)
        f = pl.read_parquet(parent.path / f"{unit.key}.parquet", columns=["ID", "VALUE"])
        (f.with_columns(pl.col("VALUE") * self.factor + self.bias).sort("ID")
         .write_parquet(ctx.out_dir / f"{unit.key}.parquet"))

    @classmethod
    def from_artifact(cls, artifact, index, **kwargs) -> ChildJob:
        hp = artifact.meta.hyperparams
        return cls(hp["parent_id"], index, hp["factor"])


@pytest.fixture(autouse=True)
def _registered_child(monkeypatch):
    # registered for these tests only: the jobs contract lists the real registry
    monkeypatch.setitem(jobs_mod._REGISTRY, ChildJob.kind, ChildJob)


def _make_child(index: DatalakeIndex, parent) -> Any:
    from datalake.jobs import JobRunner

    runner = JobRunner(index, handle_signals=False, allow_dirty=True, log_file=False)
    return runner.start(ChildJob(parent.artifact_id, index))


@pytest.fixture
def lake(index):
    old_parent = _make_parent(index, src="v1")
    child = _make_child(index, old_parent)
    new_parent = _make_parent(index, "toy_parent_v2", shuffle=True, rename=True, extra=True,
                              src="v2")
    return index, old_parent, child, new_parent


PROJECTION = ProjectionEqual(["ID", "TEXT", "VALUE"], rename={"TEXT": "HEADLINE"})
KEYS = KeysCover("ID")
KC = {"toy_parent": "toy_parent_v2"}          # the toy re-ingest changes kind


# ---------------------------------------------------------------------------
# relink
# ---------------------------------------------------------------------------

class TestRelink:
    def test_relink_registers_hard_linked_files_under_new_inputs(self, lake):
        index, old_parent, child, new_parent = lake
        res = index.relink(child.artifact_id,
                           replace={old_parent.artifact_id: new_parent.artifact_id},
                           checks=[PROJECTION, KEYS], kind_changes=KC, notes="re-ingested parent")
        assert res.registered and res.passed and res.new_id != child.artifact_id
        new = index.get(res.new_id)
        assert new.meta.sources == [new_parent.artifact_id]
        assert new.meta.hyperparams == {"parent_id": new_parent.artifact_id, "factor": 2.0}
        assert res.hyperparams_diff == {"parent_id": (old_parent.artifact_id,
                                                      new_parent.artifact_id)}
        assert new.file_hashes == child.file_hashes and not new.partial
        for name in child.file_hashes:
            a, b = (child.path / name).stat(), (new.path / name).stat()
            assert a.st_ino == b.st_ino and a.st_nlink == 2
        assert new.meta.relink["from"] == child.artifact_id
        assert [c["passed"] for c in new.meta.relink["checks"]] == [True, True]
        assert "## Relinked" in (new.path / README_FILENAME).read_text()
        assert index.parents(res.new_id) == [new_parent.artifact_id]   # not the old child
        assert not index.get(child.artifact_id).deprecated
        assert verify(index, artifact_ids=[res.new_id], check_content=False).ok

    def test_failed_check_registers_nothing_and_leaves_no_directory(self, index):
        old_parent = _make_parent(index, src="v1")
        child = _make_child(index, old_parent)
        edited = _make_parent(index, "toy_parent_v2", rename=True, change="2010-02", src="v2")
        n_before = len(index.list(include_partial=True))
        with pytest.raises(RelinkError, match="equivalence check") as err:
            index.relink(child.artifact_id, replace={old_parent.artifact_id: edited.artifact_id},
                         checks=[PROJECTION, KEYS], kind_changes=KC)
        assert not err.value.results[0].passed and "2010-02" in err.value.results[0].details
        assert len(index.list(include_partial=True)) == n_before
        assert sorted(p.name for p in child.path.parent.iterdir()) == [child.artifact_id]

    def test_dry_run_writes_nothing(self, lake):
        index, old_parent, child, new_parent = lake
        res = index.relink(child.artifact_id,
                           replace={old_parent.artifact_id: new_parent.artifact_id},
                           checks=[PROJECTION, KEYS], kind_changes=KC, dry_run=True)
        assert res.passed and not res.registered and not res.path.exists()
        assert res.n_files == 2 and not index.exists(res.new_id)

    def test_refusals(self, lake):
        index, old_parent, child, new_parent = lake
        rep = {old_parent.artifact_id: new_parent.artifact_id}
        with pytest.raises(RelinkError, match="no equivalence check"):
            index.relink(child.artifact_id, replace=rep)
        with pytest.raises(RelinkError, match="not recorded sources"):
            index.relink(child.artifact_id, replace={"nope": new_parent.artifact_id},
                         checks=[PROJECTION, KEYS], kind_changes=KC)
        # (the message names the rule: not a recorded source)
        with pytest.raises(RelinkError, match="nothing to relink"):
            index.relink(child.artifact_id, replace={}, checks=[PROJECTION, KEYS], kind_changes=KC)
        index.deprecate(new_parent.artifact_id, "test")
        with pytest.raises(RelinkError, match="deprecated"):
            index.relink(child.artifact_id, replace=rep, checks=[PROJECTION, KEYS], kind_changes=KC)

    def test_deprecate_old_only_on_request(self, lake):
        index, old_parent, child, new_parent = lake
        res = index.relink(child.artifact_id,
                           replace={old_parent.artifact_id: new_parent.artifact_id},
                           checks=[PROJECTION, KEYS], kind_changes=KC, deprecate_old=True)
        old = index.get(child.artifact_id)
        assert old.deprecated and res.new_id in old.meta.deprecation_reason
        assert (old.path / next(iter(old.file_hashes))).exists()     # files kept

    def test_new_code_version_with_rerun(self, lake):
        index, old_parent, child, new_parent = lake
        rep = {old_parent.artifact_id: new_parent.artifact_id}
        res = index.relink(child.artifact_id, replace=rep, pipeline_version="v0.1.1",
                           checks=[PROJECTION, KEYS, RerunSample(["2010-02"])],
                           kind_changes=KC, dry_run=True)
        assert res.passed, res.checks
        ChildJob.bias = 1.0                              # the code now computes something else
        try:
            with pytest.raises(RelinkError, match="outputs differ"):
                index.relink(child.artifact_id, replace=rep, pipeline_version="v0.1.1",
                             checks=[PROJECTION, KEYS, RerunSample(["2010-02"])], kind_changes=KC)
        finally:
            ChildJob.bias = 0.0
        with pytest.raises(RelinkError, match="unknown unit"):
            index.relink(child.artifact_id, replace=rep,
                         checks=[PROJECTION, KEYS, RerunSample(["1999-01"])], kind_changes=KC)

    def test_chain_and_relink_order(self, index):
        p = _make_parent(index, src="v1")
        c = _make_child(index, p)
        with index.run(kind="toy_grandchild", pipeline=PIPELINE, pipeline_version=VERSION,
                       hyperparams={"child": c.artifact_id, "parent": p.artifact_id},
                       sources=[c.artifact_id, p.artifact_id]) as run:
            (run.out_dir / "series.parquet").write_bytes(b"g")       # not a period partition
        g = index.get(run.artifact_id)
        assert relink_order(index, p.artifact_id) == [c.artifact_id, g.artifact_id]
        p2 = _make_parent(index, "toy_parent_v2", src="v2")
        c2 = index.relink(c.artifact_id, replace={p.artifact_id: p2.artifact_id},
                          checks=[FilesIdentical(), KEYS], kind_changes=KC)
        g2 = index.relink(g.artifact_id, replace={c.artifact_id: c2.new_id,
                                                  p.artifact_id: p2.artifact_id},
                          checks=[FilesIdentical(c.artifact_id), FilesIdentical(p.artifact_id)],
                          kind_changes=KC)
        assert index.get(g2.new_id).meta.hyperparams == {"child": c2.new_id,
                                                         "parent": p2.artifact_id}
        assert sorted(index.parents(g2.new_id)) == sorted([c2.new_id, p2.artifact_id])

    def test_relinked_artifact_rebuilds_its_job(self, lake):
        index, old_parent, child, new_parent = lake
        res = index.relink(child.artifact_id,
                           replace={old_parent.artifact_id: new_parent.artifact_id},
                           checks=[PROJECTION, KEYS], kind_changes=KC)
        job = ChildJob.from_artifact(index.get(res.new_id), index)
        assert job.parent_id == new_parent.artifact_id and job.params() == \
            index.get(res.new_id).meta.hyperparams

    def test_embedded_ids_are_warned(self, lake, caplog):
        index, old_parent, child, new_parent = lake
        assert substitute({"a": [old_parent.artifact_id], "b": {"c": "x"}},
                          {old_parent.artifact_id: "NEW"}) == {"a": ["NEW"], "b": {"c": "x"}}
        with caplog.at_level(logging.WARNING, logger="datalake.relink"):
            index.relink(child.artifact_id,
                         replace={old_parent.artifact_id: new_parent.artifact_id},
                         set_params={"note": f"from {old_parent.artifact_id}"},
                         checks=[PROJECTION, KEYS], kind_changes=KC, dry_run=True)
        assert "embed replaced id" in caplog.text


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

class TestChecks:
    def _ctx(self, index, child, old_parent, new_parent):
        from datalake.equivalence import RelinkContext

        return RelinkContext(index, child, child, {old_parent.artifact_id: new_parent.artifact_id})

    def test_projection_by_stem_and_by_all(self, lake):
        index, old_parent, child, new_parent = lake
        ctx = self._ctx(index, child, old_parent, new_parent)
        assert PROJECTION(ctx).passed
        assert ProjectionEqual(["ID", "TEXT"], rename={"TEXT": "HEADLINE"}, by="all")(ctx).passed
        missing = ProjectionEqual(["ID", "NOPE"])(ctx)
        assert not missing.passed and "NOPE" in missing.details
        unrenamed = ProjectionEqual(["ID", "TEXT"])(ctx)            # TEXT is HEADLINE now
        assert not unrenamed.passed

    def test_projection_detects_a_moved_row_by_stem_not_by_all(self, index):
        old_parent = _make_parent(index, src="v1")
        child = _make_child(index, old_parent)
        with index.run(kind="toy_parent_v2", pipeline=PIPELINE, pipeline_version=VERSION,
                       hyperparams={"src": "v2"}) as run:
            a, b = _parent_rows("2010-01"), _parent_rows("2010-02")
            moved = a.tail(1)
            a.head(4).write_parquet(run.out_dir / "2010-01.parquet")
            pl.concat([b, moved]).write_parquet(run.out_dir / "2010-02.parquet")
        ctx = self._ctx(index, child, old_parent, index.get(run.artifact_id))
        assert not ProjectionEqual(["ID", "TEXT"])(ctx).passed
        assert ProjectionEqual(["ID", "TEXT"], by="all")(ctx).passed

    def test_keys_cover(self, lake, index):
        _, old_parent, child, new_parent = lake
        ctx = self._ctx(index, child, old_parent, new_parent)
        assert KeysCover("ID")(ctx).passed
        short = _make_parent(index, "toy_parent_v3", change=None, src="v3")
        f = pl.read_parquet(short.path / "2010-01.parquet").head(3)
        f.write_parquet(short.path / "2010-01.parquet")
        ctx2 = self._ctx(index, child, old_parent, short)
        res = KeysCover("ID")(ctx2)
        assert not res.passed and "2010-01" in res.details

    def test_files_identical(self, lake, index):
        _, old_parent, child, new_parent = lake
        assert not FilesIdentical()(self._ctx(index, child, old_parent, new_parent)).passed
        copy = _make_parent(index, "toy_parent_copy", src="v1")
        assert FilesIdentical()(self._ctx(index, child, old_parent, copy)).passed

    def test_parse_check(self):
        assert parse_check("files") == FilesIdentical()
        assert parse_check("files@P") == FilesIdentical("P")
        p = parse_check("projection@P=ID,TEXT:HEADLINE,by=all")
        assert (p.columns, p.rename, p.by, p.parent) == (["ID", "TEXT"], {"TEXT": "HEADLINE"},
                                                         "all", "P")
        k = parse_check("keys=RP_STORY_ID:STORY_ID")
        assert (k.key, k.parent_key, k.by) == ("RP_STORY_ID", "STORY_ID", "stem")
        r = parse_check("rerun=2010-01,2010-02,rtol=1e-6")
        assert (r.units, r.rtol) == (["2010-01", "2010-02"], 1e-6)
        with pytest.raises(Exception, match="bad check spec"):
            parse_check("projection")

    def test_read_meta_keeps_relink_record(self, lake):
        index, old_parent, child, new_parent = lake
        res = index.relink(child.artifact_id,
                           replace={old_parent.artifact_id: new_parent.artifact_id},
                           checks=[PROJECTION, KEYS], kind_changes=KC)
        meta, _ = read_meta(res.path)
        assert meta.relink["replace"] == {old_parent.artifact_id: new_parent.artifact_id}
        index.reindex()
        assert index.get(res.new_id).meta.relink["from"] == child.artifact_id


# ---------------------------------------------------------------------------
# Replace validation (before any check runs) and asserted lineage
# ---------------------------------------------------------------------------

def _plain(index, kind, frames: dict[str, pl.DataFrame], sources=None, **hp):
    """A complete artifact holding ``frames`` as {stem}.parquet."""
    with index.run(kind=kind, pipeline=PIPELINE, pipeline_version=VERSION, hyperparams=hp,
                   sources=sources) as run:
        for stem, f in frames.items():
            f.write_parquet(run.out_dir / f"{stem}.parquet")
    return index.get(run.artifact_id)


def _ids(month: str) -> pl.DataFrame:
    return _parent_rows(month).select("ID")


class TestValidation:
    def _refused(self, index, child, rep, checks, match, **kw):
        with pytest.raises(RelinkError, match=match):
            index.relink(child.artifact_id, replace=rep, checks=checks, **kw)
        assert not index.exists(f"{child.artifact_id}x")       # nothing registered
        assert sorted(p.name for p in child.path.parent.iterdir()) == [child.artifact_id]

    def test_needs_a_content_check_per_replaced_parent(self, lake):
        index, old_parent, child, new_parent = lake
        self._refused(index, child, {old_parent.artifact_id: new_parent.artifact_id}, [KEYS],
                      r"\[checks\] no content check", kind_changes=KC)

    def test_needs_a_keys_check_for_partitioned_children(self, lake):
        index, old_parent, child, new_parent = lake
        self._refused(index, child, {old_parent.artifact_id: new_parent.artifact_id},
                      [PROJECTION], r"keys check is required", kind_changes=KC)

    def test_kind_change_must_be_declared(self, lake):
        index, old_parent, child, new_parent = lake
        self._refused(index, child, {old_parent.artifact_id: new_parent.artifact_id},
                      [PROJECTION, KEYS], r"\[kind\]")

    def test_layout_mismatch_and_missing_partition(self, lake):
        index, old_parent, child, _ = lake
        daily = _plain(index, "toy_parent_v2", {"2010-01-15": _parent_rows("2010-01")}, d=1)
        self._refused(index, child, {old_parent.artifact_id: daily.artifact_id},
                      [PROJECTION, KEYS], r"\[layout\] partition frequencies differ",
                      kind_changes=KC)
        short = _plain(index, "toy_parent_v2", {"2010-01": _parent_rows("2010-01")}, d=2)
        self._refused(index, child, {old_parent.artifact_id: short.artifact_id},
                      [PROJECTION, KEYS], r"lacks 1 partition\(s\) the child holds",
                      kind_changes=KC)

    def test_asserted_source(self, index):
        old_parent = _make_parent(index, src="v1")
        new_parent = _make_parent(index, "toy_parent_v2", src="v2")
        legacy = _plain(index, "toy_legacy", {m: _ids(m) for m in MONTHS})   # no sources
        rep = {old_parent.artifact_id: new_parent.artifact_id}
        checks = [ProjectionEqual(["ID", "TEXT"]), KeysCover("ID")]
        self._refused(index, legacy, rep, checks, "not recorded sources", kind_changes=KC,
                      pipeline_version="v0.2.0")
        res = index.relink(legacy.artifact_id, replace=rep, checks=checks, kind_changes=KC,
                           assert_sources=True, pipeline_version="v0.2.0")
        new = index.get(res.new_id)
        assert new.meta.sources == [new_parent.artifact_id]
        assert new.meta.relink["asserted"] == [old_parent.artifact_id]
        assert "asserted: never recorded" in (new.path / README_FILENAME).read_text()

    def test_asserted_source_with_other_row_counts(self, index):
        old_parent = _make_parent(index, src="v1")
        new_parent = _make_parent(index, "toy_parent_v2", src="v2")
        legacy = _plain(index, "toy_legacy", {"2010-01": _ids("2010-01").head(3),
                                              "2010-02": _ids("2010-02")})
        self._refused(index, legacy, {old_parent.artifact_id: new_parent.artifact_id},
                      [ProjectionEqual(["ID"]), KeysCover("ID")], r"\[asserted\] row counts",
                      kind_changes=KC, assert_sources=True, pipeline_version="v0.2.0")


class TestProjectionOptions:
    def _ctx(self, index, old, new):
        from datalake.equivalence import RelinkContext

        return RelinkContext(index, old, old, {old.artifact_id: new.artifact_id})

    def test_as_date_text_vs_datetime_and_time_zones(self, index):
        text = pl.DataFrame({"ID": ["a", "b"], "TS": ["2010-01-01 23:30:00.000",
                                                      "2010-01-02 08:00:00"]})
        dt = text.with_columns(pl.col("TS").str.to_datetime("%Y-%m-%d %H:%M:%S%.f",
                                                            time_zone="UTC"))
        old = _plain(index, "p_old", {"2010-01": text})
        new = _plain(index, "p_new", {"2010-01": dt})
        ctx = self._ctx(index, old, new)
        for tz in ("UTC", "Asia/Tokyo", "America/New_York"):
            assert ProjectionEqual(["ID", "TS"], as_date=["TS"], tz=tz)(ctx).passed
        shifted = _plain(index, "p_shift", {"2010-01": dt.with_columns(
            pl.col("TS") + pl.duration(hours=1))})
        ctx2 = self._ctx(index, old, shifted)
        assert not ProjectionEqual(["ID", "TS"], as_date=["TS"])(ctx2).passed   # 23:30 -> 00:30
        assert ProjectionEqual(["ID", "TS"], as_date=["TS"], tz="America/New_York")(ctx2).passed

    def test_old_nulls(self, index):
        old = _plain(index, "p_old", {"2010-01": pl.DataFrame({"ID": ["a", "b"],
                                                               "H": ["<NA>", "x"]})})
        new = _plain(index, "p_new", {"2010-01": pl.DataFrame({"ID": ["a", "b"],
                                                               "H": [None, "x"]})})
        ctx = self._ctx(index, old, new)
        assert not ProjectionEqual(["ID", "H"])(ctx).passed
        res = ProjectionEqual(["ID", "H"], old_nulls=["<NA>"])(ctx)
        assert res.passed and "1 old value(s)" in res.details

    def test_parse_check_options(self):
        p = parse_check("projection@P=ID,TS,as_date=TS,tz=Asia/Tokyo,old_nulls=<NA>+None")
        assert (p.as_date, p.tz, p.old_nulls) == (["TS"], "Asia/Tokyo", ["<NA>", "None"])


def test_sample_stems_spreads_and_includes_the_ends():
    from datalake.equivalence import sample_stems

    keys = [f"20{y:02d}-01" for y in range(26)]
    assert sample_stems(keys, 6) == ["2000-01", "2005-01", "2010-01", "2015-01", "2020-01",
                                     "2025-01"]
    assert sample_stems(keys, 0) == keys and sample_stems(keys, 100) == keys
    assert sample_stems(keys, 1) == ["2000-01"]


def test_checks_sample_partitions_and_name_them(lake):
    index, old_parent, child, new_parent = lake
    from datalake.equivalence import RelinkContext

    ctx = RelinkContext(index, child, child, {old_parent.artifact_id: new_parent.artifact_id})
    one = ProjectionEqual(["ID", "TEXT"], rename={"TEXT": "HEADLINE"}, sample=1)(ctx)
    assert one.passed and "sample 1/2 partitions: 2010-01" in one.details
    full = ProjectionEqual(["ID", "TEXT"], rename={"TEXT": "HEADLINE"}, sample=0)(ctx)
    assert full.passed and "sample" not in full.details
    assert parse_check("keys=ID,sample=3").sample == 3
