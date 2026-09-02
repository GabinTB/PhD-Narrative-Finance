"""Tests for the datalake layer.

Covers slug determinism, the run() context manager (including the crash path),
lookups, lineage traversal, deprecation, reindexing from sidecars, and
verification of corrupted or incomplete artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from datalake.artifact import (
    ModelCard,
    RunMeta,
    param_slug,
    slugify,
)
from datalake.index import DatalakeError, DatalakeIndex
from datalake.meta import (
    META_FILENAME,
    README_FILENAME,
    hash_file,
    read_meta,
    render_readme,
)
from datalake.verify import verify

PIPELINE = "PhD-Narrative-Finance"
VERSION = "v0.1.0"


@pytest.fixture
def index(tmp_path: Path) -> DatalakeIndex:
    with DatalakeIndex(tmp_path / "datalake") as idx:
        yield idx


@pytest.fixture
def card() -> ModelCard:
    return ModelCard(
        model_id="ravenbert",
        version="1.2",
        repo="https://github.com/GabinTB/ravenbert",
        weights_public=False,
        weights_sha256="a" * 64,
        dim=384,
        pooling="cls",
    )


def _run_producing(index: DatalakeIndex, kind: str, n_files: int = 2, **kwargs):
    """Helper: complete a run that writes n_files parquet-ish files."""
    with index.run(kind=kind, pipeline=PIPELINE, pipeline_version=VERSION, **kwargs) as run:
        for i in range(n_files):
            (run.out_dir / f"2010-{i + 1:02d}.parquet").write_bytes(b"data" * (i + 1))
    return index.latest(kind)


# ---------------------------------------------------------------------------
# Slug construction
# ---------------------------------------------------------------------------

class TestSlugs:
    def test_slugify_replaces_unsafe_characters(self):
        assert slugify("a b/c") == "a-b-c"

    def test_slugify_empty_becomes_none(self):
        assert slugify("") == "none"

    def test_param_slug_is_order_independent(self):
        a = param_slug({"tau": 0.2, "rho": 97.5})
        b = param_slug({"rho": 97.5, "tau": 0.2})
        assert a == b

    def test_param_slug_normalises_float_representation(self):
        assert param_slug({"tau": 0.20}) == param_slug({"tau": 0.2})

    def test_param_slug_empty_dict(self):
        assert param_slug({}) == ""

    def test_param_slug_bools_are_lowercase(self):
        assert "true" in param_slug({"purge": True})

    def test_long_value_is_digested_not_inlined(self):
        long_path = "/mnt/storage/some/very/long/path/to/the/raw/data/directory"
        slug = param_slug({"raw_dir": long_path})
        assert long_path not in slug
        assert len(slug) < 40

    def test_long_value_digest_is_deterministic(self):
        params = {"columns": ",".join(f"COLUMN_NAME_{i}" for i in range(20))}
        assert param_slug(params) == param_slug(params)

    def test_distinct_long_values_yield_distinct_slugs(self):
        a = param_slug({"raw_dir": "/mnt/storage/data/vintage_a_long_enough_to_hash"})
        b = param_slug({"raw_dir": "/mnt/storage/data/vintage_b_long_enough_to_hash"})
        assert a != b

    def test_slug_stays_within_filename_limit(self):
        """Many long parameters must still produce a usable directory name."""
        params = {f"param_{i}": f"value_{'x' * 50}_{i}" for i in range(20)}
        meta = RunMeta(
            kind="primitive_scores", pipeline=PIPELINE, pipeline_version=VERSION,
            hyperparams=params, created="2026-09-01T12:00:00+00:00",
        )
        assert len(meta.artifact_id.encode()) < 255

    def test_collapsed_slug_distinguishes_parameter_sets(self):
        """Two overlong sets sharing a prefix must not collide."""
        base = {f"param_{i}": "x" * 40 for i in range(10)}
        other = dict(base, extra="y" * 40)
        assert param_slug(base) != param_slug(other)

    def test_artifact_id_is_deterministic(self):
        kwargs = dict(
            kind="primitive_scores", pipeline=PIPELINE, pipeline_version=VERSION,
            hyperparams={"tau": 0.2}, created="2026-09-01T12:00:00+00:00",
        )
        assert RunMeta(**kwargs).artifact_id == RunMeta(**kwargs).artifact_id

    def test_artifact_id_includes_model_when_present(self, card):
        meta = RunMeta(
            kind="headline_embeddings", pipeline=PIPELINE, pipeline_version=VERSION,
            model_card=card, created="2026-09-01T12:00:00+00:00",
        )
        assert "ravenbert-1.2" in meta.artifact_id

    def test_artifact_id_differs_by_hyperparams(self):
        base = dict(kind="scores", pipeline=PIPELINE, pipeline_version=VERSION,
                    created="2026-09-01T12:00:00+00:00")
        a = RunMeta(**base, hyperparams={"tau": 0.2}).artifact_id
        b = RunMeta(**base, hyperparams={"tau": 0.3}).artifact_id
        assert a != b


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------

class TestModelCard:
    def test_roundtrip_dict(self, card):
        assert ModelCard.from_dict(card.to_dict()) == card

    def test_private_weights_without_hash_gets_warning_note(self):
        card = ModelCard(
            model_id="m", version="1", repo="r",
            weights_public=False, weights_sha256=None,
        )
        assert "warning" in card.notes.lower()

    def test_public_weights_no_warning(self):
        card = ModelCard(model_id="m", version="1", repo="r", weights_public=True)
        assert card.notes == ""


# ---------------------------------------------------------------------------
# run() context manager
# ---------------------------------------------------------------------------

class TestRun:
    def test_creates_out_dir(self, index):
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION) as run:
            assert run.out_dir.is_dir()

    def test_writes_both_sidecars(self, index):
        artifact = _run_producing(index, "k")
        assert (artifact.path / META_FILENAME).is_file()
        assert (artifact.path / README_FILENAME).is_file()

    def test_partial_cleared_on_success(self, index):
        artifact = _run_producing(index, "k")
        assert not artifact.partial

    def test_run_end_set_on_success(self, index):
        artifact = _run_producing(index, "k")
        assert artifact.meta.run_end is not None

    def test_files_hashed_on_success(self, index):
        artifact = _run_producing(index, "k", n_files=3)
        assert len(artifact.file_hashes) == 3

    def test_sidecars_exclude_themselves_from_hashes(self, index):
        artifact = _run_producing(index, "k")
        assert META_FILENAME not in artifact.file_hashes
        assert README_FILENAME not in artifact.file_hashes

    def test_partial_visible_before_completion(self, index):
        """Sidecar and index row exist while the run is still in flight."""
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION) as run:
            assert (run.out_dir / META_FILENAME).is_file()
            registered = index.list("k", include_partial=True)
            assert len(registered) == 1
            assert registered[0].partial

    def test_crash_leaves_artifact_partial(self, index):
        with pytest.raises(ValueError):
            with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION) as run:
                (run.out_dir / "half.parquet").write_bytes(b"x")
                raise ValueError("boom")

        partial = index.list("k", include_partial=True)
        assert len(partial) == 1
        assert partial[0].partial

    def test_crashed_run_not_returned_by_latest(self, index):
        with pytest.raises(ValueError):
            with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION):
                raise ValueError("boom")

        with pytest.raises(DatalakeError):
            index.latest("k")

    def test_note_persisted_in_sidecar(self, index):
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION) as run:
            run.note("scored 311 of 312 months")
            (run.out_dir / "f.parquet").write_bytes(b"x")

        artifact = index.latest("k")
        # note() writes to the execution's record, not the artifact-level notes
        assert "311" in artifact.meta.runs[-1].notes

    def test_hyperparams_survive_roundtrip(self, index):
        artifact = _run_producing(index, "k", hyperparams={"tau": 0.2, "rho": 97.5})
        assert artifact.meta.hyperparams == {"tau": 0.2, "rho": 97.5}

    def test_model_card_survives_roundtrip(self, index, card):
        artifact = _run_producing(index, "k", model_card=card)
        assert artifact.meta.model_card is not None
        assert artifact.meta.model_card.model_id == "ravenbert"
        assert artifact.meta.model_card.dim == 384


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

class TestLookups:
    def test_get_by_id(self, index):
        artifact = _run_producing(index, "k")
        assert index.get(artifact.artifact_id).artifact_id == artifact.artifact_id

    def test_get_unknown_raises(self, index):
        with pytest.raises(DatalakeError):
            index.get("does-not-exist")

    def test_latest_with_no_match_raises(self, index):
        with pytest.raises(DatalakeError):
            index.latest("nothing_here")

    def test_list_excludes_partial_by_default(self, index):
        with pytest.raises(ValueError):
            with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION):
                raise ValueError("boom")
        assert index.list("k") == []
        assert len(index.list("k", include_partial=True)) == 1

    def test_list_filters_by_kind(self, index):
        _run_producing(index, "alpha")
        _run_producing(index, "beta")
        assert len(index.list("alpha")) == 1

    def test_list_filters_by_model(self, index, card):
        _run_producing(index, "k", model_card=card)
        assert len(index.list("k", model="ravenbert")) == 1
        assert len(index.list("k", model="other")) == 0

    def test_exists(self, index):
        artifact = _run_producing(index, "k")
        assert index.exists(artifact.artifact_id)
        assert not index.exists("nope")

    def test_artifact_glob_string(self, index):
        artifact = _run_producing(index, "k")
        assert artifact.glob().endswith("*.parquet")
        assert str(artifact.path) in artifact.glob()


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

class TestLineage:
    def test_parents_recorded(self, index):
        parent = _run_producing(index, "raw_kind")
        child = _run_producing(index, "child_kind", sources=[parent])
        assert index.parents(child.artifact_id) == [parent.artifact_id]

    def test_children_recorded(self, index):
        parent = _run_producing(index, "raw_kind")
        child = _run_producing(index, "child_kind", sources=[parent])
        assert index.children(parent.artifact_id) == [child.artifact_id]

    def test_descendants_are_transitive(self, index):
        a = _run_producing(index, "a")
        b = _run_producing(index, "b", sources=[a])
        c = _run_producing(index, "c", sources=[b])
        assert set(index.descendants(a.artifact_id)) == {b.artifact_id, c.artifact_id}

    def test_ancestors_are_transitive(self, index):
        a = _run_producing(index, "a")
        b = _run_producing(index, "b", sources=[a])
        c = _run_producing(index, "c", sources=[b])
        assert set(index.ancestors(c.artifact_id)) == {a.artifact_id, b.artifact_id}

    def test_descendants_of_leaf_is_empty(self, index):
        a = _run_producing(index, "a")
        assert index.descendants(a.artifact_id) == []

    def test_sources_accept_plain_ids(self, index):
        parent = _run_producing(index, "a")
        child = _run_producing(index, "b", sources=[parent.artifact_id])
        assert index.parents(child.artifact_id) == [parent.artifact_id]

    def test_unregistered_parent_is_allowed(self, index):
        """External inputs may be cited without being registered."""
        child = _run_producing(index, "b", sources=["some-external-dataset"])
        assert index.parents(child.artifact_id) == ["some-external-dataset"]


# ---------------------------------------------------------------------------
# Deprecation
# ---------------------------------------------------------------------------

class TestDeprecate:
    def test_deprecated_excluded_from_latest(self, index):
        artifact = _run_producing(index, "k")
        index.deprecate(artifact.artifact_id, "argpartition bug")
        with pytest.raises(DatalakeError):
            index.latest("k")

    def test_deprecated_still_retrievable_by_id(self, index):
        artifact = _run_producing(index, "k")
        index.deprecate(artifact.artifact_id, "bad run")
        assert index.get(artifact.artifact_id).deprecated

    def test_deprecation_reason_recorded(self, index):
        artifact = _run_producing(index, "k")
        index.deprecate(artifact.artifact_id, "scoring bug")
        assert index.get(artifact.artifact_id).meta.deprecation_reason == "scoring bug"

    def test_files_not_deleted(self, index):
        artifact = _run_producing(index, "k", n_files=2)
        index.deprecate(artifact.artifact_id, "superseded")
        assert len(list(artifact.path.glob("*.parquet"))) == 2

    def test_sidecar_updated_on_disk(self, index):
        artifact = _run_producing(index, "k")
        index.deprecate(artifact.artifact_id, "superseded")
        meta, _ = read_meta(artifact.path)
        assert meta.deprecated


# ---------------------------------------------------------------------------
# Reindex
# ---------------------------------------------------------------------------

class TestReindex:
    def test_rebuilds_from_sidecars(self, index):
        a = _run_producing(index, "a")
        b = _run_producing(index, "b", sources=[a])

        index._conn.execute("DELETE FROM artifacts")
        index._conn.commit()
        assert index.list("a", include_partial=True) == []

        count = index.reindex()
        assert count == 2
        assert index.exists(a.artifact_id)
        assert index.exists(b.artifact_id)

    def test_reindex_restores_lineage(self, index):
        a = _run_producing(index, "a")
        b = _run_producing(index, "b", sources=[a])
        index.reindex()
        assert index.parents(b.artifact_id) == [a.artifact_id]

    def test_reindex_restores_file_hashes(self, index):
        artifact = _run_producing(index, "k", n_files=3)
        index.reindex()
        assert len(index.get(artifact.artifact_id).file_hashes) == 3

    def test_reindex_on_empty_root(self, index):
        assert index.reindex() == 0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class TestVerify:
    def test_clean_datalake_passes(self, index):
        _run_producing(index, "k")
        report = verify(index)
        assert report.ok

    def test_detects_modified_file(self, index):
        artifact = _run_producing(index, "k")
        target = next(artifact.path.glob("*.parquet"))
        target.write_bytes(b"tampered")

        report = verify(index)
        assert not report.ok
        assert any("digest mismatch" in f.message for f in report.errors)

    def test_detects_deleted_file(self, index):
        artifact = _run_producing(index, "k", n_files=2)
        next(artifact.path.glob("*.parquet")).unlink()

        report = verify(index)
        assert not report.ok
        assert any("missing from disk" in f.message for f in report.errors)

    def test_detects_missing_directory(self, index):
        import shutil
        artifact = _run_producing(index, "k")
        shutil.rmtree(artifact.path)

        report = verify(index)
        assert not report.ok

    def test_flags_partial_run_as_warning(self, index):
        with pytest.raises(ValueError):
            with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION):
                raise ValueError("boom")

        report = verify(index)
        assert any("incomplete" in f.message for f in report.warnings)

    def test_flags_stray_tmp_file(self, index):
        artifact = _run_producing(index, "k")
        (artifact.path / "2010-03.parquet.tmp").write_bytes(b"partial")

        report = verify(index)
        assert any("temporary file" in f.message for f in report.warnings)

    def test_flags_unrecorded_file(self, index):
        artifact = _run_producing(index, "k")
        (artifact.path / "surprise.parquet").write_bytes(b"where did this come from")

        report = verify(index)
        assert any("not recorded" in f.message for f in report.warnings)

    def test_errors_on_build_from_partial_input(self, index):
        """A completed artifact citing an incomplete input is an error."""
        with pytest.raises(ValueError):
            with index.run(kind="upstream", pipeline=PIPELINE,
                           pipeline_version=VERSION) as run:
                (run.out_dir / "x.parquet").write_bytes(b"x")
                raise ValueError("boom")

        broken = index.list("upstream", include_partial=True)[0]
        _run_producing(index, "downstream", sources=[broken])

        report = verify(index)
        assert any("incomplete input" in f.message for f in report.errors)

    def test_warns_on_build_from_deprecated_input(self, index):
        parent = _run_producing(index, "upstream")
        _run_producing(index, "downstream", sources=[parent])
        index.deprecate(parent.artifact_id, "bad")

        report = verify(index)
        assert any("deprecated input" in f.message for f in report.warnings)

    def test_no_hashes_mode_skips_file_checks(self, index):
        artifact = _run_producing(index, "k")
        next(artifact.path.glob("*.parquet")).write_bytes(b"tampered")

        report = verify(index, check_hashes=False)
        assert report.n_files_checked == 0


# ---------------------------------------------------------------------------
# Sidecar rendering
# ---------------------------------------------------------------------------

class TestSidecars:
    def test_meta_json_is_valid_json(self, index):
        artifact = _run_producing(index, "k")
        payload = json.loads((artifact.path / META_FILENAME).read_text())
        assert payload["artifact_id"] == artifact.artifact_id
        assert payload["partial"] is False

    def test_meta_roundtrip(self, index, card):
        artifact = _run_producing(
            index, "k", hyperparams={"tau": 0.2}, model_card=card,
        )
        meta, hashes = read_meta(artifact.path)
        assert meta.kind == "k"
        assert meta.hyperparams == {"tau": 0.2}
        assert meta.model_card.model_id == "ravenbert"
        assert len(hashes) == 2

    def test_readme_flags_partial_run(self):
        from datalake.artifact import RunRecord
        meta = RunMeta(kind="k", pipeline=PIPELINE, pipeline_version=VERSION)
        meta.runs.append(RunRecord(run_start="2026-09-01T12:00:00+00:00",
                                   pipeline_version=VERSION, partial=True))
        assert meta.partial
        assert "INCOMPLETE" in render_readme(meta)

    def test_readme_flags_deprecation(self):
        from datalake.artifact import RunRecord
        meta = RunMeta(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       deprecated=True, deprecation_reason="superseded by v2")
        meta.runs.append(RunRecord(run_start="2026-09-01T12:00:00+00:00",
                                   pipeline_version=VERSION, partial=False,
                                   run_end="2026-09-01T13:00:00+00:00"))
        rendered = render_readme(meta)
        assert "DEPRECATED" in rendered
        assert "superseded by v2" in rendered

    def test_readme_documents_private_weights(self, card):
        meta = RunMeta(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       model_card=card)
        assert "private" in render_readme(meta)

    def test_readme_lists_inputs(self):
        meta = RunMeta(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       sources=["upstream-artifact-id"])
        assert "upstream-artifact-id" in render_readme(meta)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

class TestHashing:
    def test_hash_is_stable(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"hello world")
        assert hash_file(path) == hash_file(path)

    def test_hash_detects_change(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"hello")
        first, _ = hash_file(path)
        path.write_bytes(b"world")
        second, _ = hash_file(path)
        assert first != second

    def test_hash_reports_size(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"x" * 1234)
        _, size = hash_file(path)
        assert size == 1234

    def test_empty_file_hashes(self, tmp_path):
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        digest, size = hash_file(path)
        assert size == 0
        assert len(digest) == 64


# ---------------------------------------------------------------------------
# RunRecord / multi-execution history
# ---------------------------------------------------------------------------

class TestRunRecords:
    def test_fresh_run_has_one_record(self, index):
        artifact = _run_producing(index, "k")
        assert len(artifact.meta.runs) == 1

    def test_record_has_commit_and_timestamps(self, index):
        artifact = _run_producing(index, "k")
        rec = artifact.meta.runs[0]
        assert rec.run_start
        assert rec.run_end is not None
        assert not rec.partial

    def test_record_produced_lists_outputs(self, index):
        artifact = _run_producing(index, "k", n_files=3)
        assert len(artifact.meta.runs[0].produced) == 3

    def test_aggregate_produced_is_union(self, index):
        artifact = _run_producing(index, "k", n_files=2)
        assert set(artifact.meta.produced) == set(artifact.meta.runs[0].produced)

    def test_created_frozen_stable_id(self, index):
        """artifact_id derives from created, stable across executions."""
        artifact = _run_producing(index, "k")
        first_id = artifact.artifact_id
        # extend it -- id must not change
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       extend=first_id) as run:
            (run.out_dir / "extra.parquet").write_bytes(b"more")
        assert index.get(first_id).artifact_id == first_id


class TestExtend:
    def test_extend_appends_record(self, index):
        artifact = _run_producing(index, "k", n_files=1)
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       extend=artifact.artifact_id) as run:
            (run.out_dir / "second.parquet").write_bytes(b"data2")

        updated = index.get(artifact.artifact_id)
        assert len(updated.meta.runs) == 2

    def test_extend_accumulates_files(self, index):
        artifact = _run_producing(index, "k", n_files=1)
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       extend=artifact.artifact_id) as run:
            (run.out_dir / "second.parquet").write_bytes(b"data2")

        updated = index.get(artifact.artifact_id)
        assert len(updated.file_hashes) == 2

    def test_cannot_extend_partial(self, index):
        with pytest.raises(ValueError):
            with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION):
                raise ValueError("boom")
        partial = index.list("k", include_partial=True)[0]

        with pytest.raises(DatalakeError):
            with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                           extend=partial.artifact_id) as run:
                (run.out_dir / "x.parquet").write_bytes(b"x")

    def test_extend_preserves_first_commit(self, index):
        artifact = _run_producing(index, "k")
        first_commit = artifact.meta.runs[0].pipeline_commit
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       extend=artifact.artifact_id) as run:
            (run.out_dir / "e.parquet").write_bytes(b"e")

        updated = index.get(artifact.artifact_id)
        assert updated.meta.runs[0].pipeline_commit == first_commit


class TestVerifierField:
    def test_verifier_recorded(self, index):
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       verifier="my_verifier") as run:
            (run.out_dir / "f.parquet").write_bytes(b"x")
        artifact = index.latest("k")
        assert artifact.meta.verifier == "my_verifier"

    def test_verifier_survives_reindex(self, index):
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       verifier="my_verifier") as run:
            (run.out_dir / "f.parquet").write_bytes(b"x")
        index.reindex()
        assert index.latest("k").meta.verifier == "my_verifier"

    def test_content_verifier_flags_missing_entrypoint(self, index):
        """An artifact declaring an uninstalled verifier is flagged."""
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       verifier="nonexistent_verifier") as run:
            (run.out_dir / "f.parquet").write_bytes(b"x")

        report = verify(index)
        assert any(
            "not installed" in f.message for f in report.warnings
        )


class TestMultiRunReindex:
    def test_reindex_preserves_run_history(self, index):
        artifact = _run_producing(index, "k")
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       extend=artifact.artifact_id) as run:
            (run.out_dir / "e.parquet").write_bytes(b"e")

        index.reindex()
        assert len(index.get(artifact.artifact_id).meta.runs) == 2


class TestProducedAttribution:
    def test_extend_produced_is_only_new_files(self, index):
        artifact = _run_producing(index, "k", n_files=2)
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       extend=artifact.artifact_id) as run:
            (run.out_dir / "new1.parquet").write_bytes(b"a")
            (run.out_dir / "new2.parquet").write_bytes(b"b")

        updated = index.get(artifact.artifact_id)
        # First execution produced 2, second produced only its own 2.
        assert len(updated.meta.runs[0].produced) == 2
        assert len(updated.meta.runs[1].produced) == 2
        assert set(updated.meta.runs[1].produced) == {"new1.parquet", "new2.parquet"}

    def test_aggregate_produced_covers_all(self, index):
        artifact = _run_producing(index, "k", n_files=2)
        with index.run(kind="k", pipeline=PIPELINE, pipeline_version=VERSION,
                       extend=artifact.artifact_id) as run:
            (run.out_dir / "new1.parquet").write_bytes(b"a")

        updated = index.get(artifact.artifact_id)
        assert len(updated.meta.produced) == 3
