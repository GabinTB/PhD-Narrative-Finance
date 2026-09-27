"""Tests for ravenpack.headlines.embed.

Synthetic data only.  No RavenBERT model is loaded -- a FakeModel exposing the
same ``encode(list[str]) -> np.ndarray`` contract (deterministic float32 vectors)
is passed directly into ``embed_month`` / ``embed_range``.

Covers:
  - EMBEDDING_SCHEMA structure (two columns, float16 array)
  - embed_month: schema, RP_STORY_ID alignment, float32 -> float16 cast, atomic
    write, null-headline handling (row kept, not dropped)
  - embed_range: one parquet per source month, resumability (skip existing),
    overwrite
  - embed_to_datalake: end to end on a temp lake with an nlp Embedder over a
    fake backend -- the card carries backend / serving / checks, backend and
    dtype are in the artifact id
  - resume guard: refuses a different backend or dtype; per-month model check
  - verify_artifact: clean set, missing month, unexpected parquet, bad schema
"""
from __future__ import annotations

import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from nlp.backends.base import Backend, EmbeddingConfig, IncompatibleModelError, unit_rows
from nlp.embedding import Embedder
from ravenpack.headlines.embed import (
    embed_month,
    embed_range,
    embed_to_datalake,
    resume_embedding,
    verify_artifact,
)
from ravenpack.headlines.schema import EMBEDDING_DIM, EMBEDDING_SCHEMA

# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class FakeModel:
    """Deterministic stand-in for RavenBERT's EmbeddingModel.

    Each text maps to a fixed pseudo-random float32 vector (seeded by a CRC of
    the text, so it is stable across processes).  Vectors are NOT L2-normalized
    -- the pipeline does not normalize, it only casts to float16.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(
        self,
        texts: list[str],
        batch_size: int = 128,
        show_progress_bar: bool = False,
        **_: object,
    ) -> np.ndarray:
        self.calls.append(list(texts))
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = zlib.crc32(t.encode("utf-8"))
            out[i] = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return out

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def vector_for(self, text: str) -> np.ndarray:
        seed = zlib.crc32(text.encode("utf-8"))
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)


def _write_source_month(path: Path, story_ids: list[str], headlines: list) -> None:
    """Write a synthetic structured (ingest) parquet -- more columns than needed."""
    pl.DataFrame(
        {
            "TIMESTAMP_UTC": ["2000-01-01 00:00:00"] * len(story_ids),
            "RP_STORY_ID": story_ids,
            "HEADLINE": headlines,
            "SOURCE_NAME": ["Reuters"] * len(story_ids),
        },
        schema={
            "TIMESTAMP_UTC": pl.String,
            "RP_STORY_ID": pl.String,
            "HEADLINE": pl.String,
            "SOURCE_NAME": pl.String,
        },
    ).write_parquet(path)


def _write_embedding_month(path: Path, story_ids: list[str]) -> None:
    """Write a schema-valid two-column embedding parquet."""
    emb = np.random.default_rng(1).standard_normal((len(story_ids), EMBEDDING_DIM))
    pl.DataFrame(
        {"RP_STORY_ID": story_ids, "EMBEDDING": list(emb.astype(np.float16))},
        schema=EMBEDDING_SCHEMA,
    ).write_parquet(path)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestEmbeddingSchema:
    def test_two_columns_only(self):
        assert list(EMBEDDING_SCHEMA.keys()) == ["RP_STORY_ID", "EMBEDDING"]

    def test_embedding_is_float16_array(self):
        emb = EMBEDDING_SCHEMA["EMBEDDING"]
        assert isinstance(emb, pl.Array)
        assert emb.inner == pl.Float16
        assert emb.size == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# embed_month
# ---------------------------------------------------------------------------


class TestEmbedMonth:
    def test_schema_and_story_id_alignment(self, tmp_path: Path):
        story_ids = [f"S{i:03d}" for i in range(7)]
        headlines = [f"headline number {i}" for i in range(7)]
        src = tmp_path / "2000-01.parquet"
        out = tmp_path / "out" / "2000-01.parquet"
        out.parent.mkdir()
        _write_source_month(src, story_ids, headlines)

        model = FakeModel()
        n = embed_month(model, src, out)

        assert n == 7
        df = pl.read_parquet(out)
        assert df.schema == EMBEDDING_SCHEMA
        assert df["RP_STORY_ID"].to_list() == story_ids  # order preserved

    def test_float32_cast_to_float16(self, tmp_path: Path):
        story_ids = ["A", "B", "C"]
        headlines = ["alpha", "bravo", "charlie"]
        src = tmp_path / "2000-01.parquet"
        out = tmp_path / "2000-01-emb.parquet"
        _write_source_month(src, story_ids, headlines)

        model = FakeModel()
        embed_month(model, src, out)

        df = pl.read_parquet(out)
        stored = np.asarray(df["EMBEDDING"].to_list(), dtype=np.float16)
        expected = np.stack([model.vector_for(h) for h in headlines]).astype(np.float16)
        assert stored.dtype == np.float16
        np.testing.assert_array_equal(stored, expected)

    def test_atomic_write_leaves_no_tmp(self, tmp_path: Path):
        src = tmp_path / "2000-01.parquet"
        out = tmp_path / "2000-01-emb.parquet"
        _write_source_month(src, ["A", "B"], ["x", "y"])
        embed_month(FakeModel(), src, out)
        assert out.exists()
        assert not (tmp_path / "2000-01-emb.parquet.tmp").exists()

    def test_null_headline_kept_not_dropped(self, tmp_path: Path):
        story_ids = ["A", "B", "C"]
        headlines = ["alpha", None, "charlie"]
        src = tmp_path / "2000-01.parquet"
        out = tmp_path / "2000-01-emb.parquet"
        _write_source_month(src, story_ids, headlines)

        n = embed_month(FakeModel(), src, out)

        assert n == 3
        df = pl.read_parquet(out)
        assert df["RP_STORY_ID"].to_list() == story_ids

    def test_chunked_write_covers_all_rows(self, tmp_path: Path):
        story_ids = [f"S{i}" for i in range(10)]
        headlines = [f"h{i}" for i in range(10)]
        src = tmp_path / "2000-01.parquet"
        out = tmp_path / "2000-01-emb.parquet"
        _write_source_month(src, story_ids, headlines)

        model = FakeModel()
        n = embed_month(model, src, out, write_chunk_rows=3)

        assert n == 10
        assert model.n_calls == 4  # ceil(10 / 3)
        assert pl.read_parquet(out)["RP_STORY_ID"].to_list() == story_ids


# ---------------------------------------------------------------------------
# embed_range
# ---------------------------------------------------------------------------


class TestEmbedRange:
    def _make_source(self, root: Path, months: list[str]) -> Path:
        src_dir = root / "src"
        src_dir.mkdir()
        for name in months:
            _write_source_month(
                src_dir / f"{name}.parquet",
                [f"{name}-S{i}" for i in range(4)],
                [f"{name} headline {i}" for i in range(4)],
            )
        return src_dir

    def test_one_output_per_present_source_month(self, tmp_path: Path):
        src_dir = self._make_source(tmp_path, ["2000-01", "2000-02", "2000-03"])
        out_dir = tmp_path / "out"

        embed_range(FakeModel(), src_dir, out_dir, 2000, 2000)

        produced = sorted(p.name for p in out_dir.glob("*.parquet"))
        assert produced == ["2000-01.parquet", "2000-02.parquet", "2000-03.parquet"]
        for name in produced:
            assert pl.read_parquet(out_dir / name).schema == EMBEDDING_SCHEMA

    def test_resumable_skips_existing_months(self, tmp_path: Path):
        src_dir = self._make_source(tmp_path, ["2000-01", "2000-02", "2000-03"])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Pre-existing month with a sentinel id -- must be left untouched.
        _write_embedding_month(out_dir / "2000-02.parquet", ["SENTINEL"])

        model = FakeModel()
        embed_range(model, src_dir, out_dir, 2000, 2000)

        assert pl.read_parquet(out_dir / "2000-02.parquet")["RP_STORY_ID"].to_list() == [
            "SENTINEL"
        ]
        embedded = {t for call in model.calls for t in call}
        assert embedded == {
            f"2000-0{m} headline {i}" for m in (1, 3) for i in range(4)
        }

    def test_overwrite_reembeds_existing(self, tmp_path: Path):
        src_dir = self._make_source(tmp_path, ["2000-01"])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _write_embedding_month(out_dir / "2000-01.parquet", ["SENTINEL"])

        embed_range(FakeModel(), src_dir, out_dir, 2000, 2000, overwrite=True)

        assert pl.read_parquet(out_dir / "2000-01.parquet")["RP_STORY_ID"].to_list() == [
            "2000-01-S0", "2000-01-S1", "2000-01-S2", "2000-01-S3",
        ]


# ---------------------------------------------------------------------------
# embed_to_datalake / resume guard / per-month model check (nlp Embedder)
# ---------------------------------------------------------------------------


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, dtype="float16"):
        super().__init__(dtype)
        self.checks = 0
        self.switched = False

    def embedding_config(self):
        return EmbeddingConfig(EMBEDDING_DIM, "cls")

    def embed(self, texts):
        return unit_rows(np.stack([FakeModel().vector_for(t) for t in texts]))

    def info(self):
        return {"engine": "fake", "served": "tiny"}

    def check_unchanged(self):
        self.checks += 1
        if self.switched:
            raise IncompatibleModelError("model changed mid-run")


def _register_source(index, root: Path):
    with index.run(kind="ravenpack_headlines", pipeline="test", pipeline_version="v0",
                   hyperparams={"start_year": 2000, "end_year": 2000}) as run:
        for name in ("2000-01", "2000-02"):
            _write_source_month(run.out_dir / f"{name}.parquet",
                                [f"{name}-S{i}" for i in range(3)],
                                [f"{name} headline {i}" for i in range(3)])
    return index.get(run.artifact_id)


class TestEmbedToDatalake:
    def test_end_to_end_card_and_id(self, tmp_path: Path):
        from datalake import DatalakeIndex

        with DatalakeIndex(tmp_path / "lake") as index:
            src = _register_source(index, tmp_path)
            backend = FakeBackend("float16")
            embedder = Embedder(backend, model_id="ravenbert")
            art = embed_to_datalake(index, "v0.2.0", embedder=embedder)
        assert sorted(p.name for p in art.path.glob("*.parquet")) == [
            "2000-01.parquet", "2000-02.parquet"]
        card = art.meta.model_card
        assert card.model_id == "ravenbert" and card.backend == "fake"
        assert card.serving["served"] == "tiny" and card.serving["checks"]["dtype"] == "float16"
        assert art.meta.hyperparams["backend"] == "fake"
        assert art.meta.hyperparams["dtype"] == "float16"
        assert "dtypefloat16" in art.artifact_id and src.artifact_id in art.meta.sources
        assert backend.checks == 2                       # one model check per month
        X = np.stack(pl.read_parquet(art.path / "2000-01.parquet")["EMBEDDING"].to_list())
        np.testing.assert_allclose(np.linalg.norm(X.astype(np.float32), axis=1), 1.0, atol=2e-3)

    def test_model_switch_mid_run_stops_the_job(self, tmp_path: Path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        for name in ("2000-01", "2000-02"):
            _write_source_month(src_dir / f"{name}.parquet", [f"{name}-S0"], [f"{name} h"])
        backend = FakeBackend()
        embedder = Embedder(backend)
        backend.switched = True
        with pytest.raises(IncompatibleModelError):
            embed_range(embedder, src_dir, tmp_path / "out", 2000, 2000)


class CrashingBackend(FakeBackend):
    """Dies on its n-th embed call (after the construction probe)."""

    def __init__(self, dtype="float16", die_at=None):
        super().__init__(dtype)
        self.calls, self.die_at = 0, die_at

    def embed(self, texts):
        self.calls += 1
        if self.die_at is not None and self.calls == self.die_at:
            raise RuntimeError("killed mid-run")
        return super().embed(texts)


class TestLayouts:
    def test_quarterly_source_gives_quarterly_embeddings(self, tmp_path: Path):
        from datalake import DatalakeIndex
        from datalake.layout import layout_of

        index = DatalakeIndex(tmp_path / "lake")
        with index.run(kind="ravenpack_headlines", pipeline="test", pipeline_version="v0",
                       hyperparams={"partition_freq": "Q", "start": "2000-01-01",
                                    "end": "2000-06-30"}) as run:
            for key in ("2000Q1", "2000Q2"):
                _write_source_month(run.out_dir / f"{key}.parquet",
                                    [f"{key}-S{i}" for i in range(3)],
                                    [f"{key} headline {i}" for i in range(3)])
        art = embed_to_datalake(index, "v0.3.0", embedder=Embedder(FakeBackend()))
        lay = layout_of(art)
        assert (lay.freq, str(lay.start), str(lay.end)) == ("Q", "2000-01-01", "2000-06-30")
        assert list(lay.existing(art.path)) == ["2000Q1", "2000Q2"]
        assert verify_artifact(art) == []


class TestResume:
    def _crashed(self, tmp_path):
        from datalake import DatalakeIndex

        index = DatalakeIndex(tmp_path / "lake")
        _register_source(index, tmp_path)
        with pytest.raises(RuntimeError, match="killed"):
            embed_to_datalake(index, "v0.2.0", embedder=Embedder(CrashingBackend(die_at=3)))
        return index, index.list("headline_embeddings", include_partial=True)[0]

    def test_resume_finishes_in_place(self, tmp_path):
        index, part = self._crashed(tmp_path)
        assert part.partial and [p.name for p in part.path.glob("*.parquet")] == ["2000-01.parquet"]
        done = resume_embedding(index, part.artifact_id, embedder=Embedder(CrashingBackend()))
        assert not done.partial and done.artifact_id == part.artifact_id
        assert sorted(done.file_hashes) == ["2000-01.parquet", "2000-02.parquet"]
        assert len(done.meta.runs) == 2 and done.meta.runs[0].partial

    def test_resume_refuses_another_model(self, tmp_path):
        index, part = self._crashed(tmp_path)
        with pytest.raises(IncompatibleModelError, match="dtype"):
            resume_embedding(index, part.artifact_id,
                             embedder=Embedder(CrashingBackend(dtype="float32")))
        assert index.get(part.artifact_id).partial                  # untouched

    def test_complete_artifact_is_not_resumed(self, tmp_path):
        from datalake import DatalakeIndex

        index = DatalakeIndex(tmp_path / "lake")
        _register_source(index, tmp_path)
        art = embed_to_datalake(index, "v0.2.0", embedder=Embedder(CrashingBackend()))
        with pytest.raises(ValueError, match="complete"):
            resume_embedding(index, art.artifact_id, embedder=Embedder(CrashingBackend()))


# ---------------------------------------------------------------------------
# verify_artifact
# ---------------------------------------------------------------------------


def _fake_artifact(path: Path, start_year: int, end_year: int):
    return SimpleNamespace(
        artifact_id="headline_embeddings__ravenbert-1.0__vtest__20260101",
        path=path,
        meta=SimpleNamespace(
            hyperparams={"start_year": start_year, "end_year": end_year}
        ),
    )


class TestVerifyArtifact:
    def _full_year(self, path: Path, year: int = 2000) -> None:
        for m in range(1, 13):
            _write_embedding_month(
                path / f"{year}-{m:02d}.parquet", [f"S{m}-{i}" for i in range(3)]
            )

    def test_clean_artifact_has_no_findings(self, tmp_path: Path):
        self._full_year(tmp_path)
        assert verify_artifact(_fake_artifact(tmp_path, 2000, 2000)) == []

    def test_missing_month_warns(self, tmp_path: Path):
        self._full_year(tmp_path)
        (tmp_path / "2000-06.parquet").unlink()
        findings = verify_artifact(_fake_artifact(tmp_path, 2000, 2000))
        assert len(findings) == 1
        assert findings[0].severity.value == "warning"
        assert "2000-06.parquet" in findings[0].message

    def test_unexpected_parquet_errors(self, tmp_path: Path):
        self._full_year(tmp_path)
        _write_embedding_month(tmp_path / "1999-12.parquet", ["X"])
        findings = verify_artifact(_fake_artifact(tmp_path, 2000, 2000))
        assert any(
            f.severity.value == "error" and "1999-12.parquet" in f.message
            for f in findings
        )

    def test_bad_schema_errors(self, tmp_path: Path):
        self._full_year(tmp_path)
        # 2000-01 is the first sampled file; give it an extra column.
        pl.DataFrame(
            {
                "RP_STORY_ID": ["A"],
                "EMBEDDING": list(np.zeros((1, EMBEDDING_DIM), dtype=np.float16)),
                "EXTRA": [1],
            },
            schema={
                "RP_STORY_ID": pl.String,
                "EMBEDDING": pl.Array(pl.Float16, EMBEDDING_DIM),
                "EXTRA": pl.Int64,
            },
        ).write_parquet(tmp_path / "2000-01.parquet")

        findings = verify_artifact(_fake_artifact(tmp_path, 2000, 2000))
        assert any(
            f.severity.value == "error" and "schema mismatch" in f.message
            for f in findings
        )

    def test_no_year_range_warns_and_returns_early(self, tmp_path: Path):
        art = SimpleNamespace(
            artifact_id="x", path=tmp_path, meta=SimpleNamespace(hyperparams={})
        )
        findings = verify_artifact(art)
        assert len(findings) == 1
        assert findings[0].severity.value == "warning"
