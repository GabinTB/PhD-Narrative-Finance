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
  - model_dir_sha256: deterministic and sensitive to content / file set
  - verify_artifact: clean set, missing month, unexpected parquet, bad schema
"""
from __future__ import annotations

import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from ravenpack.headlines.embed import (
    embed_month,
    embed_range,
    model_dir_sha256,
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
# model_dir_sha256
# ---------------------------------------------------------------------------


class TestModelDirSha256:
    def _make_model_dir(self, root: Path) -> Path:
        d = root / "model"
        (d / "sub").mkdir(parents=True)
        (d / "config.json").write_bytes(b'{"dim": 384}')
        (d / "sub" / "pytorch_model.bin").write_bytes(b"\x00\x01\x02weights")
        return d

    def test_deterministic(self, tmp_path: Path):
        d = self._make_model_dir(tmp_path)
        assert model_dir_sha256(d) == model_dir_sha256(d)

    def test_changes_when_file_bytes_change(self, tmp_path: Path):
        d = self._make_model_dir(tmp_path)
        before = model_dir_sha256(d)
        (d / "config.json").write_bytes(b'{"dim": 385}')
        assert model_dir_sha256(d) != before

    def test_changes_when_file_added(self, tmp_path: Path):
        d = self._make_model_dir(tmp_path)
        before = model_dir_sha256(d)
        (d / "vocab.txt").write_bytes(b"hello")
        assert model_dir_sha256(d) != before

    def test_raises_on_empty_dir(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            model_dir_sha256(tmp_path / "empty")


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
