"""Tests for ravenpack.headlines.mu_asof (the RavenPack corpus job).

No real GDrive/production data needed. ``compute_daily_stats`` (pass 1) is
exercised both with DuckDB monkeypatched out (merge logic) and against real
DuckDB on tiny parquet files (SUM / MIN / MAX per day). Pass 2 is
``nlp.reference_vector.ReferenceVector``, tested in tests/nlp, including
its equivalence with the original mu_asof algorithm.

Covers:
  - compute_daily_stats: datetime.date dict keys vs. the pd.Timestamp days
    index returned (regression for a KeyError on lookup), month merging,
    per-pooling SQL aggregates
  - output_name: historical name for mean, suffixed for min / max
  - verify_artifact: unit-norm check, gap detection, N > 0 check
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ravenpack.headlines.mu_asof import compute_daily_stats, output_name, verify_artifact
from ravenpack.headlines.schema import EMBEDDING_DIM, MU_ASOF_SCHEMA

# ---------------------------------------------------------------------------
# compute_daily_stats
# ---------------------------------------------------------------------------


class _FakeDuckDBConn:
    """Stand-in for duckdb.connect(): returns one canned polars result per
    call to .sql(...).pl(), regardless of the query text, and ignores
    PRAGMA execute() calls."""

    def __init__(self, result: pl.DataFrame):
        self._result = result

    def execute(self, *args, **kwargs):
        return None

    def sql(self, _query: str):
        return self

    def pl(self) -> pl.DataFrame:
        return self._result

    def close(self):
        return None


class TestComputeDailyStats:
    def test_date_key_type_does_not_raise(self, tmp_path: Path, monkeypatch):
        """Regression: DuckDB's CAST(... AS DATE) comes back as datetime.date
        via polars, so day_stat/day_cnt are keyed by datetime.date. Building
        the returned index as pd.DatetimeIndex(sorted(day_sum.keys())) yields
        pd.Timestamp entries -- indexing day_sum[d] with those Timestamps
        directly raised KeyError (Timestamp != date, even for the same day).
        """
        headlines_dir = tmp_path / "headlines"
        embeddings_dir = tmp_path / "embeddings"
        headlines_dir.mkdir()
        embeddings_dir.mkdir()
        (headlines_dir / "2000-01.parquet").touch()
        (embeddings_dir / "2000-01.parquet").touch()

        result = pl.DataFrame({
            "day": [date(2000, 1, 1), date(2000, 1, 2)],
            "day_stat": [[1.0] * EMBEDDING_DIM, [2.0] * EMBEDDING_DIM],
            "n": [3, 5],
        })
        monkeypatch.setattr(
            "ravenpack.headlines.mu_asof.duckdb.connect",
            lambda: _FakeDuckDBConn(result),
        )

        days, sums, counts = compute_daily_stats(headlines_dir, embeddings_dir, threads=1)

        assert isinstance(days, pd.DatetimeIndex)
        assert list(days.date) == [date(2000, 1, 1), date(2000, 1, 2)]
        assert counts.tolist() == [3, 5]
        assert sums.shape == (2, EMBEDDING_DIM)
        np.testing.assert_array_equal(sums[0], np.full(EMBEDDING_DIM, 1.0))
        np.testing.assert_array_equal(sums[1], np.full(EMBEDDING_DIM, 2.0))

    def test_merges_across_months(self, tmp_path: Path, monkeypatch):
        """Two monthly files, each contributing distinct days -- output
        should cover both months' days without collapsing or dropping any."""
        headlines_dir = tmp_path / "headlines"
        embeddings_dir = tmp_path / "embeddings"
        headlines_dir.mkdir()
        embeddings_dir.mkdir()
        for name in ("2000-01.parquet", "2000-02.parquet"):
            (headlines_dir / name).touch()
            (embeddings_dir / name).touch()

        results = {
            "2000-01.parquet": pl.DataFrame({
                "day": [date(2000, 1, 15)],
                "day_stat": [[1.0] * EMBEDDING_DIM],
                "n": [4],
            }),
            "2000-02.parquet": pl.DataFrame({
                "day": [date(2000, 2, 15)],
                "day_stat": [[2.0] * EMBEDDING_DIM],
                "n": [6],
            }),
        }

        # duckdb.connect() carries no info about which month is being
        # queried, so route by inspecting which files currently exist to
        # read -- simplest correct stand-in is a counter over the sorted
        # (deterministic) glob order compute_daily_stats iterates in.
        call_order = iter(sorted(results))
        monkeypatch.setattr(
            "ravenpack.headlines.mu_asof.duckdb.connect",
            lambda: _FakeDuckDBConn(results[next(call_order)]),
        )

        days, sums, counts = compute_daily_stats(headlines_dir, embeddings_dir, threads=1)

        assert list(days.date) == [date(2000, 1, 15), date(2000, 2, 15)]
        assert counts.tolist() == [4, 6]


    @pytest.mark.parametrize("pooling,op", [("mean", np.sum), ("min", np.min), ("max", np.max)])
    def test_real_duckdb_per_day_aggregates(self, tmp_path: Path, pooling, op):
        """The SQL aggregate matches numpy per day, for every pooling."""
        headlines_dir, embeddings_dir = tmp_path / "h", tmp_path / "e"
        headlines_dir.mkdir()
        embeddings_dir.mkdir()
        rng = np.random.default_rng(0)
        ids = [f"s{i}" for i in range(6)]
        stamps = [pd.Timestamp("2000-01-03 09:00") + pd.Timedelta(hours=10 * i) for i in range(6)]
        emb = rng.normal(size=(6, EMBEDDING_DIM)).astype(np.float16)
        pl.DataFrame({"RP_STORY_ID": ids, "TIMESTAMP_UTC": stamps}).write_parquet(
            headlines_dir / "2000-01.parquet")
        pl.DataFrame({"RP_STORY_ID": ids, "EMBEDDING": list(emb)},
                     schema={"RP_STORY_ID": pl.String,
                             "EMBEDDING": pl.Array(pl.Float16, EMBEDDING_DIM)}).write_parquet(
            embeddings_dir / "2000-01.parquet")

        days, stats, counts = compute_daily_stats(headlines_dir, embeddings_dir,
                                                  pooling=pooling, threads=1)

        day_of = np.array([s.date() for s in stamps])
        assert list(days.date) == sorted(set(day_of))
        for i, d in enumerate(days.date):
            rows = emb[day_of == d].astype(np.float64)
            assert counts[i] == len(rows)
            np.testing.assert_allclose(stats[i], op(rows, axis=0), rtol=1e-6, atol=1e-6)

    def test_rejects_unknown_pooling(self, tmp_path: Path):
        with pytest.raises(ValueError, match="pooling"):
            compute_daily_stats(tmp_path, tmp_path, pooling="median")


def test_output_name_keeps_historical_mean_name():
    assert output_name("1d", "expanding") == "mu_asof_delay-1d_mode-expanding.parquet"
    assert output_name("1d", "8W", "max") == "mu_asof_delay-1d_mode-8W_pooling-max.parquet"


# ---------------------------------------------------------------------------
# verify_artifact
# ---------------------------------------------------------------------------


def _write_mu_asof(
    path: Path,
    n_rows: int = 10,
    *,
    start: str = "2000-01-01",
    gap_after: int | None = None,
    bad_norm_at: int | None = None,
    bad_n_at: int | None = None,
) -> None:
    """Write a schema-valid mu_asof parquet, optionally with an injected flaw."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n_rows, freq="D")
    if gap_after is not None:
        # Push every date after `gap_after` forward by 10 days -- opens a gap.
        dates = pd.DatetimeIndex(
            list(dates[: gap_after + 1]) + list(dates[gap_after + 1 :] + pd.Timedelta(days=10))
        )

    mu = rng.standard_normal((n_rows, EMBEDDING_DIM)).astype(np.float32)
    mu_hat = (mu / np.linalg.norm(mu, axis=1, keepdims=True)).astype(np.float32)
    n = np.full(n_rows, 5, dtype=np.int64)

    if bad_norm_at is not None:
        mu_hat[bad_norm_at] = mu_hat[bad_norm_at] * 2.0  # norm 2, not 1
    if bad_n_at is not None:
        n[bad_n_at] = 0

    pl.DataFrame(
        {
            "DATE": list(dates.date),
            "MU": list(mu),
            "MU_HAT": list(mu_hat),
            "N": n.tolist(),
        },
        schema=MU_ASOF_SCHEMA,
    ).write_parquet(path / "mu_asof_delay-1M_mode-expanding.parquet")


def _fake_artifact(path: Path, *, sources: bool = True):
    hp = {"delay": "1M", "mode": "expanding", "dim": EMBEDDING_DIM}
    if sources:
        hp.update({"source_headlines": "hl__1", "source_embeddings": "emb__1"})
    return SimpleNamespace(
        artifact_id="mu_asof__v0.1.0__20260101",
        path=path,
        meta=SimpleNamespace(hyperparams=hp),
    )


class TestVerifyArtifact:
    def test_clean_artifact_has_no_findings(self, tmp_path: Path):
        _write_mu_asof(tmp_path)
        assert verify_artifact(_fake_artifact(tmp_path)) == []

    def test_missing_source_hyperparams_errors(self, tmp_path: Path):
        _write_mu_asof(tmp_path)
        findings = verify_artifact(_fake_artifact(tmp_path, sources=False))
        assert any(
            f.severity.value == "error" and "source_headlines" in f.message
            for f in findings
        )

    def test_n_nonpositive_errors(self, tmp_path: Path):
        _write_mu_asof(tmp_path, bad_n_at=3)
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert any(
            f.severity.value == "error" and "N <= 0" in f.message for f in findings
        )

    def test_non_unit_mu_hat_errors(self, tmp_path: Path):
        _write_mu_asof(tmp_path, bad_norm_at=2)
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert any(
            f.severity.value == "error" and "unit-norm" in f.message for f in findings
        )

    def test_large_gap_warns(self, tmp_path: Path):
        _write_mu_asof(tmp_path, n_rows=10, gap_after=4)
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert any(
            f.severity.value == "warning" and "gap" in f.message for f in findings
        )

    def test_small_gap_within_tolerance_ok(self, tmp_path: Path):
        # Consecutive daily dates -> max gap is 1 day, well under the 5-day
        # tolerance for weekends/holidays.
        _write_mu_asof(tmp_path, n_rows=10)
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert not any("gap" in f.message for f in findings)

    def test_no_parquet_errors(self, tmp_path: Path):
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert len(findings) >= 1
        assert any(f.severity.value == "error" and "no parquet" in f.message for f in findings)

    def test_empty_file_errors(self, tmp_path: Path):
        pl.DataFrame(schema=MU_ASOF_SCHEMA).write_parquet(
            tmp_path / "mu_asof_delay-1M_mode-expanding.parquet"
        )
        findings = verify_artifact(_fake_artifact(tmp_path))
        assert any(f.severity.value == "error" and "empty" in f.message for f in findings)


def test_killed_job_resumes_from_checkpoints(tmp_path, monkeypatch):
    """A job killed in its 2nd month resumes: month 1 comes from its checkpoint (not
    re-queried), the series equals an uninterrupted run, checkpoints are removed."""
    import ravenpack.headlines.mu_asof as mod
    from datalake import DatalakeIndex

    def lake(root):
        index = DatalakeIndex(root)
        rng = np.random.default_rng(0)
        with index.run(kind="ravenpack_headlines", pipeline="t", pipeline_version="v0") as h, \
             index.run(kind="headline_embeddings", pipeline="t", pipeline_version="v0") as e:
            for m in (1, 2, 3):
                ids = [f"{m}-{i}" for i in range(20)]
                stamps = [pd.Timestamp(f"2000-0{m}-01") + pd.Timedelta(hours=37 * i)
                          for i in range(20)]
                pl.DataFrame({"RP_STORY_ID": ids, "TIMESTAMP_UTC": stamps}).write_parquet(
                    h.out_dir / f"2000-0{m}.parquet")
                emb = rng.normal(size=(20, EMBEDDING_DIM)).astype(np.float16)
                pl.DataFrame({"RP_STORY_ID": ids, "EMBEDDING": list(emb)},
                             schema={"RP_STORY_ID": pl.String,
                                     "EMBEDDING": pl.Array(pl.Float16, EMBEDDING_DIM)}
                             ).write_parquet(e.out_dir / f"2000-0{m}.parquet")
        return index

    ref_index = lake(tmp_path / "ref")
    ref = mod.mu_asof_to_datalake(ref_index, "1d", "expanding", threads=1)
    want = pl.read_parquet(next(ref.path.glob("*.parquet")))

    index = lake(tmp_path / "lake")
    real, queried = mod._month_stats, []

    def killed_in_february(hl_file, emb_file, pooling, threads):
        queried.append(emb_file.name)
        if emb_file.name == "2000-02.parquet" and len(queried) == 2:
            raise RuntimeError("killed")
        return real(hl_file, emb_file, pooling, threads)

    monkeypatch.setattr(mod, "_month_stats", killed_in_february)
    with pytest.raises(RuntimeError, match="killed"):
        mod.mu_asof_to_datalake(index, "1d", "expanding", threads=1)
    part = index.list("mu_asof", include_partial=True)[0]
    assert part.partial
    assert [p.name for p in (part.path / mod.CHECKPOINT_DIR).glob("*.npz")] == ["2000-01.npz"]

    queried.clear()
    done = mod.resume_mu_asof(index, part.artifact_id, threads=1)
    assert queried == ["2000-02.parquet", "2000-03.parquet"]          # January not re-queried
    assert not done.partial and not (done.path / mod.CHECKPOINT_DIR).exists()
    got = pl.read_parquet(next(done.path.glob("*.parquet")))
    assert got["DATE"].to_list() == want["DATE"].to_list() and got["N"].equals(want["N"])
    np.testing.assert_array_equal(np.stack(got["MU"].to_list()), np.stack(want["MU"].to_list()))
