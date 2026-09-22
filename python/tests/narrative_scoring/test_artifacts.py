"""The one scoring path on a sandbox datalake: cold start, monthly tau job, TEMP marking."""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from datalake import DatalakeError, DatalakeIndex
from narrative_scoring import artifacts as A
from narrative_scoring.config import ScoringConfig
from narrative_scoring.partitions import month_range_days
from narrative_scoring.schema import (
    DAY_DIAGNOSTICS_SCHEMA,
    EMBEDDING_DIM,
    F0_PARTITION_SCHEMA,
    NARRATIVE_DAILY_SCHEMA,
    TAU_ASOF_SCHEMA,
)
from narrative_scoring.streaming import InMemoryHeadlineSource

from .test_pipeline import _headlines, _mu_df

EARLIEST = date(2008, 1, 1)


def _write_headline_month(dir_: Path, name: str, first_day: date) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"RP_STORY_ID": ["a"], "TIMESTAMP_UTC": [f"{first_day} 09:00:00.000"]}) \
        .write_parquet(dir_ / name)


@pytest.fixture
def lake(tmp_path: Path):
    """Sandbox root holding the upstream families the scorer reads (never production)."""
    dl = DatalakeIndex(tmp_path / "lake")
    with dl.run(kind=A.KIND_HEADLINES, pipeline="t", pipeline_version="v0") as r:
        _write_headline_month(r.out_dir, "2008-01.parquet", EARLIEST)   # earliest data
    with dl.run(kind=A.KIND_EMBEDDINGS, pipeline="t", pipeline_version="v0") as r:
        (r.out_dir / "2008-01.parquet").write_bytes(b"0")
    with dl.run(kind=A.KIND_MU_ASOF, pipeline="t", pipeline_version="v0") as r:
        # first mu row one month after earliest data (mu_asof's own delay)
        _mu_df([date(2008, 2, 1), date(2008, 7, 1)]).write_parquet(r.out_dir / "mu_asof.parquet")
    yield dl
    dl.close()


@pytest.fixture
def cfg():
    return ScoringConfig(q=0.75, null_draws_per_headline=4, min_month_draws=1)


def _source(table, P, months, n=12):
    rng = np.random.default_rng(1)
    X = {}
    for (y, m) in months:
        for d in month_range_days(y, m)[::2]:
            X[d] = _headlines(rng, table, P, n)
    return InMemoryHeadlineSource(X, chunk_size=5, source_id="toy")


class TestScoreRange:
    def test_cold_start_replay(self, lake, toy_table, toy_embeddings, cfg, caplog):
        months = [(2008, m) for m in range(1, 7)]
        source = _source(toy_table, toy_embeddings, months)
        with caplog.at_level(logging.INFO):
            s = A.score_range_to_datalake(
                lake, date(2008, 1, 1), date(2008, 6, 30), cfg, table=toy_table, P=toy_embeddings,
                source=source, window="5Y", use_kernel=False, temp=True)
        # start shift, loudly
        assert s["start"] == date(2008, 3, 1) and s["earliest_data"] == EARLIEST
        warn = [r for r in caplog.records if "start shifted" in r.message]
        assert warn and warn[0].levelno == logging.WARNING
        # Jan: no mu row -> days skipped entirely, no partition. Feb: mu but no tau ->
        # null-only partition. Mar 1: cutoff <= Feb 1 -> no partition old enough -> null-only.
        # Apr 1: cutoff <= Mar 1 -> the Feb partition -> first tau row -> April scored.
        assert s["months_finalised"] == [f"2008-0{m}.parquet" for m in (2, 3, 4, 5, 6)]
        assert s["n_days_null_only"] == 15 + 16          # Feb (15 days w/ data) + Mar (16)
        assert s["n_days_scored"] == 15 + 16 + 15        # Apr, May, Jun
        null_only = [r for r in caplog.records if "null-only day (partitions fed" in r.message]
        assert len(null_only) == 31
        # January ends before the first mu row: not walked at all
        assert not any("month 2008-01" in r.message for r in caplog.records)
        assert any("month 2008-02: null-only" in r.message for r in caplog.records)

        nd = lake.get(s["narrative_daily_id"])
        dg = lake.get(s["day_diagnostics_id"])
        pt = lake.get(s["partitions_id"])
        ta = lake.get(s["tau_asof_id"])
        # TEMP marking on everything the agent registered
        for art in (nd, dg, pt, ta):
            assert "__TEMP" in art.artifact_id and art.meta.hyperparams["agent_created"] is True
            assert A.is_agent_created(art)
        assert sorted(nd.file_hashes) == ["2008-04.parquet", "2008-05.parquet",
                                          "2008-06.parquet", "run_metadata.json"]
        assert pl.read_parquet(nd.path / "2008-04.parquet").schema == NARRATIVE_DAILY_SCHEMA
        assert pl.read_parquet(dg.path / "2008-04.parquet").schema == DAY_DIAGNOSTICS_SCHEMA
        assert sorted(pt.file_hashes) == [f"2008-0{m}.parquet" for m in (2, 3, 4, 5, 6)]
        parts = pl.read_parquet(pt.path / "2008-02.parquet")
        assert parts.schema == F0_PARTITION_SCHEMA
        assert parts["N_DAYS_CLOSED"][0] == 29 and parts["COVERAGE"][0] == 1.0
        # short-window tau rows, valid and identifiable
        tau = pl.read_parquet(ta.path / A.TAU_ASOF_FILE)
        assert tau.schema == TAU_ASOF_SCHEMA
        # the last job ran as of Jun 1 -> cutoffs <= May 1 -> rows through April
        assert tau["MONTH_END"].to_list() == [date(2008, 2, 29), date(2008, 3, 31),
                                              date(2008, 4, 30)]
        assert tau["N_PARTITIONS"].to_list() == [1, 2, 3]
        assert tau["WINDOW_MONTHS_USED"].to_list() == [1, 2, 3]
        assert (tau["N_EFF"] > 0).all() and tau["MU_DATE"].to_list()[0] == date(2008, 2, 1)
        assert tau["MU_ARTIFACT_ID"][0] == lake.latest(A.KIND_MU_ASOF).artifact_id
        # the days used the row old enough at the time: April -> Feb row, May -> Mar row
        diag = pl.concat([pl.read_parquet(p) for p in dg.files()]).sort("DATE")
        by_month = diag.group_by(pl.col("DATE").dt.month()).agg(pl.col("TAU_MONTH_END").first())
        got = dict(zip(by_month["DATE"].to_list(), by_month["TAU_MONTH_END"].to_list()))
        assert got == {4: date(2008, 2, 29), 5: date(2008, 3, 31), 6: date(2008, 4, 30)}
        assert (diag["TAU_SOURCE_ID"] == ta.artifact_id).all()
        assert set(ta.meta.sources) >= {pt.artifact_id, lake.latest(A.KIND_MU_ASOF).artifact_id}
        assert set(nd.meta.sources) >= {lake.latest(A.KIND_HEADLINES).artifact_id}
        for f in (A.verify_partitions(pt), A.verify_tau_asof(ta), A.verify_narrative_daily(nd),
                  A.verify_day_diagnostics(dg)):
            assert f == []

        # a second run continues: partitions extended, tau extended, no rebuild of history
        s2 = A.score_range_to_datalake(
            lake, date(2008, 7, 1), date(2008, 7, 31), cfg, table=toy_table, P=toy_embeddings,
            source=_source(toy_table, toy_embeddings, [(2008, 7)]), use_kernel=False, temp=True)
        pt2 = lake.get(s2["partitions_id"])
        assert pt2.artifact_id == pt.artifact_id and len(pt2.meta.runs) == 2
        assert sorted(pt2.file_hashes) == [f"2008-0{m}.parquet" for m in (2, 3, 4, 5, 6, 7)]
        assert s2["n_days_null_only"] == 0 and s2["n_days_scored"] == 16
        tau2 = pl.read_parquet(lake.get(s2["tau_asof_id"]).path / A.TAU_ASOF_FILE)
        assert tau2["MONTH_END"].to_list()[-1] == date(2008, 5, 31)   # Jul 1 - 1M = Jun 1
        assert tau2.head(3).equals(tau)                          # history untouched
        assert tau2["MU_DATE"].to_list()[-1] == date(2008, 2, 1)   # Jul 1 row not yet usable

    def test_tau_job_deterministic_and_rebuild_is_new_temp_artifact(self, lake, toy_table,
                                                                   toy_embeddings, cfg):
        months = [(2008, m) for m in range(1, 5)]
        A.score_range_to_datalake(lake, date(2008, 1, 1), date(2008, 4, 30), cfg,
                                  table=toy_table, P=toy_embeddings,
                                  source=_source(toy_table, toy_embeddings, months),
                                  use_kernel=False, temp=True)
        base = A.find_tau_asof(lake, cfg, toy_table, "5Y", 0, temp=True)
        a = pl.read_parquet(base.path / A.TAU_ASOF_FILE)
        assert a["MONTH_END"].to_list() == [date(2008, 2, 29)]   # built as of April 1
        time.sleep(1.1)                       # artifact timestamps have second precision
        rebuilt = A.build_tau_asof(lake, cfg, toy_table, toy_embeddings, today=date(2008, 4, 1),
                                   rebuild=True, temp=True)
        assert rebuilt.artifact_id != base.artifact_id and "__TEMP" in rebuilt.artifact_id
        b = pl.read_parquet(rebuilt.path / A.TAU_ASOF_FILE)
        assert a.equals(b)                                      # byte-identical rows
        assert A.find_tau_asof(lake, cfg, toy_table, "5Y", 0, temp=True).artifact_id == \
            rebuilt.artifact_id
        assert A.build_tau_asof(lake, cfg, toy_table, toy_embeddings, today=date(2008, 4, 1),
                                temp=True).artifact_id == rebuilt.artifact_id   # up to date
        # the series keeps extending from the rebuilt artifact, with more cutoffs
        later = A.build_tau_asof(lake, cfg, toy_table, toy_embeddings, today=date(2008, 6, 1),
                                 temp=True)
        assert later.artifact_id == rebuilt.artifact_id
        assert pl.read_parquet(later.path / A.TAU_ASOF_FILE)["MONTH_END"].to_list() == [
            date(2008, 2, 29), date(2008, 3, 31), date(2008, 4, 30)]

    def test_temp_marking_and_no_delete(self, lake, toy_table, toy_embeddings, cfg):
        src = _source(toy_table, toy_embeddings, [(2008, m) for m in (1, 2, 3)])
        A.score_range_to_datalake(lake, date(2008, 3, 1), date(2008, 3, 31), cfg,
                                  table=toy_table, P=toy_embeddings, source=src,
                                  use_kernel=False, temp=False)          # an "owner" run
        owner = A.find_partitions(lake, cfg, toy_table, temp=False)
        assert owner is not None and "__TEMP" not in owner.artifact_id
        assert owner.meta.hyperparams["agent_created"] is False
        assert not A.is_agent_created(owner)
        assert A.find_partitions(lake, cfg, toy_table, temp=True) is None   # separate lineage
        marked = A.mark_temp_deprecated(lake, owner.artifact_id, "superseded by owner run")
        assert marked.deprecated and marked.meta.deprecation_reason.startswith(
            "superseded by owner run")
        assert A.TEMP_NOTE in marked.meta.notes and A.is_agent_created(marked)
        assert marked.artifact_id == owner.artifact_id             # ids are frozen
        assert owner.path.exists() and marked.file_hashes == owner.file_hashes  # nothing deleted
        assert A.find_partitions(lake, cfg, toy_table, temp=False) is None   # deprecated hidden
        assert "delete" not in A.__all__ and not hasattr(A, "delete")

    def test_end_before_shifted_start_scores_nothing(self, lake, toy_table, toy_embeddings, cfg,
                                                     caplog):
        src = _source(toy_table, toy_embeddings, [(2008, 1), (2008, 2)])
        with caplog.at_level(logging.WARNING):
            s = A.score_range_to_datalake(
                lake, date(2008, 1, 1), date(2008, 2, 10), cfg, table=toy_table,
                P=toy_embeddings, source=src, use_kernel=False, temp=True)
        assert s["n_days_scored"] == 0
        assert any("nothing to score" in r.message for r in caplog.records)
        with pytest.raises(ValueError, match="end before start"):
            A.score_range_to_datalake(lake, date(2008, 5, 1), date(2008, 4, 1), cfg,
                                      table=toy_table, P=toy_embeddings, use_kernel=False)


def test_latest_matching_and_missing(lake, toy_table, cfg):
    with pytest.raises(DatalakeError):
        A.latest_matching(lake, A.KIND_TAU_ASOF, f0_config_id="nope")
    assert A.find_tau_asof(lake, cfg, toy_table) is None
    P = np.zeros((toy_table.n_primitives * toy_table.n_texts, EMBEDDING_DIM), np.float32)
    assert A.build_tau_asof(lake, cfg, toy_table, P) is None
    assert A.earliest_headline_day(lake.latest(A.KIND_HEADLINES)) == EARLIEST
