"""The one scoring path on a sandbox datalake: cold start, monthly tau job, TEMP marking."""
from __future__ import annotations

import json
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


def _make_lake(root: Path) -> DatalakeIndex:
    """Sandbox root holding the upstream families the scorer reads (never production)."""
    dl = DatalakeIndex(root)
    with dl.run(kind=A.KIND_HEADLINES, pipeline="t", pipeline_version="v0") as r:
        _write_headline_month(r.out_dir, "2008-01.parquet", EARLIEST)   # earliest data
    with dl.run(kind=A.KIND_EMBEDDINGS, pipeline="t", pipeline_version="v0") as r:
        (r.out_dir / "2008-01.parquet").write_bytes(b"0")
    with dl.run(kind=A.KIND_MU_ASOF, pipeline="t", pipeline_version="v0") as r:
        # first mu row one month after earliest data (mu_asof's own delay)
        _mu_df([date(2008, 2, 1), date(2008, 7, 1)]).write_parquet(r.out_dir / "mu_asof.parquet")
    return dl


@pytest.fixture
def lake(tmp_path: Path):
    dl = _make_lake(tmp_path / "lake")
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
                                          "2008-06.parquet", A.PROVENANCE_FILE,
                                          A.RUN_CONFIG_FILE, "run_metadata.json"]
        # embedding provenance next to every artifact of the run, and in the run metadata
        digest = A.embeddings_digest(toy_embeddings)
        key = A.embedding_key(toy_embeddings, None)
        for art in (nd, dg, pt, ta):
            prov = json.loads((art.path / A.PROVENANCE_FILE).read_text())["runs"][-1]
            assert prov["primitive_embeddings"]["digest"] == digest
            assert prov["embedding_key"] == key
            assert prov["headline_embeddings"]["artifact_id"] == \
                lake.latest(A.KIND_EMBEDDINGS).artifact_id
            assert art.meta.hyperparams["embeddings_id"] == key
        run_meta = json.loads((nd.path / "run_metadata.json").read_text())
        assert run_meta["embeddings_provenance"]["primitive_embeddings"]["digest"] == digest
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
        key = A.embedding_key(toy_embeddings, None)
        base = A.find_tau_asof(lake, cfg, toy_table, key, "5Y", 0, temp=True)
        a = pl.read_parquet(base.path / A.TAU_ASOF_FILE)
        assert a["MONTH_END"].to_list() == [date(2008, 2, 29)]   # built as of April 1
        time.sleep(1.1)                       # artifact timestamps have second precision
        rebuilt = A.build_tau_asof(lake, cfg, toy_table, toy_embeddings, today=date(2008, 4, 1),
                                   rebuild=True, temp=True)
        assert rebuilt.artifact_id != base.artifact_id and "__TEMP" in rebuilt.artifact_id
        b = pl.read_parquet(rebuilt.path / A.TAU_ASOF_FILE)
        assert a.equals(b)                                      # byte-identical rows
        assert A.find_tau_asof(lake, cfg, toy_table, key, "5Y", 0,
                               temp=True).artifact_id == \
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
        key = A.embedding_key(toy_embeddings, None)
        owner = A.find_partitions(lake, cfg, toy_table, key, temp=False)
        assert owner is not None and "__TEMP" not in owner.artifact_id
        assert owner.meta.hyperparams["agent_created"] is False
        assert not A.is_agent_created(owner)
        assert A.find_partitions(lake, cfg, toy_table, key,
                                 temp=True) is None                  # separate lineage
        marked = A.mark_temp_deprecated(lake, owner.artifact_id, "superseded by owner run")
        assert marked.deprecated and marked.meta.deprecation_reason.startswith(
            "superseded by owner run")
        assert A.TEMP_NOTE in marked.meta.notes and A.is_agent_created(marked)
        assert marked.artifact_id == owner.artifact_id             # ids are frozen
        assert owner.path.exists() and marked.file_hashes == owner.file_hashes  # nothing deleted
        assert A.find_partitions(lake, cfg, toy_table, key,
                                 temp=False) is None                 # deprecated hidden
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


def _meta(recipe: str) -> dict:
    return {"identity_digest": recipe, "identity": {"backend": {"dtype": recipe}}}


def test_embedding_recipe_decides_reuse(lake, toy_table, toy_embeddings, cfg, caplog):
    """Partitions / tau are extended only with embeddings of the same recipe (model,
    backend, dtype, ...). Same recipe with different bytes (a TEI cache rebuild)
    keeps extending, with a warning and a provenance history entry; another
    recipe opens its own artifacts."""
    months = [(2008, m) for m in range(1, 5)]
    s = A.score_range_to_datalake(lake, date(2008, 1, 1), date(2008, 3, 31), cfg,
                                  table=toy_table, P=toy_embeddings,
                                  source=_source(toy_table, toy_embeddings, months),
                                  use_kernel=False, temp=True, primitive_meta=_meta("tei-fp16"))
    noisy = toy_embeddings.copy()
    noisy[0, 0] += 5e-4                                      # same recipe, rebuilt bytes
    with caplog.at_level(logging.WARNING):
        s2 = A.score_range_to_datalake(lake, date(2008, 4, 1), date(2008, 4, 30), cfg,
                                       table=toy_table, P=noisy,
                                       source=_source(toy_table, noisy, months),
                                       use_kernel=False, temp=True,
                                       primitive_meta=_meta("tei-fp16"))
    assert s2["partitions_id"] == s["partitions_id"]                  # extended, not rebuilt
    assert any("different bytes" in r.message for r in caplog.records)
    history = json.loads((lake.get(s2["partitions_id"]).path / A.PROVENANCE_FILE)
                         .read_text())["runs"]
    assert [h["primitive_embeddings"]["digest"] for h in history] == [
        A.embeddings_digest(toy_embeddings), A.embeddings_digest(noisy)]

    assert A.find_partitions(lake, cfg, toy_table, "local-fp32", temp=True) is None
    assert A.find_tau_asof(lake, cfg, toy_table, "local-fp32", temp=True) is None
    s3 = A.score_range_to_datalake(lake, date(2008, 1, 1), date(2008, 3, 31), cfg,
                                   table=toy_table, P=toy_embeddings,
                                   source=_source(toy_table, toy_embeddings, months),
                                   use_kernel=False, temp=True,
                                   primitive_meta=_meta("local-fp32"))
    assert s3["partitions_id"] != s["partitions_id"]


class _Killed(Exception):
    pass


class _DiesOn(InMemoryHeadlineSource):
    """An in-memory source whose read of ``kill_day`` kills the run."""

    kill_day = None

    def iter_day(self, day):
        if day == self.kill_day:
            raise _Killed(f"killed on {day}")
        yield from super().iter_day(day)


def _frames(art) -> dict:
    return {p.name: pl.read_parquet(p) for p in sorted(art.path.glob("*.parquet"))}


def test_killed_scoring_run_resumes_to_identical_outputs(tmp_path, toy_table, toy_embeddings,
                                                         cfg):
    """Kill a run in the middle of a month (after its partition checkpointed some closed
    days and after earlier months were written), resume it in place: every
    narrative_daily / day_diagnostics / partition / tau file equals an uninterrupted run."""
    months = [(2008, m) for m in range(1, 7)]
    kw = dict(table=toy_table, P=toy_embeddings, window="5Y", use_kernel=False, temp=True)

    ref_lake = _make_lake(tmp_path / "ref")
    ref = A.score_range_to_datalake(ref_lake, date(2008, 1, 1), date(2008, 6, 30), cfg,
                                    source=_source(toy_table, toy_embeddings, months), **kw)

    lake = _make_lake(tmp_path / "lake")
    base = _source(toy_table, toy_embeddings, months)
    dying = _DiesOn(base.days, chunk_size=5, source_id="toy")
    dying.kill_day = date(2008, 5, 15)
    with pytest.raises(_Killed):
        A.score_range_to_datalake(lake, date(2008, 1, 1), date(2008, 6, 30), cfg,
                                  source=dying, **kw)
    nd_part = lake.list(A.KIND_NARRATIVE_DAILY, include_partial=True)[0]
    assert nd_part.partial and sorted(_frames(nd_part)) == ["2008-04.parquet"]
    rc = json.loads((nd_part.path / A.RUN_CONFIG_FILE).read_text())
    pt_part = lake.get(rc["artifacts"]["partitions"])
    assert pt_part.partial and (pt_part.path / "_open").exists()     # May checkpointed

    s = A.score_range_to_datalake(
        lake, date.fromisoformat(rc["start"]), date.fromisoformat(rc["end"]), cfg,
        source=_source(toy_table, toy_embeddings, months), inputs=rc["inputs"],
        resume=rc["artifacts"], **kw)
    assert s["narrative_daily_id"] == nd_part.artifact_id
    for kind_id, ref_id in ((s["narrative_daily_id"], ref["narrative_daily_id"]),
                            (s["day_diagnostics_id"], ref["day_diagnostics_id"]),
                            (s["partitions_id"], ref["partitions_id"]),
                            (s["tau_asof_id"], ref["tau_asof_id"])):
        got, want = _frames(lake.get(kind_id)), _frames(ref_lake.get(ref_id))
        assert sorted(got) == sorted(want)
        for name in want:
            drop = [c for c in ("RSS_GB_BEFORE", "RSS_GB_AFTER", "SECONDS", "RSS_PEAK_GB",
                                "CODE_VERSION")                 # run-specific, not results
                    if c in want[name].columns]
            assert got[name].drop(drop).equals(want[name].drop(drop)), (kind_id, name)
    done = lake.get(s["narrative_daily_id"])
    assert not done.partial and len(done.meta.runs) == 2


def test_resume_scoring_refuses_runs_without_run_config(lake, tmp_path, toy_table,
                                                        toy_embeddings, cfg):
    with pytest.raises(_Killed):
        dying = _DiesOn(_source(toy_table, toy_embeddings, [(2008, 3)]).days, chunk_size=5)
        dying.kill_day = date(2008, 3, 3)
        A.score_range_to_datalake(lake, date(2008, 3, 1), date(2008, 3, 31), cfg,
                                  table=toy_table, P=toy_embeddings, source=dying,
                                  use_kernel=False, temp=True)
    nd = lake.list(A.KIND_NARRATIVE_DAILY, include_partial=True)[0]
    (nd.path / A.RUN_CONFIG_FILE).unlink()
    with pytest.raises(ValueError, match="cannot be resumed"):
        A.resume_scoring(lake, nd.artifact_id, cache_dir=tmp_path)


def test_latest_matching_and_missing(lake, toy_table, cfg):
    with pytest.raises(DatalakeError):
        A.latest_matching(lake, A.KIND_TAU_ASOF, f0_config_id="nope")
    P = np.zeros((toy_table.n_primitives * toy_table.n_texts, EMBEDDING_DIM), np.float32)
    assert A.find_tau_asof(lake, cfg, toy_table, A.embedding_key(P, None)) is None
    assert A.build_tau_asof(lake, cfg, toy_table, P) is None
    assert A.earliest_headline_day(lake.latest(A.KIND_HEADLINES)) == EARLIEST


class TestTaxonomyRegistration:
    def _write(self, tmp_path, name, rows=None):
        from .conftest import write_toy_taxonomy

        return write_toy_taxonomy(tmp_path / "src", name=name, rows=rows,
                                  styles=("headline", "semantic"))

    def test_register_resolve_load_and_idempotent(self, lake, tmp_path):
        from .conftest import vendor_rows

        src = self._write(tmp_path, "Ever")
        self._write(tmp_path, "Vendor", rows=vendor_rows())
        ev = A.register_taxonomy(lake, src, "Ever", temp=True)
        vd = A.register_taxonomy(lake, src, "Vendor", temp=True)
        assert ev.layer == "raw" and ev.kind == A.KIND_TAXONOMY and "__TEMP" in ev.artifact_id
        assert sorted(ev.file_hashes) == sorted(A.taxonomy_files("Ever"))
        assert ev.meta.hyperparams["has_observability_channel"] is True
        assert vd.meta.hyperparams["has_observability_channel"] is False
        assert A.verify_taxonomy(ev) == [] and A.verify_taxonomy(vd) == []
        assert A.register_taxonomy(lake, src, "Ever", temp=True).artifact_id == ev.artifact_id
        got = A.resolve_taxonomy([lake], name="Vendor")
        assert got.artifact_id == vd.artifact_id
        assert A.resolve_taxonomy([lake], artifact_id=ev.artifact_id).artifact_id == ev.artifact_id
        t = A.load_registered_table(vd, "semantic")
        assert t.name == "Vendor" and not t.has_observability_channel
        assert t.frame["observability_channel"].unique().to_list() == [""]
        with pytest.raises(DatalakeError, match="register-taxonomy"):
            A.resolve_taxonomy([lake], name="Nope")
        with pytest.raises(DatalakeError, match="not a narrative_taxonomy"):
            A.resolve_taxonomy([lake], artifact_id=lake.latest(A.KIND_MU_ASOF).artifact_id)

    def test_invalid_taxonomy_is_never_registered(self, lake, tmp_path):
        import json as _json

        src = self._write(tmp_path, "Bad")
        p = src / "Bad-primitive_semantic_paraphrases.jsonl"
        recs = [_json.loads(line) for line in p.read_text().splitlines()]
        recs[0]["id"] = "0" * 40
        p.write_text("\n".join(_json.dumps(r) for r in recs) + "\n")
        with pytest.raises(ValueError, match="bijection"):
            A.register_taxonomy(lake, src, "Bad", temp=True)
        assert lake.list(A.KIND_TAXONOMY, include_partial=True) == []
        (src / "Bad-primitive_headline_paraphrases.jsonl").unlink()
        with pytest.raises(FileNotFoundError):
            A.register_taxonomy(lake, src, "Bad", temp=True)

    def test_scoring_cites_the_taxonomy(self, lake, toy_embeddings, cfg, tmp_path):
        from .conftest import TAX_NAME

        src = self._write(tmp_path, TAX_NAME)
        tax = A.register_taxonomy(lake, src, TAX_NAME, temp=True)
        table = A.load_registered_table(tax, "headline")
        s = A.score_range_to_datalake(
            lake, date(2008, 3, 1), date(2008, 4, 30), cfg, table=table, P=toy_embeddings,
            source=_source(table, toy_embeddings, [(2008, m) for m in (2, 3, 4)]),
            use_kernel=False, temp=True, taxonomy_id=tax.artifact_id)
        nd = lake.get(s["narrative_daily_id"])
        assert tax.artifact_id in nd.meta.sources
        assert nd.meta.hyperparams["taxonomy_artifact_id"] == tax.artifact_id
        assert tax.artifact_id in lake.get(s["tau_asof_id"]).meta.sources
        assert tax.artifact_id in lake.get(s["partitions_id"]).meta.sources
        assert nd.artifact_id in lake.descendants(tax.artifact_id)


# ---------------------------------------------------------------------------
# Sentiment split through the one scoring path
# ---------------------------------------------------------------------------

SPLIT = {"sentiment_split": "sign", "sentiment_source": "toy", "sentiment_column": "SENT_A",
         "neutral_eps": 0.2}


def _register_sentiment(dl: DatalakeIndex, columns=("SENT_A", "SENT_B")):
    from ravenpack.headlines.sentiment import ingest_to_datalake

    def produce(path: Path, tag: str) -> pl.DataFrame:
        ids = pl.read_parquet(path)["RP_STORY_ID"]
        return pl.DataFrame({"RP_STORY_ID": ids,
                             **{c: pl.Series([0.5] * ids.len(), dtype=pl.Float32)
                                for c in columns}})

    return ingest_to_datalake(dl, source="toy", columns=list(columns), produce=produce,
                              headlines=dl.latest(A.KIND_HEADLINES), start_year=2008,
                              end_year=2008, extra_hyperparams={}, temp=True)


def _sentiment_source(table, P, months, n=12, seed=4):
    """The toy source plus fully-labelled sentiment (so the labels reconcile with 'all')."""
    src = _source(table, P, months, n)
    rng = np.random.default_rng(seed)
    src.sentiment = {d: rng.uniform(-1, 1, size=X.shape[0]).astype(np.float32)
                     for d, X in src.days.items()}
    return src


class TestSentimentSplit:
    MONTHS = [(2008, m) for m in range(1, 6)]

    def test_split_run_cites_sentiment_and_keeps_all_rows(self, lake, toy_table, toy_embeddings,
                                                          tmp_path):
        from narrative_scoring.validation import combine_sentiment

        sent = _register_sentiment(lake)
        cfg_split = ScoringConfig(q=0.75, null_draws_per_headline=4, min_month_draws=1, **SPLIT)
        s = A.score_range_to_datalake(
            lake, date(2008, 3, 1), date(2008, 5, 31), cfg_split, table=toy_table,
            P=toy_embeddings, source=_sentiment_source(toy_table, toy_embeddings, self.MONTHS),
            use_kernel=False, temp=True, sentiment=sent)
        nd, dg = lake.get(s["narrative_daily_id"]), lake.get(s["day_diagnostics_id"])
        for art in (nd, dg):
            assert sent.artifact_id in art.meta.sources
            hp = art.meta.hyperparams
            assert (hp["sentiment_artifact_id"], hp["sentiment_column"],
                    hp["sentiment_source"]) == (sent.artifact_id, "SENT_A", "toy")
        assert sent.artifact_id not in lake.get(s["partitions_id"]).meta.sources
        assert A.verify_day_diagnostics(dg) == []
        diag = pl.concat([pl.read_parquet(p) for p in dg.files() if p.suffix == ".parquet"])
        assert diag["SENTIMENT_SOURCE_ID"].unique().to_list() == [f"{sent.artifact_id}:SENT_A"]

        # the same days without a split, in a separate sandbox: identical "all" rows
        other = _make_lake(tmp_path / "other")
        s0 = A.score_range_to_datalake(
            other, date(2008, 3, 1), date(2008, 5, 31), ScoringConfig(
                q=0.75, null_draws_per_headline=4, min_month_draws=1),
            table=toy_table, P=toy_embeddings,
            source=_sentiment_source(toy_table, toy_embeddings, self.MONTHS), use_kernel=False,
            temp=True)
        read = lambda art: pl.concat([pl.read_parquet(p) for p in art.files()   # noqa: E731
                                      if p.suffix == ".parquet"]).sort(["DATE", "narrative_key"])
        with_split, without = read(nd), read(other.get(s0["narrative_daily_id"]))
        assert with_split.filter(pl.col("SENTIMENT") == "all").equals(without)
        # fully labelled: the labels recombine into the "all" rows
        both = combine_sentiment(with_split, ["pos", "neg", "neu"]).sort(["DATE",
                                                                          "narrative_key"])
        np.testing.assert_array_equal(both["SUPPORT"].to_numpy(), without["SUPPORT"].to_numpy())
        other.close()

    def test_incompatible_sentiment_is_refused(self, lake, toy_table, toy_embeddings, cfg):
        sent = _register_sentiment(lake)
        hl_id = lake.latest(A.KIND_HEADLINES).artifact_id
        with pytest.raises(DatalakeError, match="no column 'SENT_Z'"):
            A.resolve_sentiment([lake], headlines_id=hl_id, column="SENT_Z", source="toy")
        with pytest.raises(DatalakeError, match="built on"):
            A.resolve_sentiment([lake], headlines_id="other", column="SENT_A",
                                artifact_id=sent.artifact_id)
        with pytest.raises(DatalakeError, match="config says"):
            A.check_sentiment(sent, headlines_id=hl_id, column="SENT_A", source="finbert")
        assert A.resolve_sentiment([lake], headlines_id=hl_id, column="SENT_B",
                                   source="toy").artifact_id == sent.artifact_id
        src = _source(toy_table, toy_embeddings, [(2008, 3)])
        with pytest.raises(ValueError, match="required iff"):
            A.score_range_to_datalake(lake, date(2008, 3, 1), date(2008, 3, 31), cfg,
                                      table=toy_table, P=toy_embeddings, source=src,
                                      use_kernel=False, temp=True, sentiment=sent)
        split = ScoringConfig(q=0.75, null_draws_per_headline=4, min_month_draws=1, **SPLIT)
        with pytest.raises(ValueError, match="required iff"):
            A.score_range_to_datalake(lake, date(2008, 3, 1), date(2008, 3, 31), split,
                                      table=toy_table, P=toy_embeddings, source=src,
                                      use_kernel=False, temp=True)


def test_day_diagnostics_verifier_accepts_the_previous_schema(tmp_path):
    from narrative_scoring.schema import DAY_DIAGNOSTICS_SCHEMA_V1

    f = tmp_path / "2008-01.parquet"
    row = {k: None for k in DAY_DIAGNOSTICS_SCHEMA_V1}
    row["DATE"] = date(2008, 1, 2)
    pl.DataFrame([row], schema=DAY_DIAGNOSTICS_SCHEMA_V1).write_parquet(f)
    assert list(A._schema_check([f], DAY_DIAGNOSTICS_SCHEMA, "dg",
                                accepted=(DAY_DIAGNOSTICS_SCHEMA_V1,))) == []
    assert list(A._schema_check([f], DAY_DIAGNOSTICS_SCHEMA, "dg")) == [
        "2008-01.parquet: schema mismatch"]


def test_default_calendar_adds_no_hyperparams(lake, toy_table, toy_embeddings, cfg):
    s = A.score_range_to_datalake(lake, date(2008, 1, 1), date(2008, 4, 30), cfg,
                                  table=toy_table, P=toy_embeddings,
                                  source=_source(toy_table, toy_embeddings,
                                                 [(2008, m) for m in range(1, 5)]),
                                  use_kernel=False, temp=True)
    for key in ("narrative_daily_id", "partitions_id", "tau_asof_id"):
        hp = lake.get(s[key]).meta.hyperparams
        assert "calibration_freq" not in hp and "calibration_delay" not in hp


def test_weekly_calibration_calendar_end_to_end_and_resume(tmp_path, toy_table, toy_embeddings,
                                                          cfg):
    """A W / 1W calendar: weekly partitions, weekly narrative_daily files, lookups kept
    apart from the monthly calendar, and kill-at-mid-run + resume == uninterrupted."""
    from narrative_scoring.tau_asof import CalibrationCalendar

    weekly = CalibrationCalendar("W", "1W")
    months = [(2008, m) for m in range(1, 5)]
    kw = dict(table=toy_table, P=toy_embeddings, window="5Y", use_kernel=False, temp=True,
              calendar=weekly)

    ref_lake = _make_lake(tmp_path / "ref")
    ref = A.score_range_to_datalake(ref_lake, date(2008, 1, 1), date(2008, 4, 30), cfg,
                                    source=_source(toy_table, toy_embeddings, months), **kw)
    pt = ref_lake.get(ref["partitions_id"])
    assert pt.meta.hyperparams["calibration_freq"] == "W"
    assert all("-W" in p.stem for p in pt.path.glob("*.parquet"))
    nd = ref_lake.get(ref["narrative_daily_id"])
    assert sorted(_frames(nd)) and all("-W" in n for n in _frames(nd))
    assert A.find_partitions(ref_lake, cfg, toy_table, A.embedding_key(toy_embeddings, None),
                             temp=True) is None                     # monthly lookup: no match
    assert A.find_partitions(ref_lake, cfg, toy_table, A.embedding_key(toy_embeddings, None),
                             temp=True, calendar=weekly) is not None

    lake = _make_lake(tmp_path / "lake")
    dying = _DiesOn(_source(toy_table, toy_embeddings, months).days, chunk_size=5)
    dying.kill_day = date(2008, 3, 19)
    with pytest.raises(_Killed):
        A.score_range_to_datalake(lake, date(2008, 1, 1), date(2008, 4, 30), cfg,
                                  source=dying, **kw)
    nd_part = lake.list(A.KIND_NARRATIVE_DAILY, include_partial=True)[0]
    rc = json.loads((nd_part.path / A.RUN_CONFIG_FILE).read_text())
    assert rc["calibration"] == {"freq": "W", "delay": "1W"}
    s = A.score_range_to_datalake(
        lake, date.fromisoformat(rc["start"]), date.fromisoformat(rc["end"]), cfg,
        source=_source(toy_table, toy_embeddings, months), inputs=rc["inputs"],
        resume=rc["artifacts"], **kw)
    drop = ("RSS_GB_BEFORE", "RSS_GB_AFTER", "SECONDS", "RSS_PEAK_GB", "CODE_VERSION")
    for got_id, want_id in ((s["narrative_daily_id"], ref["narrative_daily_id"]),
                            (s["partitions_id"], ref["partitions_id"]),
                            (s["tau_asof_id"], ref["tau_asof_id"])):
        got, want = _frames(lake.get(got_id)), _frames(ref_lake.get(want_id))
        assert sorted(got) == sorted(want)
        for name in want:
            cols = [c for c in drop if c in want[name].columns]
            assert got[name].drop(cols).equals(want[name].drop(cols)), (got_id, name)


def test_scoring_job_pauses_at_a_period_boundary_and_resumes_identically(
        tmp_path, toy_table, toy_embeddings, cfg, monkeypatch):
    """The jobs lifecycle on scoring: a PAUSE request stops the run after its current
    calibration period (siblings left partial, status paused, job.json not hashed);
    the resume from the run's own ids gives the uninterrupted outputs."""
    from datalake.jobs import PAUSE_FILE, STATE_FILE, JobRunner, JobState

    months = [(2008, m) for m in range(1, 7)]
    kw = dict(table=toy_table, P=toy_embeddings, window="5Y", use_kernel=False, temp=True)
    ref_lake = _make_lake(tmp_path / "ref")
    ref = A.score_range_to_datalake(ref_lake, date(2008, 1, 1), date(2008, 6, 30), cfg,
                                    source=_source(toy_table, toy_embeddings, months), **kw)

    lake = _make_lake(tmp_path / "lake")
    real = A.ScoringJob.run_unit

    def pause_after_march(self, unit, ctx):
        real(self, unit, ctx)
        if unit.key == "2008-03":
            (ctx.out_dir / PAUSE_FILE).write_text("stop")

    monkeypatch.setattr(A.ScoringJob, "run_unit", pause_after_march)
    job = A.ScoringJob(lake, date(2008, 1, 1), date(2008, 6, 30), cfg,
                       source=_source(toy_table, toy_embeddings, months), **kw)
    runner = JobRunner(lake, allow_dirty=True, handle_signals=False)
    nd = runner.start(job)
    assert nd.partial and runner.status(nd.artifact_id).status == "paused"
    state = JobState.read(nd.path)
    assert state.units_done < state.units_total
    rc = json.loads((nd.path / A.RUN_CONFIG_FILE).read_text())
    assert lake.get(rc["artifacts"]["day_diagnostics"]).partial
    monkeypatch.setattr(A.ScoringJob, "run_unit", real)

    s = A.score_range_to_datalake(
        lake, date.fromisoformat(rc["start"]), date.fromisoformat(rc["end"]), cfg,
        source=_source(toy_table, toy_embeddings, months), inputs=rc["inputs"],
        resume=rc["artifacts"], **kw)
    for got_id, ref_id in ((s["narrative_daily_id"], ref["narrative_daily_id"]),
                           (s["day_diagnostics_id"], ref["day_diagnostics_id"]),
                           (s["partitions_id"], ref["partitions_id"]),
                           (s["tau_asof_id"], ref["tau_asof_id"])):
        got, want = _frames(lake.get(got_id)), _frames(ref_lake.get(ref_id))
        assert sorted(got) == sorted(want)
        for name in want:
            drop = [c for c in ("RSS_GB_BEFORE", "RSS_GB_AFTER", "SECONDS", "RSS_PEAK_GB",
                                "CODE_VERSION") if c in want[name].columns]
            assert got[name].drop(drop).equals(want[name].drop(drop)), (got_id, name)
    done = lake.get(s["narrative_daily_id"])
    assert not done.partial and STATE_FILE not in done.file_hashes
    assert JobState.read(done.path).status == "complete"


# ---------------------------------------------------------------------------
# Lineage pre-flight: never combine artifacts built from different inputs
# ---------------------------------------------------------------------------

def _score(lake, start, end, cfg, table, P):
    months = [(2008, m) for m in range(1, 7)]
    return A.score_range_to_datalake(lake, start, end, cfg, table=table, P=P,
                                     source=_source(table, P, months), window="5Y",
                                     use_kernel=False, temp=True)


def test_new_embeddings_refuse_to_extend_partitions_and_tau_built_from_the_old(
        tmp_path, toy_table, toy_embeddings, cfg):
    from datalake.lineage import LineageError

    lake = _make_lake(tmp_path / "lake")
    _score(lake, date(2008, 1, 1), date(2008, 4, 30), cfg, toy_table, toy_embeddings)
    hl = lake.latest(A.KIND_HEADLINES)
    with lake.run(kind=A.KIND_EMBEDDINGS, pipeline="t", pipeline_version="v1",
                  sources=[hl]) as r:                      # re-embedded corpus
        (r.out_dir / "2008-01.parquet").write_bytes(b"1")
    new_em = lake.get(r.artifact_id)
    with lake.run(kind=A.KIND_MU_ASOF, pipeline="t", pipeline_version="v1",
                  sources=[hl, new_em]) as r:              # its mu_asof, recomputed
        _mu_df([date(2008, 2, 1), date(2008, 7, 1)]).write_parquet(r.out_dir / "mu.parquet")
    n_runs = len(lake.list(A.KIND_NARRATIVE_DAILY, include_partial=True))
    with pytest.raises(LineageError) as exc:
        _score(lake, date(2008, 5, 1), date(2008, 6, 30), cfg, toy_table, toy_embeddings)
    msg = str(exc.value)
    assert A.KIND_PARTITIONS in msg and new_em.artifact_id in msg    # partitions: old em
    assert "tau_asof" in msg                                         # tau: old mu_asof
    assert len(lake.list(A.KIND_NARRATIVE_DAILY, include_partial=True)) == n_runs


def test_mu_asof_built_from_other_embeddings_is_refused(tmp_path, toy_table,
                                                         toy_embeddings, cfg):
    from datalake.lineage import LineageError

    lake = _make_lake(tmp_path / "lake")
    hl, old_em = lake.latest(A.KIND_HEADLINES), lake.latest(A.KIND_EMBEDDINGS)
    with lake.run(kind=A.KIND_EMBEDDINGS, pipeline="t", pipeline_version="v1",
                  sources=[hl]) as r:
        (r.out_dir / "2008-01.parquet").write_bytes(b"1")
    with lake.run(kind=A.KIND_MU_ASOF, pipeline="t", pipeline_version="v1",
                  sources=[hl, old_em]) as r:              # latest mu_asof, stale source
        _mu_df([date(2008, 2, 1), date(2008, 7, 1)]).write_parquet(r.out_dir / "mu.parquet")
    with pytest.raises(LineageError, match=r"mu_asof.*built from headline_embeddings"):
        _score(lake, date(2008, 3, 1), date(2008, 3, 31), cfg, toy_table, toy_embeddings)


def test_unrecorded_lineage_warns_and_runs(tmp_path, toy_table, toy_embeddings, cfg, caplog):
    lake = _make_lake(tmp_path / "lake")          # upstream registered without sources
    with caplog.at_level(logging.WARNING, logger="datalake.lineage"):
        s = _score(lake, date(2008, 3, 1), date(2008, 3, 31), cfg, toy_table, toy_embeddings)
    assert s["narrative_daily_id"]
    assert "records no ravenpack_headlines source" in caplog.text
