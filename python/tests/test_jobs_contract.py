"""The Job contract, for every registered job at D / M / Q partitions.

Each case builds a small lake at the requested frequency (the storage partition for
the corpus jobs, the calibration period for scoring) and returns a fresh job, a way
to resume it and a comparable view of its outputs. Every job must then:

  * resume after a kill to outputs identical to an uninterrupted run;
  * stop at a unit boundary on a PAUSE request (status ``paused``) and resume;
  * refuse a second live process and take over a stale lock;
  * fail with the reason when its served model changes (jobs with a model);
  * jobs that support updates: add new data to the complete artifact as a new
    execution, resume a killed update to the uninterrupted result, add nothing
    when nothing changed, and refuse another major.minor.

A registered job without a case here fails ``test_every_registered_job_has_a_case``.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from datalake import DatalakeIndex
from datalake.artifact import utc_now_iso
from datalake.jobs import (
    LOCK_FILE,
    PAUSE_FILE,
    Job,
    JobLocked,
    JobRunner,
    JobState,
    registered_jobs,
)
from datalake.layout import Layout
from datalake.periods import period_key

FREQS = ["D", "M", "Q"]
START, END = date(2008, 1, 1), date(2008, 6, 30)


class Killed(RuntimeError):
    pass


@dataclass
class Env:
    index: DatalakeIndex
    new_job: Callable[[], Job]
    resume: Callable[[JobRunner, str], Any]
    outputs: Callable[[Any], dict[str, Any]]
    switch_model: Callable[[Job], None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    mutate: Callable[[], None] | None = None                 # new upstream data
    update_job: Callable[[str], Job] | None = None           # the update of a complete id


def _runner(index: DatalakeIndex) -> JobRunner:
    return JobRunner(index, allow_dirty=True, handle_signals=False, heartbeat_s=0.05)


def _parquet_bytes(art) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(art.path.glob("*.parquet"))}


# ---------------------------------------------------------------------------
# Synthetic upstream data
# ---------------------------------------------------------------------------

def _stories() -> list[tuple[str, datetime, str]]:
    """Two stories every third day of [START, END]."""
    out, d = [], START
    while d <= END:
        for i in range(2):
            out.append((f"{d:%Y%m%d}-{i}", datetime(d.year, d.month, d.day, 9 + i),
                        f"headline {d} number {i} " * (1 + i)))
        d += timedelta(days=3)
    return out


def _register_headlines(index: DatalakeIndex, freq: str, *, with_embeddings: bool = False):
    """ravenpack_headlines (and optionally headline_embeddings) partitioned at ``freq``."""
    from ravenpack.headlines.schema import EMBEDDING_DIM

    hp = Layout(freq, START, END).hyperparams()
    by_key: dict[str, list] = {}
    for sid, ts, text in _stories():
        by_key.setdefault(period_key(ts.date(), freq), []).append((sid, ts, text))
    with index.run(kind="ravenpack_headlines", pipeline="t", pipeline_version="v0",
                   hyperparams=hp) as run:
        for key, rows in by_key.items():
            pl.DataFrame({"RP_STORY_ID": [r[0] for r in rows],
                          "TIMESTAMP_UTC": [r[1] for r in rows],
                          "HEADLINE": [r[2] for r in rows]}
                         ).write_parquet(run.out_dir / f"{key}.parquet")
    headlines = index.get(run.artifact_id)
    if not with_embeddings:
        return headlines, None
    rng = np.random.default_rng(3)
    with index.run(kind="headline_embeddings", pipeline="t", pipeline_version="v0",
                   hyperparams=hp) as run:
        for key, rows in by_key.items():
            emb = rng.normal(size=(len(rows), EMBEDDING_DIM)).astype(np.float16)
            pl.DataFrame({"RP_STORY_ID": [r[0] for r in rows], "EMBEDDING": list(emb)},
                         schema={"RP_STORY_ID": pl.String,
                                 "EMBEDDING": pl.Array(pl.Float16, EMBEDDING_DIM)}
                         ).write_parquet(run.out_dir / f"{key}.parquet")
    return headlines, index.get(run.artifact_id)


def _vendor_raw(root: Path) -> Path:
    raw = root / "vendor_raw"
    raw.mkdir(parents=True)
    stem = "RavenPackAnalytics_AllEntities_1.0_2008"
    months: dict[int, list] = {}
    for sid, ts, text in _stories():
        months.setdefault(ts.month, []).append((sid, ts, text))
    with zipfile.ZipFile(raw / f"{stem}.zip", "w") as zf:
        for month, rows in months.items():
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["TIMESTAMP_UTC", "RP_STORY_ID", "RP_ENTITY_ID", "CSS",
                        "EVENT_SENTIMENT_SCORE", "EVENT_RELEVANCE", "CATEGORY", "HEADLINE"])
            for k, (sid, ts, text) in enumerate(rows):
                w.writerow([ts.isoformat(sep=" "), sid, "E1", f"{(k % 9 - 4) / 10:.2f}",
                            "0.50" if k % 2 else "", "100" if k % 2 else "", "c", text])
            zf.writestr(f"{stem}/2008-{month:02d}.csv", buf.getvalue())
    return raw


# ---------------------------------------------------------------------------
# One case per job kind
# ---------------------------------------------------------------------------

def case_ravenpack_headlines(root: Path, freq: str) -> Env:
    from ravenpack.headlines.ingest import IngestJob
    from tests.ravenpack.headlines.test_ingest import _make_zip, _month_df

    raw = root / "raw"
    if not raw.exists():
        raw.mkdir(parents=True)
        for m in range(1, 7):
            _make_zip(raw, 2008, m, _month_df(2008, m))
    index = DatalakeIndex(root / "lake")
    return Env(index, lambda: IngestJob(raw, Layout(freq, START, END), temp=True),
               lambda runner, aid: runner.resume(aid, raw_dir=raw), _parquet_bytes)


def case_headline_sentiment(root: Path, freq: str) -> Env:
    from ravenpack.headlines import sentiment_vendor
    from ravenpack.headlines.sentiment import sentiment_layout

    index = DatalakeIndex(root / "lake")
    headlines, _ = _register_headlines(index, freq)
    raw = _vendor_raw(root)
    return Env(index,
               lambda: sentiment_vendor.job(headlines, sentiment_layout(headlines), raw,
                                            temp=True),
               lambda runner, aid: runner.resume(aid, raw_dir=raw), _parquet_bytes)


def case_headline_embeddings(root: Path, freq: str) -> Env:
    from nlp.embedding import Embedder
    from ravenpack.headlines.embed import EmbedJob, embedding_layout
    from tests.ravenpack.headlines.test_embed import FakeBackend

    index = DatalakeIndex(root / "lake")
    headlines, _ = _register_headlines(index, freq)

    def switch(job: Job) -> None:
        job.embedder.backend.switched = True

    return Env(index,
               lambda: EmbedJob(headlines, embedding_layout(headlines),
                                Embedder(FakeBackend()), temp=True),
               lambda runner, aid: runner.resume(aid, embedder=Embedder(FakeBackend())),
               _parquet_bytes, switch_model=switch)


def case_mu_asof(root: Path, freq: str) -> Env:
    from ravenpack.headlines.mu_asof import MuAsofJob

    index = DatalakeIndex(root / "lake")
    headlines, embeddings = _register_headlines(index, freq, with_embeddings=True)
    return Env(index,
               lambda: MuAsofJob(headlines, embeddings, "1d", "expanding", threads=1,
                                 temp=True),
               lambda runner, aid: runner.resume(aid, threads=1), _parquet_bytes)


_CALENDARS = {"D": ("D", "7d"), "M": ("M", "1M"), "Q": ("Q", "1M")}
_RUN_SPECIFIC = ("RSS_GB_BEFORE", "RSS_GB_AFTER", "SECONDS", "RSS_PEAK_GB", "CODE_VERSION")


def case_narrative_daily(root: Path, freq: str) -> Env:
    from narrative_scoring import artifacts as A
    from narrative_scoring.assets import AssetUniverse
    from narrative_scoring.config import ScoringConfig
    from narrative_scoring.primitives import load_primitive_table
    from narrative_scoring.tau_asof import CalibrationCalendar
    from tests.narrative_scoring.conftest import TAX_NAME, unit_rows, write_toy_taxonomy
    from tests.narrative_scoring.test_artifacts import _make_lake, _register_universe, _source

    table = load_primitive_table(write_toy_taxonomy(root / "tax"), TAX_NAME, "headline")
    P = unit_rows(np.random.default_rng(11), table.n_primitives * table.n_texts)
    cfg = ScoringConfig(q=0.75, null_draws_per_headline=4, min_month_draws=1)
    calendar = CalibrationCalendar(*_CALENDARS[freq])
    months = [(2008, m) for m in range(1, 7)]
    index = _make_lake(root / "lake")
    universe = _register_universe(index)                  # the asset tables are siblings too
    assets = AssetUniverse.from_artifact(universe)
    kw = dict(table=table, P=P, window="5Y", use_kernel=False, temp=True, calendar=calendar,
              assets=assets, universe=universe)

    def source():
        src = _source(table, P, months)
        rng = np.random.default_rng(4)
        src.assets = {d: [[(int(a), int(rng.integers(0, 101)))]
                          for a in rng.integers(0, assets.n_assets, X.shape[0])]
                      for d, X in src.days.items()}
        return src

    def new_job() -> Job:
        return A.ScoringJob(index, START, END, cfg, source=source(), **kw)

    def resume(runner: JobRunner, aid: str):
        rc = json.loads((index.get(aid).path / A.RUN_CONFIG_FILE).read_text())
        job = A.ScoringJob(index, date.fromisoformat(rc["start"]),
                           date.fromisoformat(rc["end"]), cfg,
                           source=source(), inputs=rc["inputs"],
                           resume=rc["artifacts"], **kw)
        return runner.resume_job(aid, job)

    def outputs(art) -> dict[str, Any]:
        rc = json.loads((art.path / A.RUN_CONFIG_FILE).read_text())
        out: dict[str, Any] = {}
        assert rc["artifacts"]["asset_attention"] and rc["artifacts"]["narrative_asset"]
        for role, aid in rc["artifacts"].items():
            for p in sorted(index.get(aid).path.glob("*.parquet")):
                df = pl.read_parquet(p)
                out[f"{role}/{p.name}"] = df.drop([c for c in _RUN_SPECIFIC if c in df.columns])
        return out

    return Env(index, new_job, resume, outputs)


def _frames(art) -> dict[str, pl.DataFrame]:
    """Data of every parquet (the speeches manifests carry real fetch times)."""
    return {p.name: pl.read_parquet(p) for p in sorted(art.path.glob("*.parquet"))}


def _bis_case(root: Path, freq: str):
    from bis_gingado.cb_speeches.speeches import SpeechesJob
    from tests.cb_speeches.fakes import BULK_URL, FakeBIS, every_third_day, speech
    from tests.cb_speeches.test_speeches import T0, Clock

    fake = FakeBIS({2008: every_third_day(START, END)})
    index = DatalakeIndex(root / "lake")

    def new_job() -> Job:
        return SpeechesJob.new(Layout(freq, START, END), url=BULK_URL, temp=True,
                               transport=fake.transport(), clock=Clock(T0), retry_wait_s=0)

    def mutate() -> None:
        fake.rows[2008][0] = dict(fake.rows[2008][0], title="retitled")
        fake.rows[2008].append(speech(date(2008, 6, 29), 7))

    def update_job(aid: str) -> Job:
        return SpeechesJob.for_update(index.get(aid), index, transport=fake.transport(),
                                      clock=Clock(T0 + timedelta(days=1)))

    return fake, index, new_job, mutate, update_job


def case_cb_speeches(root: Path, freq: str) -> Env:
    fake, index, new_job, mutate, update_job = _bis_case(root, freq)
    return Env(index, new_job, lambda runner, aid: runner.resume(aid, transport=fake.transport()),
               _frames, mutate=mutate, update_job=update_job)


def case_cb_speech_ner(root: Path, freq: str) -> Env:
    from bis_gingado.cb_speeches.ner import NerJob
    from tests.cb_speeches.fakes import CHAT_URL, FakeChat, chat_backend
    from tests.cb_speeches.test_speeches import T0, Clock

    fake, index, speeches_job, bis_mutate, bis_update = _bis_case(root, freq)
    speeches = _runner(index).start(speeches_job())
    chat = FakeChat()
    backend_kw = dict(http_client=chat.client(), base_url=CHAT_URL, api_key="k", max_retries=0)

    def switch(job: Job) -> None:
        chat.model = "fake-llm-2026-06-01"

    def mutate() -> None:
        bis_mutate()
        _runner(index).update_job(speeches.artifact_id, bis_update(speeches.artifact_id))

    def update_job(aid: str) -> Job:
        return NerJob.for_update(index.get(aid), index, clock=Clock(T0 + timedelta(days=2)),
                                 **backend_kw)

    return Env(index,
               lambda: NerJob.new(index.get(speeches.artifact_id), chat_backend(chat),
                                  temp=True, clock=Clock(T0 + timedelta(hours=1))),
               lambda runner, aid: runner.resume(aid, **backend_kw), _frames,
               switch_model=switch, mutate=mutate, update_job=update_job)


CASES: dict[str, Callable[[Path, str], Env]] = {
    "ravenpack_headlines": case_ravenpack_headlines,
    "headline_sentiment": case_headline_sentiment,
    "headline_embeddings": case_headline_embeddings,
    "mu_asof": case_mu_asof,
    "narrative_daily": case_narrative_daily,
    "cb_speeches": case_cb_speeches,
    "cb_speech_ner": case_cb_speech_ner,
}
WITH_MODEL = [k for k in CASES if k in ("headline_embeddings", "cb_speech_ner")]
UPDATABLE = ["cb_speeches", "cb_speech_ner"]


def test_every_registered_job_has_a_case():
    assert sorted(set(registered_jobs()) - {"toy_series", "toy_cli"}) == sorted(CASES)


def test_job_groups_are_valid_folder_names():
    from datalake.index import _GROUP_NAME

    for kind, cls in registered_jobs().items():
        assert cls.group is None or _GROUP_NAME.match(cls.group), (kind, cls.group)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

def _fail_on_call(job: Job, n: int, make_exc: Callable[[], BaseException]) -> None:
    """The ``n``-th run_unit call raises before doing anything (a kill mid-run)."""
    real, calls = job.run_unit, [0]

    def run_unit(unit, ctx):
        calls[0] += 1
        if calls[0] == n:
            raise make_exc()
        real(unit, ctx)

    job.run_unit = run_unit


def _after_call(job: Job, n: int, action: Callable[[Any], None]) -> None:
    real, calls = job.run_unit, [0]

    def run_unit(unit, ctx):
        real(unit, ctx)
        calls[0] += 1
        if calls[0] == n:
            action(ctx)

    job.run_unit = run_unit


def _assert_same(got: dict[str, Any], want: dict[str, Any]) -> None:
    assert sorted(got) == sorted(want)
    for name, value in want.items():
        other = got[name]
        same = other.equals(value) if isinstance(value, pl.DataFrame) else other == value
        assert same, name


def _killed(env: Env):
    job = env.new_job()
    _fail_on_call(job, 2, lambda: Killed("killed"))
    with pytest.raises(Killed):
        _runner(env.index).start(job)
    part = env.index.list(job.kind, include_partial=True)[0]
    assert part.partial
    return part


@pytest.fixture(params=FREQS)
def freq(request):
    return request.param


@pytest.mark.parametrize("kind", sorted(CASES))
def test_kill_then_resume_equals_an_uninterrupted_run(kind, freq, tmp_path):
    ref_env = CASES[kind](tmp_path / "ref", freq)
    ref = _runner(ref_env.index).start(ref_env.new_job())
    assert not ref.partial

    env = CASES[kind](tmp_path / "run", freq)
    part = _killed(env)
    state = JobState.read(part.path)
    assert state.status == "failed" and "killed" in state.last_error
    done = env.resume(_runner(env.index), part.artifact_id)
    assert not done.partial and done.artifact_id == part.artifact_id
    assert JobState.read(done.path).status == "complete"
    _assert_same(env.outputs(done), ref_env.outputs(ref))


@pytest.mark.parametrize("kind", sorted(CASES))
def test_pause_stops_at_a_unit_boundary_then_resumes(kind, freq, tmp_path):
    env = CASES[kind](tmp_path, freq)
    job = env.new_job()
    _after_call(job, 1, lambda ctx: (ctx.out_dir / PAUSE_FILE).write_text("stop"))
    runner = _runner(env.index)
    art = runner.start(job)
    st = runner.status(art.artifact_id)
    assert art.partial and st.status == "paused" and st.units_done < st.units_total
    done = env.resume(runner, art.artifact_id)
    assert not done.partial and not (done.path / PAUSE_FILE).exists()


@pytest.mark.parametrize("kind", sorted(CASES))
def test_live_lock_blocks_and_stale_lock_is_taken_over(kind, freq, tmp_path):
    env = CASES[kind](tmp_path, freq)
    part = _killed(env)
    runner = _runner(env.index)
    lock = part.path / LOCK_FILE
    lock.write_text(json.dumps({"host": "elsewhere", "pid": 1, "heartbeat_at": utc_now_iso()}))
    assert runner.status(part.artifact_id).status == "running"
    with pytest.raises(JobLocked):
        env.resume(runner, part.artifact_id)
    lock.write_text(json.dumps({"host": "elsewhere", "pid": 1,
                                "heartbeat_at": "2000-01-01T00:00:00+00:00"}))
    assert runner.status(part.artifact_id).status == "stale"
    assert not env.resume(runner, part.artifact_id).partial


@pytest.mark.parametrize("kind", WITH_MODEL)
def test_model_switch_fails_the_job_with_its_reason(kind, freq, tmp_path):
    env = CASES[kind](tmp_path, freq)
    job = env.new_job()
    _after_call(job, 1, lambda ctx: env.switch_model(job))
    with pytest.raises(Exception, match="model"):
        _runner(env.index).start(job)
    part = env.index.list(job.kind, include_partial=True)[0]
    state = JobState.read(part.path)
    assert state.status == "failed" and "model" in state.last_error
    assert state.units_done == 1


# ---------------------------------------------------------------------------
# Updates (jobs implementing Job.for_update)
# ---------------------------------------------------------------------------

def test_updatable_cases_are_the_jobs_that_support_updates():
    from datalake.jobs import Job as Base

    supports = sorted(k for k, cls in registered_jobs().items()
                      if cls.for_update.__func__ is not Base.for_update.__func__
                      and k not in ("toy_series", "toy_cli"))
    assert supports == sorted(UPDATABLE)


def _complete(env: Env):
    art = _runner(env.index).start(env.new_job())
    assert not art.partial
    return art


@pytest.mark.parametrize("kind", UPDATABLE)
def test_killed_update_resumes_to_the_uninterrupted_update(kind, freq, tmp_path):
    ref_env = CASES[kind](tmp_path / "ref", freq)
    ref = _complete(ref_env)
    ref_env.mutate()
    ref_done = _runner(ref_env.index).update_job(ref.artifact_id,
                                                 ref_env.update_job(ref.artifact_id))
    assert len(ref_done.meta.runs) == 2 and not ref_done.partial

    env = CASES[kind](tmp_path / "run", freq)
    art = _complete(env)
    before = env.outputs(art)
    env.mutate()
    job = env.update_job(art.artifact_id)
    _fail_on_call(job, 2 if len(job.units()) > 1 else 1, lambda: Killed("killed"))
    with pytest.raises(Killed):
        _runner(env.index).update_job(art.artifact_id, job)
    assert env.index.get(art.artifact_id).partial
    done = env.resume(_runner(env.index), art.artifact_id)
    assert not done.partial and done.artifact_id == art.artifact_id
    got = env.outputs(done)
    _assert_same(got, ref_env.outputs(ref_done))
    _assert_same({k: got[k] for k in before}, before)     # earlier files never rewritten
    assert len(got) > len(before)


@pytest.mark.parametrize("kind", UPDATABLE)
def test_update_without_new_data_adds_no_rows(kind, freq, tmp_path):
    env = CASES[kind](tmp_path, freq)
    art = _complete(env)
    before = env.outputs(art)
    done = _runner(env.index).update_job(art.artifact_id, env.update_job(art.artifact_id))
    after = env.outputs(done)
    _assert_same({k: after[k] for k in before}, before)
    assert all(after[k].height == 0 for k in set(after) - set(before))


@pytest.mark.parametrize("kind", UPDATABLE)
def test_update_refuses_another_major_minor(kind, tmp_path, monkeypatch):
    from datalake.jobs import JobError, job_class

    env = CASES[kind](tmp_path, "M")
    art = _complete(env)
    monkeypatch.setattr(job_class(kind), "pipeline_version", "v9.0.0")
    with pytest.raises(JobError, match="major.minor"):
        _runner(env.index).update(art.artifact_id)
