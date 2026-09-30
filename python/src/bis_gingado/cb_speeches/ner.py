"""Speech provenance NER -> ``cb_speech_ner`` artifact (a table keyed by speech_id).

An LLM behind any OpenAI-compatible provider (``nlp.llm``) reads each speech's
metadata (date, author, title, description; not the text) and returns the
prototype's four fields (Zenodo 18157708, ``gigando_speech_NER.ipynb``):
author, organization, country_code (ISO 3166-1 alpha-2) and sentiment
(hawkish / dovish / neutral). Replies are validated against the schema; a
reply that still fails after the retries is an explicit ``error`` row.

Identity (hyperparams): the speeches artifact, provider, model, sampling and
``prompt_sha256`` (system prompt + user template + schema). The source is read
as of a pinned speeches vintage (``plan-<vintage>.json``), so a resume sees the
same speeches even if the source was updated in between.

Base run: one parquet per source partition key holding speeches. Update
(``jobs update <ner_id>``): speeches whose (speech_id, input_sha256) is not in
the table yet -- new speeches, or revised metadata -- as of the source's latest
vintage, into ``update-<vintage>-<period key>.parquet``.

``read_ner(artifact)`` gives the latest row per speech_id;
``speeches_with_ner`` joins it to the speeches on speech_id.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

from bis_gingado.cb_speeches.schema import NER_SCHEMA
from bis_gingado.cb_speeches.speeches import (
    KIND as SPEECHES_KIND,
)
from bis_gingado.cb_speeches.speeches import (
    PLAN_PREFIX,
    UPDATE_PREFIX,
    Clock,
    _write_atomic,
    data_files,
    parse_vintage,
    plan_files,
    read_speeches,
    read_state,
    utc_now,
    vintage_tag,
    write_table,
)
from datalake.jobs import TEMP_SUFFIX, Job, JobContext, JobError, Unit
from datalake.layout import layout_from_hyperparams
from datalake.periods import partition_file, period_key

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex, ModelCard
    from nlp.llm import ChatBackend

log = logging.getLogger(__name__)

KIND = "cb_speech_ner"
PIPELINE_VERSION = "v0.1.0"

SYSTEM_PROMPT = (
    "You are an expert in natural language processing and financial analysis. Your task is to "
    "extract key information including the author's name, his/her organization, the "
    "organization country code, and the overall macroeconomic sentiment. Restrict your "
    "knowledge to what was available up to the date of the provided speech date.")

USER_TEMPLATE = """Extract information for the following speech metadata:
Date: '{date}'
Author: '{author}'
Title: '{title}'
Description: '{description}'

Your response:
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "author": {"type": "string", "description":
                   "First and last name of the author of the speech (None if not provided)."},
        "organization": {"type": "string", "description":
                         "The organization the author is affiliated with (try to guess if "
                         "not explicit)."},
        "country_code": {"type": "string", "description":
                         "The ISO 3166-1 alpha-2 country code of the organization (try to "
                         "guess if not explicit)."},
        "sentiment": {"type": "string", "description":
                      "Overall sentiment of the speech regarding macroeconomy.",
                      "enum": ["hawkish", "dovish", "neutral"]},
    },
    "required": ["author", "organization", "country_code", "sentiment"],
    "additionalProperties": False,
}
SCHEMA_NAME = "speech_information_extraction"
FIELDS = tuple(RESPONSE_SCHEMA["required"])
INPUT_FIELDS = ("date", "author", "title", "description")


def prompt_sha256() -> str:
    payload = json.dumps([SYSTEM_PROMPT, USER_TEMPLATE, RESPONSE_SCHEMA], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _inputs(row: dict[str, Any]) -> dict[str, str]:
    return {k: "" if row[k] is None else str(row[k]) for k in INPUT_FIELDS}


def input_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_inputs(row), sort_keys=True).encode()).hexdigest()


def user_prompt(row: dict[str, Any]) -> str:
    return USER_TEMPLATE.format(**_inputs(row))


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_ner(artifact: Artifact, as_of: datetime | None = None) -> pl.DataFrame:
    """The latest NER row per speech_id (recorded at or before ``as_of``)."""
    frames = [pl.read_parquet(p).with_columns(pl.lit(i).alias("_order"))
              for i, p in enumerate(data_files(artifact.path))]
    if not frames:
        return pl.from_arrow(NER_SCHEMA.empty_table())
    df = pl.concat(frames, how="vertical_relaxed")
    if as_of is not None:
        df = df.filter(pl.col("vintage") <= as_of)
    return (df.sort(["vintage", "_order"], maintain_order=True)
              .group_by("speech_id", maintain_order=True).last().drop("_order")
              .sort(["date", "speech_id"]))


def speeches_with_ner(speeches: Artifact, ner: Artifact,
                      as_of: datetime | None = None) -> pl.DataFrame:
    """Speeches (as of ``as_of``) left-joined to their NER fields on speech_id."""
    right = read_ner(ner, as_of).select(["speech_id", *FIELDS, "input_sha256", "error"])
    return read_speeches(speeches, as_of).join(right, on="speech_id", how="left")


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

class NerJob(Job):
    """LLM provenance NER over a cb_speeches artifact -> cb_speech_ner."""

    kind = KIND
    pipeline_version = PIPELINE_VERSION

    def __init__(self, speeches: Artifact, backend: ChatBackend, plan: dict[str, Any], *,
                 temp: bool = False, done_before: pl.DataFrame | None = None) -> None:
        if speeches.kind != SPEECHES_KIND:
            raise JobError(f"{speeches.artifact_id} is a {speeches.kind}, not {SPEECHES_KIND}")
        self.speeches, self.backend, self.plan, self.temp = speeches, backend, plan, temp
        self.layout = layout_from_hyperparams(speeches.meta.hyperparams)
        self.vintage = parse_vintage(plan["vintage"])
        todo = read_speeches(speeches, parse_vintage(plan["source_vintage"]),
                             columns=["speech_id", *INPUT_FIELDS])
        rows = todo.to_dicts()
        todo = todo.with_columns(
            pl.Series("input_sha256", [input_sha256(r) for r in rows], dtype=pl.String),
            pl.Series("key", [period_key(r["date"], self.layout.freq) for r in rows],
                      dtype=pl.String))
        if done_before is not None and done_before.height:
            todo = todo.join(done_before.select(["speech_id", "input_sha256"]),
                             on=["speech_id", "input_sha256"], how="anti")
        self.todo = todo
        self._counts = {"rows": 0, "errors": 0}

    # -- construction -------------------------------------------------------

    @staticmethod
    def make_plan(mode: str, speeches: Artifact, vintage: datetime) -> dict[str, Any]:
        plans = plan_files(speeches.path)
        if not plans:
            raise JobError(f"{speeches.artifact_id} has no plan file (not a cb_speeches "
                           "artifact written by this pipeline)")
        return {"mode": mode, "vintage": vintage_tag(vintage),
                "source_vintage": json.loads(plans[-1].read_text())["vintage"]}

    @classmethod
    def new(cls, speeches: Artifact, backend: ChatBackend, *, temp: bool = False,
            clock: Clock = utc_now) -> NerJob:
        if speeches.partial:
            raise JobError(f"{speeches.artifact_id} is partial")
        return cls(speeches, backend, cls.make_plan("base", speeches, clock()), temp=temp)

    @property
    def mode(self) -> str:
        return self.plan["mode"]

    def params(self) -> dict[str, Any]:
        return {"speeches_id": self.speeches.artifact_id, "provider": self.backend.provider.name,
                "model": self.backend.model, "temperature": self.backend.temperature,
                "seed": self.backend.seed, "prompt_sha256": prompt_sha256(),
                **self.layout.hyperparams()}

    def sources(self) -> list[Any]:
        return [self.speeches]

    def model_card(self) -> ModelCard:
        from datalake import ModelCard

        served = self.backend.served
        return ModelCard(
            model_id=self.backend.model,
            version=served[:19] if served else "api",
            repo=f"{self.backend.provider.name}:{self.backend.info()['endpoint']}",
            weights_public=False, architecture="chat-completion",
            notes="generative extraction through an OpenAI-compatible endpoint",
            backend=self.backend.provider.name, serving=self.backend.info())

    def backends(self) -> list[Any]:
        return [self.backend]

    def notes(self) -> str:
        return (f"{self.mode} vintage={self.plan['vintage']} "
                f"source_vintage={self.plan['source_vintage']}")

    # -- units ---------------------------------------------------------------

    def _path(self, out_dir: Path, key: str) -> Path:
        if self.mode == "update":
            return out_dir / f"{UPDATE_PREFIX}{self.plan['vintage']}-{partition_file(key)}"
        return out_dir / partition_file(key)

    def units(self) -> list[Unit]:
        return [Unit(k) for k in sorted(set(self.todo["key"].to_list()))]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return self._path(out_dir, unit.key).exists()

    @contextmanager
    def session(self, ctx: JobContext) -> Iterator[None]:
        path = ctx.out_dir / f"{PLAN_PREFIX}{self.plan['vintage']}.json"
        if not path.exists():
            _write_atomic(path, lambda tmp: tmp.write_text(
                json.dumps(self.plan, indent=2, sort_keys=True)))
        yield

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        rows = self.todo.filter(pl.col("key") == unit.key).sort(["date", "speech_id"]).to_dicts()
        results = self.backend.complete_many(
            [(SYSTEM_PROMPT, user_prompt(r)) for r in rows], RESPONSE_SCHEMA, name=SCHEMA_NAME)
        out = pd.DataFrame([{
            "speech_id": r["speech_id"], "input_sha256": r["input_sha256"], "date": r["date"],
            **{f: (res.data or {}).get(f) for f in FIELDS},
            "provider": self.backend.provider.name, "response_model": res.response_model,
            "system_fingerprint": res.system_fingerprint, "finish_reason": res.finish_reason,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "attempts": res.attempts, "error": res.error,
            "vintage": pd.Timestamp(self.vintage)} for r, res in zip(rows, results)],
            columns=NER_SCHEMA.names)
        write_table(self._path(ctx.out_dir, unit.key), out, NER_SCHEMA, {})
        n_err = int(out["error"].notna().sum())
        self._counts["rows"] += len(out)
        self._counts["errors"] += n_err
        ctx.log.info("%s: %d speech(es), %d error(s)", unit.key, len(out), n_err)

    def finalize(self, ctx: JobContext) -> None:
        ctx.note(f"{self.mode} {self.plan['vintage']}: {self._counts['rows']} row(s) this "
                 f"execution, {self._counts['errors']} error(s); served={self.backend.served}")

    # -- rebuild -------------------------------------------------------------

    @staticmethod
    def _backend(artifact: Artifact, **kwargs: Any) -> ChatBackend:
        """The recorded provider/model, pinned to the recorded snapshot or digest."""
        from nlp.llm import ChatBackend

        hp = artifact.meta.hyperparams
        backend = ChatBackend(hp["provider"], hp["model"], temperature=hp["temperature"],
                              seed=hp["seed"], **kwargs)
        served = None
        if backend.provider.name == "ollama":
            card = artifact.meta.model_card
            served = (card.serving or {}).get("served") if card else None
        else:
            for path in data_files(artifact.path):
                models = pl.read_parquet(path, columns=["response_model"])["response_model"]
                models = models.drop_nulls()
                if models.len():
                    served = models[0]
                    break
        if served:
            backend.expect_served(served)
        return backend

    @staticmethod
    def _check_prompt(artifact: Artifact) -> None:
        recorded = artifact.meta.hyperparams["prompt_sha256"]
        if recorded != prompt_sha256():
            raise JobError(f"{artifact.artifact_id} was produced with prompt {recorded[:12]}, "
                           f"this code has {prompt_sha256()[:12]}: bump the pipeline version")

    @staticmethod
    def _done(artifact: Artifact, before: str | None) -> pl.DataFrame:
        """(speech_id, input_sha256) recorded by executions before vintage ``before``."""
        frames = [pl.read_parquet(p, columns=["speech_id", "input_sha256", "vintage"])
                  for p in data_files(artifact.path)]
        if not frames:
            return pl.DataFrame(schema={"speech_id": pl.String, "input_sha256": pl.String})
        df = pl.concat(frames)
        if before is not None:
            df = df.filter(pl.col("vintage") < parse_vintage(before))
        return df.select(["speech_id", "input_sha256"])

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: DatalakeIndex, **kwargs: Any) -> NerJob:
        cls._check_prompt(artifact)
        speeches = index.get(artifact.meta.hyperparams["speeches_id"])
        backend = cls._backend(artifact, **kwargs)
        plans = plan_files(artifact.path)
        temp = artifact.meta.pipeline_version.endswith(TEMP_SUFFIX)
        if not plans:
            return cls.new(speeches, backend, temp=temp)
        plan = json.loads(plans[-1].read_text())
        done = cls._done(artifact, plan["vintage"]) if plan["mode"] == "update" else None
        return cls(speeches, backend, plan, temp=temp, done_before=done)

    @classmethod
    def for_update(cls, artifact: Artifact, index: DatalakeIndex, *, clock: Clock = utc_now,
                   **kwargs: Any) -> NerJob:
        """NER for speeches not yet in the table, as of the source's latest vintage."""
        from datalake.lineage import require_lineage

        cls._check_prompt(artifact)
        speeches = index.get(artifact.meta.hyperparams["speeches_id"])
        if speeches.partial:
            raise JobError(f"{speeches.artifact_id} is partial: finish its update first")
        require_lineage([(artifact, {SPEECHES_KIND: speeches.artifact_id})],
                        what=f"{KIND} update")
        plan = cls.make_plan("update", speeches, clock())
        return cls(speeches, cls._backend(artifact, **kwargs), plan,
                   temp=artifact.meta.pipeline_version.endswith(TEMP_SUFFIX),
                   done_before=cls._done(artifact, None))

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        from nlp.llm import PROVIDERS

        parser.add_argument("--speeches-id", required=True, help="cb_speeches artifact id")
        parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS),
                            help="endpoint and key are read from .env by provider")
        parser.add_argument("--model", required=True, help="served model name (no default)")
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--seed", type=int, default=0, help="sent where supported")
        parser.add_argument("--workers", type=int, default=8, help="concurrent requests")
        parser.add_argument("--max-attempts", type=int, default=3,
                            help="tries per speech when the reply fails validation")
        parser.add_argument("--temp", action="store_true", help="agent-created (__TEMP)")

    @classmethod
    def from_args(cls, args: Any, index: DatalakeIndex) -> NerJob:
        from nlp.llm import ChatBackend

        backend = ChatBackend(args.provider, args.model, temperature=args.temperature,
                              seed=args.seed, workers=args.workers,
                              max_attempts=args.max_attempts)
        return cls.new(index.get(args.speeches_id), backend, temp=args.temp)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def verify_artifact(artifact: Artifact) -> list:
    """Schema, ids subset of the source, value domains, uniqueness, error rate."""
    import re

    import pyarrow.parquet as pq

    from datalake import DatalakeError, DatalakeIndex
    from datalake.verify import Finding, Severity

    aid = artifact.artifact_id
    findings: list[Finding] = []

    def add(sev: Severity, msg: str) -> None:
        findings.append(Finding(sev, aid, msg))

    frames = []
    for path in data_files(artifact.path):
        if not pq.read_schema(path).remove_metadata().equals(NER_SCHEMA):
            add(Severity.ERROR, f"{path.name}: schema differs from NER_SCHEMA")
            continue
        frames.append(pl.read_parquet(path))
    if not frames:
        add(Severity.ERROR, "no NER files")
        return findings
    df = pl.concat(frames)
    if df.select(["speech_id", "input_sha256"]).is_duplicated().any():
        add(Severity.ERROR, "duplicate (speech_id, input_sha256) rows")
    ok = df.filter(pl.col("error").is_null())
    if (ok.select([pl.col(f).is_null().any() for f in FIELDS]).row(0)).count(True):
        add(Severity.ERROR, "rows without error have null fields")
    if not set(ok["sentiment"].unique().to_list()) <= set(
            RESPONSE_SCHEMA["properties"]["sentiment"]["enum"]):
        add(Severity.ERROR, "sentiment outside the enum")
    iso = re.compile(r"^[A-Z]{2}$")
    bad_cc = sum(1 for c in ok["country_code"].to_list() if not iso.match(c or ""))
    if bad_cc:
        add(Severity.WARNING, f"{bad_cc} country_code value(s) are not ISO alpha-2")
    n_err = df.height - ok.height
    if n_err:
        add(Severity.WARNING, f"{n_err}/{df.height} row(s) failed extraction")
    root = artifact.path.parents[2]
    try:
        with DatalakeIndex(root, create=False) as index:
            source = index.get(artifact.meta.hyperparams["speeches_id"])
            ids = set(read_state(source.path, columns=("speech_id",))["speech_id"].to_list())
            ids |= {i for p in data_files(source.path)
                    for i in pl.read_parquet(p, columns=["speech_id"])["speech_id"].to_list()}
    except (DatalakeError, OSError) as exc:
        add(Severity.WARNING, f"source not checked: {exc}")
        return findings
    missing = set(df["speech_id"].to_list()) - ids
    if missing:
        add(Severity.ERROR, f"{len(missing)} speech_id(s) not in the source, e.g. "
                            f"{sorted(missing)[:5]}")
    return findings


__all__ = ["KIND", "PIPELINE_VERSION", "NerJob", "prompt_sha256", "read_ner",
           "speeches_with_ner", "verify_artifact"]
