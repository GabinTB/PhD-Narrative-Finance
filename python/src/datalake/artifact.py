"""Datalake artifact metadata types.

Four dataclasses:

  ModelCard  -- describes a model used to produce an artifact.  For models on
                the HuggingFace Hub, prefer pulling the Hub's own metadata;
                this class is the fallback for models whose weights live
                locally and cannot be published.

  RunRecord  -- one execution that contributed to an artifact.  Carries its
                own git commit, timestamps, and the list of outputs it wrote.
                An artifact may have several: a completed artifact that later
                gains more data appends a new RunRecord rather than mutating
                history, so each execution's code version stays auditable.

  RunMeta    -- the artifact as a whole: identity (kind, hyperparams, model),
                lineage sources, a verifier reference, and the list of
                RunRecords.  Serialised to meta.json.

  Artifact   -- a registered, on-disk artifact: RunMeta plus its resolved path
                and per-file hashes.  Returned by DatalakeIndex lookups.

Identity
--------
An artifact's ID is a human-readable slug, deterministic given its identity
fields (kind, model, pipeline_version, hyperparams, creation date):

    {kind}__{model_id}-{model_version}__{pipeline_version}__{param_slug}__{YYYYMMDD}

Slugs sort sensibly, are greppable, and survive being read out loud.  They are
NOT content hashes: two artifacts with identical parameters created on
different days get different IDs.  Content identity lives in the file hashes.

Completeness
------------
An artifact is complete when its last RunRecord has partial=False.  Any
artifact whose last record is still partial is a crashed or in-progress run:
downstream code must ignore it, and verification flags it.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Slug construction
# ---------------------------------------------------------------------------

_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_SEP = "__"

_MAX_PARAM_VALUE_CHARS = 24
_MAX_PARAM_SLUG_CHARS = 96
_DIGEST_CHARS = 8


def slugify(value: str) -> str:
    """Reduce an arbitrary string to a slug-safe component."""
    cleaned = _SLUG_SAFE.sub("-", str(value)).strip("-")
    return cleaned or "none"


def _short_digest(text: str) -> str:
    """Stable short digest for slug use (process-independent)."""
    import hashlib
    return hashlib.blake2b(text.encode(), digest_size=_DIGEST_CHARS // 2).hexdigest()


def _format_param_value(value: Any) -> str:
    """Render a hyperparameter value compactly for a slug."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return slugify(value)


def param_slug(hyperparams: dict[str, Any]) -> str:
    """Deterministic, length-bounded slug for a hyperparameter dict.

    Keys are sorted so construction does not depend on insertion order.  Values
    too long to inline are digested; if the whole slug is still too long it
    collapses to one digest of the canonical dict.  Both fallbacks stay
    deterministic.  Full values are always recorded in meta.json regardless.
    """
    if not hyperparams:
        return ""

    parts: list[str] = []
    for key, value in sorted(hyperparams.items()):
        rendered = _format_param_value(value)
        if len(rendered) > _MAX_PARAM_VALUE_CHARS:
            rendered = _short_digest(f"{key}={value!r}")
        parts.append(f"{slugify(key)}{rendered}")

    slug = "_".join(parts)
    if len(slug) > _MAX_PARAM_SLUG_CHARS:
        canonical = repr(sorted(hyperparams.items()))
        slug = f"params{_short_digest(canonical)}"
    return slug


# ---------------------------------------------------------------------------
# Time helper
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Current UTC time as ISO8601 with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """Describes a model used to produce an artifact."""

    model_id: str
    version: str
    repo: str
    weights_public: bool
    weights_sha256: str | None = None
    commit: str | None = None
    architecture: str | None = None
    dim: int | None = None
    pooling: str | None = None
    trained_on: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.weights_public and not self.weights_sha256:
            self.notes = (
                (self.notes + " " if self.notes else "")
                + "[warning: private weights with no weights_sha256; "
                "third parties cannot verify weight identity]"
            ).strip()

    @property
    def slug(self) -> str:
        return f"{slugify(self.model_id)}-{slugify(self.version)}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """One execution that contributed outputs to an artifact.

    `produced` lists the output file names this execution wrote (just this
    run's contribution, not the whole artifact).  The union of produced lists
    across all records is what a verifier checks against the declared scope.

    `partial` is True from the moment the record is created until the execution
    completes cleanly.  A record left partial is a crashed run.
    """

    run_start: str
    pipeline_version: str
    pipeline_commit: str | None = None
    run_end: str | None = None
    partial: bool = True
    produced: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# RunMeta
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    """Reproducibility record for one artifact across all its executions."""

    kind: str
    pipeline: str
    pipeline_version: str
    pipeline_repo: str | None = None
    hyperparams: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    model_card: ModelCard | None = None
    verifier: str | None = None            # entry-point name for content verification
    runs: list[RunRecord] = field(default_factory=list)
    deprecated: bool = False
    deprecation_reason: str | None = None
    notes: str = ""
    # Frozen at creation so the artifact_id stays stable across executions.
    created: str = ""

    def __post_init__(self) -> None:
        if not self.created:
            self.created = utc_now_iso()

    # -- aggregate state ---------------------------------------------------

    @property
    def partial(self) -> bool:
        """An artifact is complete only when its last run finished cleanly."""
        if not self.runs:
            return True
        return self.runs[-1].partial

    @property
    def run_start(self) -> str:
        """Start of the first execution."""
        return self.runs[0].run_start if self.runs else self.created

    @property
    def run_end(self) -> str | None:
        """End of the last execution, or None if still partial."""
        if not self.runs:
            return None
        return self.runs[-1].run_end

    @property
    def pipeline_commit(self) -> str | None:
        """Commit of the last execution."""
        return self.runs[-1].pipeline_commit if self.runs else None

    @property
    def produced(self) -> list[str]:
        """Union of all outputs across every execution, sorted and deduped."""
        seen: set[str] = set()
        for record in self.runs:
            seen.update(record.produced)
        return sorted(seen)

    # -- identity ----------------------------------------------------------

    @property
    def artifact_id(self) -> str:
        parts: list[str] = [slugify(self.kind)]
        if self.model_card is not None:
            parts.append(self.model_card.slug)
        parts.append(slugify(self.pipeline_version))
        params = param_slug(self.hyperparams)
        if params:
            parts.append(params)
        parts.append(self.created[:10].replace("-", ""))
        return _SEP.join(parts)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pipeline": self.pipeline,
            "pipeline_version": self.pipeline_version,
            "pipeline_repo": self.pipeline_repo,
            "hyperparams": dict(self.hyperparams),
            "sources": list(self.sources),
            "model_card": self.model_card.to_dict() if self.model_card else None,
            "verifier": self.verifier,
            "runs": [r.to_dict() for r in self.runs],
            "deprecated": self.deprecated,
            "deprecation_reason": self.deprecation_reason,
            "notes": self.notes,
            "created": self.created,
            # Denormalised aggregates for human/tool convenience (never read back).
            "partial": self.partial,
            "run_start": self.run_start,
            "run_end": self.run_end,
            "pipeline_commit": self.pipeline_commit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMeta:
        payload = dict(data)
        card = payload.pop("model_card", None)
        runs_raw = payload.pop("runs", [])
        for derived in ("partial", "run_start", "run_end", "pipeline_commit"):
            payload.pop(derived, None)
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in payload.items() if k in known}
        return cls(
            **filtered,
            model_card=ModelCard.from_dict(card) if card else None,
            runs=[RunRecord.from_dict(r) for r in runs_raw],
        )


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

@dataclass
class Artifact:
    """A registered artifact on disk: its metadata, path, and file hashes."""

    artifact_id: str
    layer: str                     # 'raw' | 'derived' | 'output'
    path: Path
    meta: RunMeta
    file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.meta.kind

    @property
    def partial(self) -> bool:
        return self.meta.partial

    @property
    def deprecated(self) -> bool:
        return self.meta.deprecated

    def glob(self, pattern: str = "*.parquet") -> str:
        """A glob string ready to hand to DuckDB's read_parquet()."""
        return str(self.path / pattern)

    def files(self, pattern: str = "*.parquet") -> list[Path]:
        return sorted(self.path.glob(pattern))

    def __repr__(self) -> str:
        flags = []
        if self.partial:
            flags.append("PARTIAL")
        if self.deprecated:
            flags.append("DEPRECATED")
        suffix = f" [{','.join(flags)}]" if flags else ""
        return f"<Artifact {self.artifact_id}{suffix}>"
