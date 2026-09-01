"""Datalake artifact metadata types.

Three dataclasses:

  ModelCard  -- describes a model used to produce an artifact.  For models on
                the HuggingFace Hub, prefer `ModelCard.from_hf()` which pulls
                the Hub's own metadata; this class is the fallback for models
                whose weights live locally and cannot be published.

  RunMeta    -- everything needed to reproduce one pipeline run: code commit,
                hyperparameters, input artifact IDs, timing.  Serialised to
                meta.json alongside the run's output files.

  Artifact   -- a registered, on-disk artifact: RunMeta plus its resolved path
                and per-file hashes.  Returned by DatalakeIndex lookups.

Identity
--------
An artifact's ID is a human-readable slug, deterministic given its RunMeta:

    {kind}__{model_id}-{model_version}__{pipeline_version}__{param_slug}__{YYYYMMDD}

Slugs sort sensibly, are greppable, and survive being read out loud.  They are
NOT content hashes: two runs with identical parameters on different days get
different IDs.  That is deliberate; re-running is a new artifact, never an
overwrite.  Content identity lives in the per-file hashes.
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

# Slug components must survive being a directory name on POSIX and a key in
# JSON.  Anything outside this set is replaced with '-'.
_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Separator between slug components.  Double underscore so single underscores
# inside a component (e.g. "pos_neu") stay unambiguous.
_SEP = "__"


def slugify(value: str) -> str:
    """Reduce an arbitrary string to a slug-safe component."""
    cleaned = _SLUG_SAFE.sub("-", str(value)).strip("-")
    return cleaned or "none"


def _format_param_value(value: Any) -> str:
    """Render a hyperparameter value compactly for use inside a slug.

    Floats drop trailing zeros so tau=0.20 and tau=0.2 produce the same slug
    (they are the same parameter).  Bools render as true/false, not True/False.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Normalise 0.20 -> 0.2, 97.50 -> 97.5, 1.0 -> 1
        text = f"{value:g}"
        return text
    return slugify(value)


# A single parameter longer than this is replaced by a short digest of its
# value.  Long values (file paths, column lists) still contribute to identity,
# but through a hash rather than by being spelled out in a directory name.
_MAX_PARAM_VALUE_CHARS = 24

# Hard ceiling on the whole parameter component.  POSIX filenames cap at 255
# bytes and the slug carries other components too, so past this the whole
# parameter set collapses to one digest.  The full values always remain
# readable in meta.json; the slug is an identifier, not the record.
_MAX_PARAM_SLUG_CHARS = 96

_DIGEST_CHARS = 8


def _short_digest(text: str) -> str:
    """Stable short digest for slug use.

    blake2b rather than hash() because Python's hash() is salted per process,
    which would make artifact IDs non-deterministic across runs.
    """
    import hashlib

    return hashlib.blake2b(
        text.encode(), digest_size=_DIGEST_CHARS // 2
    ).hexdigest()


def param_slug(hyperparams: dict[str, Any]) -> str:
    """Deterministic, length-bounded slug for a hyperparameter dict.

    Keys are sorted so slug construction does not depend on dict insertion
    order.  An empty dict yields an empty string, which the caller drops.

    Values too long to inline are replaced by a digest of the value, and if
    the assembled slug is still too long the whole set collapses to a single
    digest.  Both fallbacks stay deterministic, so the same parameters always
    yield the same ID.  The unabbreviated values are always recorded in
    meta.json regardless.
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
        # Digest the canonical rendering of the whole dict, not the truncated
        # slug, so two distinct parameter sets cannot collide by sharing a
        # prefix.
        canonical = repr(sorted(hyperparams.items()))
        slug = f"params{_short_digest(canonical)}"
    return slug


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """Describes a model used to produce an artifact.

    weights_public distinguishes two reproducibility regimes:
      True  -- weights are fetchable from `repo`; anyone can rerun.
      False -- weights are local-only; `weights_sha256` lets someone who
               obtains them independently verify they have the same ones.

    `commit` is the model repo's git SHA, filled at run time where available.
    """

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
            # Not fatal: a local model may be described before its weights are
            # hashed.  But it removes the only verification handle a third
            # party would have, so make the gap visible rather than silent.
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
# RunMeta
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    """Reproducibility record for one pipeline run.

    `partial` is True from run start until the run completes cleanly.  Any
    artifact left with partial=True is a crashed run: downstream code must
    ignore it, and `datalake ls` flags it.

    `sources` holds the artifact IDs this run consumed.  The index turns these
    into lineage edges, which is what makes "which scores are stale after I
    re-embed?" answerable.
    """

    kind: str
    pipeline: str
    pipeline_version: str
    pipeline_commit: str | None = None
    pipeline_repo: str | None = None
    hyperparams: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    model_card: ModelCard | None = None
    run_start: str = ""
    run_end: str | None = None
    partial: bool = True
    deprecated: bool = False
    deprecation_reason: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.run_start:
            self.run_start = utc_now_iso()

    @property
    def artifact_id(self) -> str:
        """Deterministic slug identifying this run."""
        parts: list[str] = [slugify(self.kind)]
        if self.model_card is not None:
            parts.append(self.model_card.slug)
        parts.append(slugify(self.pipeline_version))
        params = param_slug(self.hyperparams)
        if params:
            parts.append(params)
        parts.append(self.run_start[:10].replace("-", ""))
        return _SEP.join(parts)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # asdict recurses into ModelCard already; keep None as None.
        if self.model_card is None:
            data["model_card"] = None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMeta:
        payload = dict(data)
        card = payload.pop("model_card", None)
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in payload.items() if k in known}
        return cls(
            **filtered,
            model_card=ModelCard.from_dict(card) if card else None,
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


# ---------------------------------------------------------------------------
# Time helper
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Current UTC time as an ISO8601 string with second precision.

    Second precision (not microsecond) because this value feeds the artifact
    slug via its date prefix, and because sub-second timing carries no
    reproducibility information.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
