"""Typed, immutable configuration and run metadata for the canonical scorer.

Every modelling choice the scorer makes is a field here with an explicit
default, and every field is copied verbatim into :class:`RunMetadata`, which
is persisted next to each run. Nothing is inferred silently: the paraphrase
pooling rule, for instance, is chosen by the caller (with ``default_pooling``
as the documented recommendation per paraphrase style) and recorded as used.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from narrative_scoring.tags import SentimentTags
from nlp.corrections import Correction

SPEC_PATH = "python/doc/narratives.md"


class ParaphraseStyle(str, Enum):
    HEADLINE = "headline"      # distinct surfacings of one mechanism
    SEMANTIC = "semantic"      # definition-preserving replicates of one sentence


class PoolRule(str, Enum):
    """How the master description + K paraphrase scores collapse to one primitive score."""

    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"


class AggRule(str, Enum):
    """How a headline's retained primitives collapse to one narrative score."""

    MEAN = "mean"
    MEDIAN = "median"


PERCENTILE_AXIS = "row_wise"   # the only axis the canonical scorer implements


def default_pooling(style: ParaphraseStyle) -> PoolRule:
    """Recommended pooling per paraphrase style: MAX for headline, MEAN for semantic."""
    return PoolRule.MAX if style is ParaphraseStyle.HEADLINE else PoolRule.MEAN


def n_candidates_for(q: float, n_primitives: int) -> int:
    """Per-row candidate budget implied by the percentile ``q``: ceil((1 - q) * P), >= 1.

    Owner decision (final): with q = 0.99 and P = 1508 this is 16. The budget
    is fixed by P alone, never by how many scores clear tau, which is what
    makes it a stable per-headline candidate set (see selection.py).
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be a fraction in (0, 1), got {q!r}")
    # epsilon guard: (1 - 0.99) * 100 is 1.0000000000000009 in binary, which must not ceil to 2
    return max(1, int(math.ceil((1.0 - q) * n_primitives - 1e-9)))


SENTIMENT_ALL = "all"          # narrative_daily SENTIMENT value of a run without tags
MODEL_SOURCES = ("ravenbert", "finbert")      # headline_sentiment grid tables (P_00..P_40)
# what a config without tags hashed as before any sentiment layer existed: kept so the
# config_id of every existing all-headlines run stays the same
_LEGACY_NO_SENTIMENT = {"sentiment_split": "none", "neutral_eps": 0.0,
                        "sentiment_source": "none", "sentiment_column": ""}
# fields of earlier config forms, read back by from_dict
_LEGACY_SPLIT = ("sentiment_split", "neutral_eps", "sentiment_column")
_LEGACY_BUCKET = ("sentiment", "sentiment_source", "sentiment_rule", "neg_max", "pos_min",
                  "min_conf")


@dataclass(frozen=True)
class ScoringConfig:
    """Everything that determines the numbers, and nothing that only determines speed.

    Attributes:
        mode: embedding correction applied identically to headlines and primitive texts.
        paraphrase_style: which paraphrase file the primitive texts came from.
        paraphrase_pooling: pool rule over master + K paraphrases; see ``default_pooling``.
        q: percentile for the per-row candidate budget (fraction, default 0.99).
        narrative_agg: primitive -> narrative rule within a headline.
        jump_cut: enable the optional largest-gap prefix cut after q + tau (OFF by default).
        jump_min_candidates: jump cut is evaluated only when the retained count EXCEEDS this.
        alpha: F0 family-wise level used to calibrate tau.
        trim_frac: per-headline top trim when pooling null draws.
        include_master: whether the master description is one of the primitive texts.
        null_draws_per_headline: kept null draws sampled per headline into the month digest.
        min_month_draws: the tau job extends its window backwards until the merged pool
            holds at least this many draws (tau_asof.py).
        gap_alert_threshold: |tau_gauss - tau_empirical| above this is logged (warning only).
        tags: the sentiment layer (``narrative_scoring.tags.SentimentTags``): score source
            (a ``headline_sentiment`` grid table, ``ravenbert`` | ``finbert``), ordinal rule
            and disjoint intervals of the score, one output label each. In ONE pass every
            headline is scored and feeds the null model; a tagged headline also feeds its
            tag's rows, an untagged one (no interval, or no score) feeds no row and is
            counted. ``None``: one label, ``all``, every headline. The tags enter
            ``digest`` only (never ``f0_digest`` / ``warmup_digest``: the null model is the
            same whatever the tags), and a config without tags hashes exactly as before
            any sentiment layer existed.
        mask_bipolar: per headline, keep one pole of every bipolar pair (two signed
            SUB_TYPEs under one TOPIC/GROUP/TYPE): the pole with the higher mean of its
            top-3 primitive scores (first pole in table order on an exact tie); every
            primitive of the other pole is set to -inf before selection, the F0 trim and the
            null draws (selection.apply_pole_mask). The two pole rows then add up exactly to
            the narrative level (validation.merge_poles). It enters ``digest``,
            ``f0_digest`` and ``warmup_digest`` only when True, so every config without it
            hashes as before it existed.
    """

    mode: Correction = Correction.R2
    paraphrase_style: ParaphraseStyle = ParaphraseStyle.HEADLINE
    paraphrase_pooling: PoolRule = PoolRule.MAX
    q: float = 0.99
    narrative_agg: AggRule = AggRule.MEAN
    jump_cut: bool = False
    jump_min_candidates: int = 10
    alpha: float = 0.01
    trim_frac: float = 0.10
    include_master: bool = True
    null_draws_per_headline: int = 64
    min_month_draws: int = 20_000_000
    gap_alert_threshold: float = 0.05
    tags: SentimentTags | None = None
    mask_bipolar: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        # accept the string form of every enum (CLI / JSON round trips)
        for name, enum_type in (("mode", Correction), ("paraphrase_style", ParaphraseStyle),
                                ("paraphrase_pooling", PoolRule), ("narrative_agg", AggRule)):
            value = getattr(self, name)
            if not isinstance(value, enum_type):
                object.__setattr__(self, name, enum_type(value))
        if not 0.0 < self.q < 1.0:
            raise ValueError(f"q must be a fraction in (0, 1), got {self.q!r}")
        if self.jump_min_candidates < 1:
            raise ValueError("jump_min_candidates must be >= 1")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 <= self.trim_frac < 1.0:
            raise ValueError("trim_frac must be in [0, 1)")
        if self.null_draws_per_headline < 1:
            raise ValueError("null_draws_per_headline must be >= 1")
        if self.min_month_draws < 1:
            raise ValueError("min_month_draws must be >= 1")
        if self.gap_alert_threshold <= 0:
            raise ValueError("gap_alert_threshold must be > 0")
        if isinstance(self.tags, dict):
            object.__setattr__(self, "tags", SentimentTags.from_dict(self.tags))
        if self.tags is not None:
            if not isinstance(self.tags, SentimentTags):
                raise ValueError(f"tags must be SentimentTags or its dict, got {self.tags!r}")
            if self.tags.source not in MODEL_SOURCES:
                raise ValueError(f"tags source must be one of {MODEL_SOURCES} (a grid table), "
                                 f"got {self.tags.source!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoringConfig:
        """A config from its ``to_dict`` form, including the forms recorded by older runs:
        the in-run split (``sentiment_split`` ...) and the bucket filter (``sentiment``
        ...), readable when they selected every headline, refused otherwise."""
        d = dict(d)
        legacy = {k: d.pop(k) for k in _LEGACY_SPLIT if k in d}
        if legacy.get("sentiment_split", "none") != "none":
            raise ValueError("this run used the removed in-run sentiment split "
                             f"({legacy}); it cannot be rebuilt")
        bucket = {k: d.pop(k) for k in _LEGACY_BUCKET if k in d}
        if bucket.get("sentiment", "none") != "none":
            raise ValueError("this run used the removed single-bucket sentiment filter "
                             f"({bucket}); it cannot be rebuilt")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        d["tags"] = self.tags.to_dict() if self.tags is not None else None
        return d

    def _hashable(self) -> dict[str, Any]:
        d = self.to_dict()
        d.pop("label")
        _drop_default_mask(d)
        if self.tags is None:
            d.pop("tags")
            d.update(_LEGACY_NO_SENTIMENT)
        return d

    def digest(self) -> str:
        """Stable hash of every numerical choice (``label`` excluded). A config without
        tags hashes as it did before any sentiment layer existed."""
        return hashlib.sha1(json.dumps(self._hashable(), sort_keys=True).encode()).hexdigest()[:16]

    def f0_digest(self) -> str:
        """Hash of the choices the null model depends on: mode, pooling, style, master,
        trim, alpha, draws per headline. Selection/aggregation choices do not enter."""
        d = {k: self.to_dict()[k] for k in (
            "mode", "paraphrase_style", "paraphrase_pooling", "include_master",
            "trim_frac", "alpha", "null_draws_per_headline", "mask_bipolar")}
        _drop_default_mask(d)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def warmup_digest(self) -> str:
        """Hash of what N_eff depends on: mode, pooling, style, master inclusion."""
        d = {k: self.to_dict()[k] for k in (
            "mode", "paraphrase_style", "paraphrase_pooling", "include_master",
            "mask_bipolar")}
        _drop_default_mask(d)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def labels(self) -> list[str]:
        """The narrative_daily SENTIMENT values of this run, in tag order: the tag names,
        or ``["all"]`` without tags."""
        return self.tags.names if self.tags is not None else [SENTIMENT_ALL]


def _drop_default_mask(d: dict[str, Any]) -> None:
    """``mask_bipolar`` off hashes as before the field existed."""
    if not d.get("mask_bipolar"):
        d.pop("mask_bipolar", None)


@dataclass
class RunMetadata:
    """Persisted with every run so a backtest is replayable exactly."""

    config: dict[str, Any]
    percentile_axis: str
    n_candidates: int
    n_primitives: int
    n_narratives: int
    n_primitive_texts: int
    k_paraphrases: int
    embedding_dim: int
    taxonomy_name: str
    taxonomy_sha1: str
    paraphrase_sha1: str
    primitive_embeddings_digest: str
    mu_asof_id: str | None
    mu_policy: str
    tau_source_id: str            # tau_asof artifact id (or a test provider's id)
    tau_policy: str
    n_eff: float                  # of the first tau row used; per-day values in day_diagnostics
    seed: int
    code_version: str | None
    sentiment_artifact_id: str | None = None   # headline_sentiment artifact behind the tags
    # what produced both sides of S = H @ P.T: primitive-text embeddings (backend,
    # serving metadata, checks) and the headline_embeddings artifact + model card
    embeddings_provenance: dict[str, Any] | None = None
    config_id: str = ""
    f0_config_id: str = ""
    spec: str = SPEC_PATH
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat(self) -> dict[str, Any]:
        """One row: config fields hoisted to the top level (for comparison tables)."""
        row = dict(self.config)
        for k, v in self.to_dict().items():
            if k not in ("config", "extra"):
                row[k] = v
        row.update(self.extra)
        return row
