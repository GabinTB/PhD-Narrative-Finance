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

from narrative_scoring.corrections import Correction

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


class SentimentSplit(str, Enum):
    NONE = "none"    # one "all" row per (day, narrative)
    SIGN = "sign"    # "all" plus "pos" / "neg" (and "neu" when neutral_eps > 0)


SENTIMENT_ALL = "all"


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
        sentiment_split: ``none`` (default) or ``sign``; see pipeline.py.
        neutral_eps: |sentiment| <= eps is "neu" (only when > 0).
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
    sentiment_split: SentimentSplit = SentimentSplit.NONE
    neutral_eps: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        # accept the string form of every enum (CLI / JSON round trips)
        for name, enum_type in (("mode", Correction), ("paraphrase_style", ParaphraseStyle),
                                ("paraphrase_pooling", PoolRule), ("narrative_agg", AggRule),
                                ("sentiment_split", SentimentSplit)):
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
        if self.neutral_eps < 0.0:
            raise ValueError("neutral_eps must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def digest(self) -> str:
        """Stable hash of every numerical choice (``label`` excluded)."""
        d = self.to_dict()
        d.pop("label")
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def f0_digest(self) -> str:
        """Hash of the choices the null model depends on: mode, pooling, style, master,
        trim, alpha, draws per headline. Selection/aggregation choices do not enter."""
        d = {k: self.to_dict()[k] for k in (
            "mode", "paraphrase_style", "paraphrase_pooling", "include_master",
            "trim_frac", "alpha", "null_draws_per_headline")}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def warmup_digest(self) -> str:
        """Hash of what N_eff depends on: mode, pooling, style, master inclusion."""
        d = {k: self.to_dict()[k] for k in (
            "mode", "paraphrase_style", "paraphrase_pooling", "include_master")}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def sentiment_labels(self) -> tuple[str, ...]:
        if self.sentiment_split is SentimentSplit.NONE:
            return ()
        return ("pos", "neg", "neu") if self.neutral_eps > 0 else ("pos", "neg")


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
