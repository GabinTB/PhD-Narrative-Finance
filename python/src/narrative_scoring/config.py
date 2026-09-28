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


class SentimentFilter(str, Enum):
    """Which headlines a run scores. ``none`` = all of them; the four others are the
    sentiment buckets, which partition the headlines (see ``ScoringConfig``)."""

    NONE = "none"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    UNSCORED = "unscored"


SENTIMENT_ALL = "all"          # narrative_daily SENTIMENT value of a ``none`` run
SENTIMENT_NONE = "none"
SENTIMENT_BUCKETS = (SentimentFilter.POSITIVE, SentimentFilter.NEUTRAL,
                     SentimentFilter.NEGATIVE, SentimentFilter.UNSCORED)
MODEL_SOURCES = ("ravenbert", "finbert")      # grid tables (P_00..P_40)
VENDOR_SOURCES = ("ravenpack",)               # one score column (SENT_CSS)
MODEL_RULES = ("argmax", "mean", "median")    # nlp.sentiment.ordinal_sql.RULES
VENDOR_RULES = ("css",)
_SENTIMENT_FIELDS = ("sentiment", "sentiment_source", "sentiment_rule", "neg_max", "pos_min",
                     "min_conf")
# what a no-filter config hashed as before the sentiment filter existed: kept so the
# config_id of every existing all-headlines run stays the same
_LEGACY_NO_SENTIMENT = {"sentiment_split": "none", "neutral_eps": 0.0,
                        "sentiment_source": "none", "sentiment_column": ""}


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
        sentiment: which headlines are scored: ``none`` (all) or one bucket. A bucket is
            selected at load time, from the ``headline_sentiment`` artifact of
            ``sentiment_source``, with SENT and CONF computed by ``sentiment_rule``
            (``nlp.sentiment.ordinal_sql``; vendor: the SENT_CSS score, no CONF):

                unscored = SENT null OR CONF < min_conf
                negative = SENT <= neg_max          (and not unscored)
                positive = SENT >= pos_min          (and not unscored)
                neutral  = neg_max < SENT < pos_min (and not unscored)

            -1 <= neg_max < pos_min <= 1, so the four buckets partition the headlines,
            and a bucket run reuses the all-headlines tau and mu: the four bucket runs
            add up exactly to the ``none`` run (validation.sum_sentiment_runs).
        sentiment_source: ``ravenbert`` | ``finbert`` (grid tables) | ``ravenpack`` (SENT_CSS).
        sentiment_rule: ``argmax`` | ``mean`` | ``median`` (models) or ``css`` (vendor).
        neg_max, pos_min, min_conf: the bucket thresholds above; ``min_conf`` applies to
            model sources only.
        The sentiment fields do not enter ``f0_digest`` / ``warmup_digest``, and a
        ``none`` config hashes exactly as before they existed.
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
    sentiment: SentimentFilter = SentimentFilter.NONE
    sentiment_source: str = SENTIMENT_NONE
    sentiment_rule: str = "mean"
    neg_max: float = -1.0 / 3.0
    pos_min: float = 1.0 / 3.0
    min_conf: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        # accept the string form of every enum (CLI / JSON round trips)
        for name, enum_type in (("mode", Correction), ("paraphrase_style", ParaphraseStyle),
                                ("paraphrase_pooling", PoolRule), ("narrative_agg", AggRule),
                                ("sentiment", SentimentFilter)):
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
        if self.sentiment is SentimentFilter.NONE:
            if self.sentiment_source != SENTIMENT_NONE:
                raise ValueError("sentiment_source is only meaningful with a sentiment filter")
            return
        if self.sentiment_source in MODEL_SOURCES:
            rules = MODEL_RULES
        elif self.sentiment_source in VENDOR_SOURCES:
            rules = VENDOR_RULES
            if self.min_conf != 0.0:
                raise ValueError(f"min_conf applies to model sources only; "
                                 f"{self.sentiment_source} has no confidence")
        else:
            raise ValueError(f"sentiment_source must be one of "
                             f"{MODEL_SOURCES + VENDOR_SOURCES}, got {self.sentiment_source!r}")
        if self.sentiment_rule not in rules:
            raise ValueError(f"sentiment_rule for {self.sentiment_source} must be one of "
                             f"{rules}, got {self.sentiment_rule!r}")
        if not -1.0 <= self.neg_max < self.pos_min <= 1.0:
            raise ValueError(f"need -1 <= neg_max < pos_min <= 1, got neg_max={self.neg_max}, "
                             f"pos_min={self.pos_min}")
        if not 0.0 <= self.min_conf <= 1.0:
            raise ValueError(f"min_conf must be in [0, 1], got {self.min_conf}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoringConfig:
        """A config from its ``to_dict`` form, including the pre-filter form recorded by
        older runs (``sentiment_split`` / ``neutral_eps`` / ``sentiment_column``)."""
        d = dict(d)
        legacy = {k: d.pop(k) for k in ("sentiment_split", "neutral_eps", "sentiment_column")
                  if k in d}
        if legacy.get("sentiment_split", "none") != "none":
            raise ValueError("this run used the removed in-run sentiment split "
                             f"({legacy}); it cannot be rebuilt")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def _hashable(self, *, with_filter: bool = True) -> dict[str, Any]:
        d = self.to_dict()
        d.pop("label")
        if self.sentiment is SentimentFilter.NONE:
            for k in _SENTIMENT_FIELDS:
                d.pop(k)
            d.update(_LEGACY_NO_SENTIMENT)
        elif not with_filter:
            d.pop("sentiment")
        return d

    def digest(self) -> str:
        """Stable hash of every numerical choice (``label`` excluded). A ``none`` config
        hashes as it did before the sentiment filter existed."""
        return hashlib.sha1(json.dumps(self._hashable(), sort_keys=True).encode()).hexdigest()[:16]

    def digest_without_filter(self) -> str:
        """``digest`` with the bucket removed: equal across the four bucket runs of one
        sentiment setup (same source, rule and thresholds)."""
        d = self._hashable(with_filter=False)
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

    @property
    def filtered(self) -> bool:
        return self.sentiment is not SentimentFilter.NONE

    @property
    def row_label(self) -> str:
        """The narrative_daily SENTIMENT value of this run: "all" or the bucket."""
        return SENTIMENT_ALL if not self.filtered else self.sentiment.value


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
    sentiment_artifact_id: str | None = None   # headline_sentiment artifact read by the split
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
