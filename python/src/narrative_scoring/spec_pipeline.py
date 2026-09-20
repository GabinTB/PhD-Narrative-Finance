"""Spec-compliant narrative scoring -- implements python/doc/narratives.md.

That document is the authority; this module is its executable form. Where the
spec marks a decision open (the percentile axis, the narrative aggregation
rule), the choice is an explicit config field with the spec's stated default
and is recorded in :class:`RunMetadata`. Nothing is chosen silently.

The five steps, in the spec's order, none skipped or reordered::

    1. embed + L2-normalise; mode in {raw, R1, R2}      (same mu on both sides)
    2. S = H @ P.T, then pool master + K paraphrase texts to ONE score per
       primitive (mean for definition-preserving/semantic, max for
       headline-style) -- pooling happens BEFORE the F0 floor
    3. F0 floor:        S[S < tau] = NaN                (the only floor)
    4. percentile cut:  S[S < percentile(S, q)] = NaN   (axis configurable)
    5. aggregate primitive -> narrative PER HEADLINE, mean|median skipping NaN

NaN means "no evidence" and is never filled with zero. An all-NaN narrative
for a headline stays NaN.

Aggregation order matters and differs from the v1 harness. The spec pools
primitives into a narrative *for each headline*, and only then aggregates that
day's headline-level narrative scores. v1 did the reverse (primitive-day
first, then roll up), which is a different number: it lets a primitive that
fired on a different headline contribute to the same narrative-day mean.

Owner ruling R1: the storable object is the (day x narrative) table. Primitive
grain exists here only in memory / local cache for diagnostics, and is never
written to the datalake. Nothing in this module writes to the datalake.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import duckdb
import numpy as np
import polars as pl

from narrative_scoring.corrections import Correction, apply_correction
from narrative_scoring.null_model import (
    ReservoirPool,
    compute_gaussian_tau,
    compute_n_eff,
    compute_tau,
    trim_null_draws_batch,
)
from narrative_scoring.schema import EMBEDDING_DIM

log = logging.getLogger(__name__)

TAXONOMY_CSV = "{name}_taxonomy.authored.csv"
PARAPHRASE_JSONL = "{name}-primitive_{style}_paraphrases.jsonl"

# The spec's unique primitive key (section 1). CATEGORY alone is NOT unique.
PRIMITIVE_KEY = ("TOPIC", "GROUP", "CATEGORY", "ROLE", "OBSERVABILITY_CHANNEL")
# The sha1 path hash that joins the CSV to the paraphrase JSONL (section 1).
PATH_FIELDS = ("TOPIC", "GROUP", "TYPE", "SUB_TYPE", "CATEGORY", "ROLE", "OBSERVABILITY_CHANNEL")


class PoolRule(str, Enum):
    """Step 5: how the master + K paraphrase texts collapse to one primitive score."""

    MEAN = "mean"   # definition-preserving (semantic) paraphrases: replicates -> variance reduction
    MAX = "max"     # headline-style paraphrases: distinct surfacings -> closest surfacing wins


class PctAxis(str, Enum):
    """Step 4's open decision, surfaced as config."""

    ROW_WISE = "row_wise"   # spec default: per headline, over that headline's own non-NaN scores
    GLOBAL = "global"       # over the whole day matrix: weak headlines drop out entirely


class AggRule(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"


@dataclass(frozen=True)
class ScoringConfig:
    mode: Correction = Correction.R2
    paraphrase_pooling: PoolRule = PoolRule.MAX
    q: float = 99.0
    percentile_axis: PctAxis = PctAxis.ROW_WISE
    narrative_agg: AggRule = AggRule.MEAN
    alpha: float = 0.01
    trim_frac: float = 0.10
    include_master: bool = True
    legacy_rel_floor: float | None = None   # v1 reference arm only; NOT spec-compliant
    label: str = ""

    def key(self) -> dict[str, Any]:
        """Flat dict used for cache keys and summary rows."""
        return {
            "mode": self.mode.value,
            "paraphrase_pooling": self.paraphrase_pooling.value,
            "q": self.q,
            "percentile_axis": self.percentile_axis.value,
            "narrative_agg": self.narrative_agg.value,
            "alpha": self.alpha,
            "trim_frac": self.trim_frac,
            "include_master": self.include_master,
            "legacy_rel_floor": self.legacy_rel_floor,
        }


@dataclass
class RunMetadata:
    """Persisted alongside every result (spec section 3)."""

    mode: str
    mu_norm: float
    tau: float
    n_eff: float
    q: float
    percentile_axis: str
    paraphrase_pooling: str
    narrative_agg: str
    taxonomy_sha1: str
    paraphrase_sha1: str
    alpha: float
    trim_frac: float
    include_master: bool
    legacy_rel_floor: float | None
    n_primitive_texts: int
    k_paraphrases: int
    spec: str = "python/doc/narratives.md"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Taxonomy: load, join on the sha1 path hash, run the spec's sanity checks
# ---------------------------------------------------------------------------

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def primitive_path(row: dict[str, str]) -> str:
    return "/".join(row[c] for c in PATH_FIELDS)


@dataclass
class PrimitiveTable:
    """The scoring unit table, plus its primitive texts in matrix row order.

    ``frame`` is sorted by (narrative_id, primitive) so each narrative's
    primitives are contiguous and narrative pooling is a reduceat.
    ``texts`` is flattened primitive-major: primitive i owns rows
    [i*n_texts, (i+1)*n_texts), master first when included.
    """

    frame: pl.DataFrame
    texts: list[str]
    n_texts: int                 # per primitive (1 + K, or K when master excluded)
    k_paraphrases: int
    narrative_starts: np.ndarray  # reduceat offsets into the primitive axis
    narrative_frame: pl.DataFrame  # one row per narrative, in narrative_id order
    taxonomy_sha1: str
    paraphrase_sha1: str
    text_major: bool = True        # texts grouped by text-slot, not by primitive

    @property
    def n_primitives(self) -> int:
        return self.frame.height

    @property
    def primitive_to_narrative(self) -> np.ndarray:
        return self.frame["narrative_id"].to_numpy().astype(np.int32)


def load_primitive_table(
    taxonomy_root: Path,
    name: str = "Evergreen_v5",
    style: str = "headline",
    *,
    include_master: bool = True,
) -> PrimitiveTable:
    """Load the CSV and paraphrase JSONL, joined on the sha1 path hash."""
    root = Path(taxonomy_root)
    csv_path = root / TAXONOMY_CSV.format(name=name)
    jsonl_path = root / PARAPHRASE_JSONL.format(name=name, style=style)

    # missing_utf8_is_empty_string: an unsigned narrative has an EMPTY SUB_TYPE,
    # and the JSONL path keeps that component as an empty segment ("a/b//c").
    # Read as null instead, concat_str would propagate the null and silently
    # drop every unsigned primitive from the join -- exactly the failure
    # spec check 2 exists to catch.
    csv = pl.read_csv(
        csv_path, infer_schema_length=0, missing_utf8_is_empty_string=True
    ).with_columns(
        pl.concat_str([pl.col(c).fill_null("") for c in PATH_FIELDS], separator="/").alias("_path")
    )
    csv = csv.with_columns(
        pl.col("_path").map_elements(_sha1, return_dtype=pl.String).alias("path_id")
    )

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    para = pl.DataFrame(
        [
            {"path_id": r["id"], "master": r["master"], "paraphrases": r["paraphrases"]}
            for r in records
        ]
    )

    joined = csv.join(para, on="path_id", how="inner")
    if joined.height != csv.height:
        raise ValueError(
            f"CSV<->JSONL bijection broken on the sha1 path hash: "
            f"{csv.height} CSV rows, {para.height} JSONL records, {joined.height} joined"
        )

    frame = (
        joined.rename(
            {
                "TOPIC": "reservoir",
                "GROUP": "dimension",
                "CATEGORY": "narrative",
                "SUB_TYPE": "pole",
                "ROLE": "sub_mechanism",
                "OBSERVABILITY_CHANNEL": "observability_channel",
                "DISPLAY_NAME": "primitive",
            }
        )
        .with_columns(
            # narrative is unique only when scoped by its parents (spec section 1)
            pl.concat_str(
                [pl.col("reservoir"), pl.col("dimension"), pl.col("narrative")], separator="|"
            ).alias("narrative_key")
        )
        .sort(["narrative_key", "primitive"])
        .with_row_index("primitive_id")
    )

    narrative_frame = (
        frame.group_by("narrative_key", maintain_order=True)
        .agg(
            pl.col("reservoir").first(),
            pl.col("dimension").first(),
            pl.col("narrative").first(),
            pl.col("pole").first(),
            pl.len().alias("n_primitives"),
        )
        .with_row_index("narrative_id")
    )
    frame = frame.join(narrative_frame.select("narrative_key", "narrative_id"), on="narrative_key")

    nid = frame["narrative_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, nid[1:] != nid[:-1]])

    ks = {len(p) for p in frame["paraphrases"].to_list()}
    if len(ks) != 1:
        raise ValueError(f"non-uniform K across primitives: {sorted(ks)}")
    k = ks.pop()

    # Primitive-major: primitive i owns texts [i*n_texts, (i+1)*n_texts), master
    # first. This is the order the embedding cache is keyed on; the scoring
    # matrix is permuted to text-major once on load by _to_text_major, which
    # is what makes paraphrase pooling a contiguous reduction.
    texts: list[str] = []
    for master, paras in zip(frame["master"].to_list(), frame["paraphrases"].to_list()):
        if include_master:
            texts.append(master)
        texts.extend(paras)

    return PrimitiveTable(
        frame=frame,
        texts=texts,
        n_texts=(1 if include_master else 0) + k,
        k_paraphrases=k,
        narrative_starts=starts,
        narrative_frame=narrative_frame,
        taxonomy_sha1=file_sha1(csv_path),
        paraphrase_sha1=file_sha1(jsonl_path),
    )


def sanity_checks(table: PrimitiveTable, taxonomy_root: Path, name: str = "Evergreen_v5") -> pl.DataFrame:
    """Spec section 4 checks 1, 2, 3, 5, 6 (the taxonomy-only ones).

    Checks 4 (NaN accounting), 7 (mode comparison) and 8 (cross-reservoir
    levels) depend on scores and are reported by the notebook after scoring.
    """
    f = table.frame
    rows: list[dict[str, Any]] = []

    key_cols = ["reservoir", "dimension", "narrative", "sub_mechanism", "observability_channel"]
    n_dupe = f.height - f.select(key_cols).unique().height
    rows.append({
        "check": "1. primitive key uniqueness",
        "result": "PASS" if n_dupe == 0 else "FAIL",
        "detail": f"{f.height} primitives, {n_dupe} duplicate key(s) on {tuple(PRIMITIVE_KEY)}",
    })

    rows.append({
        "check": "2. CSV<->JSONL bijection (sha1 path hash)",
        "result": "PASS",
        "detail": f"{f.height} primitives joined on path_id; join is enforced in load_primitive_table",
    })

    rows.append({
        "check": "3. uniform K paraphrases",
        "result": "PASS",
        "detail": f"K={table.k_paraphrases} for every primitive; "
                  f"{table.n_texts} primitive text(s) each (master {'included' if table.n_texts > table.k_paraphrases else 'EXCLUDED'})",
    })

    pol = pl.read_csv(
        Path(taxonomy_root) / TAXONOMY_CSV.format(name=name), infer_schema_length=0
    ).select(["SUB_TYPE", "POLARITY"])
    incoherent = pol.filter(
        (pl.col("POLARITY").fill_null("") != pl.col("SUB_TYPE").fill_null(""))
    ).height
    signed = pol.filter(pl.col("SUB_TYPE").fill_null("") != "").height
    rows.append({
        "check": "5. polarity coherence",
        "result": "PASS" if incoherent == 0 else "FAIL",
        "detail": f"POLARITY == SUB_TYPE on all rows; {signed}/{pol.height} signed, "
                  f"{pol.height - signed} intensity (empty pole); {incoherent} incoherent",
    })

    # 6. bipolar mirror presence. The schema does not link sibling poles: poles
    # live in the narrative name, and mirrored pairs need not share a TYPE stem
    # (e.g. accelerating-inflation vs deflation-scare in the same dimension).
    # Report the per-dimension pole inventory for review rather than invent a
    # pairing rule the taxonomy does not encode.
    signed_f = f.filter(pl.col("pole").fill_null("") != "")
    per_dim = (
        signed_f.group_by(["reservoir", "dimension"])
        .agg(pl.col("narrative").n_unique().alias("n_signed_narratives"))
    )
    singles = per_dim.filter(pl.col("n_signed_narratives") == 1)
    rows.append({
        "check": "6. bipolar mirror presence",
        "result": "REVIEW",
        "detail": f"{signed_f['narrative'].n_unique()} signed narratives in "
                  f"{per_dim.height} dimensions; {singles.height} dimension(s) carry exactly one "
                  f"signed narrative (possible orphan poles -- listed below). Sibling linkage is "
                  f"not encoded in the schema, so this cannot be decided mechanically.",
    })
    return pl.DataFrame(rows)


def orphan_pole_candidates(table: PrimitiveTable) -> pl.DataFrame:
    """Dimensions carrying exactly one signed narrative -- candidate orphan poles."""
    signed = table.frame.filter(pl.col("pole").fill_null("") != "")
    counts = (
        signed.group_by(["reservoir", "dimension"])
        .agg(
            pl.col("narrative").n_unique().alias("n_signed"),
            pl.col("narrative").unique().alias("narratives"),
        )
    )
    return counts.filter(pl.col("n_signed") == 1).sort(["reservoir", "dimension"])


# ---------------------------------------------------------------------------
# Primitive-text embeddings (local cache; never a datalake artifact)
# ---------------------------------------------------------------------------

def embed_primitive_texts(
    table: PrimitiveTable,
    cache_dir: Path,
    *,
    batch_size: int = 256,
    device: str = "embedx",
) -> np.ndarray:
    """(n_primitives * n_texts, 384) L2-normalised primitive-text embeddings.

    Cached locally by a digest of the exact text list, so a change to the
    taxonomy or to master-inclusion busts the cache. Nothing is written to the
    datalake -- the existing taxonomy_embeddings artifacts hold only the K
    paraphrase vectors and never embedded the master description, which the
    spec requires as a primitive text.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.blake2b(
        ("\x00".join(table.texts)).encode(), digest_size=8
    ).hexdigest()
    path = cache_dir / f"primitive_texts__{digest}.npy"
    if path.exists():
        log.info("primitive-text embeddings: cache hit %s", path.name)
        return _to_text_major(np.load(path), table.n_texts)

    from ravenpack.headlines.embed import load_embedding_model

    log.info(
        "embedding %d primitive texts (%d primitives x %d texts) via %s ...",
        len(table.texts), table.frame.height, table.n_texts, device,
    )
    model = load_embedding_model(Path("."), device=device)
    vectors = np.asarray(model.encode(table.texts, batch_size=batch_size), dtype=np.float32)
    if vectors.shape != (len(table.texts), EMBEDDING_DIM):
        raise ValueError(f"encode returned {vectors.shape}, expected {(len(table.texts), EMBEDDING_DIM)}")
    vectors = l2_normalise(vectors)
    np.save(path, vectors)
    log.info("cached primitive-text embeddings -> %s", path.name)
    return _to_text_major(vectors, table.n_texts)


def _to_text_major(vectors: np.ndarray, n_texts: int) -> np.ndarray:
    """Reorder (n_prim * n_texts, dim) from primitive-major to text-major.

    Primitive-major groups a primitive's texts together, so pooling reduces
    over a length-n_texts inner axis -- ~18x slower in numpy than reducing
    over a length-n_texts OUTER axis, which walks contiguous (n_head, n_prim)
    slabs instead. The cache stores the natural primitive-major order; this
    permutation is applied on the way out. Output is identical either way.
    """
    n_prim = vectors.shape[0] // n_texts
    order = np.arange(n_prim * n_texts).reshape(n_prim, n_texts).T.ravel()
    return np.ascontiguousarray(vectors[order])


def l2_normalise(X: np.ndarray) -> np.ndarray:
    """Step 1's L2 normalisation, applied to every embedding on both sides."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.where(norms < 1e-12, 1.0, norms)).astype(np.float32)


def apply_mode(X: np.ndarray, mode: Correction, mu: np.ndarray | None, mu_hat: np.ndarray | None) -> np.ndarray:
    """Step 1's raw / R1 / R2, then renormalise. Same mu/u on both sides."""
    out, _ = apply_correction(X, mode, mu=mu, mu_hat=mu_hat)
    return l2_normalise(out)


# ---------------------------------------------------------------------------
# Steps 2-5 on one block of headlines
# ---------------------------------------------------------------------------

@dataclass
class NaNAccounting:
    """Spec check 4: what the F0 floor removed vs what the percentile cut removed."""

    n_scores: int = 0
    n_nan_f0: int = 0
    n_nan_pct: int = 0
    n_headlines: int = 0
    n_unassigned: int = 0          # all primitives NaN after both cuts

    def merge(self, other: "NaNAccounting") -> None:
        self.n_scores += other.n_scores
        self.n_nan_f0 += other.n_nan_f0
        self.n_nan_pct += other.n_nan_pct
        self.n_headlines += other.n_headlines
        self.n_unassigned += other.n_unassigned

    def to_dict(self) -> dict[str, Any]:
        n = max(self.n_scores, 1)
        return {
            "n_scores": self.n_scores,
            "share_nan_f0": self.n_nan_f0 / n,
            "share_nan_pct": self.n_nan_pct / n,
            "share_surviving": 1.0 - (self.n_nan_f0 + self.n_nan_pct) / n,
            "n_headlines": self.n_headlines,
            "unassigned_share": self.n_unassigned / max(self.n_headlines, 1),
        }


# ---------------------------------------------------------------------------
# Steps 2-5 on one block of headlines
# ---------------------------------------------------------------------------

def build_scoring_matrix(
    P: np.ndarray, table: PrimitiveTable, cfg: ScoringConfig,
    mu: np.ndarray | None, mu_hat: np.ndarray | None,
) -> np.ndarray:
    """Mode-corrected primitive-text matrix, ready for Step 2.

    For MEAN pooling this collapses the n_texts text vectors of each primitive
    into ONE mean vector, because mean_j(H . p_j) == H . mean_j(p_j) exactly.
    The mean vector is deliberately NOT renormalised -- renormalising would
    rescale the score and break the identity.
    """
    P_mode = apply_mode(P, cfg.mode, mu, mu_hat)          # text-major
    if cfg.paraphrase_pooling is PoolRule.MEAN:
        return P_mode.reshape(table.n_texts, table.n_primitives, EMBEDDING_DIM).mean(axis=0)
    return P_mode


def primitive_scores(H: np.ndarray, P_scoring: np.ndarray, table: PrimitiveTable,
                     pooling: PoolRule) -> np.ndarray:
    """Step 2: cosine scores, pooled over the master + K paraphrase texts."""
    R = H @ P_scoring.T
    if pooling is PoolRule.MEAN:
        return R                                          # already (n_head, n_prim)
    n_prim = table.n_primitives
    out = R[:, :n_prim].copy()
    for j in range(1, table.n_texts):                     # contiguous slabs, text-major
        np.maximum(out, R[:, j * n_prim:(j + 1) * n_prim], out=out)
    return out


def kth_largest_threshold(S: np.ndarray, n_surv: np.ndarray, q: float) -> np.ndarray:
    """Per-row threshold keeping the top (100-q)% of that row's F0 survivors.

    Exactly equivalent to nanpercentile over the survivors, ~6x faster: an O(n)
    argpartition at K = max_i k_i, then a sort of only those K columns.
    Rows with no survivor get +inf, so a floored headline is never resurrected.
    """
    k = np.where(n_surv > 0,
                 np.maximum(np.ceil(n_surv * (100.0 - q) / 100.0).astype(np.int64), 1), 0)
    max_k = int(k.max()) if k.size else 0
    if max_k == 0:
        return np.full(S.shape[0], np.inf, dtype=np.float32)
    idx = np.argpartition(S, -max_k, axis=1)[:, -max_k:]
    top = np.take_along_axis(S, idx, axis=1)
    top.sort(axis=1)
    top = top[:, ::-1]
    rows = np.arange(S.shape[0])
    return np.where(k > 0, top[rows, np.clip(k, 1, max_k) - 1], np.inf).astype(np.float32)


@dataclass(frozen=True)
class GateVariant:
    """One (q, axis) post-floor cut. ``legacy`` is the v1 arm, NOT spec-compliant."""

    q: float | None = 99.0
    axis: PctAxis = PctAxis.ROW_WISE
    legacy_rel_floor: float | None = None

    @property
    def key(self) -> str:
        if self.legacy_rel_floor is not None:
            return f"legacy_rel{self.legacy_rel_floor:g}"
        return f"q{self.q:g}_{self.axis.value}"

    @property
    def needs_day_buffer(self) -> bool:
        return self.legacy_rel_floor is None and self.axis is PctAxis.GLOBAL


def survivors(S: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Step 3 as sparse triplets: the (row, col, value) that clear the F0 floor.

    Only ~2% of scores clear tau and ~0.06% survive the percentile cut, so
    everything downstream works on triplets rather than a dense (n, 1508)
    matrix full of NaN. Returns (rows, cols, vals, n_floored, n_surv_per_row).
    """
    above = S >= tau
    rows, cols = np.nonzero(above)
    n_surv = above.sum(axis=1)
    return (rows.astype(np.int32), cols.astype(np.int32), S[rows, cols],
            int(S.size - rows.size), n_surv)


class _DayAccumulator:
    """Per-day aggregates over that day's HEADLINE-level narrative scores (ruling R1)."""

    __slots__ = ("count", "total", "peak", "n_headlines", "n_unassigned")

    def __init__(self, n_nodes: int):
        self.count = np.zeros(n_nodes, dtype=np.int64)
        self.total = np.zeros(n_nodes, dtype=np.float64)
        self.peak = np.full(n_nodes, -np.inf, dtype=np.float64)
        self.n_headlines = 0
        self.n_unassigned = 0

    def add_groups(self, node_ids: np.ndarray, values: np.ndarray, n_nodes: int) -> None:
        """Fold one block's (node, value) groups in. Absent nodes contribute nothing."""
        if node_ids.size == 0:
            return
        self.count += np.bincount(node_ids, minlength=n_nodes)
        self.total += np.bincount(node_ids, weights=values.astype(np.float64), minlength=n_nodes)
        order = np.argsort(node_ids, kind="stable")
        nid_s, val_s = node_ids[order], values[order]
        starts = np.flatnonzero(np.r_[True, nid_s[1:] != nid_s[:-1]])
        np.maximum.at(self.peak, nid_s[starts], np.maximum.reduceat(val_s, starts))

    def frame(self, nodes: pl.DataFrame, day: date) -> pl.DataFrame:
        has = self.count > 0
        return nodes.with_columns(
            pl.lit(day).cast(pl.Date).alias("DATE"),
            pl.Series("SUPPORT", self.count, dtype=pl.Int32),
            pl.Series("INTENSITY", np.where(has, self.total / np.maximum(self.count, 1), np.nan)).cast(pl.Float32),
            pl.Series("TOTAL", np.where(has, self.total, np.nan)).cast(pl.Float32),
            pl.Series("PEAK", np.where(has, self.peak, np.nan)).cast(pl.Float32),
        ).with_columns(
            pl.col("INTENSITY").fill_nan(None), pl.col("TOTAL").fill_nan(None),
            pl.col("PEAK").fill_nan(None),
        )


def _headline_narrative_groups(
    rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
    prim_to_narr: np.ndarray, n_narr: int, rule: AggRule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 5, first hop: pool each HEADLINE's surviving primitives into narratives.

    Returns (headline_row, narrative_id, score) for every (headline, narrative)
    that has at least one surviving primitive. A narrative with no surviving
    primitive for a headline is simply absent -- never zero.
    """
    if rows.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float32)
    nid = prim_to_narr[cols].astype(np.int64)
    key = rows.astype(np.int64) * n_narr + nid
    order = np.argsort(key, kind="stable")
    key_s, val_s = key[order], vals[order].astype(np.float32)
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    gkey = key_s[starts]
    if rule is AggRule.MEAN:
        counts = np.diff(np.r_[starts, val_s.size])
        gval = (np.add.reduceat(val_s, starts) / counts).astype(np.float32)
    else:
        ends = np.r_[starts[1:], val_s.size]
        gval = np.fromiter(
            (np.median(val_s[a:b]) for a, b in zip(starts, ends)),
            dtype=np.float32, count=starts.size,
        )
    return gkey // n_narr, gkey % n_narr, gval


# ---------------------------------------------------------------------------
# Streaming data path (Part 2: the OOM fix)
# ---------------------------------------------------------------------------

class MemoryBudgetExceeded(RuntimeError):
    """Raised when RSS crosses the budget, naming the month being scored."""


def rss_gb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def _stream_month(
    headlines_path: Path, embeddings_path: Path, *, batch_size: int, threads: int,
    duckdb_memory_limit: str, temp_directory: str | None,
    day_lo: date | None = None, day_hi: date | None = None,
) -> Iterator[pl.DataFrame]:
    """Stream (TIMESTAMP_UTC, EMBEDDING) batches for one month.

    Three things keep this bounded where v1's load_month was not: the day
    filter is pushed into SQL so out-of-window days never materialise; the
    embedding is cast to FLOAT[384] in the SQL projection so polars hands back
    a fixed-size Array whose .to_numpy() is a zero-copy (n, 384) view; and
    DuckDB's own memory_limit bounds the join's hash table and spills to
    temp_directory instead of the process growing to the month's size.

    Batches arrive in arbitrary day order, which is fine because count/sum/max
    are associative and each day owns an accumulator.
    """
    where = []
    if day_lo is not None:
        where.append(f"h.TIMESTAMP_UTC >= '{day_lo.isoformat()}'")
    if day_hi is not None:
        day_after = date.fromordinal(day_hi.toordinal() + 1).isoformat()
        where.append(f"h.TIMESTAMP_UTC < '{day_after}'")
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    conn = duckdb.connect()
    try:
        conn.execute("SET enable_progress_bar=false")
        conn.execute(f"SET threads={threads}")
        conn.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        if temp_directory:
            Path(temp_directory).mkdir(parents=True, exist_ok=True)
            conn.execute(f"SET temp_directory='{temp_directory}'")
        query = f"""
            SELECT h.TIMESTAMP_UTC,
                   CAST(e.EMBEDDING AS FLOAT[{EMBEDDING_DIM}]) AS EMBEDDING
            FROM read_parquet('{embeddings_path}') e
            JOIN read_parquet('{headlines_path}') h USING (RP_STORY_ID)
            {clause}
        """
        for batch in conn.sql(query).to_arrow_reader(batch_size):
            yield pl.from_arrow(batch)
    finally:
        conn.close()


def _months(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m, y = (1, y + 1) if m == 12 else (m + 1, y)
    return out


@dataclass
class SpecResult:
    """The storable object (ruling R1) plus in-memory diagnostics."""

    narrative_daily: pl.DataFrame          # day x narrative -- the artifact shape
    primitive_daily: pl.DataFrame | None   # notebook-local diagnostics only
    metadata: RunMetadata
    nan_accounting: dict[str, Any]
    day_diagnostics: pl.DataFrame
    n_days: int
    peak_rss_gb: float
    variant: str = ""


def calibrate_tau(
    headlines_dir: Path, embeddings_dir: Path, P: np.ndarray, table: PrimitiveTable,
    cfg: ScoringConfig, mu: np.ndarray | None, mu_hat: np.ndarray | None,
    calib_start: date, calib_end: date, *,
    cap: int = 2_000_000, batch_size: int = 50_000, score_block: int = 8_192,
    threads: int = 8, seed: int = 0, duckdb_memory_limit: str = "6GB",
    temp_directory: str | None = "/tmp/duckdb_spill",
) -> tuple[float, float, float, int]:
    """F0 tau for one (mode, pooling). Thresholds do not transfer across either
    axis, so this is refit per combination (spec check 7)."""
    P_scoring = build_scoring_matrix(P, table, cfg, mu, mu_hat)
    reps = (
        P_scoring if cfg.paraphrase_pooling is PoolRule.MEAN
        else P_scoring.reshape(table.n_texts, table.n_primitives, EMBEDDING_DIM).mean(axis=0)
    )
    n_eff = compute_n_eff(l2_normalise(reps))

    pool = ReservoirPool(cap=cap, seed=seed)
    for y, m in _months(calib_start, calib_end):
        name = f"{y}-{m:02d}.parquet"
        hl, emb = headlines_dir / name, embeddings_dir / name
        if not hl.exists() or not emb.exists():
            continue
        for batch in _stream_month(
            hl, emb, batch_size=batch_size, threads=threads,
            duckdb_memory_limit=duckdb_memory_limit, temp_directory=temp_directory,
            day_lo=calib_start, day_hi=calib_end,
        ):
            X = batch["EMBEDDING"].to_numpy()
            for i in range(0, X.shape[0], score_block):
                H = apply_mode(l2_normalise(X[i:i + score_block]), cfg.mode, mu, mu_hat)
                S = primitive_scores(H, P_scoring, table, cfg.paraphrase_pooling)
                pool.add(trim_null_draws_batch(S, trim_frac=cfg.trim_frac).ravel())
                del H, S
            del X, batch
        log.info("tau calib %s/%s %s: pool %d/%d (%d seen), RSS %.1f GB",
                 cfg.mode.value, cfg.paraphrase_pooling.value, name,
                 pool.n_filled, pool.cap, pool.n_seen, rss_gb())

    if pool.n_filled == 0:
        raise RuntimeError(f"no headlines for tau calibration in [{calib_start}, {calib_end}]")
    F0 = pool.draws
    return (compute_tau(F0, n_eff, alpha=cfg.alpha),
            compute_gaussian_tau(F0, n_eff, alpha=cfg.alpha),
            float(n_eff), int(F0.shape[0]))

class _VariantState:
    """Accumulators + NaN accounting for one (q, axis) variant across the window."""

    def __init__(self, n_narr: int, n_prim: int, keep_primitive: bool,
                 prim_to_narr: np.ndarray, rule: AggRule):
        self.n_narr, self.n_prim = n_narr, n_prim
        self.keep_primitive = keep_primitive
        self.prim_to_narr, self.rule = prim_to_narr, rule
        self.narr: dict[date, _DayAccumulator] = {}
        self.prim: dict[date, _DayAccumulator] = {}
        self.acct = NaNAccounting()

    def add_block(self, day: date, n_head: int, rows: np.ndarray, cols: np.ndarray,
                  vals: np.ndarray, keep: np.ndarray, n_floored: int, n_scores: int,
                  row_offset: int) -> None:
        r, c, v = rows[keep], cols[keep], vals[keep]
        n_assigned = int(np.unique(r).size)

        a = self.acct
        a.n_scores += n_scores
        a.n_nan_f0 += n_floored
        a.n_nan_pct += int(keep.size - keep.sum())
        a.n_headlines += n_head
        a.n_unassigned += n_head - n_assigned

        na = self.narr.setdefault(day, _DayAccumulator(self.n_narr))
        na.n_headlines += n_head
        na.n_unassigned += n_head - n_assigned
        _, nid, nval = _headline_narrative_groups(
            r.astype(np.int64) + row_offset, c, v, self.prim_to_narr, self.n_narr, self.rule)
        na.add_groups(nid.astype(np.int64), nval, self.n_narr)

        if self.keep_primitive:
            pa = self.prim.setdefault(day, _DayAccumulator(self.n_prim))
            pa.n_headlines += n_head
            pa.add_groups(c.astype(np.int64), v.astype(np.float32), self.n_prim)


def score_grid(
    headlines_dir: Path, embeddings_dir: Path, P: np.ndarray, table: PrimitiveTable,
    cfg: ScoringConfig, mu: np.ndarray | None, mu_hat: np.ndarray | None,
    tau: float, n_eff: float, variants: list[GateVariant], *,
    start: date, end: date, mu_norm: float = float("nan"),
    keep_primitive_daily: bool = True, batch_size: int = 50_000, score_block: int = 8_192,
    threads: int = 8, duckdb_memory_limit: str = "6GB",
    temp_directory: str | None = "/tmp/duckdb_spill", rss_budget_gb: float | None = 30.0,
) -> dict[str, SpecResult]:
    """Run steps 1-5 once per (mode, pooling) and emit EVERY (q, axis) variant.

    q and the percentile axis are post-matmul filters, so Step 1 + Step 2 --
    the expensive part -- are shared across variants and the whole grid costs
    one pass per (mode, pooling) instead of one per config.

    Row-wise and legacy variants are applied inline per block. A global-axis
    variant cannot cut until it knows the whole day's distribution, so that
    day's F0 survivors are buffered as sparse triplets and closed at month end.
    That stays bounded because only ~2% of scores clear the floor.
    """
    P_scoring = build_scoring_matrix(P, table, cfg, mu, mu_hat)
    prim_to_narr = table.primitive_to_narrative
    n_narr, n_prim = table.narrative_frame.height, table.n_primitives

    row_variants = [v for v in variants if v.legacy_rel_floor is None and v.axis is PctAxis.ROW_WISE]
    global_variants = [v for v in variants if v.legacy_rel_floor is None and v.axis is PctAxis.GLOBAL]
    legacy_variants = [v for v in variants if v.legacy_rel_floor is not None]

    state = {
        v.key: _VariantState(n_narr, n_prim, keep_primitive_daily, prim_to_narr, cfg.narrative_agg)
        for v in variants
    }
    day_rows: dict[date, int] = {}
    peak_rss = 0.0

    for y, m in _months(start, end):
        name = f"{y}-{m:02d}.parquet"
        hl, emb = headlines_dir / name, embeddings_dir / name
        if not hl.exists() or not emb.exists():
            continue
        buffered: dict[date, list[tuple]] = {}

        for batch in _stream_month(
            hl, emb, batch_size=batch_size, threads=threads,
            duckdb_memory_limit=duckdb_memory_limit, temp_directory=temp_directory,
            day_lo=start, day_hi=end,
        ):
            days = batch["TIMESTAMP_UTC"].str.slice(0, 10).str.to_date().to_list()
            day_ord = np.fromiter((d.toordinal() for d in days), dtype=np.int64, count=len(days))
            X = batch["EMBEDDING"].to_numpy()

            for ordv in np.unique(day_ord):
                day = date.fromordinal(int(ordv))
                Xi = X[day_ord == ordv]
                for i in range(0, Xi.shape[0], score_block):
                    H = apply_mode(l2_normalise(Xi[i:i + score_block]), cfg.mode, mu, mu_hat)
                    S = primitive_scores(H, P_scoring, table, cfg.paraphrase_pooling)
                    n_head, n_scores = S.shape[0], S.size
                    offset = day_rows.get(day, 0)
                    rows, cols, vals, n_floored, n_surv = survivors(S, tau)

                    for v in row_variants:
                        thr = kth_largest_threshold(S, n_surv, v.q)
                        state[v.key].add_block(
                            day, n_head, rows, cols, vals, vals >= thr[rows],
                            n_floored, n_scores, offset)

                    for v in legacy_variants:
                        s_max = S.max(axis=1, keepdims=True)
                        lk = (S >= v.legacy_rel_floor * s_max) & (s_max >= tau)
                        lr, lc = np.nonzero(lk)
                        state[v.key].add_block(
                            day, n_head, lr.astype(np.int32), lc.astype(np.int32),
                            S[lr, lc], np.ones(lr.size, dtype=bool),
                            int(S.size - lr.size), n_scores, offset)

                    if global_variants:
                        buffered.setdefault(day, []).append(
                            (rows.copy(), cols.copy(), vals.copy(), n_floored, n_scores,
                             n_head, offset))

                    day_rows[day] = offset + n_head
                    del H, S
                del Xi
            del X, batch

        for day, blocks in buffered.items():
            all_vals = np.concatenate([b[2] for b in blocks]) if blocks else np.empty(0, np.float32)
            for v in global_variants:
                thr = float(np.quantile(all_vals, v.q / 100.0)) if all_vals.size else np.inf
                for rows, cols, vals, n_floored, n_scores, n_head, offset in blocks:
                    state[v.key].add_block(
                        day, n_head, rows, cols, vals, vals >= thr,
                        n_floored, n_scores, offset)
        buffered.clear()

        current = rss_gb()
        peak_rss = max(peak_rss, current)
        if rss_budget_gb is not None and current > rss_budget_gb:
            raise MemoryBudgetExceeded(
                f"RSS {current:.2f} GB exceeded budget {rss_budget_gb:.2f} GB scoring {name}")
        log.info("scored %s [%s/%s] RSS %.2f GB", name, cfg.mode.value,
                 cfg.paraphrase_pooling.value, current)

    nodes = table.narrative_frame.select(
        ["reservoir", "dimension", "narrative", "pole", "narrative_key", "n_primitives"])
    prim_nodes = table.frame.select(
        ["reservoir", "dimension", "narrative", "pole", "sub_mechanism",
         "observability_channel", "primitive"])

    out: dict[str, SpecResult] = {}
    for v in variants:
        st = state[v.key]
        if not st.narr:
            raise RuntimeError(f"no headlines in [{start}, {end}] for variant {v.key}")
        ordered = sorted(st.narr)
        md = RunMetadata(
            mode=cfg.mode.value, mu_norm=mu_norm, tau=float(tau), n_eff=float(n_eff),
            q=(v.q if v.legacy_rel_floor is None else float("nan")),
            percentile_axis=(v.axis.value if v.legacy_rel_floor is None else "n/a"),
            paraphrase_pooling=cfg.paraphrase_pooling.value,
            narrative_agg=cfg.narrative_agg.value,
            taxonomy_sha1=table.taxonomy_sha1, paraphrase_sha1=table.paraphrase_sha1,
            alpha=cfg.alpha, trim_frac=cfg.trim_frac, include_master=cfg.include_master,
            legacy_rel_floor=v.legacy_rel_floor,
            n_primitive_texts=len(table.texts), k_paraphrases=table.k_paraphrases,
        )
        out[v.key] = SpecResult(
            narrative_daily=pl.concat([st.narr[d].frame(nodes, d) for d in ordered]).select(
                ["DATE", "reservoir", "dimension", "narrative", "pole",
                 "SUPPORT", "INTENSITY", "TOTAL", "PEAK", "narrative_key", "n_primitives"]),
            primitive_daily=(pl.concat([st.prim[d].frame(prim_nodes, d) for d in ordered])
                             if keep_primitive_daily else None),
            metadata=md, nan_accounting=st.acct.to_dict(),
            day_diagnostics=pl.DataFrame([{
                "DATE": d, "n_headlines": st.narr[d].n_headlines,
                "n_unassigned": st.narr[d].n_unassigned,
                "unassigned_share": st.narr[d].n_unassigned / max(st.narr[d].n_headlines, 1),
                "narratives_touched": int((st.narr[d].count > 0).sum()),
            } for d in ordered]),
            n_days=len(ordered), peak_rss_gb=peak_rss, variant=v.key,
        )
    return out


def summarize(result: SpecResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One comparison row per config, with EVERY config field always present.

    v1's summarize merged only what the caller passed, so a substudy that
    varied one axis printed nulls for the others (the B2 semantic row showed a
    null correction and use_f0). Here the row is built from the full
    RunMetadata, so no column is null merely because a call did not restate a
    default.
    """
    row: dict[str, Any] = dict(result.metadata.to_dict())
    row["variant"] = result.variant
    row.update(result.nan_accounting)
    diag = result.day_diagnostics
    row["n_days"] = result.n_days
    row["mean_unassigned_share"] = float(diag["unassigned_share"].mean())
    row["mean_narratives_touched"] = float(diag["narratives_touched"].mean())
    nd = result.narrative_daily
    row["mean_support"] = float(nd["SUPPORT"].mean())
    row["mean_intensity_present"] = float(nd["INTENSITY"].drop_nulls().mean() or float("nan"))
    row["narrative_day_rows"] = nd.height
    row["peak_rss_gb"] = result.peak_rss_gb
    if result.primitive_daily is not None:
        row["retained_per_headline"] = (
            float(result.primitive_daily["SUPPORT"].sum())
            / max(float(diag["n_headlines"].sum()), 1.0)
        )
    row.update(extra or {})
    return row


def attention_share(narrative_daily: pl.DataFrame) -> pl.DataFrame:
    """Derived downstream view (never stored): each narrative's share of the day's SUPPORT.

    Share rather than raw SUPPORT because headline volume grows ~35x across
    2004-2022, so a raw count conflates narrative attention with corpus size.
    Per spec section 3 this is derived at analysis time and never persisted.
    """
    return narrative_daily.with_columns(
        (pl.col("SUPPORT") / pl.col("SUPPORT").sum().over("DATE")).alias("SHARE")
    )
