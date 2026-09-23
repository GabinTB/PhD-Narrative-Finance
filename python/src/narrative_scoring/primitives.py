"""The primitive table: taxonomy CSV + paraphrase JSONL, joined on the sha1 path hash.

This is the scoring-unit view of a frozen taxonomy (spec section 1):

    reservoir -> dimension -> narrative -> sub-mechanism -> primitive

One primitive = one CSV row = one master description + K paraphrases. The
JSONL is joined on the sha1 of the primitive *path*, never on DISPLAY_NAME
or CATEGORY (both have collided historically and silently emptied a join).
The join is asserted to be a bijection, not assumed.

Also here: the primitive-text embedding cache, the mode-corrected scoring
matrix, and the pooled primitive score S[h, p] = pool_k cos(h, text[p, k]).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from narrative_scoring.config import PoolRule
from narrative_scoring.corrections import Correction, apply_mode, l2_normalise
from narrative_scoring.schema import EMBEDDING_DIM

log = logging.getLogger(__name__)

TAXONOMY_CSV = "{name}_taxonomy.authored.csv"
PARAPHRASE_JSONL = "{name}-primitive_{style}_paraphrases.jsonl"

# Strict vendor (RavenPack) schema, plus the one column the vendor lacks. Anything else in
# the CSV (e.g. a POLARITY column) is dropped on load; polarity IS SUB_TYPE.
VENDOR_COLUMNS = ("TOPIC", "GROUP", "TYPE", "SUB_TYPE", "ROLE", "CATEGORY", "DISPLAY_NAME",
                  "DESCRIPTION", "SCHEDULED", "VALID_ENTITY_TYPES", "TAGS")
CHANNEL_COLUMN = "OBSERVABILITY_CHANNEL"
REQUIRED_COLUMNS = ("TOPIC", "GROUP", "TYPE", "SUB_TYPE", "ROLE", "CATEGORY", "DISPLAY_NAME")
# The spec's unique primitive key. CATEGORY alone is NOT unique.
PRIMITIVE_KEY = ("TOPIC", "GROUP", "CATEGORY", "ROLE", CHANNEL_COLUMN)
# The path whose sha1 keys the paraphrase JSONL: ALWAYS these 7 fields, a missing channel
# being an empty trailing segment ("a/b/c/d/e/f/"), so every taxonomy hashes on one base.
# The authoring notebook (author_primitives.ipynb, cell 9) uses the same rule.
PATH_FIELDS = ("TOPIC", "GROUP", "TYPE", "SUB_TYPE", "CATEGORY", "ROLE", CHANNEL_COLUMN)

NARRATIVE_COLUMNS = ["reservoir", "dimension", "narrative", "pole", "narrative_key", "n_primitives"]
PRIMITIVE_COLUMNS = [
    "reservoir", "dimension", "narrative", "pole", "sub_mechanism",
    "observability_channel", "primitive", "narrative_key",
]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def primitive_path(row: dict[str, str]) -> str:
    return "/".join(row[c] for c in PATH_FIELDS)


@dataclass
class PrimitiveTable:
    """The scoring-unit table plus its primitive texts.

    ``frame`` is sorted by (narrative_key, primitive) so each narrative's
    primitives are contiguous; ``primitive_id`` is the row index, which is
    the column index of the score matrix. ``texts`` is primitive-major:
    primitive i owns texts [i*n_texts, (i+1)*n_texts), master first when
    included.
    """

    name: str
    style: str
    frame: pl.DataFrame
    texts: list[str]
    n_texts: int
    k_paraphrases: int
    narrative_frame: pl.DataFrame
    taxonomy_sha1: str
    paraphrase_sha1: str
    has_observability_channel: bool = True

    @property
    def n_primitives(self) -> int:
        return self.frame.height

    @property
    def n_narratives(self) -> int:
        return self.narrative_frame.height

    @property
    def primitive_to_narrative(self) -> np.ndarray:
        """(n_primitives,) int32: narrative_id of each primitive, in score-column order."""
        return self.frame["narrative_id"].to_numpy().astype(np.int32)

    @property
    def narrative_nodes(self) -> pl.DataFrame:
        return self.narrative_frame.select(NARRATIVE_COLUMNS)

    @property
    def primitive_nodes(self) -> pl.DataFrame:
        return self.frame.select(PRIMITIVE_COLUMNS)


def load_primitive_table(
    taxonomy_root: Path,
    name: str = "Evergreen_v5",
    style: str = "headline",
    *,
    include_master: bool = True,
) -> PrimitiveTable:
    """Load the CSV and paraphrase JSONL, joined on the sha1 path hash.

    Only ``VENDOR_COLUMNS`` + ``OBSERVABILITY_CHANNEL`` are kept; the channel is
    added as an empty column when the vendor file has none (before hashing, so
    it contributes the empty trailing path segment). Raises ``ValueError`` when
    a required column is missing, the join is not a bijection, K is not
    uniform, or the primitive key is not unique.
    """
    root = Path(taxonomy_root)
    csv_path = root / TAXONOMY_CSV.format(name=name)
    jsonl_path = root / PARAPHRASE_JSONL.format(name=name, style=style)

    # missing_utf8_is_empty_string: an unsigned narrative has an EMPTY SUB_TYPE
    # and the JSONL path keeps it as an empty segment ("a/b//c"). Read as
    # null, concat_str would propagate the null and silently drop every
    # unsigned primitive from the join.
    csv = pl.read_csv(
        csv_path, infer_schema_length=0, missing_utf8_is_empty_string=True
    )
    missing = [c for c in REQUIRED_COLUMNS if c not in csv.columns]
    if missing:
        raise ValueError(f"taxonomy CSV {csv_path.name} lacks column(s) {missing}")
    has_channel = CHANNEL_COLUMN in csv.columns
    dropped = [c for c in csv.columns if c not in VENDOR_COLUMNS and c != CHANNEL_COLUMN]
    if dropped:
        log.info("%s: non-vendor column(s) ignored: %s", csv_path.name, dropped)
    csv = csv.select([c for c in (*VENDOR_COLUMNS, CHANNEL_COLUMN) if c in csv.columns])
    if not has_channel:
        csv = csv.with_columns(pl.lit("").alias(CHANNEL_COLUMN))
    csv = csv.with_columns(
        pl.concat_str([pl.col(c).fill_null("") for c in PATH_FIELDS], separator="/").alias("_path")
    ).with_columns(
        pl.col("_path").map_elements(_sha1, return_dtype=pl.String).alias("path_id")
    )

    n_dupe_key = csv.height - csv.select(list(PRIMITIVE_KEY)).unique().height
    if n_dupe_key:
        raise ValueError(f"{n_dupe_key} duplicate primitive key(s) on {PRIMITIVE_KEY}")

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    para = pl.DataFrame(
        [{"path_id": r["id"], "master": r["master"], "paraphrases": r["paraphrases"]}
         for r in records],
        schema={"path_id": pl.String, "master": pl.String, "paraphrases": pl.List(pl.String)},
    )
    if para["path_id"].n_unique() != para.height:
        raise ValueError("paraphrase JSONL has duplicate ids")

    joined = csv.join(para, on="path_id", how="inner")
    if joined.height != csv.height or joined.height != para.height:
        raise ValueError(
            "CSV<->JSONL bijection broken on the sha1 path hash: "
            f"{csv.height} CSV rows, {para.height} JSONL records, {joined.height} joined"
        )

    frame = (
        joined.rename({
            "TOPIC": "reservoir", "GROUP": "dimension", "CATEGORY": "narrative",
            "SUB_TYPE": "pole", "ROLE": "sub_mechanism",
            "OBSERVABILITY_CHANNEL": "observability_channel", "DISPLAY_NAME": "primitive",
        })
        .with_columns(
            # a narrative is unique only when scoped by its parents (spec section 1)
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
            pl.col("reservoir").first(), pl.col("dimension").first(),
            pl.col("narrative").first(), pl.col("pole").first(),
            pl.len().alias("n_primitives"),
        )
        .with_row_index("narrative_id")
    )
    frame = frame.join(narrative_frame.select("narrative_key", "narrative_id"), on="narrative_key")

    ks = {len(p) for p in frame["paraphrases"].to_list()}
    if len(ks) != 1:
        raise ValueError(f"non-uniform K across primitives: {sorted(ks)}")
    k = ks.pop()

    texts: list[str] = []
    for master, paras in zip(frame["master"].to_list(), frame["paraphrases"].to_list()):
        if include_master:
            texts.append(master)
        texts.extend(paras)

    return PrimitiveTable(
        name=name, style=style, frame=frame, texts=texts,
        n_texts=(1 if include_master else 0) + k, k_paraphrases=k,
        narrative_frame=narrative_frame,
        taxonomy_sha1=file_sha1(csv_path), paraphrase_sha1=file_sha1(jsonl_path),
        has_observability_channel=has_channel,
    )


# ---------------------------------------------------------------------------
# Spec section 4: taxonomy-only sanity checks (1, 2, 3, 6). Check 5 (a POLARITY
# column coherent with SUB_TYPE) is gone: the schema is strict vendor, polarity IS SUB_TYPE.
# ---------------------------------------------------------------------------

def sanity_checks(table: PrimitiveTable) -> pl.DataFrame:
    """Checks 1, 2, 3, 6 of the spec. Score-dependent checks live in validation.py."""
    f = table.frame
    rows: list[dict[str, Any]] = []

    key_cols = ["reservoir", "dimension", "narrative", "sub_mechanism", "observability_channel"]
    n_dupe = f.height - f.select(key_cols).unique().height
    rows.append({
        "check": "1. primitive key uniqueness",
        "result": "PASS" if n_dupe == 0 else "FAIL",
        "detail": f"{f.height} primitives, {n_dupe} duplicate key(s) on {PRIMITIVE_KEY}",
    })
    rows.append({
        "check": "2. CSV<->JSONL bijection (sha1 path hash)",
        "result": "PASS",
        "detail": f"{f.height} primitives joined on path_id; enforced in load_primitive_table",
    })
    rows.append({
        "check": "3. uniform K paraphrases",
        "result": "PASS",
        "detail": f"K={table.k_paraphrases}; {table.n_texts} text(s) per primitive "
                  f"(master {'included' if table.n_texts > table.k_paraphrases else 'EXCLUDED'})",
    })

    orphans = orphan_pole_candidates(table)
    signed_f = f.filter(pl.col("pole").fill_null("") != "")
    rows.append({
        "check": "6. bipolar mirror presence",
        "result": "REVIEW",
        "detail": f"{signed_f['narrative'].n_unique()} signed narratives; {orphans.height} "
                  f"dimension(s) carry exactly one signed narrative (candidate orphan poles). "
                  f"Sibling linkage is not encoded in the schema; review by hand.",
    })
    return pl.DataFrame(rows)


def orphan_pole_candidates(table: PrimitiveTable) -> pl.DataFrame:
    """Dimensions carrying exactly one signed narrative: candidate orphan poles."""
    signed = table.frame.filter(pl.col("pole").fill_null("") != "")
    counts = signed.group_by(["reservoir", "dimension"]).agg(
        pl.col("narrative").n_unique().alias("n_signed"),
        pl.col("narrative").unique().alias("narratives"),
    )
    return counts.filter(pl.col("n_signed") == 1).sort(["reservoir", "dimension"])


# ---------------------------------------------------------------------------
# Primitive-text embeddings (local cache; never a datalake artifact)
# ---------------------------------------------------------------------------

def texts_digest(table: PrimitiveTable) -> str:
    return hashlib.blake2b("\x00".join(table.texts).encode(), digest_size=8).hexdigest()


def embed_primitive_texts(
    table: PrimitiveTable,
    cache_dir: Path,
    *,
    batch_size: int = 256,
    device: str = "embedx",
) -> np.ndarray:
    """(n_primitives * n_texts, dim) L2-normalised primitive-text embeddings, primitive-major.

    Cached locally by a digest of the exact text list, so any taxonomy or
    master-inclusion change busts the cache.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"primitive_texts__{texts_digest(table)}.npy"
    if path.exists():
        log.info("primitive-text embeddings: cache hit %s", path.name)
        return np.load(path)

    from ravenpack.headlines.embed import load_embedding_model

    log.info("embedding %d primitive texts via %s ...", len(table.texts), device)
    model = load_embedding_model(Path("."), device=device)
    vectors = np.asarray(model.encode(table.texts, batch_size=batch_size), dtype=np.float32)
    expected = (len(table.texts), EMBEDDING_DIM)
    if vectors.shape != expected:
        raise ValueError(f"encode returned {vectors.shape}, expected {expected}")
    vectors = l2_normalise(vectors)
    np.save(path, vectors)
    return vectors


def embeddings_digest(P: np.ndarray) -> str:
    raw = np.ascontiguousarray(P, dtype=np.float32).tobytes()
    return hashlib.blake2b(raw, digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Scoring matrix and pooled primitive scores (steps 1-2 on the target side)
# ---------------------------------------------------------------------------

def to_text_major(vectors: np.ndarray, n_texts: int) -> np.ndarray:
    """(n_prim * n_texts, dim) primitive-major -> text-major (slab j = text j of every primitive).

    Pooling then reduces over contiguous (n_head, n_prim) slabs instead of a
    strided inner axis, which is ~18x faster in numpy. Output is identical.
    """
    n_prim = vectors.shape[0] // n_texts
    order = np.arange(n_prim * n_texts).reshape(n_prim, n_texts).T.ravel()
    return np.ascontiguousarray(vectors[order])


def scoring_matrix(
    P: np.ndarray, table: PrimitiveTable, mode: Correction, pooling: PoolRule,
    mu: np.ndarray | None, mu_hat: np.ndarray | None,
) -> np.ndarray:
    """Mode-corrected primitive-text matrix ready for ``S = H @ P_scoring.T``.

    ``P`` is primitive-major as returned by ``embed_primitive_texts``. For MEAN
    pooling the n_texts corrected vectors of each primitive collapse into one
    mean vector, because mean_j(h . p_j) == h . mean_j(p_j) exactly; that
    vector is deliberately NOT renormalised, which would break the identity.
    MAX and MEDIAN keep every text (text-major).
    """
    if P.shape != (table.n_primitives * table.n_texts, EMBEDDING_DIM):
        raise ValueError(
            f"P has shape {P.shape}, expected {(table.n_primitives * table.n_texts, EMBEDDING_DIM)}"
        )
    P_mode = apply_mode(to_text_major(P, table.n_texts), mode, mu, mu_hat)
    if pooling is PoolRule.MEAN:
        return np.ascontiguousarray(
            P_mode.reshape(table.n_texts, table.n_primitives, EMBEDDING_DIM).mean(axis=0)
        )
    return P_mode


def primitive_scores(
    H: np.ndarray, P_scoring: np.ndarray, n_primitives: int, n_texts: int, pooling: PoolRule,
) -> np.ndarray:
    """Step 2: S[h, p] = pool_k cos(h, text[p, k]) as a contiguous float32 (n_head, n_prim)."""
    R = H @ P_scoring.T
    if pooling is PoolRule.MEAN:
        return np.ascontiguousarray(R, dtype=np.float32)
    if pooling is PoolRule.MAX:
        out = R[:, :n_primitives].copy()
        for j in range(1, n_texts):
            np.maximum(out, R[:, j * n_primitives:(j + 1) * n_primitives], out=out)
        return out
    if pooling is PoolRule.MEDIAN:
        return np.ascontiguousarray(
            np.median(R.reshape(R.shape[0], n_texts, n_primitives), axis=1), dtype=np.float32
        )
    raise ValueError(f"unknown pooling: {pooling!r}")


def representative_matrix(P: np.ndarray, table: PrimitiveTable, mode: Correction,
                          mu: np.ndarray | None, mu_hat: np.ndarray | None) -> np.ndarray:
    """One unit-norm vector per primitive (mean of its corrected texts, renormalised).

    Used only to compute N_eff (f0.gram_spectrum) in the monthly tau job.
    """
    P_mode = apply_mode(to_text_major(P, table.n_texts), mode, mu, mu_hat)
    mean = P_mode.reshape(table.n_texts, table.n_primitives, EMBEDDING_DIM).mean(axis=0)
    return l2_normalise(mean)
