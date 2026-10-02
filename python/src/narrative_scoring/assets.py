"""The asset layer of a scoring pass: universe entities, membership as of a day, accumulators.

An asset is a RavenPack entity (``rp_entity_id``) of a registered ``universe`` artifact
(``universe.ingest``; ids filled by ``universe.enrich`` from the RavenPack reference).
Coverage is every entity EVER in the universe; membership is applied as of each day
(the latest snapshot on or before it, no look-ahead) and written as ``IN_UNIVERSE``
(null before the first snapshot): masking is the reader's choice. Universe rows with no
``rp_entity_id`` (ambiguous or unmatched) cannot be counted; how many there are as of
the day is reported (``N_UNMAPPED``).

A headline's assets come from its aligned RP_ENTITY_ID / RELEVANCE lists (one entry per
raw detection, so an entity detected in several events of a story appears several
times): they are collapsed to (headline, asset) with RELEVANCE = max BEFORE anything is
counted, so a story counts once per asset.

Every headline tagging an asset counts, at any relevance. ``min_relevance`` (a fraction,
default 0.6: RELEVANCE >= 60 on the vendor's 0-100 scale) is always applied, as extra
``*_REL`` columns restricted to the headlines at or above it; w = RELEVANCE / 100.

Per day, sentiment label and asset (``asset_attention_daily``):

    N_STORIES                headlines tagging the asset
    N_STORIES_REL            of which RELEVANCE >= min_relevance
    REL_SUM                  sum of w

Per day, label, narrative and asset (``narrative_asset_daily``), over the headlines that
tag the asset AND have a narrative score s (the pooled score narrative_daily sums), the
statistics narrative_daily keeps, as additive sums plus the peak:

    SUPPORT, TOTAL_SCORE, SUMSQ, PEAK            every such headline: count, sum s,
                                                 sum s^2, max s
    SUPPORT_REL, TOTAL_SCORE_REL, SUMSQ_REL,     the same over headlines with
    PEAK_REL                                     RELEVANCE >= min_relevance (PEAK_REL null
                                                 when SUPPORT_REL is 0)
    TOTAL_SCORE_RELW, SUMSQ_RELW                 relevance-weighted: sum s*w, sum (s*w)^2

INTENSITY = TOTAL / SUPPORT and STD = sqrt(SUMSQ / SUPPORT - INTENSITY^2) (ddof=0, the
narrative_daily statistics) follow from the sums over any window or set of labels; PEAK
aggregates by max. Both tables are sparse (no row = no support).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from datalake import Artifact

ENTITY_COL = "RP_ENTITY_ID"
ASSET_COL = "_asset"

ASSET_ATTENTION_SCHEMA: pl.Schema = pl.Schema({
    "DATE": pl.Date, "SENTIMENT": pl.String, "RP_ENTITY_ID": pl.String,
    "N_STORIES": pl.Int32, "N_STORIES_REL": pl.Int32, "REL_SUM": pl.Float64,
    "IN_UNIVERSE": pl.Boolean, "N_HEADLINES": pl.Int32, "N_LABELLED": pl.Int32,
})
NARRATIVE_ASSET_SCHEMA: pl.Schema = pl.Schema({
    "DATE": pl.Date, "SENTIMENT": pl.String, "narrative_key": pl.String,
    "RP_ENTITY_ID": pl.String,
    "SUPPORT": pl.Int32, "TOTAL_SCORE": pl.Float64, "SUMSQ": pl.Float64, "PEAK": pl.Float32,
    "SUPPORT_REL": pl.Int32, "TOTAL_SCORE_REL": pl.Float64, "SUMSQ_REL": pl.Float64,
    "PEAK_REL": pl.Float32,
    "TOTAL_SCORE_RELW": pl.Float64, "SUMSQ_RELW": pl.Float64,
    "IN_UNIVERSE": pl.Boolean, "N_HEADLINES": pl.Int32, "N_LABELLED": pl.Int32,
})
MIN_RELEVANCE = 0.6


@dataclass
class AssetUniverse:
    """Assets (sorted entity ids) and their membership snapshot by snapshot."""

    artifact_id: str
    entity_ids: np.ndarray                 # str, sorted unique, one per asset
    snapshots: list[date]                  # sorted
    member: np.ndarray                     # bool (n_snapshots, n_assets)
    n_unmapped: np.ndarray                 # int64 per snapshot: rows without an entity id
    min_relevance: float = MIN_RELEVANCE   # fraction of the vendor's 0-100 RELEVANCE

    def __post_init__(self) -> None:
        self.min_relevance = float(self.min_relevance)
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError(f"min_relevance is a fraction in [0, 1] (0.6 = RELEVANCE 60), "
                             f"got {self.min_relevance}")
        if self.member.shape != (len(self.snapshots), len(self.entity_ids)):
            raise ValueError("membership matrix does not match snapshots x assets")

    @property
    def n_assets(self) -> int:
        return int(len(self.entity_ids))

    @property
    def relevance_floor(self) -> int:
        """The smallest vendor RELEVANCE (0-100 integer) at or above ``min_relevance``."""
        return relevance_floor(self.min_relevance)

    @classmethod
    def from_frame(cls, frame: pl.DataFrame, *, artifact_id: str,
                   min_relevance: float = MIN_RELEVANCE) -> AssetUniverse:
        """From universe rows (``snapshot_date``, ``rp_entity_id``)."""
        missing = {"snapshot_date", "rp_entity_id"} - set(frame.columns)
        if missing:
            raise ValueError(f"universe {artifact_id} has no column(s) {sorted(missing)}; "
                             "register it with RavenPack entity ids (universe.enrich)")
        f = frame.select(pl.col("snapshot_date").cast(pl.Date), "rp_entity_id")
        ids = np.array(sorted(f["rp_entity_id"].drop_nulls().unique().to_list()), dtype=object)
        if not len(ids):
            raise ValueError(f"universe {artifact_id} maps no row to a RavenPack entity")
        snaps = sorted(f["snapshot_date"].unique().to_list())
        pos = {s: i for i, s in enumerate(snaps)}
        col = {e: j for j, e in enumerate(ids)}
        member = np.zeros((len(snaps), len(ids)), dtype=bool)
        mapped = f.drop_nulls("rp_entity_id").unique()
        member[[pos[s] for s in mapped["snapshot_date"].to_list()],
               [col[e] for e in mapped["rp_entity_id"].to_list()]] = True
        unm = (f.group_by("snapshot_date")
               .agg(pl.col("rp_entity_id").null_count().alias("n")))
        n_unmapped = np.zeros(len(snaps), dtype=np.int64)
        for s, n in unm.iter_rows():
            n_unmapped[pos[s]] = n
        return cls(artifact_id, ids, snaps, member, n_unmapped, min_relevance)

    @classmethod
    def from_artifact(cls, art: Artifact, *,
                      min_relevance: float = MIN_RELEVANCE) -> AssetUniverse:
        files = art.files("*.parquet")
        if not files:
            raise ValueError(f"{art.artifact_id} holds no parquet file")
        frame = pl.concat([pl.read_parquet(p) for p in files], how="diagonal_relaxed")
        return cls.from_frame(frame, artifact_id=art.artifact_id, min_relevance=min_relevance)

    def asset_map(self) -> pl.DataFrame:
        """RP_ENTITY_ID -> asset index (Int32), for joining entity lists."""
        return pl.DataFrame({ENTITY_COL: list(self.entity_ids),
                             ASSET_COL: np.arange(self.n_assets, dtype=np.int32)},
                            schema={ENTITY_COL: pl.String, ASSET_COL: pl.Int32})

    def as_of(self, day: date) -> tuple[np.ndarray | None, int | None]:
        """(IN_UNIVERSE per asset, N_UNMAPPED) at the latest snapshot <= ``day``; (None,
        None) before the first snapshot."""
        i = bisect.bisect_right(self.snapshots, day) - 1
        if i < 0:
            return None, None
        return self.member[i], int(self.n_unmapped[i])

    def params(self) -> dict[str, Any]:
        """Identity of the asset outputs (never of narrative_daily)."""
        return {"universe_artifact_id": self.artifact_id, "min_relevance": self.min_relevance}


def relevance_floor(min_relevance: float) -> int:
    """The smallest integer RELEVANCE r with r / 100 >= ``min_relevance`` (0.6 -> 60)."""
    return int(np.ceil(round(float(min_relevance) * 100, 9)))


def headline_assets(rows: pl.DataFrame, asset_map: pl.DataFrame,
                    n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR (indptr int64 [n_rows + 1], asset int32, relevance uint8) of each row's assets.

    ``rows`` holds ``_r`` (the row position, 0..n_rows-1) and the aligned RP_ENTITY_ID /
    RELEVANCE lists; entities outside ``asset_map`` are dropped and every (row, asset)
    is kept once, with the maximum RELEVANCE, assets in ascending order within a row."""
    ent = (rows.select("_r", ENTITY_COL, "RELEVANCE")
           .explode([ENTITY_COL, "RELEVANCE"], empty_as_null=True)
           .drop_nulls(ENTITY_COL)
           .join(asset_map, on=ENTITY_COL, how="inner")
           .group_by("_r", ASSET_COL).agg(pl.col("RELEVANCE").max())
           .sort("_r", ASSET_COL))
    if ent["RELEVANCE"].null_count():
        raise ValueError(f"{ent['RELEVANCE'].null_count()} detection(s) of universe entities "
                         "without RELEVANCE")
    r = ent["_r"].to_numpy().astype(np.int64)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(r, minlength=n_rows), out=indptr[1:])
    return (indptr, ent[ASSET_COL].to_numpy().astype(np.int32),
            ent["RELEVANCE"].to_numpy().astype(np.uint8))


@dataclass
class AssetDay:
    """One day's asset accumulators, for ``n_labels`` sentiment labels; ``min_relevance``
    is the fraction of the vendor's 0-100 RELEVANCE the *_REL columns require."""

    n_labels: int
    n_narr: int
    n_assets: int
    min_relevance: float = MIN_RELEVANCE
    n_stories: np.ndarray = field(init=False)
    n_rel: np.ndarray = field(init=False)
    rel_sum: np.ndarray = field(init=False)
    _cross: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list,
                                                                     init=False)

    def __post_init__(self) -> None:
        shape = (self.n_labels, self.n_assets)
        self.n_stories = np.zeros(shape, dtype=np.int64)
        self.n_rel = np.zeros(shape, dtype=np.int64)
        self.rel_sum = np.zeros(shape, dtype=np.float64)
        self._floor = relevance_floor(self.min_relevance)

    def add_attention(self, label: np.ndarray, indptr: np.ndarray, asset: np.ndarray,
                      rel: np.ndarray) -> None:
        """Rows with label >= 0 count once per asset they tag."""
        per_row = np.diff(indptr)
        lab = np.repeat(label.astype(np.int64), per_row)
        keep = lab >= 0
        if not keep.any():
            return
        key = lab[keep] * self.n_assets + asset[keep]
        size = self.n_labels * self.n_assets
        r = rel[keep]
        self.n_stories += np.bincount(key, minlength=size).reshape(self.n_stories.shape)
        self.n_rel += np.bincount(key, weights=(r >= self._floor),
                                  minlength=size).astype(np.int64).reshape(self.n_rel.shape)
        self.rel_sum += np.bincount(key, weights=r.astype(np.float64) / 100.0,
                                    minlength=size).reshape(self.rel_sum.shape)

    def add_cross(self, label: np.ndarray, trip_n: np.ndarray, trip_narr: np.ndarray,
                  trip_score: np.ndarray, indptr: np.ndarray, asset: np.ndarray,
                  rel: np.ndarray) -> None:
        """(row, narrative, s) triplets x the row's assets, rows with label >= 0."""
        rows = np.repeat(np.arange(len(trip_n), dtype=np.int64), trip_n)
        if not rows.size:
            return
        tn = trip_n.astype(np.int64)
        cols = np.arange(rows.size) - np.repeat(np.cumsum(tn) - tn, tn)
        narr = trip_narr[rows, cols].astype(np.int64)
        s = trip_score[rows, cols]
        lab = label[rows].astype(np.int64)
        k_assets = indptr[rows + 1] - indptr[rows]
        ok = (lab >= 0) & (k_assets > 0)
        rows, narr, s, lab, k_assets = rows[ok], narr[ok], s[ok], lab[ok], k_assets[ok]
        if not rows.size:
            return
        # expand every (row, narrative) by the row's assets
        rep = np.repeat(np.arange(rows.size), k_assets)
        start = np.repeat(indptr[rows], k_assets)
        first = np.repeat(np.cumsum(k_assets) - k_assets, k_assets)
        pos = start + (np.arange(rep.size) - first)
        a, w8 = asset[pos].astype(np.int64), rel[pos]
        sv, w = s[rep], w8.astype(np.float64) / 100.0
        hi = (w8 >= self._floor).astype(np.float64)
        key = (lab[rep] * self.n_narr + narr[rep]) * self.n_assets + a
        sums = np.stack([np.ones_like(sv), sv, sv * sv,
                         hi, sv * hi, sv * sv * hi,
                         sv * w, (sv * w) ** 2], axis=1)
        peaks = np.stack([sv, np.where(hi > 0, sv, -np.inf)], axis=1)
        self._cross.append(_reduce(key, sums, peaks))

    def cross(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(keys, sums, peaks) of the day. sums columns: SUPPORT, TOTAL_SCORE, SUMSQ,
        SUPPORT_REL, TOTAL_SCORE_REL, SUMSQ_REL, TOTAL_SCORE_RELW, SUMSQ_RELW; peaks: PEAK,
        PEAK_REL (-inf where no headline clears the relevance threshold)."""
        if not self._cross:
            return np.empty(0, dtype=np.int64), np.empty((0, 8)), np.empty((0, 2))
        keys = np.concatenate([k for k, _, _ in self._cross])
        sums = np.concatenate([v for _, v, _ in self._cross])
        peaks = np.concatenate([m for _, _, m in self._cross])
        return _reduce(keys, sums, peaks)

    def frames(self, day: date, labels: list[str], narrative_keys: list[str],
               entity_ids: np.ndarray, in_universe: np.ndarray | None,
               n_headlines: int, n_labelled: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
        """The day's asset_attention_daily and narrative_asset_daily rows (sparse)."""
        lab_idx, a_idx = np.nonzero(self.n_stories)
        inu = (pl.Series("IN_UNIVERSE", in_universe[a_idx], dtype=pl.Boolean)
               if in_universe is not None
               else pl.Series("IN_UNIVERSE", [None] * len(a_idx), dtype=pl.Boolean))
        att = pl.DataFrame({
            "DATE": [day] * len(a_idx),
            "SENTIMENT": [labels[i] for i in lab_idx],
            "RP_ENTITY_ID": entity_ids[a_idx].tolist(),
            "N_STORIES": self.n_stories[lab_idx, a_idx],
            "N_STORIES_REL": self.n_rel[lab_idx, a_idx],
            "REL_SUM": self.rel_sum[lab_idx, a_idx],
        }).with_columns(inu, pl.lit(n_headlines).alias("N_HEADLINES"),
                        pl.Series("N_LABELLED", [n_labelled[i] for i in lab_idx]))
        keys, sums, peaks = self.cross()
        a = keys % self.n_assets
        ln = keys // self.n_assets
        nid, lab = ln % self.n_narr, ln // self.n_narr
        inu2 = (pl.Series("IN_UNIVERSE", in_universe[a], dtype=pl.Boolean)
                if in_universe is not None
                else pl.Series("IN_UNIVERSE", [None] * len(a), dtype=pl.Boolean))
        peak_rel = np.where(sums[:, 3] > 0, peaks[:, 1], np.nan)
        cross = pl.DataFrame({
            "DATE": [day] * len(keys),
            "SENTIMENT": [labels[i] for i in lab],
            "narrative_key": [narrative_keys[i] for i in nid],
            "RP_ENTITY_ID": entity_ids[a].tolist(),
            "SUPPORT": sums[:, 0], "TOTAL_SCORE": sums[:, 1], "SUMSQ": sums[:, 2],
            "PEAK": peaks[:, 0],
            "SUPPORT_REL": sums[:, 3], "TOTAL_SCORE_REL": sums[:, 4], "SUMSQ_REL": sums[:, 5],
            "PEAK_REL": peak_rel,
            "TOTAL_SCORE_RELW": sums[:, 6], "SUMSQ_RELW": sums[:, 7],
        }).with_columns(inu2, pl.lit(n_headlines).alias("N_HEADLINES"),
                        pl.Series("N_LABELLED", [n_labelled[i] for i in lab]),
                        pl.col("PEAK_REL").fill_nan(None))
        return (att.select([pl.col(c).cast(t) for c, t in ASSET_ATTENTION_SCHEMA.items()]),
                cross.select([pl.col(c).cast(t) for c, t in NARRATIVE_ASSET_SCHEMA.items()]))


def _reduce(keys: np.ndarray, sums: np.ndarray,
            peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per unique key (sorted): the column sums of ``sums`` (float64) and the column maxima
    of ``peaks``."""
    if not keys.size:
        return keys.astype(np.int64), np.empty((0, sums.shape[1])), np.empty((0, peaks.shape[1]))
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    inv = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, k.size]))
    out = np.stack([np.bincount(inv, weights=sums[order, c], minlength=starts.size)
                    for c in range(sums.shape[1])], axis=1)
    mx = np.stack([np.maximum.reduceat(peaks[order, c], starts)
                   for c in range(peaks.shape[1])], axis=1)
    return k[starts], out, mx
