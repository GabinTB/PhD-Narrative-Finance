"""Validation analytics for the v2 notebooks (crisis batteries, CIs, controls).

Everything here is DERIVED and never stored: per spec section 3 the artifact is
the raw (day x narrative) table, and attention shares, z-scores and rankings are
computed downstream at analysis time.

The statistics are deliberately conservative about two things the v1 notebooks
got wrong:

  * comparability -- narrative attention is reported as a SHARE of the day's
    total support, because RavenPack headline volume grows ~35x between 2004
    and 2022 and a raw count conflates attention with corpus size;
  * dependence -- daily narrative series are strongly autocorrelated, so
    confidence intervals come from a WEEK-BLOCK bootstrap rather than an iid
    resample, which would understate the interval badly.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

# Match on the NARRATIVE NAME only. Searching sub_mechanism text as well was
# far too permissive: "transfer" pulled coup-and-unconstitutional-transfer into
# a fiscal-stimulus battery, and "restriction" pulled in export controls.
SEARCH_COLUMNS = ("narrative",)


# ---------------------------------------------------------------------------
# Predeclared crisis batteries
# ---------------------------------------------------------------------------

@dataclass
class Battery:
    """A predeclared must-fire / must-not-fire list, resolved against the taxonomy."""

    label: str
    patterns: tuple[str, ...]
    matched: pl.DataFrame          # narrative_key | reservoir | dimension | narrative | pole
    unmatched_patterns: tuple[str, ...]

    @property
    def keys(self) -> list[str]:
        return self.matched["narrative_key"].to_list()

    def __len__(self) -> int:
        return self.matched.height


def resolve_battery(
    table: Any, label: str, patterns: Sequence[str], exclude: Sequence[str] = ()
) -> Battery:
    """Resolve narrative-name patterns to narrative keys, for printing BEFORE scoring.

    Matching is a case-insensitive substring test against the narrative name.
    ``exclude`` removes names afterwards, which is how an opposite pole is kept
    out: "contagion" alone would sweep in contagion-CONTAINMENT, and
    "credit-deterioration" would sweep in credit-STRENGTHENING.

    Patterns that match nothing are reported rather than silently dropped -- a
    battery that quietly resolves to zero narratives makes every downstream
    check vacuously pass.
    """
    base = ["narrative_key", "reservoir", "dimension", "narrative", "pole"]
    cols = [c for c in SEARCH_COLUMNS if c in table.frame.columns]
    hay = table.frame.select(
        base + [c for c in cols if c not in base]      # SEARCH_COLUMNS overlaps base
    ).with_columns(
        pl.concat_str([pl.col(c).fill_null("") for c in cols], separator=" ").str.to_lowercase().alias("_hay")
    )
    matched_keys: set[str] = set()
    unmatched: list[str] = []
    for pat in patterns:
        hits = hay.filter(pl.col("_hay").str.contains(pat.lower(), literal=False))
        if hits.height == 0:
            unmatched.append(pat)
        matched_keys.update(hits["narrative_key"].to_list())
    if exclude:
        drop = set()
        for pat in exclude:
            drop.update(
                hay.filter(pl.col("_hay").str.contains(pat.lower(), literal=False))["narrative_key"].to_list()
            )
        matched_keys -= drop

    matched = (
        hay.filter(pl.col("narrative_key").is_in(list(matched_keys)))
        .select(["narrative_key", "reservoir", "dimension", "narrative", "pole"])
        .unique(subset=["narrative_key"])
        .sort(["reservoir", "dimension", "narrative"])
    )
    return Battery(label, tuple(patterns), matched, tuple(unmatched))


# ---------------------------------------------------------------------------
# Attention-share series and crisis z-scores
# ---------------------------------------------------------------------------

def battery_series(share: pl.DataFrame, keys: Iterable[str]) -> pl.DataFrame:
    """Mean SHARE per day across a battery's narratives."""
    keys = list(keys)
    return (
        share.filter(pl.col("narrative_key").is_in(keys))
        .group_by("DATE").agg(pl.col("SHARE").mean().alias("SHARE"))
        .sort("DATE")
    )


def _window(series: pl.DataFrame, lo: date, hi: date) -> np.ndarray:
    return series.filter(pl.col("DATE").is_between(pl.lit(lo), pl.lit(hi)))["SHARE"].to_numpy()


def crisis_z(series: pl.DataFrame, baseline: tuple[date, date], event: tuple[date, date]) -> float:
    """Event-window mean expressed in baseline standard deviations."""
    b, e = _window(series, *baseline), _window(series, *event)
    if b.size == 0 or e.size == 0:
        return float("nan")
    sd = b.std(ddof=1)
    return float("nan") if not sd else float((e.mean() - b.mean()) / sd)


def ratio(series: pl.DataFrame, baseline: tuple[date, date], event: tuple[date, date]) -> float:
    b, e = _window(series, *baseline), _window(series, *event)
    if b.size == 0 or e.size == 0 or not b.mean():
        return float("nan")
    return float(e.mean() / b.mean())


# ---------------------------------------------------------------------------
# Week-block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    series: pl.DataFrame,
    stat: Callable[[pl.DataFrame], float],
    *,
    block_days: int = 7,
    n_boot: int = 500,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float, float]:
    """(point, lo, hi) for ``stat`` under a moving-block bootstrap.

    Contiguous ``block_days`` blocks are resampled with replacement to rebuild a
    series of the original length, preserving within-week dependence. An iid
    day bootstrap would treat 1,096 autocorrelated days as 1,096 independent
    observations and report intervals far too tight.
    """
    point = stat(series)
    df = series.sort("DATE")
    n = df.height
    if n < block_days * 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_days))
    starts = np.arange(n - block_days + 1)
    dates = df["DATE"].to_numpy()
    shares = df["SHARE"].to_numpy()

    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(starts, size=n_blocks, replace=True)
        idx = (pick[:, None] + np.arange(block_days)[None, :]).ravel()[:n]
        draws[b] = stat(pl.DataFrame({"DATE": dates[idx], "SHARE": shares[idx]}))
    lo, hi = np.nanpercentile(draws, ci)
    return point, float(lo), float(hi)


def battery_report(
    share: pl.DataFrame, battery: Battery,
    baseline: tuple[date, date], event: tuple[date, date],
    *, n_boot: int = 500, seed: int = 0,
) -> dict[str, Any]:
    """Point z and ratio for a battery, each with a week-block bootstrap CI."""
    series = battery_series(share, battery.keys)
    z, z_lo, z_hi = block_bootstrap_ci(
        series, lambda s: crisis_z(s, baseline, event), n_boot=n_boot, seed=seed)
    r, r_lo, r_hi = block_bootstrap_ci(
        series, lambda s: ratio(s, baseline, event), n_boot=n_boot, seed=seed)
    return {
        "battery": battery.label, "n_narratives": len(battery),
        "z": z, "z_lo": z_lo, "z_hi": z_hi,
        "ratio": r, "ratio_lo": r_lo, "ratio_hi": r_hi,
    }


# ---------------------------------------------------------------------------
# Full-panel ranking
# ---------------------------------------------------------------------------

def rank_narratives(
    share: pl.DataFrame, baseline: tuple[date, date], event: tuple[date, date],
    *, n_boot: int = 200, seed: int = 0, top: int | None = None,
) -> pl.DataFrame:
    """Rank EVERY narrative by attention-share change at the crisis peak.

    Ranking the whole panel rather than hand-picked categories is the point:
    a spurious narrative in the top ranks is only visible if nothing was
    pre-filtered out.
    """
    base = (
        share.filter(pl.col("DATE").is_between(pl.lit(baseline[0]), pl.lit(baseline[1])))
        .group_by("narrative_key").agg(pl.col("SHARE").mean().alias("base"),
                                       pl.col("SHARE").std(ddof=1).alias("base_sd"))
    )
    peak = (
        share.filter(pl.col("DATE").is_between(pl.lit(event[0]), pl.lit(event[1])))
        .group_by("narrative_key").agg(pl.col("SHARE").mean().alias("peak"))
    )
    meta = share.select(["narrative_key", "reservoir", "dimension", "narrative", "pole"]).unique(
        subset=["narrative_key"])
    out = (
        base.join(peak, on="narrative_key", how="inner").join(meta, on="narrative_key", how="left")
        .with_columns(
            (pl.col("peak") / pl.col("base")).alias("ratio"),
            ((pl.col("peak") - pl.col("base")) / pl.col("base_sd")).alias("z"),
        )
        .sort("z", descending=True, nulls_last=True)
    )
    if top:
        out = pl.concat([out.head(top), out.tail(top)])
    return out


def kendall_tau_matrix(rankings: dict[str, pl.DataFrame], key: str = "z") -> pl.DataFrame:
    """Kendall tau between each pair of configs' narrative rankings.

    Low tau means the choice of config, not the data, is driving which
    narratives look important.
    """
    from scipy.stats import kendalltau

    names = list(rankings)
    aligned = {
        n: rankings[n].select(["narrative_key", key]).sort("narrative_key")[key].to_numpy()
        for n in names
    }
    rows = []
    for a in names:
        row: dict[str, Any] = {"config": a}
        for b in names:
            x, y = aligned[a], aligned[b]
            ok = np.isfinite(x) & np.isfinite(y)
            row[b] = float(kendalltau(x[ok], y[ok]).statistic) if ok.sum() > 2 else float("nan")
        rows.append(row)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

NON_FINANCIAL_TEXTS: tuple[str, ...] = (
    "The cat sat on the warm windowsill all afternoon.",
    "Recipe: boil the pasta for nine minutes, then drain.",
    "Local library extends opening hours on Saturdays.",
    "He forgot his umbrella again this morning.",
    "The museum's new wing opens to visitors next spring.",
    "Traffic lights at the junction were replaced overnight.",
    "She is learning to play the clarinet.",
    "Weekend weather: mild, with occasional light showers.",
    "The football match ended in a goalless draw.",
    "Gardening club meets on the first Tuesday of the month.",
)


def sample_window_headlines(
    headlines_dir: Any, start: date, end: date, n: int, *, seed: int = 0,
    threads: int = 8,
) -> list[str]:
    """Draw a random sample of real headline TEXT from the window, for the junk battery."""
    import duckdb

    months, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append(f"{y}-{m:02d}.parquet")
        m, y = (1, y + 1) if m == 12 else (m + 1, y)
    files = [str(headlines_dir / mm) for mm in months if (headlines_dir / mm).exists()]
    if not files:
        return []
    conn = duckdb.connect()
    try:
        conn.execute("SET enable_progress_bar=false")
        conn.execute(f"SET threads={threads}")
        lst = ", ".join(f"'{f}'" for f in files)
        rows = conn.sql(
            f"SELECT HEADLINE FROM read_parquet([{lst}]) "
            f"WHERE HEADLINE IS NOT NULL USING SAMPLE {int(n)} ROWS (reservoir, {seed})"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows if r[0]]


def shuffle_tokens(texts: Sequence[str], seed: int = 0) -> list[str]:
    """Token-shuffle each headline: same vocabulary, destroyed syntax and meaning."""
    rng = np.random.default_rng(seed)
    out = []
    for t in texts:
        toks = re.findall(r"\S+", t)
        rng.shuffle(toks)
        out.append(" ".join(toks))
    return out


def junk_battery(
    real_texts: Sequence[str], P_scoring: Any, table: Any, cfg: Any, tau: float,
    mu, mu_hat, *, encode: Callable[[Sequence[str]], np.ndarray], seed: int = 0,
) -> dict[str, Any]:
    """Score token-shuffled headlines + short non-financial text against the taxonomy.

    Under the null the F0 gate admits at most alpha of pure noise, so the
    realized activation rate on junk is the direct empirical check on tau.
    Returns the realized rate alongside the max-score distributions needed for
    the junk-vs-real histogram.
    """
    from narrative_scoring import spec_pipeline as sp

    junk = shuffle_tokens(real_texts, seed=seed) + list(NON_FINANCIAL_TEXTS)

    def score(texts: Sequence[str]) -> np.ndarray:
        X = sp.l2_normalise(np.asarray(encode(list(texts)), dtype=np.float32))
        H = sp.apply_mode(X, cfg.mode, mu, mu_hat)
        return sp.primitive_scores(H, P_scoring, table, cfg.paraphrase_pooling)

    S_junk, S_real = score(junk), score(real_texts)
    junk_max, real_max = S_junk.max(axis=1), S_real.max(axis=1)
    return {
        "alpha": cfg.alpha,
        "n_junk": len(junk),
        "n_real": len(real_texts),
        "junk_activation_rate": float((junk_max >= tau).mean()),
        "real_activation_rate": float((real_max >= tau).mean()),
        "junk_max": junk_max,
        "real_max": real_max,
        "tau": float(tau),
    }


# ---------------------------------------------------------------------------
# Spec check 7 -- mode comparison, judged on margins not levels
# ---------------------------------------------------------------------------

def mode_comparison(
    sample_embeddings: np.ndarray, P: np.ndarray, table: Any, cfg_factory: Callable[[Any], Any],
    modes: Sequence[Any], mu, mu_hat,
) -> pl.DataFrame:
    """Score distribution and top-1 margin per embedding mode on a fixed sample.

    R2 is expected to LOWER absolute scores while WIDENING the top-1 margin.
    Judge on the margin: levels are not comparable across modes, which is also
    why tau and q are recalibrated per mode rather than carried over.
    """
    from narrative_scoring import spec_pipeline as sp

    rows = []
    for mode in modes:
        cfg = cfg_factory(mode)
        P_scoring = sp.build_scoring_matrix(P, table, cfg, mu, mu_hat)
        H = sp.apply_mode(sp.l2_normalise(sample_embeddings), mode, mu, mu_hat)
        S = sp.primitive_scores(H, P_scoring, table, cfg.paraphrase_pooling)
        part = np.partition(S, -2, axis=1)
        top1, top2 = part[:, -1], part[:, -2]
        rows.append({
            "mode": mode.value,
            "mean_score": float(S.mean()),
            "p99_score": float(np.quantile(S, 0.99)),
            "mean_top1": float(top1.mean()),
            "mean_top1_margin": float((top1 - top2).mean()),
            "median_top1_margin": float(np.median(top1 - top2)),
            "margin_over_sd": float((top1 - top2).mean() / S.std()),
        })
    return pl.DataFrame(rows)


def cross_reservoir_note(share: pl.DataFrame) -> pl.DataFrame:
    """Spec check 8: show per-reservoir primitive counts so level comparison is not attempted.

    Macro and monetary reservoirs own far more sub-mechanisms and channels than
    event-driven ones, so unnormalised narrative LEVELS are not comparable
    across reservoirs. Shares within a day are; levels are not.
    """
    return (
        share.group_by("reservoir")
        .agg(pl.col("n_primitives").mean().alias("mean_primitives_per_narrative"),
             pl.col("narrative_key").n_unique().alias("n_narratives"),
             pl.col("SHARE").mean().alias("mean_share"))
        .sort("n_narratives", descending=True)
    )
