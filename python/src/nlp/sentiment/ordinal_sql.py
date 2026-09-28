"""Scores and confidences of a stored grid distribution, computed lazily in DuckDB.

A ``headline_sentiment`` model artifact stores P_00..P_40 (Float16), a distribution on
s = linspace(-1, 1, 41) (``nlp.sentiment.base``). This module turns such a relation into
(RP_STORY_ID, SENT, CONF) with one of three rules, as plain SQL (no unnest, no Python
UDF). Every P_i is cast to DOUBLE and renormalised, q_i = P_i / sum_j P_j, before any
arithmetic: DuckDB reads float16 as FLOAT, and float32 over 41 terms would drift.

    rule     SENT in [-1, 1]                              CONF in [0, 1]
    argmax   mean of s_i over every i with q_i = max q    sum of q_i over those i
    mean     sum s_i q_i                                  1 - sqrt(sum s_i^2 q_i - SENT^2)
    median   q-quantile 1/2 of the grid distribution,     1 - (Q(3/4) - Q(1/4)) / 2
             each s_i uniform on its bin (base.GRID_EDGES)

Ties in argmax are averaged (a FinBERT pos = neg tie gives 0). A row with null P_*
(no model output) gives null SENT and CONF. Quantiles: j = first index with cumulative
mass c_j >= level, Q = lo_j + (level - c_{j-1}) / q_j * width_j (c_{-1} = 0), cumulative
sums taken left to right. The numpy functions below are the reference implementation
the SQL is tested against; they use the same summation order.
"""
from __future__ import annotations

import numpy as np

from nlp.sentiment.base import GRID, GRID_BIN_HI, GRID_BIN_LO, GRID_COLUMNS, N_GRID

RULES = ("argmax", "mean", "median")
ID_COL = "RP_STORY_ID"


def _lit(x: float) -> str:
    return f"CAST({float(x)!r} AS DOUBLE)"


def _sum(terms: list[str]) -> str:
    return "(" + " + ".join(terms) + ")"


def _quantile_sql(level: float) -> str:
    """CASE ladder over the cumulative columns c_00..c_40 and the q_* columns."""
    whens = []
    for j in range(N_GRID):
        prev = _lit(0.0) if j == 0 else f"c_{j - 1:02d}"
        width = GRID_BIN_HI[j] - GRID_BIN_LO[j]
        whens.append(f"WHEN c_{j:02d} >= {_lit(level)} THEN {_lit(GRID_BIN_LO[j])} + "
                     f"({_lit(level)} - {prev}) / q_{j:02d} * {_lit(width)}")
    return "CASE " + " ".join(whens) + f" ELSE {_lit(1.0)} END"


def grid_select_sql(relation: str, rule: str) -> str:
    """``SELECT RP_STORY_ID, SENT, CONF`` over ``relation`` (a table, view or
    ``read_parquet(...)``) holding RP_STORY_ID and P_00..P_40."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    d = [f"CAST({c} AS DOUBLE)" for c in GRID_COLUMNS]
    norm = (f"SELECT {ID_COL}, tot, "
            + ", ".join(f"{d[j]} / tot AS q_{j:02d}" for j in range(N_GRID))
            + f" FROM (SELECT {ID_COL}, " + ", ".join(GRID_COLUMNS)
            + f", {_sum(d)} AS tot FROM {relation})")
    q = [f"q_{j:02d}" for j in range(N_GRID)]

    def out(sent: str, conf: str, inner: str) -> str:
        # a row with null P_* has a null total: both outputs null ("no model output")
        return (f"SELECT {ID_COL}, CASE WHEN tot IS NULL THEN NULL ELSE {sent} END AS SENT, "
                f"CASE WHEN tot IS NULL THEN NULL ELSE {conf} END AS CONF FROM ({inner})")

    if rule == "mean":
        m1 = _sum([f"{_lit(GRID[j])} * {q[j]}" for j in range(N_GRID)])
        m2 = _sum([f"{_lit(GRID[j] ** 2)} * {q[j]}" for j in range(N_GRID)])
        return out("m1", f"{_lit(1.0)} - sqrt(greatest(m2 - m1 * m1, {_lit(0.0)}))",
                   f"SELECT {ID_COL}, tot, {m1} AS m1, {m2} AS m2 FROM ({norm})")
    if rule == "argmax":
        at = [f"CASE WHEN {q[j]} = mx THEN" for j in range(N_GRID)]
        num = _sum([f"{at[j]} {_lit(GRID[j])} ELSE {_lit(0.0)} END" for j in range(N_GRID)])
        cnt = _sum([f"{at[j]} {_lit(1.0)} ELSE {_lit(0.0)} END" for j in range(N_GRID)])
        mass = _sum([f"{at[j]} {q[j]} ELSE {_lit(0.0)} END" for j in range(N_GRID)])
        return out(f"{num} / {cnt}", mass,
                   f"SELECT *, greatest({', '.join(q)}) AS mx FROM ({norm})")
    cums = ", ".join(f"{_sum(q[:j + 1])} AS c_{j:02d}" for j in range(N_GRID))
    return out(_quantile_sql(0.5),
               f"{_lit(1.0)} - ({_quantile_sql(0.75)} - {_quantile_sql(0.25)}) / {_lit(2.0)}",
               f"SELECT *, {cums} FROM ({norm})")


def score_select_sql(relation: str, column: str) -> str:
    """``SELECT RP_STORY_ID, SENT, CONF`` for a one-score source (e.g. vendor SENT_CSS):
    SENT = the column in DOUBLE, NaN -> NULL ("no score"); CONF is NULL (none exists)."""
    return (f"SELECT {ID_COL}, CASE WHEN isnan({column}) THEN NULL "
            f"ELSE CAST({column} AS DOUBLE) END AS SENT, CAST(NULL AS DOUBLE) AS CONF "
            f"FROM {relation}")


# ---------------------------------------------------------------------------
# numpy reference (tests), same definitions and summation order as the SQL
# ---------------------------------------------------------------------------

def normalise(P: np.ndarray) -> np.ndarray:
    """(N, 41) stored values -> q in float64, row sums taken left to right."""
    d = np.asarray(P, dtype=np.float64)
    tot = np.cumsum(d, axis=1)[:, -1]
    return d / tot[:, None]


def ordinal_mean(q: np.ndarray) -> np.ndarray:
    return np.cumsum(q * GRID, axis=1)[:, -1]


def ordinal_dispersion(q: np.ndarray) -> np.ndarray:
    m1 = ordinal_mean(q)
    m2 = np.cumsum(q * GRID ** 2, axis=1)[:, -1]
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def ordinal_quantile(q: np.ndarray, level: float) -> np.ndarray:
    """Quantile of (N, 41) grid distributions, each point uniform on its clipped bin."""
    c = np.cumsum(q, axis=1)
    hit = c >= level
    out = np.full(q.shape[0], 1.0)
    rows = np.flatnonzero(hit.any(axis=1))
    j = np.argmax(hit[rows], axis=1)
    prev = np.where(j > 0, c[rows, np.maximum(j - 1, 0)], 0.0)
    width = (GRID_BIN_HI - GRID_BIN_LO)[j]
    out[rows] = GRID_BIN_LO[j] + (level - prev) / q[rows, j] * width
    return out


def ordinal_median(q: np.ndarray) -> np.ndarray:
    return ordinal_quantile(q, 0.5)


def grid_argmax(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(SENT, CONF) of the argmax rule, ties averaged."""
    at = q == q.max(axis=1, keepdims=True)
    return (at * GRID).sum(axis=1) / at.sum(axis=1), (at * q).sum(axis=1)


def reference(P: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """(SENT, CONF) of every row of stored values ``P`` (all-NaN rows -> NaN, NaN)."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    P = np.asarray(P, dtype=np.float64)
    ok = ~np.isnan(P).any(axis=1)
    sent = np.full(P.shape[0], np.nan)
    conf = np.full(P.shape[0], np.nan)
    if ok.any():
        q = normalise(P[ok])
        if rule == "mean":
            s, c = ordinal_mean(q), 1.0 - ordinal_dispersion(q)
        elif rule == "argmax":
            s, c = grid_argmax(q)
        else:
            s = ordinal_median(q)
            c = 1.0 - (ordinal_quantile(q, 0.75) - ordinal_quantile(q, 0.25)) / 2.0
        sent[ok], conf[ok] = s, c
    return sent, conf
