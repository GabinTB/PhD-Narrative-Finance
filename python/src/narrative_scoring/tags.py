"""Sentiment tags: which output rows a headline feeds in ONE scoring pass.

A headline's sentiment score s comes from a ``headline_sentiment`` grid table (P_00..P_40)
through one of the ordinal rules (``nlp.sentiment.ordinal_sql.reference``: mean, median,
argmax). A tag is a name and an interval of s; every headline whose s lies in a tag's
interval feeds that tag's rows, every headline (tagged or not) feeds the null model.

Conditions are written as short strings and parsed through a whitelisted AST, never
evaluated: comparisons of ``x`` with numeric constants (``<``, ``<=``, ``>``, ``>=``),
chained (``-1/3 < x < 1/3``) or joined by ``&`` with each side in parentheses
(``(x > -1/3) & (x < 1/3)``, as in numpy: ``&`` binds tighter than ``<``), constants
built from numbers with ``+ - * /`` and unary minus. Anything else is refused. Each
condition becomes a canonical interval ``(lo, hi, lo_closed, hi_closed)``, an unbounded
side being +-inf (open); a dict with those four keys is accepted directly. Identity is
computed on the canonical intervals, so ``"x>=1/3"`` and ``"1/3 <= x"`` hash the same.

Tags must be pairwise disjoint (checked once, at construction: an end point shared by
two closed sides is an overlap). They need not cover [-1, 1]: a headline in no interval,
or with no score (null grid), is untagged; it is counted, scored for the null model and
written nowhere.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from nlp.sentiment.ordinal_sql import RULES

UNTAGGED = -1
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv}
# x OP c, read as a bound on x: (is_lower, closed)
_BOUND = {ast.Gt: (True, False), ast.GtE: (True, True), ast.Lt: (False, False),
          ast.LtE: (False, True)}
_FLIP = {ast.Gt: ast.Lt, ast.GtE: ast.LtE, ast.Lt: ast.Gt, ast.LtE: ast.GtE}


@dataclass(frozen=True)
class Interval:
    """A (possibly unbounded) interval of the sentiment score."""

    lo: float = -math.inf
    hi: float = math.inf
    lo_closed: bool = False
    hi_closed: bool = False

    def __post_init__(self) -> None:
        for name in ("lo", "hi"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or math.isnan(v):
                raise ValueError(f"interval {name} must be a number, got {v!r}")
            object.__setattr__(self, name, float(v))
        # an infinite side is open
        if math.isinf(self.lo):
            object.__setattr__(self, "lo_closed", False)
        if math.isinf(self.hi):
            object.__setattr__(self, "hi_closed", False)
        if self.lo > self.hi or (self.lo == self.hi and not (self.lo_closed and self.hi_closed)):
            raise ValueError(f"empty interval {self}")

    def contains(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            above = s >= self.lo if self.lo_closed else s > self.lo
            below = s <= self.hi if self.hi_closed else s < self.hi
        return above & below

    def overlaps(self, other: Interval) -> bool:
        a, b = sorted((self, other), key=lambda i: (i.lo, not i.lo_closed))
        if b.lo < a.hi:
            return True
        return b.lo == a.hi and a.hi_closed and b.lo_closed

    def to_dict(self) -> dict[str, Any]:
        return {"lo": _num(self.lo), "hi": _num(self.hi),
                "lo_closed": self.lo_closed, "hi_closed": self.hi_closed}

    def __str__(self) -> str:
        return (f"{'[' if self.lo_closed else '('}{self.lo:g}, {self.hi:g}"
                f"{']' if self.hi_closed else ')'}")


def _num(v: float) -> float | str:
    """JSON-safe float (infinities as strings)."""
    return v if math.isfinite(v) else ("inf" if v > 0 else "-inf")


def _const(node: ast.AST, text: str) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const(node.operand, text)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_const(node.left, text), _const(node.right, text))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        raise ValueError(f"condition {text!r}: & binds tighter than a comparison; "
                         "parenthesise each side: (x > a) & (x < b)")
    raise ValueError(f"condition {text!r}: only numbers and + - * / are allowed in a bound")


def _is_x(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "x"


def _bounds(node: ast.AST, text: str) -> list[tuple[bool, float, bool]]:
    """(is_lower, value, closed) for every comparison in ``node``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        return _bounds(node.left, text) + _bounds(node.right, text)
    if not isinstance(node, ast.Compare):
        raise ValueError(f"condition {text!r}: expected comparisons of x with numbers, "
                         "chained or joined by &")
    out = []
    terms = [node.left, *node.comparators]
    for left, op, right in zip(terms, node.ops, terms[1:]):
        if type(op) not in _BOUND:
            raise ValueError(f"condition {text!r}: only < <= > >= are allowed")
        if _is_x(left) and not _is_x(right):
            is_lower, closed = _BOUND[type(op)]
            out.append((is_lower, _const(right, text), closed))
        elif _is_x(right) and not _is_x(left):
            is_lower, closed = _BOUND[_FLIP[type(op)]]
            out.append((is_lower, _const(left, text), closed))
        else:
            raise ValueError(f"condition {text!r}: each comparison must have x on exactly "
                             "one side")
    return out


def parse_condition(text: str) -> Interval:
    """A condition string -> its canonical interval (see the module docstring)."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"condition {text!r}: {exc.msg}") from None
    lo, lo_closed, hi, hi_closed = -math.inf, False, math.inf, False
    for is_lower, v, closed in _bounds(tree.body, text):
        if is_lower and (v > lo or (v == lo and not closed)):
            lo, lo_closed = v, closed
        elif not is_lower and (v < hi or (v == hi and not closed)):
            hi, hi_closed = v, closed
    return Interval(lo, hi, lo_closed, hi_closed)


def as_interval(spec: str | dict[str, Any] | Interval) -> Interval:
    if isinstance(spec, Interval):
        return spec
    if isinstance(spec, str):
        return parse_condition(spec)
    if isinstance(spec, dict):
        unknown = set(spec) - {"lo", "hi", "lo_closed", "hi_closed"}
        if unknown:
            raise ValueError(f"interval keys {sorted(unknown)} not understood")
        d = {k: (float(v) if k in ("lo", "hi") else bool(v)) for k, v in spec.items()}
        return Interval(**d)
    raise ValueError(f"a tag condition is a string or an interval dict, got {spec!r}")


@dataclass(frozen=True)
class SentimentTags:
    """The sentiment layer of a scoring pass: score source and rule, and the tags.

    ``tags`` is a tuple of (name, Interval), in the order given (the order of the output
    rows); names are unique and the intervals pairwise disjoint."""

    source: str
    rule: str
    tags: tuple[tuple[str, Interval], ...]

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if not self.tags:
            raise ValueError("at least one tag is needed (no sentiment layer: pass None)")
        names = [n for n, _ in self.tags]
        if len(set(names)) != len(names):
            raise ValueError(f"tag names must be unique, got {names}")
        if any(not n or n == "all" for n in names):
            raise ValueError("a tag name must be non-empty and not 'all' (the untagged run)")
        for i, (a, ia) in enumerate(self.tags):
            for b, ib in self.tags[i + 1:]:
                if ia.overlaps(ib):
                    raise ValueError(f"tags {a!r} {ia} and {b!r} {ib} overlap")

    @classmethod
    def build(cls, source: str, rule: str,
              tags: list[dict[str, Any]] | list[tuple[str, Any]]) -> SentimentTags:
        """From ``[{"tag": "neg", "condition": "x <= -1/3"}, ...]`` (a condition string or
        an interval dict) or ``[("neg", "x <= -1/3"), ...]``."""
        pairs = []
        for t in tags:
            name, cond = (t["tag"], t["condition"]) if isinstance(t, dict) else t
            pairs.append((str(name), as_interval(cond)))
        return cls(source, rule, tuple(pairs))

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.tags]

    def codes(self, sent: np.ndarray) -> np.ndarray:
        """int8 per headline: the index of its tag, ``UNTAGGED`` (-1) when none (or NaN)."""
        if len(self.tags) > 127:
            raise ValueError("at most 127 tags")
        out = np.full(len(sent), UNTAGGED, dtype=np.int8)
        for k, (_, interval) in enumerate(self.tags):
            out[interval.contains(sent)] = k
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "rule": self.rule,
                "tags": [{"tag": n, "condition": i.to_dict()} for n, i in self.tags]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SentimentTags:
        return cls.build(d["source"], d["rule"], d["tags"])

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
