"""Sentiment tags: whitelisted condition parsing, canonical intervals, disjointness, codes."""
from __future__ import annotations

import math

import numpy as np
import pytest

from narrative_scoring.tags import UNTAGGED, Interval, SentimentTags, parse_condition

THIRD = 1 / 3
OBJECTIVE = [{"tag": "neg", "condition": "x <= -1/3"},
             {"tag": "neu", "condition": "-1/3 < x < 1/3"},
             {"tag": "pos", "condition": "x >= 1/3"}]


@pytest.mark.parametrize("text, want", [
    ("x <= -1/3", Interval(-math.inf, -THIRD, False, True)),
    ("-1/3 >= x", Interval(-math.inf, -THIRD, False, True)),
    ("x >= 1/3", Interval(THIRD, math.inf, True, False)),
    ("1/3 <= x", Interval(THIRD, math.inf, True, False)),
    ("-1/3 < x < 1/3", Interval(-THIRD, THIRD, False, False)),
    ("(x > -1/3) & (x < 1/3)", Interval(-THIRD, THIRD, False, False)),
    ("(x>=0.5*(1-0.5)) & (x<=1)", Interval(0.25, 1.0, True, True)),
    ("(x > -0.2) & (x >= -0.1)", Interval(-0.1, math.inf, True, False)),   # tighter wins
    ("(x <= 0) & (x < 0)", Interval(-math.inf, 0.0, False, False)),       # open wins on a tie
])
def test_conditions_parse_to_canonical_intervals(text, want):
    assert parse_condition(text) == want


@pytest.mark.parametrize("text", [
    "x and 1", "x <= y", "abs(x) < 1", "__import__('os').system('true')", "x == 0",
    "x < 1 or x > 2", "1 < 2", "x < x", "x <= 'a'", "x <= True", "lambda: x", "x <=",
    "x >= 0.25 & x <= 1",                   # & before <: must be parenthesised
])
def test_anything_outside_the_whitelist_is_refused(text):
    with pytest.raises(ValueError):
        parse_condition(text)


def test_empty_intervals_are_refused():
    with pytest.raises(ValueError, match="empty"):
        parse_condition("(x > 0.5) & (x < 0.1)")
    with pytest.raises(ValueError, match="empty"):
        parse_condition("(x >= 0.5) & (x < 0.5)")
    assert parse_condition("(x >= 0.5) & (x <= 0.5)") == Interval(0.5, 0.5, True, True)


def test_equivalent_conditions_hash_equal():
    a = SentimentTags.build("ravenbert", "mean", OBJECTIVE)
    b = SentimentTags.build("ravenbert", "mean", [
        ("neg", "-1/3 >= x"), ("neu", "(x > -1/3) & (x < 0.3333333333333333)"),
        ("pos", {"lo": THIRD, "hi": math.inf, "lo_closed": True, "hi_closed": False})])
    assert a.digest() == b.digest()
    assert SentimentTags.from_dict(a.to_dict()) == a
    other = SentimentTags.build("ravenbert", "median", OBJECTIVE)
    assert other.digest() != a.digest()


def test_overlapping_tags_are_refused_at_construction():
    with pytest.raises(ValueError, match="overlap"):        # -1/3 in both
        SentimentTags.build("ravenbert", "mean", [("neg", "x <= -1/3"),
                                                  ("neu", "-1/3 <= x <= 1/3")])
    with pytest.raises(ValueError, match="overlap"):
        SentimentTags.build("ravenbert", "mean", [("a", "x < 0.2"), ("b", "x > 0.1")])
    # touching at a point closed on one side only is fine
    SentimentTags.build("ravenbert", "mean", [("neg", "x < 0"), ("pos", "x >= 0")])


def test_tag_set_validation():
    with pytest.raises(ValueError, match="unique"):
        SentimentTags.build("ravenbert", "mean", [("a", "x < 0"), ("a", "x > 0")])
    with pytest.raises(ValueError, match="all"):
        SentimentTags.build("ravenbert", "mean", [("all", "x < 0")])
    with pytest.raises(ValueError, match="rule"):
        SentimentTags.build("ravenbert", "mode", OBJECTIVE)
    with pytest.raises(ValueError, match="at least one"):
        SentimentTags.build("ravenbert", "mean", [])


def test_codes_follow_the_intervals_and_leave_the_rest_untagged():
    tags = SentimentTags.build("ravenbert", "mean", OBJECTIVE)
    s = np.array([-1.0, -THIRD, -0.3, 0.0, 0.3, THIRD, 1.0, np.nan])
    assert tags.codes(s).tolist() == [0, 0, 1, 1, 1, 2, 2, UNTAGGED]
    one = SentimentTags.build("ravenbert", "mean", [("pos", "x >= 1/3")])
    assert one.codes(s).tolist() == [UNTAGGED] * 5 + [0, 0, UNTAGGED]
    two = SentimentTags.build("ravenbert", "mean", [("neg", "x < 0"), ("pos", "x >= 0")])
    assert two.codes(s).tolist() == [0, 0, 0, 1, 1, 1, 1, UNTAGGED]
