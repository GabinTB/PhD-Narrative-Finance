"""Tests for Deutsche Boerse HFT analytics core.

Tests cover pure functions only -- no API calls, no disk I/O, no dbg-cdm
imports at module level.  The analytics core is fully deterministic given
its inputs.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from deutsche_boerse.analytics.hft_event_detection import (
    classify_latency,
    compute_event_analytics,
    signal_noise,
    w2w_latency,
)
from deutsche_boerse.schema import HFT_ANALYTICS_SCHEMA, HFT_CLASSES

# ---------------------------------------------------------------------------
# Nanosecond constants (duplicated here to avoid dbg_cdm import at test time)
# ---------------------------------------------------------------------------

_NANOS_PER_MICRO = 1_000


# ---------------------------------------------------------------------------
# w2w_latency
# ---------------------------------------------------------------------------

class TestW2WLatency:
    def test_basic(self):
        assert w2w_latency(2720, 10_000, 5_000) == 10_000 - 5_000 - 2720

    def test_zero(self):
        assert w2w_latency(0, 100, 100) == 0

    def test_negative_produces_noise(self):
        assert w2w_latency(5000, 6000, 5000) < 0


# ---------------------------------------------------------------------------
# classify_latency
# ---------------------------------------------------------------------------

class TestClassifyLatency:
    def test_negative_is_noise(self):
        assert classify_latency(-1) == "Noise"

    def test_zero_is_uft(self):
        assert classify_latency(0) == "UFT"

    def test_just_below_micro_is_uft(self):
        assert classify_latency(_NANOS_PER_MICRO - 1) == "UFT"

    def test_one_micro_is_hft(self):
        assert classify_latency(_NANOS_PER_MICRO) == "HFT"

    def test_ten_micros_is_other(self):
        assert classify_latency(10 * _NANOS_PER_MICRO) == "Other"

    def test_all_classes_reachable(self):
        inputs = [-1, 0, _NANOS_PER_MICRO, 10 * _NANOS_PER_MICRO]
        classes = {classify_latency(x) for x in inputs}
        assert classes == set(HFT_CLASSES)


# ---------------------------------------------------------------------------
# signal_noise
# ---------------------------------------------------------------------------

class TestSignalNoise:
    def test_none_inputs_return_zeros(self):
        assert signal_noise(None, 1.0, 1.0) == (0.0, 0.0)
        assert signal_noise(1.0, None, 1.0) == (0.0, 0.0)
        assert signal_noise(1.0, 1.0, None) == (0.0, 0.0)

    def test_move_toward_markout_is_pure_signal(self):
        # trigger=100, reaction=105, markout=110: move toward markout
        sig, noi = signal_noise(100.0, 105.0, 110.0)
        assert noi == 0.0
        assert sig == pytest.approx(abs(105 - 100) / 105)

    def test_move_away_from_markout_is_pure_noise(self):
        # trigger=110, reaction=105, markout=120: move away from markout
        sig, noi = signal_noise(110.0, 105.0, 120.0)
        assert sig == 0.0
        assert noi == pytest.approx(abs(105 - 110) / 105)

    def test_overshoot_splits_signal_and_noise(self):
        # trigger=100, markout=105, reaction=110: overshoots
        sig, noi = signal_noise(100.0, 110.0, 105.0)
        assert sig == pytest.approx(abs(105 - 100) / 110)
        assert noi == pytest.approx(abs(110 - 105) / 110)

    def test_symmetric_downward(self):
        # same logic but prices falling
        sig, noi = signal_noise(110.0, 105.0, 100.0)
        assert noi == 0.0
        assert sig == pytest.approx(abs(105 - 110) / 105)

    def test_normalised_by_reaction_price(self):
        sig, _ = signal_noise(100.0, 200.0, 300.0)
        assert sig == pytest.approx(100.0 / 200.0)


# ---------------------------------------------------------------------------
# compute_event_analytics
# ---------------------------------------------------------------------------

def _make_microprices(timestamps: list[int], prices: list[float]) -> pl.DataFrame:
    return pl.DataFrame({
        "timestamp": pl.Series(timestamps, dtype=pl.Int64),
        "microprice": pl.Series(prices, dtype=pl.Float64),
        "midprice": pl.Series(prices, dtype=pl.Float64),
        "last_trade_price": pl.Series(prices, dtype=pl.Float64),
        "trade_event": pl.Series([False] * len(timestamps), dtype=pl.Boolean),
    })


def _make_trigger_events(t_9d: list[int], exec_id: list[int]) -> pl.DataFrame:
    return pl.DataFrame({
        "t_9d": pl.Series(t_9d, dtype=pl.Int64),
        "exec_id": pl.Series(exec_id, dtype=pl.Int64),
        "t_3a": pl.Series(t_9d, dtype=pl.Int64),
        "priority_ts": pl.Series(t_9d, dtype=pl.Int64),
        "message": pl.Series([1] * len(t_9d), dtype=pl.Int32),
        "side": pl.Series([1] * len(t_9d), dtype=pl.Int8),
        "price": pl.Series([100.0] * len(t_9d), dtype=pl.Float64),
        "qty": pl.Series([1.0] * len(t_9d), dtype=pl.Float64),
    })


def _make_reaction_events(t_3a: list[int], exec_id: list[int]) -> pl.DataFrame:
    return pl.DataFrame({
        "t_3a": pl.Series(t_3a, dtype=pl.Int64),
        "exec_id": pl.Series(exec_id, dtype=pl.Int64),
        "t_9d": pl.Series(t_3a, dtype=pl.Int64),
        "priority_ts": pl.Series(t_3a, dtype=pl.Int64),
        "message": pl.Series([1] * len(t_3a), dtype=pl.Int32),
        "side": pl.Series([1] * len(t_3a), dtype=pl.Int8),
        "price": pl.Series([100.0] * len(t_3a), dtype=pl.Float64),
        "qty": pl.Series([1.0] * len(t_3a), dtype=pl.Float64),
    })


class TestComputeEventAnalytics:
    def _exec_id_for_date(self, ccyymmdd: int, offset_ns: int) -> int:
        """Produce a realistic exec_id (nanoseconds since epoch) for a given date."""
        import pandas as pd
        midnight = int(pd.Timestamp(str(ccyymmdd), tz="UTC").timestamp() * 1e9)
        return midnight + offset_ns

    def test_output_schema_matches(self):
        date = 20240101
        base = self._exec_id_for_date(date, 0)
        ts = [base + i * 1000 for i in range(5)]
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]

        microprices     = _make_microprices(ts, prices)
        trigger_events  = _make_trigger_events([base + 100], [base + 100])
        reaction_events = _make_reaction_events([base + 5000], [base + 2000])

        result = compute_event_analytics(
            microprices=microprices,
            trigger_events=trigger_events,
            reaction_events=reaction_events,
            markout_period=10_000,
            t9d_to_t3a_min_ns=2720,
            t1_to_t3a_latency=907,
        )
        for col, dtype in HFT_ANALYTICS_SCHEMA.items():
            assert col in result.columns, f"missing column: {col}"
            assert result[col].dtype == dtype, f"{col}: expected {dtype}, got {result[col].dtype}"

    def test_output_row_count_matches_reaction_events(self):
        date = 20240101
        base = self._exec_id_for_date(date, 0)
        ts = [base + i * 1000 for i in range(10)]
        prices = [100.0 + i for i in range(10)]

        microprices     = _make_microprices(ts, prices)
        trigger_events  = _make_trigger_events([base + 100], [base + 100])
        reaction_events = _make_reaction_events(
            [base + 5000, base + 7000], [base + 3000, base + 4000]
        )

        result = compute_event_analytics(
            microprices=microprices,
            trigger_events=trigger_events,
            reaction_events=reaction_events,
            markout_period=1000,
            t9d_to_t3a_min_ns=2720,
            t1_to_t3a_latency=907,
        )
        assert len(result) == 2

    def test_class_values_are_valid(self):
        date = 20240101
        base = self._exec_id_for_date(date, 0)
        ts = [base + i * 1000 for i in range(10)]
        prices = [100.0 + i for i in range(10)]

        microprices     = _make_microprices(ts, prices)
        trigger_events  = _make_trigger_events([base + 100], [base + 100])
        reaction_events = _make_reaction_events([base + 5000], [base + 3000])

        result = compute_event_analytics(
            microprices=microprices,
            trigger_events=trigger_events,
            reaction_events=reaction_events,
            markout_period=1000,
            t9d_to_t3a_min_ns=2720,
            t1_to_t3a_latency=907,
        )
        assert result["class"][0] in HFT_CLASSES

    def test_empty_inputs_raise(self):
        with pytest.raises(AssertionError):
            compute_event_analytics(
                microprices=pl.DataFrame(),
                trigger_events=_make_trigger_events([1], [1]),
                reaction_events=_make_reaction_events([2], [2]),
                markout_period=1000,
                t9d_to_t3a_min_ns=2720,
                t1_to_t3a_latency=907,
            )
