"""Deutsche Boerse data schemas.

Four data types, two layers:

Raw (collected from DBAG APIs, never modified):
  - hpt zips          -> trades parquet (cache)
  - hpt_all zips      -> events parquet (cache)
  - microprice        -> raw parquet via A7 algo API
  - orderbook         -> raw parquet via REST API

Derived (datalake artifacts, produced by analytics pipelines):
  - hft_analytics     -> w2w latency, classification, signal/noise

Cache parquets are keyed by (MIC, ccyymmdd) for trades and
(MIC, market_segment_id, security_id, ccyymmdd) for events.
They are reproducible from raw zips and never pushed to GDrive.
"""
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Trades (from hpt zips)
# ---------------------------------------------------------------------------

TRADES_SCHEMA = pl.Schema({
    "market_seg_id": pl.Int64,
    "sec_id":        pl.Int64,
    "t_3a":          pl.Int64,   # nullable
    "exec_id":       pl.Int64,
    "t_9d":          pl.Int64,
    "side":          pl.Int8,    # 1=buy, 2=sell
    "price":         pl.Float64,
    "qty":           pl.Float64,
})

# ---------------------------------------------------------------------------
# Events (from hpt_all zips, filtered to one security)
# ---------------------------------------------------------------------------

EVENTS_SCHEMA = pl.Schema({
    "t_3a":        pl.Int64,
    "exec_id":     pl.Int64,
    "t_9d":        pl.Int64,
    "priority_ts": pl.Int64,
    "message":     pl.Int32,
    "side":        pl.Int8,
    "price":       pl.Float64,
    "qty":         pl.Float64,
})

# ---------------------------------------------------------------------------
# Microprice (from A7 algo API)
# ---------------------------------------------------------------------------

MICROPRICE_SCHEMA = pl.Schema({
    "timestamp":         pl.Int64,
    "microprice":        pl.Float64,
    "midprice":          pl.Float64,
    "last_trade_price":  pl.Float64,
    "trade_event":       pl.Boolean,
})

# ---------------------------------------------------------------------------
# Orderbook (from REST API)
# ---------------------------------------------------------------------------

ORDERBOOK_SCHEMA = pl.Schema({
    "timestamp":    pl.Int64,
    "transact_time": pl.Int64,
    "prev_ts":      pl.Int64,
    "next_ts":      pl.Int64,
    "bid_cnt":      pl.Int64,
    "bid_qty":      pl.Int64,
    "bid_price":    pl.Int64,
    "ask_price":    pl.Int64,
    "ask_qty":      pl.Int64,
    "ask_cnt":      pl.Int64,
    "tick_size":    pl.Float64,  # nullable
    "price_prec":   pl.Float64,  # nullable
})

# ---------------------------------------------------------------------------
# HFT analytics (derived datalake artifact)
# ---------------------------------------------------------------------------

HFT_ANALYTICS_SCHEMA = pl.Schema({
    "t_3a":         pl.Int64,
    "exec_id":      pl.Int64,
    "t_9d":         pl.Int64,
    "priority_ts":  pl.Int64,
    "message":      pl.Int32,
    "side":         pl.Int8,
    "price":        pl.Float64,
    "qty":          pl.Float64,
    "w2w_latency":  pl.Int64,
    "class":        pl.String,
    "signal":       pl.Float64,
    "noise":        pl.Float64,
})

# HFT classification labels
HFT_CLASSES = ("Noise", "UFT", "HFT", "Other")
