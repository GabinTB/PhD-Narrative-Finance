"""Orderbook collection: DBAG REST API -> raw parquet.

Paginates through the orderbook REST endpoint for a specific
(MIC, market_segment_id, security_id, date, time_range) and writes
a raw parquet.  Subsequent calls for the same key return the stored
parquet without hitting the API.

Raw layout:
    $RAW_DATA_PATH/Deutsche_Boerse/orderbook/{MIC}/{market_segment_id}_{security_id}_{ccyymmdd}_{from_}_{to_}.parquet

Time window strings (from_, to_) use HH:MM:SS format and are included in
the filename so different windows for the same security/date are stored
separately.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import polars as pl
import requests
from dbg_cdm.a7_utils import (
    get_security_price_precision,
    get_tick_ladder,
)

from deutsche_boerse import RAW_ROOT
from deutsche_boerse.schema import ORDERBOOK_SCHEMA

log = logging.getLogger(__name__)

_DEFAULT_LIMIT = 10_000
_DEFAULT_LEVELS = 10
_DEFAULT_ORDERBOOK_MODE = "complete"


def _raw_path(
    mic: str,
    market_segment_id: int,
    security_id: int,
    ccyymmdd: int,
    from_: str,
    to_: str,
) -> Path:
    path = RAW_ROOT / "orderbook" / mic
    path.mkdir(parents=True, exist_ok=True)
    tag = f"{from_.replace(':', '')}_{to_.replace(':', '')}"
    return path / f"{market_segment_id}_{security_id}_{ccyymmdd}_{tag}.parquet"


def _ts_ns(ccyymmdd: int, time_str: str) -> int:
    return int(
        pd.Timestamp(f"{ccyymmdd} {time_str}", unit="ns", tz="UTC").timestamp() * 1e9
    )


def _fetch_from_api(
    mic: str,
    ccyymmdd: int,
    market_segment_id: int,
    security_id: int,
    from_: str,
    to_: str,
    limit: int,
    levels: int,
    orderbook_mode: str,
    trades: bool,
    indicatives: bool,
    pause_s: float,
) -> pl.DataFrame:
    token = os.environ.get("A7_TOKEN")
    if not token:
        raise RuntimeError("A7_TOKEN environment variable not set")
    base_url = os.environ.get("OB_API")
    if not base_url:
        raise RuntimeError("OB_API environment variable not set")

    price_precision = get_security_price_precision(
        mic, ccyymmdd, market_segment_id, security_id
    )
    price_precision = None if price_precision is None else float(price_precision)

    try:
        tickladder = get_tick_ladder(mic, ccyymmdd, market_segment_id, security_id)
    except Exception as exc:
        log.warning("could not retrieve tick ladder: %s", exc)
        tickladder = None

    path = f"{mic}/{ccyymmdd}/{market_segment_id}/{security_id}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}
    from_ns = _ts_ns(ccyymmdd, from_)
    to_ns = _ts_ns(ccyymmdd, to_)

    records: list[dict] = []
    next_from = from_ns

    while True:
        params = {
            "from": next_from,
            "to": to_ns,
            "limit": limit,
            "levels": levels,
            "orderbook": orderbook_mode,
            "trades": str(trades).lower(),
            "indicatives": str(indicatives).lower(),
        }
        r = requests.get(f"{base_url}/{path}", headers=headers, params=params, timeout=30)
        r.raise_for_status()

        chunk_raw = r.json()
        if not chunk_raw:
            break

        chunk: list[dict] = []
        for c in chunk_raw:
            try:
                tick_size = (
                    None if tickladder is None
                    else float(tickladder.get_ticks_between_prices(
                        int(c["Buy"][0]["Price"]), int(c["Sell"][0]["Price"])
                    ))
                )
            except Exception:
                tick_size = None

            chunk.append({
                "timestamp":     int(c["Timestamp"]),
                "transact_time": int(c["TransactTime"]),
                "prev_ts":       int(c["prev"]),
                "next_ts":       int(c["next"]),
                "bid_cnt":       int(c["Buy"][0]["OrderCount"]),
                "bid_qty":       int(c["Buy"][0]["Quantity"]),
                "bid_price":     int(c["Buy"][0]["Price"]),
                "ask_price":     int(c["Sell"][0]["Price"]),
                "ask_qty":       int(c["Sell"][0]["Quantity"]),
                "ask_cnt":       int(c["Sell"][0]["OrderCount"]),
                "tick_size":     tick_size,
                "price_prec":    price_precision,
            })

        records.extend(chunk)
        log.debug("fetched %d records, total %d", len(chunk), len(records))

        if len(chunk) < limit:
            break
        next_from = int(chunk[-1]["timestamp"]) + 1
        if next_from >= to_ns:
            break

        if pause_s > 0:
            time.sleep(pause_s)

    if not records:
        return pl.DataFrame(schema=ORDERBOOK_SCHEMA)

    return pl.DataFrame(records).cast(ORDERBOOK_SCHEMA).sort("timestamp")


def get_orderbook(
    mic: str,
    ccyymmdd: int,
    market_segment_id: int,
    security_id: int,
    from_: str,
    to_: str,
    *,
    limit: int = _DEFAULT_LIMIT,
    levels: int = _DEFAULT_LEVELS,
    orderbook_mode: str = _DEFAULT_ORDERBOOK_MODE,
    trades: bool = False,
    indicatives: bool = False,
    pause_s: float = 0.0,
) -> pl.DataFrame:
    """Return orderbook snapshots for one (MIC, security, date, time window).

    Returns the stored raw parquet if it exists; otherwise fetches from the
    REST API, writes the parquet, and returns it.

    Args:
        mic:                Market identifier code.
        ccyymmdd:           Date as integer.
        market_segment_id:  Market segment identifier.
        security_id:        Security identifier.
        from_:              Start time as "HH:MM:SS".
        to_:                End time as "HH:MM:SS".
        limit:              Page size for API pagination.
        levels:             Order book depth.
        orderbook_mode:     "complete" or "delta".
        trades:             Include trade events.
        indicatives:        Include indicative prices.
        pause_s:            Sleep between API pages (rate limiting).

    Returns:
        Orderbook DataFrame sorted by timestamp.
    """
    raw_path = _raw_path(mic, market_segment_id, security_id, ccyymmdd, from_, to_)
    if raw_path.exists():
        log.debug("orderbook raw hit: %s", raw_path)
        return pl.read_parquet(raw_path)

    log.info(
        "fetching orderbook from API for %s %d sec=%d %s-%s",
        mic, ccyymmdd, security_id, from_, to_,
    )
    df = _fetch_from_api(
        mic, ccyymmdd, market_segment_id, security_id,
        from_, to_, limit, levels, orderbook_mode,
        trades, indicatives, pause_s,
    )

    if df.is_empty():
        log.warning(
            "API returned no orderbook data for %s %d sec=%d %s-%s",
            mic, ccyymmdd, security_id, from_, to_,
        )
        return df

    tmp = raw_path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(raw_path)
    log.info(
        "stored %d orderbook rows for %s %d sec=%d -> %s",
        len(df), mic, ccyymmdd, security_id, raw_path,
    )
    return df
