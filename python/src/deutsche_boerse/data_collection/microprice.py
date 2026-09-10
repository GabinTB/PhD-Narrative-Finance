"""Microprice collection: A7 MicropriceTimeSeries algo -> raw parquet.

Calls the A7 algorithm API in batches to retrieve microprice time series
for a specific (MIC, market_segment_id, security_id, date).  Results are
written to the raw microprice directory (not the cache -- this is API data,
not derived).

Raw layout:
    $RAW_DATA_PATH/Deutsche_Boerse/microprice/{MIC}/{market_segment_id}_{security_id}_{ccyymmdd}.parquet

Subsequent calls for the same key return the stored parquet without hitting
the API again.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import polars as pl
from a7 import A7Client
from dbg_cdm.time_utils import (
    NANOS_PER_DAY,
    NANOS_PER_HOUR,
    NANOS_PER_MINUTE,
    midnight_timestamp,
)

from deutsche_boerse import RAW_ROOT
from deutsche_boerse.schema import MICROPRICE_SCHEMA

log = logging.getLogger(__name__)

_A7_ALGORITHM = "MicropriceTimeSeries"
_A7_OWNER = "gabitai"
_DEFAULT_BATCH_MINS = 30


def _raw_path(
    mic: str,
    market_segment_id: int,
    security_id: int,
    ccyymmdd: int,
) -> Path:
    path = RAW_ROOT / "microprice" / mic
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{market_segment_id}_{security_id}_{ccyymmdd}.parquet"


def _fetch_from_api(
    mic: str,
    ccyymmdd: int,
    market_segment_id: int,
    security_id: int,
    batch_mins: int,
) -> pl.DataFrame:
    token = os.environ.get("A7_TOKEN")
    if not token:
        raise RuntimeError("A7_TOKEN environment variable not set")

    client = A7Client(token=token)
    dt = NANOS_PER_MINUTE * batch_mins
    start_ts = midnight_timestamp(ccyymmdd)
    end_ts = start_ts + NANOS_PER_DAY

    df = pl.DataFrame(schema=MICROPRICE_SCHEMA)
    while start_ts + NANOS_PER_HOUR < end_ts:
        raw = client.algo.run(
            owner=_A7_OWNER,
            algorithm=_A7_ALGORITHM,
            params={
                "date": ccyymmdd,
                "marketId": mic,
                "marketSegmentId": market_segment_id,
                "securityId": security_id,
                "start_timestamp": start_ts,
                "end_timestamp": start_ts + dt,
            },
        )
        if raw is None:
            log.warning(
                "A7 returned None for %s %d seg=%d sec=%d at ts=%d",
                mic, ccyymmdd, market_segment_id, security_id, start_ts,
            )
            break

        assert len(raw) == 1, f"unexpected A7 response length: {len(raw)}"
        assert "series" in raw[0], "A7 response missing 'series'"
        assert len(raw[0]["series"]) == 1
        assert raw[0]["series"][0]["name"] == "Microprice"

        batch = pl.DataFrame(raw[0]["series"][0]["content"], strict=False)
        batch = batch.select(list(MICROPRICE_SCHEMA.keys()))
        df = pl.concat([df, batch], how="vertical_relaxed")
        log.debug(
            "fetched %d rows, total %d | ts=%d",
            len(batch), len(df), start_ts,
        )
        start_ts += dt

    return df.unique(subset=["timestamp"]).sort("timestamp")


def get_microprice(
    mic: str,
    ccyymmdd: int,
    market_segment_id: int,
    security_id: int,
    *,
    batch_mins: int = _DEFAULT_BATCH_MINS,
) -> pl.DataFrame:
    """Return microprice time series for one (MIC, security, date).

    Returns the stored raw parquet if it exists; otherwise fetches from the
    A7 API, writes the parquet, and returns it.

    Args:
        mic:                Market identifier code.
        ccyymmdd:           Date as integer.
        market_segment_id:  Market segment identifier.
        security_id:        Security identifier.
        batch_mins:         Duration of each API batch request in minutes.

    Returns:
        Microprice DataFrame sorted by timestamp.
    """
    raw_path = _raw_path(mic, market_segment_id, security_id, ccyymmdd)
    if raw_path.exists():
        log.debug("microprice raw hit: %s", raw_path)
        return pl.read_parquet(raw_path)

    log.info(
        "fetching microprice from A7 for %s %d seg=%d sec=%d",
        mic, ccyymmdd, market_segment_id, security_id,
    )
    df = _fetch_from_api(mic, ccyymmdd, market_segment_id, security_id, batch_mins)

    if df.is_empty():
        log.warning(
            "A7 returned no data for %s %d sec=%d", mic, ccyymmdd, security_id
        )
        return df

    tmp = raw_path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(raw_path)
    log.info(
        "stored %d microprice rows for %s %d sec=%d -> %s",
        len(df), mic, ccyymmdd, security_id, raw_path,
    )
    return df
