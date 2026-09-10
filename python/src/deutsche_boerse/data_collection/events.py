"""Events collection: hpt_all zip -> structured parquet cache.

Reads raw hpt_all zip files (one per partition per day per MIC) and extracts
all order book events for a specific security.  Writes a zstd-compressed
parquet to the local cache.

Raw layout (managed by dbg-cdm):
    $RAW_DATA_PATH/Deutsche_Boerse/hpt_all/{MIC}/{archive_filename}.zip

Cache layout:
    $CACHE_PATH/Deutsche_Boerse/events/{MIC}/{market_segment_id}_{security_id}_{ccyymmdd}.parquet

The zip is deleted after extraction.  Set keep_zip=True to retain it.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import polars as pl
from dbg_cdm.a7_utils import partition_for_market_segment_id
from dbg_cdm.datashop_file_api import retrieve_hptall_file
from dbg_cdm.eobi_utils import PRICE_MULTIPLIER, VOLUME_MULTIPLIER
from dbg_cdm.hpt_utils import (
    EXEC_ID_COLUMN_NAME,
    FIELD_SEPERATOR,
    PRICE_COLUMN_NAME,
    PRIORITY_COLUMN_NAME,
    QTY_COLUMN_NAME,
    SECURITY_ID_COLUMN_NAME,
    SIDE_COLUMN_NAME,
    T3A_COLUMN_NAME,
    T9D_COLUMN_NAME,
    TEMPLATE_ID_COLUMN_NAME,
    hptall_archive,
    hptall_filename,
)

from deutsche_boerse import CACHE_ROOT
from deutsche_boerse.schema import EVENTS_SCHEMA

log = logging.getLogger(__name__)


def _cache_path(
    mic: str,
    market_segment_id: int,
    security_id: int,
    ccyymmdd: int,
) -> Path:
    path = CACHE_ROOT / "events" / mic
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{market_segment_id}_{security_id}_{ccyymmdd}.parquet"


def _extract_events(
    zip_path: Path,
    mic: str,
    ccyymmdd: int,
    partition_id: int,
    security_id: int,
    options: bool,
) -> pl.DataFrame:
    """Extract and cast one hpt_all zip into an events DataFrame."""
    fn = hptall_filename(mic, ccyymmdd, partition_id, options=options)
    with zipfile.ZipFile(zip_path) as zf:
        if fn not in zf.namelist():
            raise FileNotFoundError(f"{fn} not found in {zip_path}")
        with zf.open(fn) as fh:
            raw = pl.read_csv(
                io.BytesIO(fh.read()),
                separator=FIELD_SEPERATOR,
                infer_schema_length=0,
            )

    return (
        raw
        .filter(
            (pl.col(SECURITY_ID_COLUMN_NAME).cast(pl.Int64) == security_id) &
            (pl.col(T3A_COLUMN_NAME).cast(pl.Int64) > 0) &
            (pl.col(T9D_COLUMN_NAME).cast(pl.Int64) > 0) &
            (pl.col(EXEC_ID_COLUMN_NAME).cast(pl.Int64) > 0)
        )
        .select([
            pl.col(T3A_COLUMN_NAME).cast(pl.Int64).alias("t_3a"),
            pl.col(EXEC_ID_COLUMN_NAME).cast(pl.Int64).alias("exec_id"),
            pl.col(T9D_COLUMN_NAME).cast(pl.Int64).alias("t_9d"),
            pl.col(PRIORITY_COLUMN_NAME).cast(pl.Int64).alias("priority_ts"),
            pl.col(TEMPLATE_ID_COLUMN_NAME).cast(pl.Int32).alias("message"),
            pl.col(SIDE_COLUMN_NAME).cast(pl.Int8).alias("side"),
            (pl.col(PRICE_COLUMN_NAME).cast(pl.Float64) / PRICE_MULTIPLIER).alias("price"),
            (pl.col(QTY_COLUMN_NAME).cast(pl.Float64) / VOLUME_MULTIPLIER).alias("qty"),
        ])
        .sort("t_3a")
        .cast(EVENTS_SCHEMA)
    )


def get_events(
    mic: str,
    ccyymmdd: int,
    market_segment_id: int,
    security_id: int,
    *,
    options: bool = False,
    keep_zip: bool = False,
) -> pl.DataFrame:
    """Return all order book events for one (MIC, security, date).

    Args:
        mic:                Market identifier code.
        ccyymmdd:           Date as integer.
        market_segment_id:  Market segment identifier.
        security_id:        Security identifier.
        options:            If True, fetch options (XEUR only).
        keep_zip:           Retain the downloaded zip after extraction.

    Returns:
        Events DataFrame sorted by t_3a.

    Raises:
        FileNotFoundError: if the zip cannot be found or downloaded.
    """
    cache = _cache_path(mic, market_segment_id, security_id, ccyymmdd)
    if cache.exists():
        log.debug("events cache hit: %s", cache)
        return pl.read_parquet(cache)

    partition_id = partition_for_market_segment_id(mic, ccyymmdd, market_segment_id)
    zip_path = Path(hptall_archive(mic, ccyymmdd, partition_id, options=options))

    if not zip_path.exists():
        log.info("downloading hpt_all for %s %d partition %d", mic, ccyymmdd, partition_id)
        zip_path = Path(retrieve_hptall_file(mic, ccyymmdd, partition_id, options=options))
        if not zip_path.exists():
            raise FileNotFoundError(f"hpt_all zip not found after download: {zip_path}")

    log.info(
        "extracting events for %s %d sec=%d", mic, ccyymmdd, security_id
    )
    df = _extract_events(zip_path, mic, ccyymmdd, partition_id, security_id, options)

    tmp = cache.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(cache)
    log.info(
        "cached %d events for %s %d sec=%d -> %s",
        len(df), mic, ccyymmdd, security_id, cache,
    )

    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    return df


def get_events_range(
    mic: str,
    from_ccyymmdd: int,
    to_ccyymmdd: int,
    market_segment_id: int,
    security_id: int,
    *,
    options: bool = False,
    keep_zip: bool = False,
    skip_missing: bool = True,
) -> pl.DataFrame:
    """Collect and concatenate events over a date range for one security.

    Returns:
        Concatenated events DataFrame sorted by t_3a.
    """
    from dbg_cdm.time_utils import SKIP_DATES, daterange, today_ccyymmdd

    frames: list[pl.DataFrame] = []
    for ccyymmdd in daterange(from_ccyymmdd, to_ccyymmdd):
        if ccyymmdd in SKIP_DATES or ccyymmdd >= today_ccyymmdd():
            continue
        try:
            frames.append(get_events(
                mic, ccyymmdd, market_segment_id, security_id,
                options=options, keep_zip=keep_zip,
            ))
        except FileNotFoundError as exc:
            if skip_missing:
                log.warning("no events for %s %d sec=%d: %s", mic, ccyymmdd, security_id, exc)
            else:
                raise

    if not frames:
        return pl.DataFrame(schema=EVENTS_SCHEMA)
    return pl.concat(frames).sort("t_3a")
