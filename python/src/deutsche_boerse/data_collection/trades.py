"""Trades collection: hpt zip -> structured parquet cache.

Reads raw hpt zip files (one per day per MIC) downloaded via dbg-cdm's
datashop file API.  Extracts, filters, casts, and writes a zstd-compressed
parquet to the local cache.  Subsequent calls for the same (MIC, ccyymmdd)
return the cached parquet without touching the zip.

Raw layout (managed by dbg-cdm):
    $RAW_DATA_PATH/Deutsche_Boerse/hpt/{MIC}/{archive_filename}.zip

Cache layout:
    $CACHE_PATH/Deutsche_Boerse/trades/{MIC}/{ccyymmdd}.parquet

The zip is deleted after extraction to keep raw disk usage bounded.
Set keep_zip=True to retain it.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import polars as pl
from dbg_cdm.datashop_file_api import retrieve_hpt_file
from dbg_cdm.eobi_utils import PRICE_MULTIPLIER, VOLUME_MULTIPLIER
from dbg_cdm.hpt_utils import (
    AGGRESSOR_SIDE_COLUMN_NAME,
    EXEC_ID_COLUMN_NAME,
    FIELD_SEPERATOR,
    LAST_PX_COLUMN_NAME,
    LAST_QTY_COLUMN_NAME,
    MARKET_SEGMENT_COLUMN_NAME,
    SECURITY_ID_COLUMN_NAME,
    T3A_COLUMN_NAME,
    T9D_COLUMN_NAME,
    hpt_archive,
    hpt_filename,
)

from deutsche_boerse import CACHE_ROOT
from deutsche_boerse.schema import TRADES_SCHEMA

log = logging.getLogger(__name__)


def _cache_path(mic: str, ccyymmdd: int) -> Path:
    path = CACHE_ROOT / "trades" / mic
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{ccyymmdd}.parquet"


def _extract_trades(zip_path: Path, mic: str, ccyymmdd: int) -> pl.DataFrame:
    """Extract and cast one hpt zip into a trades DataFrame."""
    fn = hpt_filename(mic, ccyymmdd)
    with zipfile.ZipFile(zip_path) as zf:
        if fn not in zf.namelist():
            raise FileNotFoundError(f"{fn} not found in {zip_path}")
        with zf.open(fn) as fh:
            raw = pl.read_csv(
                io.BytesIO(fh.read()),
                separator=FIELD_SEPERATOR,
                infer_schema_length=0,  # read all as string, cast explicitly
            )

    return (
        raw
        .filter(
            (pl.col(T9D_COLUMN_NAME).cast(pl.Int64) > 0) &
            (pl.col(LAST_QTY_COLUMN_NAME).cast(pl.Int64) > 0)
        )
        .select([
            pl.col(MARKET_SEGMENT_COLUMN_NAME).cast(pl.Int64).alias("market_seg_id"),
            pl.col(SECURITY_ID_COLUMN_NAME).cast(pl.Int64).alias("sec_id"),
            pl.when(pl.col(T3A_COLUMN_NAME).cast(pl.Int64) > 0)
              .then(pl.col(T3A_COLUMN_NAME).cast(pl.Int64))
              .otherwise(None)
              .alias("t_3a"),
            pl.col(EXEC_ID_COLUMN_NAME).cast(pl.Int64).alias("exec_id"),
            pl.col(T9D_COLUMN_NAME).cast(pl.Int64).alias("t_9d"),
            pl.col(AGGRESSOR_SIDE_COLUMN_NAME).cast(pl.Int8).alias("side"),
            (pl.col(LAST_PX_COLUMN_NAME).cast(pl.Float64) / PRICE_MULTIPLIER).alias("price"),
            (pl.col(LAST_QTY_COLUMN_NAME).cast(pl.Float64) / VOLUME_MULTIPLIER).alias("qty"),
        ])
        .sort("t_9d")
        .cast(TRADES_SCHEMA)
    )


def get_trades(
    mic: str,
    ccyymmdd: int,
    *,
    keep_zip: bool = False,
) -> pl.DataFrame:
    """Return trades for one (MIC, date), using the cache if available.

    Args:
        mic:        Market identifier code (e.g. "XETR", "XEUR").
        ccyymmdd:   Date as integer (e.g. 20240101).
        keep_zip:   If False (default), delete the zip after extraction.

    Returns:
        Trades DataFrame sorted by t_9d.

    Raises:
        FileNotFoundError: if the zip cannot be found or downloaded.
        RuntimeError:      on extraction failure.
    """
    cache = _cache_path(mic, ccyymmdd)
    if cache.exists():
        log.debug("trades cache hit: %s", cache)
        return pl.read_parquet(cache)

    # Check for already-downloaded zip before hitting the API.
    zip_path = Path(hpt_archive(mic, ccyymmdd))
    if not zip_path.exists():
        log.info("downloading hpt for %s %d", mic, ccyymmdd)
        zip_path = Path(retrieve_hpt_file(mic, ccyymmdd))
        if not zip_path.exists():
            raise FileNotFoundError(f"hpt zip not found after download: {zip_path}")

    log.info("extracting trades for %s %d", mic, ccyymmdd)
    df = _extract_trades(zip_path, mic, ccyymmdd)

    tmp = cache.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(cache)
    log.info("cached %d trades for %s %d -> %s", len(df), mic, ccyymmdd, cache)

    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    return df


def get_trades_range(
    mic: str,
    from_ccyymmdd: int,
    to_ccyymmdd: int,
    *,
    keep_zip: bool = False,
    skip_missing: bool = True,
) -> pl.DataFrame:
    """Collect and concatenate trades over a date range.

    Args:
        mic:            Market identifier code.
        from_ccyymmdd:  Start date inclusive.
        to_ccyymmdd:    End date inclusive.
        keep_zip:       Retain downloaded zips.
        skip_missing:   If True, log and skip dates with no data.
                        If False, raise on missing data.

    Returns:
        Concatenated trades DataFrame sorted by t_9d.
    """
    from dbg_cdm.time_utils import SKIP_DATES, daterange, today_ccyymmdd

    frames: list[pl.DataFrame] = []
    for ccyymmdd in daterange(from_ccyymmdd, to_ccyymmdd):
        if ccyymmdd in SKIP_DATES or ccyymmdd >= today_ccyymmdd():
            log.debug("skip date %d", ccyymmdd)
            continue
        try:
            frames.append(get_trades(mic, ccyymmdd, keep_zip=keep_zip))
        except FileNotFoundError as exc:
            if skip_missing:
                log.warning("no trades for %s %d: %s", mic, ccyymmdd, exc)
            else:
                raise

    if not frames:
        return pl.DataFrame(schema=TRADES_SCHEMA)
    return pl.concat(frames).sort("t_9d")
