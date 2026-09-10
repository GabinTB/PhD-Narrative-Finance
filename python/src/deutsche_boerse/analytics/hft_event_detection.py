"""HFT analytics pipeline.

Computes wire-to-wire (w2w) latency, HFT classification, and signal/noise
decomposition for reaction events relative to trigger trades.

Pure analytics functions (w2w_latency, classify_latency, signal_noise,
compute_event_analytics) have NO external dependencies beyond numpy, polars
and pandas -- importable without dbg_cdm or any API credentials.

I/O-bound functions (resolve_most_liquid, iter_jobs, run_hft_analytics)
import dbg_cdm lazily at call time, so the module is still importable in
environments where dbg_cdm is not installed (e.g. CI, pure-analytics tests).
"""
from __future__ import annotations

import logging
import pickle
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
import polars as pl
from pandas import Interval

from deutsche_boerse.schema import HFT_ANALYTICS_SCHEMA

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex

log = logging.getLogger(__name__)

_NANOS_PER_MICRO = 1_000

_NOISE = Interval(-np.inf, 0,                    closed="left")
_UFT   = Interval(0,                _NANOS_PER_MICRO, closed="left")
_HFT   = Interval(_NANOS_PER_MICRO, 10 * _NANOS_PER_MICRO, closed="left")
_OTHER = Interval(10 * _NANOS_PER_MICRO, np.inf, closed="left")

# ---------------------------------------------------------------------------
# Pure analytics functions
# ---------------------------------------------------------------------------

def w2w_latency(
    t9d_to_t3a_min_ns: int,
    reaction_t_3a: int,
    trigger_t_9d: int,
) -> int:
    return reaction_t_3a - trigger_t_9d - t9d_to_t3a_min_ns


def classify_latency(latency: int) -> str:
    if latency in _NOISE:
        return "Noise"
    if latency in _UFT:
        return "UFT"
    if latency in _HFT:
        return "HFT"
    if latency in _OTHER:
        return "Other"
    raise ValueError(f"unhandled latency: {latency}")


def signal_noise(
    trigger_price: float | None,
    reaction_price: float | None,
    markout_price: float | None,
) -> tuple[float, float]:
    if trigger_price is None or reaction_price is None or markout_price is None:
        return 0.0, 0.0

    if (trigger_price <= reaction_price <= markout_price or
            trigger_price >= reaction_price >= markout_price):
        return abs(reaction_price - trigger_price) / reaction_price, 0.0
    elif (reaction_price <= trigger_price <= markout_price or
            reaction_price >= trigger_price >= markout_price):
        return 0.0, abs(reaction_price - trigger_price) / reaction_price
    elif (trigger_price <= markout_price <= reaction_price or
            reaction_price <= markout_price <= trigger_price):
        sig = abs(markout_price - trigger_price) / reaction_price
        noi = abs(reaction_price - markout_price) / reaction_price
        return sig, noi
    raise ValueError(
        f"unhandled signal/noise case: "
        f"trigger={trigger_price}, reaction={reaction_price}, markout={markout_price}"
    )


def compute_event_analytics(
    microprices: pl.DataFrame,
    trigger_events: pl.DataFrame,
    reaction_events: pl.DataFrame,
    markout_period: int,
    t9d_to_t3a_min_ns: int,
    t1_to_t3a_latency: int,
    epsilon: int = 10,
) -> pl.DataFrame:
    """Compute w2w latency, classification and signal/noise for reaction events.

    The two latency constants (t9d_to_t3a_min_ns, t1_to_t3a_latency) are passed
    in rather than imported, keeping this function free of dbg_cdm so it can be
    tested and reused independently.  The pipeline resolves them from
    dbg_cdm.hpt_utils per trading day.
    """
    assert not microprices.is_empty(),     "microprices is empty"
    assert not reaction_events.is_empty(), "reaction_events is empty"
    assert not trigger_events.is_empty(),  "trigger_events is empty"

    ob_ts   = microprices["timestamp"].to_list()
    ob_mp   = microprices["microprice"].to_list()
    trig_ts = trigger_events["t_9d"].to_list()
    trig_ei = trigger_events["exec_id"].to_list()
    N = len(ob_ts)

    latencies:       list[int]   = []
    classifications: list[str]   = []
    signals:         list[float] = []
    noises:          list[float] = []

    for event in reaction_events.rows(named=True):
        event_t_9d_min = event["t_3a"] - t9d_to_t3a_min_ns - epsilon

        idx_mp_event = bisect_left(ob_ts, event["exec_id"])
        mp_event = None if idx_mp_event >= N else ob_mp[idx_mp_event]

        idx_trigger = max(0, bisect_right(trig_ts, event_t_9d_min) - 1)
        trigger_t_9d_val = trig_ts[idx_trigger]

        idx_mp_trigger = bisect_left(ob_ts, trig_ei[idx_trigger])
        mp_trigger = None if idx_mp_trigger >= N else ob_mp[idx_mp_trigger]

        markout_ts = event["t_3a"] - t1_to_t3a_latency + markout_period
        if markout_ts > ob_ts[-1]:
            mp_markout = None
        else:
            idx_mp_markout = bisect_left(ob_ts, markout_ts)
            mp_markout = None if idx_mp_markout >= N else ob_mp[idx_mp_markout]

        lat = w2w_latency(t9d_to_t3a_min_ns, event["t_3a"], trigger_t_9d_val)
        cls = classify_latency(lat)
        sig, noi = signal_noise(mp_trigger, mp_event, mp_markout)

        latencies.append(lat)
        classifications.append(cls)
        signals.append(sig)
        noises.append(noi)

    return reaction_events.with_columns(
        pl.Series("w2w_latency", latencies),
        pl.Series("class",       classifications),
        pl.Series("signal",      signals),
        pl.Series("noise",       noises),
    ).cast(HFT_ANALYTICS_SCHEMA)


# ---------------------------------------------------------------------------
# Most-liquid-instrument resolution (analytics layer)
# ---------------------------------------------------------------------------

def _most_liquid_cache_path(mic: str, product: str) -> Path:
    from deutsche_boerse import RAW_ROOT
    path = RAW_ROOT / "most_liquid_instrument" / mic
    path.mkdir(parents=True, exist_ok=True)
    return path / f"most_liquid_instrument.{product}.pickle"


def _load_most_liquid(mic: str, product: str) -> dict[int, int | None]:
    p = _most_liquid_cache_path(mic, product)
    if p.exists():
        with p.open("rb") as fh:
            return pickle.load(fh)
    return {}


def _save_most_liquid(mic: str, product: str, data: dict[int, int | None]) -> None:
    with _most_liquid_cache_path(mic, product).open("wb") as fh:
        pickle.dump(data, fh)


def resolve_most_liquid(
    mic: str,
    ccyymmdd: int,
    product: str,
) -> tuple[int, int] | None:
    from dbg_cdm.a7_utils import get_market_segment_id, get_most_liquid_contract

    market_segment_id = get_market_segment_id(mic, ccyymmdd, product)
    cache = _load_most_liquid(mic, product)

    if ccyymmdd in cache:
        security_id = cache[ccyymmdd]
    else:
        log.info("resolving most liquid for %s %s %d", mic, product, ccyymmdd)
        security_id = get_most_liquid_contract(mic, ccyymmdd, market_segment_id)
        cache[ccyymmdd] = security_id
        _save_most_liquid(mic, product, cache)

    if security_id is None:
        log.warning("no most liquid contract for %s %s %d", mic, product, ccyymmdd)
        return None

    return market_segment_id, security_id


# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------

@dataclass
class HFTJob:
    mic: str
    ccyymmdd: int
    product: str
    market_segment_id: int
    security_id: int
    markout_period: int


def iter_jobs(
    mic: str,
    product: str,
    from_ccyymmdd: int,
    to_ccyymmdd: int,
    markout_period: int,
) -> Iterator[HFTJob]:
    from dbg_cdm.time_utils import SKIP_DATES, daterange, today_ccyymmdd

    for ccyymmdd in daterange(from_ccyymmdd, to_ccyymmdd):
        if ccyymmdd in SKIP_DATES or ccyymmdd >= today_ccyymmdd():
            continue
        result = resolve_most_liquid(mic, ccyymmdd, product)
        if result is None:
            continue
        market_segment_id, security_id = result
        yield HFTJob(
            mic=mic,
            ccyymmdd=ccyymmdd,
            product=product,
            market_segment_id=market_segment_id,
            security_id=security_id,
            markout_period=markout_period,
        )


# ---------------------------------------------------------------------------
# Datalake-aware pipeline entry point
# ---------------------------------------------------------------------------

def run_hft_analytics(
    index: "DatalakeIndex",
    mic: str,
    product: str,
    from_ccyymmdd: int,
    to_ccyymmdd: int,
    markout_period: int,
    *,
    pipeline: str = "PhD-Narrative-Finance",
    pipeline_version: str,
    pipeline_repo: str | None = None,
    epsilon: int = 10,
    skip_missing: bool = True,
) -> "Artifact":
    from dbg_cdm.hpt_utils import T1_TO_T3A_LATENCY, t9d_to_t3a_latency_min_ns

    from deutsche_boerse.data_collection.events import get_events
    from deutsche_boerse.data_collection.microprice import get_microprice
    from deutsche_boerse.data_collection.trades import get_trades

    hyperparams = {
        "mic":            mic,
        "product":        product,
        "from_ccyymmdd":  from_ccyymmdd,
        "to_ccyymmdd":    to_ccyymmdd,
        "markout_period": markout_period,
        "epsilon":        epsilon,
    }

    with index.run(
        kind="hft_analytics",
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        hyperparams=hyperparams,
        layer="derived",
    ) as run:
        n_days = 0
        for job in iter_jobs(mic, product, from_ccyymmdd, to_ccyymmdd, markout_period):
            try:
                trades = get_trades(job.mic, job.ccyymmdd)
                events = get_events(
                    job.mic, job.ccyymmdd, job.market_segment_id, job.security_id
                )
                microprices = get_microprice(
                    job.mic, job.ccyymmdd, job.market_segment_id, job.security_id
                )
            except FileNotFoundError as exc:
                if skip_missing:
                    log.warning("skip %s %d: %s", job.mic, job.ccyymmdd, exc)
                    continue
                raise

            if trades.is_empty() or events.is_empty() or microprices.is_empty():
                log.warning("skip %s %d: empty data", job.mic, job.ccyymmdd)
                continue

            result = compute_event_analytics(
                microprices=microprices,
                trigger_events=trades,
                reaction_events=events,
                markout_period=job.markout_period,
                t9d_to_t3a_min_ns=t9d_to_t3a_latency_min_ns(job.ccyymmdd),
                t1_to_t3a_latency=T1_TO_T3A_LATENCY,
                epsilon=epsilon,
            )

            out = run.out_dir / f"{job.ccyymmdd}.parquet"
            result.write_parquet(out, compression="zstd")
            n_days += 1
            log.info("%s %d: %d events -> %s", job.mic, job.ccyymmdd, len(result), out.name)

        if n_days == 0:
            raise RuntimeError(
                f"hft_analytics produced no output for {mic}/{product} "
                f"{from_ccyymmdd}-{to_ccyymmdd}"
            )
        run.note(f"{n_days} days written for {mic}/{product}")

    return index.get(run.artifact_id)
