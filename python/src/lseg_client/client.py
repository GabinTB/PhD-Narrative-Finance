"""LSEG data through the LSEG Data Library (``lseg.data``): index members, identifiers,
fundamentals, prices and news.

    with LsegClient.connect() as lseg:          # desktop (tunnel) first, platform fallback
        members = lseg.index_members(".STOXX")
        ids = lseg.identifiers(members["ric"])

What the entitlements give (probed 2026-09-29 through Workspace):

    index_members   CURRENT members only. A dated chain ("0#.SPX(2010-01-04)") returns
                    today's members unchanged and a dated TR.IndexConstituentRIC returns
                    nothing, so no date argument is offered. MSCI member chains do not
                    resolve (index levels only). History comes from WRDS
                    (``wrds_client.indices``).
    identifiers     current ISIN / SEDOL / CUSIP / organisation PermID / LEI / MIC per RIC
    rics_from_isins ISIN -> RIC through symbol conversion; unresolved ISINs (e.g. delisted)
                    are absent from the result, never guessed
    fundamentals    TR.* fields over a date range at a frequency (FY, FQ, ...)
    prices          interday / intraday history (``ld.get_history``)
    headlines       news headlines (back to at least 2008); the service returns at most
                    100 per call, so a date range is paged backwards
    story           one story's body

Requests are chunked (``CHUNK`` instruments per call) and retried with backoff on
throttling.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any, Iterator

import pandas as pd

from lseg_client.session import DESKTOP, PLATFORM, LsegConfig, open_session

log = logging.getLogger(__name__)

CHUNK = 200
NEWS_PAGE = 100
_RETRIES = 3
_THROTTLED = ("429", "too many", "rate limit", "throttl")

IDENTIFIER_FIELDS = {
    "TR.ISIN": "isin", "TR.SEDOL": "sedol", "TR.CUSIP": "cusip",
    "TR.OrganizationID": "org_permid", "TR.LegalEntityIdentifier": "lei",
    "TR.ExchangeMarketIdCode": "mic", "TR.CommonName": "name",
}


def _chunks(items: Sequence[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield list(items[i:i + size])


def _unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(v) for v in values if v is not None and not pd.isna(v)))


class LsegClient:
    """Thin, tested wrapper over ``lseg.data``; ``ld`` is the module (injectable)."""

    def __init__(self, ld: Any, session: Any = None, kind: str | None = None):
        self._ld, self.session, self.kind = ld, session, kind

    @classmethod
    @contextmanager
    def connect(cls, config: LsegConfig | None = None, *,
                prefer: tuple[str, ...] = (DESKTOP, PLATFORM)) -> Iterator[LsegClient]:
        import lseg.data as ld

        session, kind = open_session(config or LsegConfig.from_env(), prefer=prefer, ld=ld)
        try:
            yield cls(ld, session, kind)
        finally:
            session.close()

    # -- plumbing -------------------------------------------------------------

    def _call(self, fn, *args, **kwargs):
        for attempt in range(_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - retried only when throttled
                if attempt + 1 < _RETRIES and any(t in str(exc).lower() for t in _THROTTLED):
                    time.sleep(2 ** attempt)
                    continue
                raise

    def _get_data(self, universe: Sequence[str], fields: Sequence[str],
                  names: Sequence[str], parameters: dict | None = None) -> pd.DataFrame:
        """``ld.get_data`` chunked, columns renamed positionally to ``instrument`` + names."""
        frames = []
        for chunk in _chunks(list(universe), CHUNK):
            df = self._call(self._ld.get_data, chunk, list(fields), parameters=parameters)
            if df.shape[1] != len(names) + 1:
                raise ValueError(f"expected {len(names) + 1} columns, got {list(df.columns)}")
            df.columns = ["instrument", *names]
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["instrument", *names])
        return pd.concat(frames, ignore_index=True)

    # -- membership and identifiers ------------------------------------------

    def index_members(self, index_ric: str) -> pd.DataFrame:
        """Current members of an index: ric, name, isin (see the module docstring)."""
        df = self._get_data([f"0#{index_ric}"], ["TR.CommonName", "TR.ISIN"], ["name", "isin"])
        return df.rename(columns={"instrument": "ric"}).replace({"": None})

    def identifiers(self, rics: Iterable[str]) -> pd.DataFrame:
        """Current identifiers per RIC: ric + ``IDENTIFIER_FIELDS`` values."""
        df = self._get_data(_unique(rics), list(IDENTIFIER_FIELDS),
                            list(IDENTIFIER_FIELDS.values()))
        return df.rename(columns={"instrument": "ric"}).replace({"": None})

    def rics_from_isins(self, isins: Iterable[str]) -> pd.DataFrame:
        """isin, ric, org_permid for the ISINs symbol conversion resolves."""
        frames = []
        for chunk in _chunks(_unique(isins), CHUNK):
            df = self._call(self._ld.discovery.convert_symbols, chunk,
                            from_symbol_type="IssueISIN")
            frames.append(df)
        df = pd.concat(frames) if frames else pd.DataFrame()
        if df.empty:
            return pd.DataFrame(columns=["isin", "ric", "org_permid"])
        out = pd.DataFrame({"isin": df.index.astype(str), "ric": df.get("RIC"),
                            "org_permid": df.get("IssuerOAPermID")}).reset_index(drop=True)
        return out.dropna(subset=["ric"]).drop_duplicates("isin")

    # -- time series ------------------------------------------------------------

    def fundamentals(self, rics: Iterable[str], fields: Sequence[str], start: date | str,
                     end: date | str, frequency: str = "FY") -> pd.DataFrame:
        """TR.* ``fields`` per RIC over [start, end] at ``frequency`` (FY, FQ, ...), with the
        period date of the first field: ric, date, one column per field."""
        date_field = f"{fields[0]}.date"
        params = {"SDate": str(start), "EDate": str(end), "Frq": frequency}
        df = self._get_data(_unique(rics), [date_field, *fields], ["date", *fields], params)
        return df.rename(columns={"instrument": "ric"})

    def prices(self, rics: Iterable[str], start: date | str, end: date | str, *,
               interval: str = "daily", fields: Sequence[str] | None = None) -> pd.DataFrame:
        """``ld.get_history`` for each chunk of RICs over [start, end], both inclusive, long
        format: date, ric, then the fields (the library's default fields when ``fields`` is
        None). The service treats ``start`` as exclusive (a daily request from 2024-01-02
        begins on 2024-01-03), so the request starts a day earlier and is cut at ``start``."""
        start_ts = pd.Timestamp(start)
        frames = []
        for chunk in _chunks(_unique(rics), CHUNK):
            df = self._call(self._ld.get_history, chunk, fields=list(fields) if fields else None,
                            interval=interval, start=str((start_ts - timedelta(days=1)).date()),
                            end=str(end))
            df = df[df.index >= start_ts]
            if isinstance(df.columns, pd.MultiIndex):
                df = df.stack(level=0, future_stack=True).rename_axis(["date", "ric"])
            else:
                df = df.assign(ric=chunk[0]).rename_axis("date").set_index("ric", append=True)
            frames.append(df.reset_index())
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out.columns.name = None
        return out

    # -- news -------------------------------------------------------------------

    def headlines(self, query: str, start: date | str | datetime, end: date | str | datetime,
                  *, max_items: int = 10_000) -> pd.DataFrame:
        """Headlines matching ``query`` in [start, end], newest first, paged backwards
        ``NEWS_PAGE`` at a time: version_created, headline, story_id, source_code."""
        start_ts = pd.Timestamp(start)
        cursor = pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        n = 0
        while n < max_items and cursor > start_ts:
            page = self._call(self._ld.news.get_headlines, query, count=NEWS_PAGE,
                              start=start_ts.isoformat(), end=cursor.isoformat())
            if page is None or page.empty:
                break
            page = page.reset_index().rename(columns={
                "versionCreated": "version_created", "storyId": "story_id",
                "sourceCode": "source_code"})
            frames.append(page)
            n += len(page)
            oldest = pd.Timestamp(page["version_created"].min())
            if len(page) < NEWS_PAGE or oldest >= cursor:
                break
            cursor = oldest - timedelta(milliseconds=1)
        if not frames:
            return pd.DataFrame(columns=["version_created", "headline", "story_id",
                                         "source_code"])
        out = pd.concat(frames, ignore_index=True).drop_duplicates("story_id")
        return out.head(max_items).reset_index(drop=True)

    def story(self, story_id: str) -> str:
        return self._call(self._ld.news.get_story, story_id)
