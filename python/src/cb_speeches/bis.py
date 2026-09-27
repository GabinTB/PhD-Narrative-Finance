"""BIS central-banker speeches: yearly zip discovery, download and parsing.

BIS publishes one zip per speech year (``speeches-YYYY.zip``, one CSV
``speeches_YYYY.csv`` inside, columns url, title, description, date, text,
author) on its download page and replaces them in place when speeches are
added or revised (ETag / Last-Modified change). gingado's ``load_CB_speeches``
(<= 0.2.7) still points at the retired ``bis.org/speeches/speeches_YYYY.zip``
URLs (404), so this module fetches the files itself and returns the same
columns.

``speech_id`` is BIS's review code, the stem of the speech URL
(``https://www.bis.org/review/r970211c.pdf`` -> ``r970211c``): stable across
text revisions and file formats. A URL without one raises.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from cb_speeches.schema import CONTENT_FIELDS, RAW_COLUMNS

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://www.bis.org/pages/download-central-bankers-speeches/"
DEFAULT_INDEX_URL = "https://www.bis.org/cbspeeches/download.htm"

_YEAR_HREF = re.compile(r"speeches-(\d{4})\.zip")
_SPEECH_ID = re.compile(r"/review/(r\d{6}[a-z]*)\.(?:pdf|htm|html)$")


class SpeechDataError(ValueError):
    """A BIS file does not have the expected content (never silently repaired)."""


def zip_name(year: int) -> str:
    return f"speeches-{year}.zip"


@dataclass(frozen=True)
class Manifest:
    """Where a raw zip came from and what it was."""

    year: int
    url: str
    etag: str | None
    last_modified: str | None
    sha256: str | None = None
    size_bytes: int | None = None
    fetched_at: str | None = None

    @property
    def zip(self) -> str:
        return zip_name(self.year)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BISClient:
    """HTTP access to the BIS download page and its yearly zips.

    Args:
        base_url:  directory of the ``speeches-YYYY.zip`` files.
        index_url: page listing them (year discovery).
        timeout:   per-request timeout (s).
        transport: injected httpx transport (tests).
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, index_url: str = DEFAULT_INDEX_URL,
                 *, timeout: float = 300.0, transport: Any = None) -> None:
        import httpx

        self.base_url = base_url.rstrip("/") + "/"
        self.index_url = index_url
        self._http = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True)

    def __repr__(self) -> str:
        return f"BISClient({self.base_url!r})"

    def url(self, year: int) -> str:
        return self.base_url + zip_name(year)

    def years(self) -> list[int]:
        """Years with a zip on the download page, ascending."""
        resp = self._http.get(self.index_url)
        resp.raise_for_status()
        years = sorted({int(y) for y in _YEAR_HREF.findall(resp.text)})
        if not years:
            raise SpeechDataError(f"no speeches-YYYY.zip link on {self.index_url}")
        return years

    def head(self, year: int) -> Manifest:
        resp = self._http.head(self.url(year))
        resp.raise_for_status()
        return Manifest(year, self.url(year), resp.headers.get("etag"),
                        resp.headers.get("last-modified"))

    def download(self, year: int, fetched_at: datetime) -> tuple[Manifest, bytes]:
        resp = self._http.get(self.url(year))
        resp.raise_for_status()
        content = resp.content
        if not zipfile.is_zipfile(io.BytesIO(content)):
            raise SpeechDataError(f"{self.url(year)} is not a zip "
                                  f"({resp.headers.get('content-type')})")
        manifest = Manifest(year, self.url(year), resp.headers.get("etag"),
                            resp.headers.get("last-modified"),
                            hashlib.sha256(content).hexdigest(), len(content),
                            fetched_at.isoformat(timespec="seconds"))
        log.info("downloaded %s (%d bytes, sha256 %s)", manifest.zip, len(content),
                 manifest.sha256[:12])
        return manifest, content

    def close(self) -> None:
        self._http.close()


def read_zip(content: bytes, year: int) -> pd.DataFrame:
    """The raw BIS rows of one yearly zip (all columns as str, nulls as None)."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        members = [n for n in zf.namelist() if n.endswith(".csv")]
        expected = f"speeches_{year}.csv"
        member = expected if expected in members else (members[0] if len(members) == 1
                                                       else None)
        if member is None:
            raise SpeechDataError(f"{zip_name(year)}: expected {expected}, found {members}")
        with zf.open(member) as fh:
            df = pd.read_csv(fh, dtype=str, keep_default_na=False, na_values=[""])
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise SpeechDataError(f"{zip_name(year)} lacks columns {sorted(missing)}")
    return df[list(RAW_COLUMNS)].astype(object).where(df[list(RAW_COLUMNS)].notna(), None)


def speech_id(url: str) -> str:
    m = _SPEECH_ID.search(url or "")
    if m is None:
        raise SpeechDataError(f"no BIS review code in speech URL {url!r}")
    return m.group(1)


def content_sha256(row: dict[str, Any]) -> str:
    """sha256 of the speech content fields (date as ISO day), None kept as null."""
    payload = json.dumps([row[k] for k in CONTENT_FIELDS], ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def normalise(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Raw BIS rows -> one row per speech_id with date and content hash.

    Exact duplicate speeches (same id and content) are collapsed and logged; the
    same id with different content raises.
    """
    df = raw.copy()
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        bad = df.loc[dates.isna(), "date"].head(3).tolist()
        raise SpeechDataError(f"{zip_name(year)}: unparseable dates {bad}")
    df["date"] = dates.dt.date
    df["speech_id"] = [speech_id(u) for u in df["url"]]
    df["content_sha256"] = [content_sha256(r) for r in df.to_dict("records")]
    dup = df.duplicated("speech_id", keep=False)
    if dup.any():
        conflicts = df[dup].groupby("speech_id")["content_sha256"].nunique()
        conflicts = conflicts[conflicts > 1]
        if len(conflicts):
            raise SpeechDataError(f"{zip_name(year)}: speech ids with different content: "
                                  f"{conflicts.index.tolist()[:10]}")
        log.warning("%s: %d exact duplicate row(s) collapsed", zip_name(year),
                    int(df.duplicated("speech_id").sum()))
        df = df.drop_duplicates("speech_id")
    df["source_zip"] = zip_name(year)
    return df.sort_values(["date", "speech_id"], kind="stable").reset_index(drop=True)
