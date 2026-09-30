"""BIS central-banker speeches: bulk download and parsing.

BIS publishes every speech since 1996 in one file, ``speeches.zip`` (one CSV
``speeches.csv`` inside, columns url, title, description, date, text, author;
~130 MB zipped, ~390 MB of CSV, 20,728 speeches on 2026-08-23), next to one
zip per year, and regenerates them all together when speeches are added or
revised (same Last-Modified on every file). gingado's ``load_CB_speeches``
(<= 0.2.7) still points at the retired ``bis.org/speeches/speeches_YYYY.zip``
URLs (404), so this module fetches the bulk file itself and returns the same
columns.

The BIS server often cuts a download short (``RemoteProtocolError`` after a
few MB). It answers byte ranges although it does not advertise them, so a cut
download resumes where it stopped (``If-Range`` on the ETag: a file replaced
meanwhile restarts from zero instead of mixing two versions).

``speech_id`` is BIS's review code, the stem of the speech URL
(``https://www.bis.org/review/r970211c.pdf`` -> ``r970211c``): stable across
text revisions and file formats. Shapes seen (whole history, 2026-09-30):
``r`` + 6 or 7 digits, then letters and/or a digit (``r970211c``,
``r9906166``, ``r1607014a``, ``r080610bc``, ``r140522e1``, ``r970106``), and
two with a dotted part (``r151221.a``, ``r150714c.copy-1``, kept whole); all
unique. A URL without one raises.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from bis_gingado.cb_speeches.schema import CONTENT_FIELDS, RAW_COLUMNS

log = logging.getLogger(__name__)

DEFAULT_URL = "https://www.bis.org/pages/download-central-bankers-speeches/speeches.zip"

_SPEECH_ID = re.compile(r"/review/(r\d{6,7}[a-z0-9]*(?:\.[a-z0-9-]+)?)\.(?:pdf|htm|html)$")
_ATTEMPTS = 20             # requests per download (each resumes where the last one stopped)


class SpeechDataError(ValueError):
    """A BIS file does not have the expected content (never silently repaired)."""


def zip_name(url: str) -> str:
    """The file name of a download URL (``speeches.zip``)."""
    return url.rstrip("/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Manifest:
    """Where a raw zip came from and what it was."""

    url: str
    etag: str | None
    last_modified: str | None
    sha256: str | None = None
    size_bytes: int | None = None
    fetched_at: str | None = None

    @property
    def zip(self) -> str:
        return zip_name(self.url)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BISClient:
    """HTTP access to BIS's bulk speeches file.

    Args:
        url:       the bulk ``speeches.zip``.
        timeout:   per-request timeout (s).
        transport: injected httpx transport (tests).
    """

    def __init__(self, url: str = DEFAULT_URL, *, timeout: float = 300.0,
                 transport: Any = None, retry_wait_s: float = 2.0) -> None:
        import httpx

        self.url = url
        self.retry_wait_s = retry_wait_s
        self._http = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True)

    def __repr__(self) -> str:
        return f"BISClient({self.url!r})"

    def _retrying(self, fn):
        import httpx

        for attempt in range(_ATTEMPTS):
            try:
                return fn()
            except httpx.TransportError:
                if attempt + 1 == _ATTEMPTS:
                    raise
                time.sleep(self.retry_wait_s * min(attempt + 1, 5))
        raise AssertionError("unreachable")

    def head(self) -> Manifest:
        """The file's ETag / Last-Modified / size, without downloading it."""
        def call():
            resp = self._http.head(self.url)
            resp.raise_for_status()
            return resp
        resp = self._retrying(call)
        size = resp.headers.get("content-length")
        return Manifest(self.url, resp.headers.get("etag"), resp.headers.get("last-modified"),
                        size_bytes=int(size) if size else None)

    def download(self, fetched_at: datetime) -> tuple[Manifest, bytes]:
        """The whole file, resumed across cut connections (see the module docstring)."""
        import httpx

        head = self.head()
        buf = bytearray()
        for attempt in range(_ATTEMPTS):
            headers = {}
            if buf:
                headers["Range"] = f"bytes={len(buf)}-"
                if head.etag:
                    headers["If-Range"] = head.etag
            try:
                with self._http.stream("GET", self.url, headers=headers) as resp:
                    resp.raise_for_status()
                    if buf and resp.status_code != 206:      # new version or no ranges
                        log.warning("%s: resume refused (HTTP %d); restarting from byte 0",
                                    self.url, resp.status_code)
                        buf.clear()
                        head = Manifest(self.url, resp.headers.get("etag"),
                                        resp.headers.get("last-modified"), size_bytes=(
                                            int(resp.headers["content-length"])
                                            if resp.headers.get("content-length") else None))
                    for chunk in resp.iter_bytes():
                        buf.extend(chunk)
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout,
                    httpx.ConnectError) as exc:
                log.info("%s: cut at %d bytes (%s), resuming", zip_name(self.url), len(buf),
                         type(exc).__name__)
                time.sleep(self.retry_wait_s * min(attempt + 1, 5) if not buf else 0)
                continue
            if head.size_bytes is None or len(buf) >= head.size_bytes:
                break
        else:
            raise SpeechDataError(f"{self.url}: incomplete after {_ATTEMPTS} requests "
                                  f"({len(buf)} of {head.size_bytes} bytes)")
        content = bytes(buf)
        if head.size_bytes is not None and len(content) != head.size_bytes:
            raise SpeechDataError(f"{self.url}: {len(content)} bytes, "
                                  f"announced {head.size_bytes}")
        if not zipfile.is_zipfile(io.BytesIO(content)):
            raise SpeechDataError(f"{self.url} is not a zip")
        manifest = Manifest(self.url, head.etag, head.last_modified,
                            hashlib.sha256(content).hexdigest(), len(content),
                            fetched_at.isoformat(timespec="seconds"))
        log.info("downloaded %s (%d bytes, sha256 %s)", manifest.zip, len(content),
                 manifest.sha256[:12])
        return manifest, content

    def close(self) -> None:
        self._http.close()


def read_zip(content: bytes, name: str = "speeches.zip") -> pd.DataFrame:
    """The raw BIS rows of a speeches zip holding one CSV (all columns as str, nulls as
    None)."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        members = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(members) != 1:
            raise SpeechDataError(f"{name}: expected one CSV, found {members}")
        with zf.open(members[0]) as fh:
            df = pd.read_csv(fh, dtype=str, keep_default_na=False, na_values=[""])
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise SpeechDataError(f"{name} lacks columns {sorted(missing)}")
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


def normalise(raw: pd.DataFrame, source: str = "speeches.zip") -> pd.DataFrame:
    """Raw BIS rows -> one row per speech_id with date and content hash; ``source`` (the
    zip read) names the file in errors and fills ``source_zip``.

    Exact duplicate speeches (same id and content) are collapsed and logged; the
    same id with different content raises.
    """
    df = raw.copy()
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        bad = df.loc[dates.isna(), "date"].head(3).tolist()
        raise SpeechDataError(f"{source}: unparseable dates {bad}")
    df["date"] = dates.dt.date
    df["speech_id"] = [speech_id(u) for u in df["url"]]
    df["content_sha256"] = [content_sha256(r) for r in df.to_dict("records")]
    dup = df.duplicated("speech_id", keep=False)
    if dup.any():
        conflicts = df[dup].groupby("speech_id")["content_sha256"].nunique()
        conflicts = conflicts[conflicts > 1]
        if len(conflicts):
            raise SpeechDataError(f"{source}: speech ids with different content: "
                                  f"{conflicts.index.tolist()[:10]}")
        log.warning("%s: %d exact duplicate row(s) collapsed", source,
                    int(df.duplicated("speech_id").sum()))
        df = df.drop_duplicates("speech_id")
    df["source_zip"] = source
    return df.sort_values(["date", "speech_id"], kind="stable").reset_index(drop=True)
