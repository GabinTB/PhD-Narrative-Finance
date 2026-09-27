"""cb_speeches.bis: ids, parsing, duplicates, discovery."""
from __future__ import annotations

from datetime import date, datetime, timezone

import httpx
import pytest

from cb_speeches.bis import (
    BISClient,
    SpeechDataError,
    content_sha256,
    normalise,
    read_zip,
    speech_id,
)
from tests.cb_speeches.fakes import BASE_URL, INDEX_URL, FakeBIS, make_zip, speech


@pytest.mark.parametrize("url, code", [
    ("https://www.bis.org/review/r970211c.pdf", "r970211c"),
    ("https://www.bis.org/review/r970106.pdf", "r970106"),       # no letter (1996 file)
    ("https://www.bis.org/review/r250924f.htm", "r250924f"),
])
def test_speech_id_is_the_bis_review_code(url, code):
    assert speech_id(url) == code


def test_url_without_review_code_raises():
    with pytest.raises(SpeechDataError, match="review code"):
        speech_id("https://www.ecb.europa.eu/press/key/date/2024/html/x.en.html")


def test_read_and_normalise_one_year():
    rows = [speech(date(2008, 1, 3), 1), speech(date(2008, 1, 2), 0)]
    rows[0]["description"] = ""
    df = normalise(read_zip(make_zip(2008, rows), 2008), 2008)
    assert df["speech_id"].tolist() == ["r080102a", "r080103b"]        # sorted by date
    assert df["date"].tolist() == [date(2008, 1, 2), date(2008, 1, 3)]
    assert df.loc[1, "description"] is None                          # empty -> null
    assert df["source_zip"].unique().tolist() == ["speeches-2008.zip"]
    assert df["content_sha256"].str.len().eq(64).all()


def test_exact_duplicates_collapse_conflicting_ones_raise():
    row = speech(date(2008, 1, 2), 0)
    df = normalise(read_zip(make_zip(2008, [row, dict(row)]), 2008), 2008)
    assert len(df) == 1
    other = dict(row, text="a revised text")
    with pytest.raises(SpeechDataError, match="different content"):
        normalise(read_zip(make_zip(2008, [row, other]), 2008), 2008)


def test_content_hash_changes_with_any_content_field():
    base = {"url": "u", "title": "t", "description": None, "date": date(2008, 1, 1),
            "author": "a", "text": "x"}
    assert content_sha256(base) != content_sha256(dict(base, text="y"))
    assert content_sha256(base) != content_sha256(dict(base, description=""))


def test_client_discovers_years_and_downloads_with_manifest():
    fake = FakeBIS({2008: [speech(date(2008, 1, 2), 0)], 2009: [speech(date(2009, 1, 2), 0)]})
    client = BISClient(BASE_URL, INDEX_URL, transport=fake.transport())
    assert client.years() == [2008, 2009]
    head = client.head(2008)
    manifest, content = client.download(2008, datetime(2026, 9, 27, tzinfo=timezone.utc))
    assert content == fake.zip(2008)
    assert manifest.etag == head.etag and manifest.size_bytes == len(content)
    assert manifest.fetched_at == "2026-09-27T00:00:00+00:00"


def test_non_zip_download_raises():
    moved = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>moved</html>"))
    client = BISClient(BASE_URL, INDEX_URL, transport=moved)
    with pytest.raises(SpeechDataError, match="not a zip"):
        client.download(2008, datetime(2026, 1, 1, tzinfo=timezone.utc))
