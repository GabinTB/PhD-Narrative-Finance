"""bis_gingado.cb_speeches.bis: ids, parsing, duplicates, resumable bulk download."""
from __future__ import annotations

from datetime import date, datetime, timezone

import httpx
import pytest

from bis_gingado.cb_speeches.bis import (
    BISClient,
    SpeechDataError,
    content_sha256,
    normalise,
    read_zip,
    speech_id,
)
from tests.cb_speeches.fakes import BULK_URL, FakeBIS, make_zip, speech


@pytest.mark.parametrize("url, code", [
    ("https://www.bis.org/review/r970211c.pdf", "r970211c"),
    ("https://www.bis.org/review/r970106.pdf", "r970106"),       # no letter (1996 file)
    ("https://www.bis.org/review/r250924f.htm", "r250924f"),
    # the other shapes of the real history (2026-09-30), all unique
    ("https://www.bis.org/review/r9906166.pdf", "r9906166"),     # 7 digits
    ("https://www.bis.org/review/r1607014a.htm", "r1607014a"),
    ("https://www.bis.org/review/r080610bc.pdf", "r080610bc"),   # two letters
    ("https://www.bis.org/review/r140522e1.htm", "r140522e1"),   # letter + digit
    ("https://www.bis.org/review/r151221.a.pdf", "r151221.a"),   # dotted part kept
    ("https://www.bis.org/review/r150714c.copy-1.htm", "r150714c.copy-1"),
])
def test_speech_id_is_the_bis_review_code(url, code):
    assert speech_id(url) == code


def test_url_without_review_code_raises():
    with pytest.raises(SpeechDataError, match="review code"):
        speech_id("https://www.ecb.europa.eu/press/key/date/2024/html/x.en.html")


def test_read_and_normalise_one_year():
    rows = [speech(date(2008, 1, 3), 1), speech(date(2008, 1, 2), 0)]
    rows[0]["description"] = ""
    df = normalise(read_zip(make_zip(rows)))
    assert df["speech_id"].tolist() == ["r080102a", "r080103b"]        # sorted by date
    assert df["date"].tolist() == [date(2008, 1, 2), date(2008, 1, 3)]
    assert df.loc[1, "description"] is None                          # empty -> null
    assert df["source_zip"].unique().tolist() == ["speeches.zip"]
    assert df["content_sha256"].str.len().eq(64).all()


def test_exact_duplicates_collapse_conflicting_ones_raise():
    row = speech(date(2008, 1, 2), 0)
    df = normalise(read_zip(make_zip([row, dict(row)])))
    assert len(df) == 1
    other = dict(row, text="a revised text")
    with pytest.raises(SpeechDataError, match="different content"):
        normalise(read_zip(make_zip([row, other])))


def test_content_hash_changes_with_any_content_field():
    base = {"url": "u", "title": "t", "description": None, "date": date(2008, 1, 1),
            "author": "a", "text": "x"}
    assert content_sha256(base) != content_sha256(dict(base, text="y"))
    assert content_sha256(base) != content_sha256(dict(base, description=""))


def _client(fake: FakeBIS) -> BISClient:
    return BISClient(BULK_URL, transport=fake.transport(), retry_wait_s=0)


T = datetime(2026, 9, 27, tzinfo=timezone.utc)


def test_client_downloads_the_bulk_file_with_manifest():
    fake = FakeBIS({2008: [speech(date(2008, 1, 2), 0)], 2009: [speech(date(2009, 1, 2), 0)]})
    client = _client(fake)
    head = client.head()
    manifest, content = client.download(T)
    assert content == fake.zip()
    assert manifest.etag == head.etag and manifest.size_bytes == len(content) == head.size_bytes
    assert manifest.zip == "speeches.zip" and manifest.fetched_at == "2026-09-27T00:00:00+00:00"
    assert read_zip(content)["url"].str.contains("r080102a|r090102a").all()


def test_cut_downloads_resume_where_they_stopped():
    fake = FakeBIS({2008: [speech(date(2008, 1, d), 0, text="x" * 5000) for d in range(1, 29)]},
                   cut=3)
    manifest, content = _client(fake).download(T)
    assert content == fake.zip() and manifest.sha256
    gets = [m for m, _ in fake.requests if m == "GET"]
    assert len(gets) == 4                               # 3 cut requests + the finishing one


def test_a_new_version_during_the_download_restarts_it():
    fake = FakeBIS({2008: [speech(date(2008, 1, 2), 0, text="x" * 5000)]}, cut=1)
    client = _client(fake)
    real = fake.handler

    def handler(request):                               # BIS replaces the file after the cut
        if request.headers.get("range"):
            fake.rows[2008].append(speech(date(2008, 1, 3), 0))
            fake.handler = real
        return real(request)
    fake.handler = handler
    manifest, content = client.download(T)
    assert len(fake.rows[2008]) == 2                    # the file did change mid-download
    assert content == fake.zip()                        # the new version, whole, not a mix
    assert manifest.etag == fake.etag()


def test_non_zip_download_raises():
    moved = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>moved</html>"))
    client = BISClient(BULK_URL, transport=moved, retry_wait_s=0)
    with pytest.raises(SpeechDataError, match="not a zip"):
        client.download(T)
