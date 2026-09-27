# cb_speeches

BIS central-banker speeches as datalake artifacts: a historical load, in-place updates, and an
LLM provenance-NER table keyed by `speech_id`. Productionises the Zenodo prototypes
[18034730](https://zenodo.org/records/18034730) (extract) and
[18157708](https://zenodo.org/records/18157708) (NER).

## Source

BIS publishes one zip per speech year on <https://www.bis.org/cbspeeches/download.htm>
(`/pages/download-central-bankers-speeches/speeches-YYYY.zip`, one `speeches_YYYY.csv`:
url, title, description, date, text, author). The files are replaced in place when speeches are
added or revised. gingado <= 0.2.7 `load_CB_speeches` still points at the retired
`bis.org/speeches/speeches_YYYY.zip` (404), so `bis.py` fetches the files itself and returns the
same columns. Both URLs are parameters (`--base-url`, `--index-url`).

`speech_id` is BIS's review code, the stem of the speech URL (`.../review/r970211c.pdf` ->
`r970211c`). It is stable across text revisions. `content_sha256` identifies a version. The
same id with different content inside one zip raises. The prototype key (`date + uuid5(title)`,
then `drop_duplicates`) silently dropped same-day, same-title speeches.

## Jobs

```bash
jobs start cb_speeches --start 1996-01-01 --end 2026-12-31 [--partition-freq Y] [--temp]
jobs update <cb_speeches id>          # new / revised / removed speeches, same artifact
jobs start cb_speech_ner --speeches-id <id> --provider ollama --model <name> [--temp]
jobs update <cb_speech_ner id>        # NER for speeches not in the table yet
jobs resume <id>                      # a killed start or update
```

`cb_speeches` files (all hashed):

| file | content |
|---|---|
| `<period key>.parquet` | base partitions by speech date (`change = base`); empty when no speech |
| `update-<vintage>-YYYY.parquet` | one delta per BIS year checked by an update (`new` / `revised` / `removed`; empty when unchanged) |
| `raw-speeches-YYYY-<sha12>.zip` | every raw zip read, content-addressed |
| `plan-<vintage>.json` | what an execution fetches: mode, years, URLs |

An update never rewrites a file. It checks every BIS year >= the artifact's start: an unchanged
ETag writes an empty delta without downloading; otherwise the zip is diffed against the current
state. Speeches dated after the declared `end` land in deltas. Each parquet records the manifest
(URL, ETag, Last-Modified, sha256, size, fetch time) of the zip(s) it was read from in its
key-value metadata (`file_sources`).

`cb_speech_ner` holds one row per (speech_id, input_sha256): the prototype's author,
organization, country_code (ISO 3166-1 alpha-2) and sentiment (hawkish / dovish / neutral),
extracted from date, author, title and description (not the text). Replies are validated
against the schema; a reply still invalid after `--max-attempts` is an `error` row. The
provider's endpoint and key come from `.env` (`nlp.llm.PROVIDERS`); the model is always
given. Identity: speeches id, provider, model, sampling, `prompt_sha256`. A changed prompt
cannot resume or update (bump the version). A different served model (Ollama digest, hosted
snapshot) stops the job.

## Reading

```python
from cb_speeches.speeches import read_speeches
from cb_speeches.ner import read_ner, speeches_with_ner

speeches = read_speeches(art)                        # latest version of every speech
speeches = read_speeches(art, as_of="2026-10-01")    # as recorded at that time
table = speeches_with_ner(speeches_art, ner_art)     # left join on speech_id
```

Point-in-time caveat: base rows carry the base load's vintage, not BIS's publication time.
`as_of` is exact about availability only from the first update on.
