# bis_gingado.cb_speeches

BIS central-banker speeches as datalake artifacts: a historical load, in-place updates, and an
LLM provenance-NER table keyed by `speech_id`. Productionises the Zenodo prototypes
[18034730](https://zenodo.org/records/18034730) (extract) and
[18157708](https://zenodo.org/records/18157708) (NER).

## Source

BIS publishes every speech since 1996 in one bulk file,
<https://www.bis.org/pages/download-central-bankers-speeches/speeches.zip> (one `speeches.csv`:
url, title, description, date, text, author; ~130 MB zipped, 20,728 speeches on 2026-08-23),
next to one zip per year, and regenerates them all together when speeches are added or revised.
The job reads the bulk file (`--url`). The BIS server often cuts downloads short; they resume
where they stopped (byte ranges, `If-Range` on the ETag so a file replaced meanwhile restarts
from zero). gingado <= 0.2.7 `load_CB_speeches` still points at the retired
`bis.org/speeches/speeches_YYYY.zip` (404), hence `bis.py`.

`speech_id` is BIS's review code, the stem of the speech URL (`.../review/r970211c.pdf` ->
`r970211c`): `r` + 6 or 7 digits, then letters and/or a digit, and two with a dotted part
(`r151221.a`, `r150714c.copy-1`, kept whole); all unique over the history. It is stable across
text revisions. `content_sha256` identifies a version. The same id with different content
raises. The prototype key (`date + uuid5(title)`, then `drop_duplicates`) silently dropped
same-day, same-title speeches. BIS's `date` is kept as published, even when wrong (two speeches
dated 2027 whose review codes say 2025-07-10 and 2026-06-03).

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
| `update-<vintage>.parquet` | one delta per update (`new` / `revised` / `removed`; empty when unchanged) |
| `raw-speeches-<sha12>.zip` | every raw zip read, content-addressed |
| `plan-<vintage>.json` | what an execution fetches: mode, URL |

An update never rewrites a file. It sends one HEAD for the bulk file: an unchanged ETag writes an
empty delta without downloading; otherwise the file is diffed against the current state
(speeches dated from the declared `start` on, including after `end`). Each parquet records the
manifest (URL, ETag, Last-Modified, sha256, size, fetch time) of the zip it was read from in its
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
from bis_gingado.cb_speeches.speeches import read_speeches
from bis_gingado.cb_speeches.ner import read_ner, speeches_with_ner

speeches = read_speeches(art)                        # latest version of every speech
speeches = read_speeches(art, as_of="2026-10-01")    # as recorded at that time
table = speeches_with_ner(speeches_art, ner_art)     # left join on speech_id
```

Point-in-time caveat: base rows carry the base load's vintage, not BIS's publication time.
`as_of` is exact about availability only from the first update on.
