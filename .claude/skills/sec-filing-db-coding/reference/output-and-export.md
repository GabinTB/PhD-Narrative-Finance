# Output schema, export, and the versioning manifest

## Section schema: long, not wide

One row per `(accession, item, tool, repr_id)`:

```
cik                 int64
accession           string        # the unique key, not (cik, date)
form                string
filing_date         date32
period_of_report    date32        # drift pairs align on fiscal period, not filing date
filer_category      string
company_name        string        # as of filing_date
is_amendment        bool

item                string
tool                string
tool_version        string
repr_id             string
preproc_id          string

text                string
start_offset        int64         # scoped to repr_id
end_offset          int64
n_words             int32

detection_method    string        # edgartools stage; null elsewhere
confidence          float32       # edgartools; null elsewhere
validation_status   string
passed_validation   bool

row_config_hash     string        # extraction-relevant config subset, for resume compatibility
```

### Why long

A wide schema (`cik, date, item_1, item_1a, ...`) has nowhere to put extractor provenance. Adding
it forces parallel columns (`item_1_extractor`, `item_1_valid`, ...) which is unmanageable at 20
items, and the table is sparse because 10-K and 10-Q have different item sets.

Long gives one row per (filing, item, tool), provenance for free, and no sparsity.

Provide `to_wide(df, form="10-K", tool=...)` as an analysis-time helper. Wide is a view, never
the storage format.

### Why `accession` is the key

`(cik, date)` is not unique. A firm can file a 10-K and a 10-Q on the same date, and can file an
original and an amendment. Minimum key columns are `cik, accession, form, filing_date,
period_of_report`.

`period_of_report` must be stored separately from `filing_date` because drift pairs align on
fiscal period, not on when the document reached EDGAR. Filing dates move around by weeks.

## Partitioning

```
extracted/{confighash}/
  config.json          # resolved config, self-describing
  MANIFEST.json
  form=10-K/year=2020/part-*.parquet
  form=10-Q/year=2020/part-*.parquet
```

Partitioned on `form` and `year` because that is how drift reads it: a firm's series within a
form type. DuckDB globs it directly with no load step and no server:

```sql
SELECT * FROM 'extracted/{hash}/**/*.parquet'
WHERE cik = 320193 AND item = '1A'
```

### Per-config vs global store

Currently **per-config**: each `get` with different items, tools, or dates is a separate
self-describing dataset. Clean provenance, trivial export ("ship this hash"), at the cost of
duplication when configs overlap.

The alternative is a global section store keyed on `(accession, item, tool, preproc_id)` where a
config is a query plus a manifest recording selected rows. Better amortisation, more work to
export.

Per-config is right for a build-and-export workflow. The row schema is identical either way, so
migration later is a migration, not a rewrite. Flagged as open decision 4 in `SKILL.md`.

## Two export formats

Different consumers, so provide both. Neither replaces the other.

```python
store.export(confighash, title=..., label=...)       # ZIP archive
store.flat_export(confighash, title=..., label=...)  # single parquet
```

### Archive (primary)

```
{title}_{confighash8}.zip
  MANIFEST.json
  README.md               # rendered from the manifest
  config.json             # resolved config
  form=10-K/year=2020/part-*.parquet
  ...
```

Self-describing, reproducible, citeable. This is what attaches to a paper, ships to LGT, and goes
to HuggingFace or Kaggle.

**Use ZIP `STORE`, not `DEFLATE`.** Parquet is already compressed; deflating it again costs CPU
and buys nothing.

### Flat (convenience)

One parquet file, all rows, for someone who wants `pd.read_parquet` and nothing else. Loses
partition pruning. Carries the same manifest as a sidecar and in the parquet schema metadata.

Both are writers over the same underlying store, not two datasets.

## The manifest

The general rule this implements, which applies beyond EDGAR (RavenPack computation, and other
sources):

> Raw and lightly-derived data live on the server, append-only and content-addressed. Every
> research project is based on a labelled, immutable extract of that store. No research project
> reads the raw store directly.

`MANIFEST.json` in the archive root, plus a rendered `README.md` alongside it.

```json
{
  "schema_version": "1",
  "title": "MSCI World 10-K/10-Q Item 1A+7, 2004-2025, drift v2",
  "content_hash": "a3f9c1e2...",
  "resolved_config": { ... },
  "code_version": {
    "edgar_tools": "git:9f2c1ab",
    "edgartools": "5.43.0",
    "inscriptis": "2.5.0",
    "itemseg": "3.4.0",
    "edgar-crawler-fork": "git:44de01f"
  },
  "source_state_ref": {
    "as_known_at": "2026-07-15T02:00:00Z",
    "edgar_state_rows": 24817302,
    "catalog_schema_version": "1"
  },
  "user_label": "quality sleeve early-warning overlay",
  "filing_count": 41883,
  "row_count": 209415,
  "checksum": "sha256:...",
  "export_timestamp": "2026-07-29T14:22:07Z"
}
```

### Field notes

- **`title`** — short human summary. Once fifty of these exist on a datalake, searching titles is
  how the right one gets found without opening each manifest.
- **`content_hash`** — the reproducibility key. Hash of the canonicalised resolved config.
- **`resolved_config`** — the full parameter set passed to `get`, with dates concrete and the CIK
  list expanded. Not the user's raw input. This subsumes what would otherwise be a separate
  "selection" field: the config *is* the selection.
- **`code_version`** — git commit of this module plus pinned versions of every dependency that
  affects output. Bumping inscriptis changes the text; bumping edgartools changes section
  detection.
- **`source_state_ref`** — which snapshot of the raw store the extract was drawn from. An extract
  is not reproducible without it, because EDGAR grows daily. Reads from `received_at`.
- **`checksum`** — over the data payload, so corruption is detectable on transfer.
- **`export_timestamp`** — when the extract was produced.
- **`schema_version`** — of the manifest format itself, so future readers can migrate.

## Central index

Write every manifest's header (title, content_hash, export_timestamp, user_label, filing_count,
row_count) into `exports/extracts_index.parquet` on every export.

This is the catalog-vs-blob split one level up: extracts are blobs, their manifests form a
searchable catalog. Querying across all research extracts then costs one parquet read instead of
walking every ZIP.

## What makes the whole thing sound

The raw store is append-only and the manifest pins a state snapshot, so any extract is
reproducible forever even as the store grows. Mutating the raw store in place would break
reproducibility silently. Content addressing plus append-only is what protects it.
