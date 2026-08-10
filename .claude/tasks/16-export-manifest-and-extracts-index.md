# Task 16 — Export, manifest, extracts index

**Goal**: Implement `Store.export()` / `Store.flat_export()` and the `MANIFEST.json` +
`extracts_index.parquet` machinery, per `reference/output-and-export.md`.

**Depends on**: 13

**Files**:
- `src/edgar_tools/export.py`
- `tests/edgar_tools/test_export.py`

**Detail**:
- Two formats, neither replaces the other:
  - `store.export(confighash, title, label)` → ZIP archive (`MANIFEST.json`, rendered
    `README.md`, `config.json`, the partitioned parquet tree). **Use ZIP `STORE`, not `DEFLATE`**
    — parquet is already compressed.
  - `store.flat_export(confighash, title, label)` → single parquet, manifest as sidecar + parquet
    schema metadata. Loses partition pruning; that's expected and fine for this format's purpose.
  - Both are writers over the same underlying store, not two separate datasets — a test should
    confirm row-for-row equivalence between the two export paths for the same confighash.
- **Never export the derived tier** — it's a cache, regenerable, a pure function of
  `(raw_sha, preproc_id)`. Assert the archive contains no `derived/` content.
- Archive naming: `archive_name()` from task 02 (`{title-slug}_{hash[:8]}.zip`) — the hash is a
  lookup key, not a decoder; the full resolved config lives inside the archive at `config.json`.
- `MANIFEST.json` fields exactly as specified: `schema_version, title, content_hash,
  resolved_config, code_version {this module's git commit + pinned dep versions that affect
  output}, source_state_ref {as_known_at, edgar_state_rows, catalog_schema_version},
  user_label, filing_count, row_count, checksum, export_timestamp`.
  - `code_version` must include this module's own commit SHA (or a bumped version string) plus
    every pinned dependency that affects extracted output (`edgartools`, `inscriptis`, `itemseg`,
    `edgar-crawler-fork` once it exists) — not just a package version number.
  - `source_state_ref.as_known_at` reads from `received_at` (task 03/05) — an extract is not
    reproducible without knowing which snapshot of the raw store it was drawn from.
  - `checksum` is over the data payload (corruption-detectable on transfer), not over the
    manifest itself.
- Render `README.md` from the manifest (human-readable summary) and place it alongside
  `MANIFEST.json` in the archive root.
- Every export appends its manifest header (`title, content_hash, export_timestamp, user_label,
  filing_count, row_count`) to a central `exports/extracts_index.parquet` — this is the
  catalog-vs-blob split one level up (extracts are blobs, their manifest headers form a
  searchable catalog).

**Tests**:
- Exporting a fixture dataset produces a ZIP with `STORE` compression method on the parquet
  entries (assert via `zipfile.ZipInfo.compress_type`), and no `derived/` path inside.
- `flat_export()` output, read back with `pd.read_parquet`, matches `export()`'s unzipped rows
  exactly (same row count, same content).
- `MANIFEST.json` round-trips through write/read with every documented field populated (no field
  silently `null` when a real value is available).
- Two exports of the same confighash both append rows to `extracts_index.parquet`; querying it
  finds both by `content_hash`.

**Done when**: four gates green; a fixture-derived archive is fully self-describing (openable and
interpretable using only its own `config.json` + `MANIFEST.json`, no external lookup needed).

**Commit**: `edgar_tools: export (archive + flat), manifest, extracts index`
