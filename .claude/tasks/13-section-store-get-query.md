# Task 13 — Section store, `get`, `query`

**Goal**: Wire up the permanent, partitioned-parquet sections tier and the public `Store.get()` /
`Store.query()` methods, per `reference/output-and-export.md` and `SKILL.md`'s public API.

**Depends on**: 06 (derived cache), 07–12 (extractors, validation, selection)

**Files**:
- `src/edgar_tools/section_store.py` — parquet writer/reader over `extracted/{confighash}/`
- `src/edgar_tools/store.py` — extend `Store` with `get()` and `query()`
- `tests/edgar_tools/test_section_store.py`
- `tests/edgar_tools/test_store_get_query.py`

**Detail**:
- **Schema is long, one row per `(accession, item, tool, repr_id)`** — exactly the columns listed
  in `reference/output-and-export.md` (`cik, accession, form, filing_date, period_of_report,
  filer_category, company_name, is_amendment, item, tool, tool_version, repr_id, preproc_id, text,
  start_offset, end_offset, n_words, detection_method, confidence, validation_status,
  passed_validation, row_config_hash`). Do not build a wide table as the storage format; provide
  `to_wide(df, form=..., tool=...)` as an analysis-time helper only.
- `accession` is the key, not `(cik, date)` — a firm can file same-day, and can file an original
  plus an amendment.
- Partition on `form` and `year` (`extracted/{confighash}/form=10-K/year=2020/part-*.parquet`) so
  DuckDB globs it with no load step: `SELECT * FROM 'extracted/{hash}/**/*.parquet' WHERE ...`.
- Currently **per-config** storage (open decision 4 — do not migrate to a global store; just don't
  paint yourself into a corner: keep the row schema identical either way).
- `Store.get(config, allow_download=False)`: **ensure raw → ensure derived → extract → persist →
  return**, per the public API in `SKILL.md`. With `allow_download=False` it must be strictly
  read-only over what already exists — no network calls, not even a check-if-newer call.
- Resume compatibility: a previously-extracted row is reusable only if its stored
  `row_config_hash` (task 02) matches the current run's extraction-relevant config subset — reuse
  only on exact match, otherwise re-extract. Do not reuse a `remove_tables: false` row into a run
  configured `remove_tables: true`.
- `Store.query(sql: str) -> DataFrame` is read-only SQL over the section store (DuckDB glob under
  the hood); it must not accept or execute anything beyond `SELECT`-shaped queries against the
  section table (guard against accidental writes through a raw SQL string).

**Tests**:
- `get()` on a fixture config produces the documented long-format schema, correctly partitioned on
  disk by `form`/`year`.
- Calling `get()` twice with an unchanged config and unchanged inputs is a no-op on the second call
  (nothing re-extracted) — assert via a spy on the extractor registry.
- Changing one `preproc.*.lib_version` and re-calling `get()` re-extracts only the rows whose
  `row_config_hash` changed, not the whole dataset.
- `get(config, allow_download=False)` against a config requiring an undownloaded filing returns
  partial results (or raises clearly) without making any network call — assert via a mock that
  fails the test if HTTP is touched.
- `query()` rejects a non-`SELECT` string.

**Done when**: four gates green; `get()` on the committed fixture filings produces a queryable
parquet dataset matching the documented schema exactly.

**Commit**: `edgar_tools: section store, get(), query()`
