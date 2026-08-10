# Task 17 — State export and `rebuild_db_from_state`

**Goal**: Implement `Store.export_state()` / `Store.rebuild_db_from_state()`, per
`reference/catalog-and-state.md`.

**Depends on**: 03, 04

**Files**:
- `src/edgar_tools/state_export.py`
- `tests/edgar_tools/test_state_export.py`

**Detail**:
- `state = store.export_state()` — catalog only: **no blobs, no parquet**. Exported content is
  exactly: the `filings` catalog, the `edgar_state` snapshot, the `companies` resolver, and
  `schema_version`. Reject any implementation that pulls in raw/derived/section content here —
  that's what `export()` (task 16) is for.
- `Store.rebuild_db_from_state(state, path)` — rehydrates the catalog only. A subsequent
  `get(config)` re-downloads and re-derives from EDGAR to reconstruct data.
  **Reproducibility is by re-execution, not by data transfer.** This is the intended default for
  open-sourcing: publish code + config + state, and anyone reconstructs the dataset themselves.
  Document this loudly in the module docstring — it's a design choice a future contributor could
  easily "fix" by accidentally shipping blobs in `export_state()`, which must not happen.
- The assumption is EDGAR still serves those accessions identically — holds given append-only +
  immutable accessions, except the rare redaction case (which `edgar_state`'s disappearance
  handling, task 04, already surfaces). Shipping actual data (HuggingFace/Kaggle) is a
  belt-and-braces fallback, never the primary path — do not build this task around it.
- `schema_version` travels with the exported state so a future schema change doesn't make an old
  export's rebuild silent guesswork.

**Tests**:
- `export_state()` on a populated fixture catalog produces a payload with zero bytes belonging to
  raw/derived/section content (assert by structural inspection of the exported object, not just
  by size).
- `rebuild_db_from_state()` on that payload reconstructs a catalog whose `filings`, `edgar_state`,
  and `companies` tables are row-for-row identical to the source (order-independent comparison).
- Rebuilding from a state export with a mismatched `schema_version` raises clearly rather than
  silently producing a catalog with unexpected columns.
- Round-trip idempotence: `rebuild_db_from_state(export_state())` applied twice produces the same
  catalog as once.

**Done when**: four gates green; a rebuilt catalog from a state-only export is provably free of
any document content, and a subsequent `get()` against it triggers real re-download/re-extraction
rather than silently serving stale cached data that shouldn't exist.

**Commit**: `edgar_tools: state export and rebuild_db_from_state`
