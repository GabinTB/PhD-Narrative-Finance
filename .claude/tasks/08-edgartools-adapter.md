# Task 08 — `edgartools` adapter

**Goal**: Implement the `edgartools`-backed `Extractor`, per `reference/extractors.md`.

**Depends on**: 07

**Files**:
- `src/edgar_tools/extractors/edgartools_adapter.py`
- `tests/edgar_tools/test_edgartools_adapter.py`
- `tests/edgar_tools/fixtures/filings/` — add/reuse fixtures per `.claude/tasks/README.md`'s test
  data list (this adapter especially needs the HTML-era fixtures)

**Detail**:
- Method: rule-based over the DOM (`HybridSectionDetector`'s 3-stage cascade — TOC anchors 0.95,
  typography 0.7-0.9, regex 0.6). No ML, no torch/transformers — confirm the dependency set stays
  httpx/pandas/pyarrow/beautifulsoup4/lxml/rich/textdistance/rank_bm25/rapidfuzz/pydantic.
- Consumes **raw bytes, parsed to DOM in-process** — its own `preproc_id` (not inscriptis, not
  edgar-crawler's strip). It is the one extractor for which sharing a text representation would
  destroy its advantage (anchors/typography need markup others strip).
- **Feed it local data via the materialised-view approach**: symlink/write raw blobs into a
  temporary `filings/{YYYYMMDD}/{accession}.nc` tree, point `EDGAR_LOCAL_DATA_DIR` at it, call
  `use_local_storage`, run, discard. Use only the public API
  (`use_local_storage`/`is_using_local_storage`/`resolve_local_filing_path`/`.text()`/`.html()`/
  `.obj()`). **Do not** construct `Filing` objects from bytes via internals — the source is dense
  with churn markers (`GH #880`, `edgartools-9hwf`).
- Store both `confidence` and `detection_method` per row (finer-grained provenance than a bare
  extractor name) — pass the size-band guardrail signal through as part of confidence rather than
  silently accepting it.
- **Pin the `edgartools` version and fold it into the config hash** (task 02's `row_config_hash`)
  — section-detection internals churn hard and size bands regenerate with test fixtures.
- Expect this to be the slowest adapter (DOM + style parsing + cascade, seconds per large filing)
  — do not optimise away correctness for speed here; timing is measured properly in task 18.

**Tests**:
- Against a committed HTML-era fixture, extracts the expected items with `detection_method` in
  `{toc, typography, regex}` and a plausible `confidence`.
- The materialised-view construction is torn down after use (temp dir removed; no crud left in
  `EDGAR_LOCAL_DATA_DIR` between test runs — assert via a tmp_path fixture, not a shared dir).
- A filing with a size-band violation (synthetic/truncated fixture) surfaces as reduced
  confidence, not a silent full-confidence wrong-content extraction.
- Adapter's `preproc_id` never equals `itemseg`/`own-regex`'s inscriptis-based one — representation
  independence is asserted, not just assumed.

**Done when**: four gates green; adapter round-trips on the committed HTML-era fixtures with no
network access in tests.

**Commit**: `edgar_tools: edgartools extractor adapter`
