# Task 02 — Config schema, resolution, canonicalisation, hashing

**Goal**: Implement the `get` config as the sole, hashable specification of a dataset, per
`reference/config-and-hashing.md`.

**Depends on**: 01

**Files**:
- `src/edgar_tools/config.py` — schema (dataclasses or pydantic), resolution, canonicalisation, hashing
- `tests/edgar_tools/test_config.py`

**Detail**:
- Accept a config as a path or a dict; canonicalise to a dict internally before anything else.
- Schema fields exactly as in `reference/config-and-hashing.md`: `schema_version`, `universe`
  (`ciks`, `company_names`, `resolve_as_of`), `filings` (`forms`, `start_date`, `end_date`),
  `items_to_extract` (form-keyed, per-form item lists — `null`/`{}` means all items), `tools`,
  `preproc` (per-tool engine + `lib_version`), `extract_options`, `healing`
  (`enabled`, `cache_version`), `validation` (`min_words`, `max_doc_fraction`,
  `reject_if_terminator_present`, `reject_boilerplate_xref`), `runtime`
  (`user_agent`, `workers`, `root`).
- **Resolution before hashing**: `end_date: null` resolves to a concrete date at call time;
  `company_names` resolve to CIKs as of `resolve_as_of` (stub the CIK resolver — real lookup is
  task 04+); CIKs normalised to zero-padded 10-char strings and sorted. The resolved config, not
  the user's input, is what gets hashed and stored.
- **Canonicalisation**: drop `runtime.*` (never hashed — user_agent, workers, root, network
  settings), sort object keys recursively, sort semantically-unordered arrays (CIK/form/item/tool
  lists), serialise with `json.dumps(..., sort_keys=True, separators=(",", ":"),
  ensure_ascii=False)`, SHA-256 the UTF-8 bytes.
- Two hash functions, not one:
  - `config_hash(resolved_config) -> str` — full dataset identity (excludes `runtime`).
  - `row_config_hash(resolved_config, tool) -> str` — the narrower per-row hash used for resume
    compatibility (task 12/13): a row's `tool`, its `preproc` entry, `extract_options`,
    `validation`, `healing.cache_version`, and the extractor code version (accept a
    `code_version: str` param here; real version wiring is task 03+).
- Archive naming helper: `archive_name(title, config_hash) -> str` producing
  `{title-slug}_{hash[:8]}.zip` form, per the example in the reference doc.
- `items_to_extract` must be form-keyed; reject (raise) a flat list — that ambiguity is exactly
  the bug being avoided (Item 7 in a 10-K vs Part I Item 2 in a 10-Q).

**Tests**:
- Same semantic config in different key order / array order hashes identically.
- Changing anything under `runtime` does not change `config_hash`.
- Changing a `preproc.*.lib_version` changes `config_hash`.
- `end_date: null` resolves to a concrete date and that date is what gets hashed (hash is stable
  across two calls made "on the same day", not literally reproducing prod nondeterminism — inject
  a clock/resolver).
- A flat (non-form-keyed) `items_to_extract` raises.
- `row_config_hash` differs when `tool` or `preproc` differs; is identical when only `universe`
  or `filings.start_date`/`end_date` differ (those don't affect a given row's text).

**Done when**: four gates green (see `.claude/tasks/README.md`); config round-trips through
resolve → canonicalise → hash deterministically.

**Commit**: `edgar_tools: config schema, resolution, canonicalisation, hashing`
