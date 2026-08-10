# Task 07 — `Extractor` protocol, registry, validation layer

**Goal**: Define the shared `Extractor` protocol, `ExtractedItem`, the validation rules and
`ValidationResult`, per `reference/extractors.md` and `reference/validation-and-selection.md`.
No concrete adapter yet — that starts at task 08.

**Depends on**: 06

**Files**:
- `src/edgar_tools/extractors/base.py` — `Extractor` protocol, `ExtractedItem` dataclass
- `src/edgar_tools/extractors/registry.py` — name → `Extractor` lookup, reads priority from
  config only (never hardcoded — rule 6)
- `src/edgar_tools/validation.py` — `ValidationStatus`, `ValidationResult`, rule evaluation
- `tests/edgar_tools/test_extractors_base.py`
- `tests/edgar_tools/test_validation.py`

**Detail**:
- `Extractor` protocol and `ExtractedItem` exactly as specified in `reference/extractors.md`:
  `extract(source, form, items) -> dict[str, ExtractedItem]`, raising on failure for a single
  document, no printing, no corpus looping, no file writes (batch orchestration is the caller's
  job — that's task 12/13). Fields: `item, repr_id, start_offset, end_offset, text, n_words,
  extractor, extractor_version, detection_method, confidence, validation`.
- `repr_id` is load-bearing: offsets are scoped to it (rule 3). Do not let any code compare
  offsets across representations.
- Validation rules applied **in order, first failure wins** (`reference/validation-and-selection.md`):
  `ABSENT_BY_RULE` → `NOT_FOUND` → `TOO_SHORT` (`min_words`, default 50) → `SPAN_OVERSIZED`
  (`max_doc_fraction`, default 0.6) → `TERMINATOR_IN_TEXT` → `BOILERPLATE_XREF` → `TOO_LONG`.
  `passed_validation = (status == VALID)`.
- `ABSENT_BY_RULE` is checked **first**, before anything else, and is not a failure: smaller
  reporting companies are not required to file Item 1A/7A. Requires `filer_category`
  (`dei:EntityFilerCategory`) on the filing row (already present in the `filings` schema from
  task 03) — wire the lookup here.
- Combined headers ("Items 1 and 2. Business and Properties") and incorporation-by-reference are
  legitimate, not failures — attribute combined spans to both items (flagged as such);
  incorporation-by-reference is `BOILERPLATE_XREF`, distinct from `TOO_SHORT`.
- Apply this module's validation rules independently of any extractor's own signals.
  `edgartools`'s `confidence`/size-band flag (task 08) is stored, not delegated to — every
  extractor is gated identically by this layer.
- Registry: `get_extractor(name: str) -> Extractor`, `run_all(config.tools, ...)` — **all
  registered extractors run on every filing, no early exit** (rule 1). This task only needs the
  registry to hold a couple of trivial fake extractors for its own tests; real adapters land in
  tasks 08–11.

**Tests**:
- Each `ValidationStatus` branch is independently triggerable and the "first failure wins"
  ordering is verified with a case that would match two rules (e.g. too-short *and* boilerplate
  xref — must resolve to whichever comes first in the documented order).
- `ABSENT_BY_RULE` pre-empts `NOT_FOUND` for a smaller reporting company missing Item 1A.
- A combined "Items 1 and 2" span is attributable to both items without double-triggering a
  validation failure.
- Registry raises a clear error for an unregistered tool name; priority order comes from a passed
  list, never a module-level constant (grep-test: no hardcoded ordering in source).

**Done when**: four gates green; validation truth table matches
`reference/validation-and-selection.md` rule-by-rule.

**Commit**: `edgar_tools: Extractor protocol, registry, validation layer`
