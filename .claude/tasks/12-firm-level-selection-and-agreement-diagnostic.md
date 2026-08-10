# Task 12 — Firm-level selection and agreement diagnostic

**Goal**: Implement firm-level extractor selection and the cross-extractor agreement diagnostic,
per `reference/validation-and-selection.md`. This is the rule that protects the drift measure —
implement it exactly, not at the filing level.

**Depends on**: 07 (validation), 08–11 (at least one real adapter plus the stub)

**Files**:
- `src/edgar_tools/selection.py`
- `src/edgar_tools/agreement.py`
- `tests/edgar_tools/test_selection.py`
- `tests/edgar_tools/test_agreement.py`

**Detail**:
- Selection algorithm, verbatim from the reference doc:
  ```
  for each (cik, item, form):
      candidates = tools where validation == VALID for EVERY filing in that firm's series
      if candidates non-empty:
          chosen = highest priority in candidates      # priority from config
      else:
          chosen = tool with the longest unbroken valid run
          usable_range = that run only
      emit selection row: (cik, form, item, chosen, coverage, run_start, run_end)
  ```
- **Never build a drift pair across an extractor switch** — this is a hard downstream rule to
  encode here as a flag/marker on the selection row (e.g. `usable_range`), even though the actual
  pairing logic belongs to a research script, not this module.
- Priority strictly from `config.tools` order or an explicit `priority` list — never hardcoded
  (rule 6). The benchmark (task 18) hasn't run yet at this point in the build; do not bake in an
  assumed ordering even as a "temporary" default.
- **Format-regime split**: priority may legitimately differ pre-2001 (plain text, no anchors/
  styles) vs. HTML era (roughly 2001+). Support a regime-keyed priority in config, applied as a
  common per-calendar-date shock — not resolved per firm.
- Agreement diagnostic: wherever ≥2 tools pass validation on the same `(accession, item)`, compute
  token-level agreement on the **normalised extracted text** (casefold, strip non-alphanumeric,
  collapse whitespace, then token-set IoU) — never by offset (offsets are `repr_id`-scoped and not
  comparable across representations, rule 3).
- Sample-selection consequence to surface, not just compute: extraction failure is expected to be
  correlated **within firm** (an unusual filing agent is used every year by that firm). Expose a
  helper that reports within-firm failure correlation and whether dropped firms differ
  systematically on size/industry/filer-status — this feeds the eventual paper's limitations
  section, not a hidden implementation detail.

**Tests**:
- A synthetic firm-series where one tool is VALID every year is selected outright, ignoring a
  higher-aggregate-but-inconsistent competitor.
- A firm-series with no tool VALID for every year selects the tool with the longest **unbroken**
  run, and `usable_range` reflects only that run (not the full series).
- Priority order is read from a config fixture, never from a module constant — a test that flips
  the config's tool order flips the selection outcome on a tied-candidates fixture.
- Agreement IoU is computed on normalised text and is invariant to whitespace/case differences
  that don't change content; a fixture with genuinely different content between two extractors
  reports low IoU.
- Regime split: a pre-2001 fixture and a post-2001 fixture with the same CIK can select different
  tools without violating the within-regime firm-level consistency rule.

**Done when**: four gates green; selection output is a clean `(cik, form, item, chosen, coverage,
run_start, run_end)` table, independently verifiable against the pseudocode above.

**Commit**: `edgar_tools: firm-level extractor selection and agreement diagnostic`
