# Task 14 — Point-in-time lookups

**Goal**: Implement `get_specific`, `get_latest_as_of`, and `get_covering_period` as three
**separate** functions, per `reference/catalog-and-state.md`. This is the module's core
look-ahead-bias defence — get the naming and separation exactly right.

**Depends on**: 13

**Files**:
- `src/edgar_tools/point_in_time.py`
- `tests/edgar_tools/test_point_in_time.py`

**Detail**:
- `get_specific(accession) -> Filing` — primary-key lookup, returns filing + metadata (CIK,
  company name **as of filing date**, form, filing_date, period_of_report, filer_category,
  is_amendment).
- `get_latest_as_of(cik, date, form="10-K") -> Filing | None` — the latest filing with
  `filing_date <= date`. **Point-in-time.** This is the only function safe for signal
  construction.
- `get_covering_period(cik, date, form="10-K") -> Filing | None` — the filing whose
  `period_of_report` interval contains `date`. **Look-ahead.** Safe only for descriptive work,
  never for backtests.
- **These must never be collapsed into one function with a default.** Rule 9 in
  `.claude/tasks/README.md`. Do not add a convenience `get_doc_at()` that silently picks one — a
  user who assumes the wrong semantics gets look-ahead bias with no error raised, which is exactly
  the failure mode this separation exists to prevent.
- Amendments must compose for free under point-in-time semantics: a `10-K/A` is just another
  filing with its own `filing_date`, so `get_latest_as_of` naturally returns the amendment once
  its `filing_date` has passed, and the original before that — no special-casing needed. Write the
  test from `reference/catalog-and-state.md`'s worked example directly (FY2015 10-K, amended
  2016-08-01; `get_latest_as_of(cik, 2016-09-01)` → amendment; `get_latest_as_of(cik, 2016-06-01)`
  → original).
- **Validity intervals are derived, not stored** — compute the `(valid_from, valid_to)` view from
  `filing_date` ordering within `(cik, form)` at query time. Do not persist it; it goes stale the
  moment a new filing or amendment arrives.

**Tests**:
- The exact amendment-composition scenario from the reference doc (dates and CIK as in the doc's
  worked example) — both query dates return the correct filing.
- At `t = 2016-01-15` with only FY2014 and FY2015 10-Ks on file (FY2015 not yet filed),
  `get_latest_as_of` returns FY2014 while `get_covering_period` would return the not-yet-existing
  FY2016 — assert these diverge exactly as documented, on a fixture built to match.
- `get_covering_period` for a `date` with no filing at all returns `None`, not an exception and
  not the nearest-by-distance filing.
- No function accepts an ambiguous/defaulted "mode" parameter — a signature-level test (introspect
  the function signature) that fails if a `get_doc_at`-style merged function ever gets added.

**Done when**: four gates green; the three functions are demonstrably distinct in behaviour on
the amendment fixture, not just distinct in name.

**Commit**: `edgar_tools: point-in-time lookups (get_specific, get_latest_as_of, get_covering_period)`
