# Task 18 — Benchmark harness

**Goal**: Determine the extractor priority order empirically, per `reference/benchmark.md`. This
settles an open decision (SKILL.md #1) that every prior task has deliberately left unresolved —
do not let this task quietly assume an answer either; it produces the answer.

**Depends on**: 08–11 (all extractor adapters, including the edgar-crawler stub/fork status at
whatever point it's reached)

**Files**:
- `src/edgar_tools/benchmark/harness.py` — text-comparison scoring (not label comparison)
- `src/edgar_tools/benchmark/run.py` — CLI/script entry point
- `tests/edgar_tools/test_benchmark_harness.py`
- `dev/benchmark/` (repo root, outside `src/`) — the benchmark report output goes here, not inside
  the package

**Detail**:
- Gold data: `itemseg` dataset (3,737 manually annotated 10-Ks, FY2001-2019,
  `im.ntu.edu.tw/~lu/data/itemseg/itemseg10kdata.7z`, 373-filing held-out test split). This is an
  external download — do not commit it; document the fetch step, gate the benchmark script behind
  its presence, and keep it out of the default test suite (tests use small synthetic/fixture
  stand-ins for the harness logic itself).
- **The blocking problem, and the reason this task exists as designed**: gold labels are
  line-level BIO tags over inscriptis output specifically. `edgar-crawler`'s line breaks come from
  a different HTML-to-text path; `edgartools` doesn't produce lines at all. **Therefore: do not
  compare labels — compare extracted text.** Build this normalisation+comparison harness first;
  everything else in the benchmark is downstream of it:
  1. Reconstruct gold item text from BIO labels.
  2. Normalise both gold and candidate: casefold, strip non-alphanumeric, collapse whitespace.
  3. Score with token-level IoU.
- Seven arms, not four — the own-regex ablation flags are separate arms:
  `edgar-crawler` upstream, `edgar-crawler + title_keyword`, `edgar-crawler + comma_lookbehind`,
  `edgar-crawler + both`, `edgartools`, `itemseg-bert`, `itemseg-crf`.
- Metrics: **report** line-level macro-F1 (comparable to the published table); **decide on**
  per-filing IoU above a threshold, stratified; **additionally compute** the fraction of firm-year
  **pairs** where both years segment correctly — this is what actually determines panel size and
  nobody in the published literature reports it.
- **Stratify** — aggregate alone will show "no difference" for the own-regex additions (they fix a
  minority-subpopulation failure mode: cross-reference contamination). Slice by format regime
  (pre-2001/HTML/post-2019 inline-XBRL), filing agent, decade, TOC-present-or-absent, filer
  category. Bootstrap CIs on all pairwise differences.
- Time every arm on a stratified ~200-filing sample spanning size/era range — wall-clock is a
  selection criterion (it multiplies by tens of thousands of filings × however many rebuilds), not
  a footnote.
- Output written to `dev/benchmark/`: aggregate table, per-stratum tables, pair-correctness metric
  per extractor, bootstrap CIs, timing distribution, agreement matrix between extractors on
  multiply-valid filings.
- **The resulting priority order goes into config** (feeds directly into task 12's
  `config.tools`/`priority`), never into code.

**Tests**:
- The text-comparison harness (steps 1–3 above) is unit-tested independently of any real gold
  data, using small synthetic BIO-labelled fixtures with known correct IoU.
- Stratification buckets are computed correctly on a synthetic filing-metadata fixture (format
  regime boundary dates, decade bucketing).
- Bootstrap CI computation is tested against a case with a known analytic answer (e.g. identical
  arms → CI straddling zero).
- The benchmark script gracefully skips (not errors) when the gold dataset isn't present locally,
  with a clear message pointing at the download step.

**Done when**: four gates green; running the harness against the real gold set (manual, outside
CI) produces `dev/benchmark/`'s full report and a config-ready priority ordering.

**Commit**: `edgar_tools: extraction benchmark harness`
