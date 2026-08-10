# Task 10 — `itemseg` adapters (BERT and CRF)

**Goal**: Implement `itemseg-bert` and `itemseg-crf` as two **separate** registered extractors,
per `reference/extractors.md`.

**Depends on**: 07

**Files**:
- `src/edgar_tools/extractors/itemseg_bert_adapter.py`
- `src/edgar_tools/extractors/itemseg_crf_adapter.py`
- `tests/edgar_tools/test_itemseg_adapters.py`

**Detail**:
- Install `itemseg` (3.4.0, requires-python >=3.8); add it to the `edgar` dependency group in
  root `pyproject.toml` (populate the placeholder group added in task 01), not to core
  `dependencies`.
- Both consume **inscriptis** text (matches training distribution) — same `preproc_id` as
  `own-regex`, which is intentional (shared representation makes their outputs comparable).
- Line-level BIO tagging (`B{item}`/`I{item}`/`O`); reconstruct `ExtractedItem` spans (character
  offsets into the inscriptis representation) from the line predictions.
- **Do not silently drop Item 1C.** The SEC added Item 1C (cybersecurity) for fiscal years ending
  on or after 2023-12-15, inside the research window. Trained models were never trained on it and
  fail *silently*, not with an error. Either exclude Item 1C explicitly when routing through these
  adapters (raise/flag rather than mis-tag) or route it to healing (task 15) by default — pick one
  and make it visible in the output row (e.g. a `known_limitation` marker), not a quiet gap.
- Implement both as genuinely separate `Extractor`s (not one class with a mode flag) — they are
  distinct arms in the benchmark (task 18) with different operational profiles: BERT needs GPU
  scheduling considerations, CRF is CPU-only and was statistically tied with BERT on non-core
  items in the published eval, which matters operationally if it also ties here.
- Item boundaries coincide with the start of a new line — this is the structural assumption both
  models rest on; do not attempt to "improve" it with sentence-level heuristics.

**Tests**:
- On a small fixture with known item boundaries, both adapters return spans with the correct
  `item` labels, `repr_id` set to the inscriptis preproc id, and `extractor` set to
  `"itemseg-bert"` / `"itemseg-crf"` respectively (never a shared/ambiguous name).
- A fixture containing an Item 1C section is either explicitly excluded or explicitly flagged —
  never silently mis-tagged as another item with no signal.
- The CRF adapter runs with no GPU/torch dependency required (assert the process doesn't attempt
  a CUDA import) — this is the "CPU-only tie" property that matters for the benchmark's
  operational conclusion.

**Done when**: four gates green; both extractors registered independently and individually
selectable via config.

**Commit**: `edgar_tools: itemseg-bert and itemseg-crf adapters`
