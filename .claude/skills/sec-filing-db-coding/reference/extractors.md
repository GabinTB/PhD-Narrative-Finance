# Extractors

## The interface

Every extractor implements one protocol. Adding a backend must not change anything else in the
module.

```python
from typing import Protocol
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class ExtractedItem:
    item: str
    repr_id: str                  # which representation the offsets index into
    start_offset: int
    end_offset: int
    text: str
    n_words: int
    extractor: str
    extractor_version: str
    detection_method: str | None  # edgartools stage; None elsewhere
    confidence: float | None      # edgartools; None elsewhere
    validation: "ValidationResult"

class Extractor(Protocol):
    name: str
    version: str
    preproc_id: str               # which representation this extractor consumes

    def extract(
        self,
        source: Path,             # raw blob or derived text, per preproc_id
        form: str,
        items: list[str],
    ) -> dict[str, ExtractedItem]: ...
```

`extract` raises on failure for a single document. It does not print, does not loop over a
corpus, and does not write files. Batch orchestration belongs to the caller.

## Per-framework preprocessing

This is a correctness requirement, not a convenience. BERT4ItemSeg was trained on inscriptis
output; feeding it edgar-crawler's stripped text puts it off-distribution, and a trained model
degrades quietly rather than failing.

| Consumer | Representation | Rationale |
|---|---|---|
| `edgartools` | raw bytes, parsed to DOM in-process | anchors, styles and agent detection all need markup |
| `itemseg` | inscriptis text | matches training distribution |
| `edgar-crawler` | its own strip plus table removal | matches what its patterns were tuned on |
| `own-regex` | inscriptis text | shares a representation with itemseg for comparability |
| LLM healing | line-numbered inscriptis text | offset prompting needs stable line IDs |

### Offsets are representation-scoped

Offset 41,203 means a different place in inscriptis text than in edgar-crawler text. Every
`ExtractedItem` carries `repr_id`, the healing cache is keyed on `(accession, item, repr_id)`,
and cross-representation comparison goes through normalised token sets on the extracted text,
never through offsets.

## edgartools

**Method: rule-based, no ML.** Dependencies are httpx, pandas, pyarrow, beautifulsoup4, lxml,
rich, textdistance, rank_bm25, rapidfuzz, pydantic. No torch, no transformers, no shipped
weights. The `edgar/ai` package is MCP server plumbing and token counting, not a model.

`HybridSectionDetector` runs its own internal three-stage cascade per filing:

| Stage | Method | Confidence |
|---|---|---|
| 1 | TOC-based, resolved through HTML anchors | 0.95 |
| 2 | Heading detection from typography and style | 0.7-0.9 |
| 3 | Regex pattern matching | 0.6 |

Stage 1 reads the table of contents, extracts `href` targets, verifies each anchor target exists
in the tree, and bounds sections between consecutive anchor positions. Filing agent is detected
from the first ~3000 characters of HTML and passed to the TOC parser, because TOC markup is
agent-specific.

It already ships two things this module would otherwise build:

- **Cross-validation between stages.** Agreement between methods raises confidence; disagreement
  lowers it. Store this as `confidence`.
- **Size guardrail.** Sections falling outside a curated per-item size band (roughly median/5 to
  median x8) get reduced confidence rather than silent acceptance. Their own documentation names
  the failure class: a 0.95-confidence section containing the wrong item, or raw HTML, or a few
  hundred characters because an anchor landed on a PART header.

Store both `confidence` and `detection_method` per row; they are finer-grained provenance than a
bare extractor name.

### Feeding it local data

`edgartools` has a full offline mode. `use_local_storage(path)` flips `is_using_local_storage()`,
and `text()` / `html()` / `.obj()` then resolve through `resolve_local_filing_path()` before
touching the network.

The expected layout is the SEC **bulk feed bundle**:

```
{EDGAR_LOCAL_DATA_DIR}/filings/{YYYYMMDD}/{accession}.nc[.gz]
```

bucketed by dissemination date, one accession per file. This is not edgar-crawler's flat
`RAW_FILINGS/` folder of stripped primary documents, and it cannot be.

**Use the materialised-view approach.** Before an `edgartools` run, symlink or write the needed
raw blobs into a temporary `filings/{YYYYMMDD}/{accession}.nc` tree, point
`EDGAR_LOCAL_DATA_DIR` at it, run, discard. This uses the public API and survives version bumps.

Do not construct `Filing` objects from bytes by reaching into internals. The source is dense
with issue-reference comments (`GH #880`, `edgartools-9hwf`) indicating heavy churn in exactly
those paths.

### Cautions

- **Pin the version and fold it into the config hash.** Section detection internals churn hard,
  and the size bands are regenerated when test fixtures rotate. A minor version bump can change
  the corpus.
- **Size bands are calibrated on large caps.** Their own caveat is that Item 8 enforcement
  assumes every filer inlines financial statements; a filer incorporating Item 8 by reference is
  legitimately small. Flag-only, so no data corruption, but expect noise if the universe widens
  beyond large caps.
- **It is the slowest.** DOM build plus style parsing plus the detection cascade is seconds per
  large filing. Inherent to what makes it accurate.

## edgar-crawler

**Method: line-anchored regex with monotonic ordering.** Patterns are generated per item from
the index alone, with no title text:

```
Item 1    \n[^\S\r\n]*ITEMS?\s*(?:I|1)[.*~\-:\s\(]
Item 1A   \n[^\S\r\n]*ITEMS?\s*1[^\S\r\n]*A[.*~\-:\s\(]
Item 7    \n[^\S\r\n]*ITEMS?\s*(?:VII|7)[.*~\-:\s\(]
Item 7A   \n[^\S\r\n]*ITEMS?\s*7[^\S\r\n]*A[.*~\-:\s\(]
SIGNATURE \n[^\S\r\n]*SIGNATURE(s|\(s\))?[.*~\-:\s\(]
```

Span regex is `{item}[.*~\-:\s\(].+?({next_item}[.*~\-:\s\(])` under `IGNORECASE | DOTALL`, with
`next_item` walked from `items_list[i+1:]` until one matches.

Three properties worth preserving in the fork:

1. **Line anchoring** (`\n[^\S\r\n]*`). Prevents mid-sentence matches on cross-references. This
   is why its preprocessing converts vertical-margin `<span>` elements to newlines rather than
   spaces: the newline is load-bearing, not cosmetic.
2. **Monotonic ordering.** A `positions` list threaded through the document rejects any candidate
   starting before the end of the previously extracted item. Items in a 10-K are strictly
   ordered; discarding that constraint is free accuracy lost. This is also most of what the
   itemseg CRF learns.
3. **Terminator generality.** Falls through successive candidate terminators rather than
   requiring one specific next heading.

### Fork requirements

Not on PyPI, no `pyproject.toml`. The fork is handled in a separate repo and pipeline. What the
fork must expose:

- A **per-document callable** that raises on failure, not a batch script that loops and prints
- Ablation flags for the own-regex additions (see below), so arms are switchable at runtime
- A stable `version` string that changes on any extraction-affecting change

Build the adapter against the `Extractor` protocol with a stub until the fork lands. The stub
raises `NotImplementedError` and the test suite skips accordingly.

### Verify on fork

Confirm whether it downloads the complete submission `.txt` or only the primary document. That
determines whether exhibit stripping is this module's problem.

## itemseg (BERT4ItemSeg and CRF)

**Method: line-level sequence labelling.** Modified BIO tags over lines: the first line of Item 1
is B1, subsequent lines I1, lines belonging to no item are O. Sentence-BERT encodes each line, a
Bi-LSTM runs over the line embeddings, which sidesteps BERT's 512-token window and handles
filings of arbitrary length. Rests on the observation that item boundaries coincide with the
start of a new line, in both HTML-derived and pure-text filings.

Reference: Lu, Chien, Yen and Chen (2025), *Utilizing Pre-trained Language Models and Large
Language Models for 10-K Items Segmentation*, arXiv:2502.08875.

**Published results** (3,737 manually annotated 10-Ks, FY2001-2019):

| Method | Core items (1, 1A, 3, 7) macro-F1 | Other items |
|---|---|---|
| BERT4ItemSeg | 0.9826 | 0.9692 |
| CRF, hand-crafted features | 0.9788 | 0.9691 |
| GPT-4o, line-ID prompting | 0.9552 | 0.9385 |
| Rule-based baseline | 0.9048 | 0.9010 |

The CRF row matters operationally. Its features are unigrams and bigrams, first-character case,
percentage of uppercase characters, word length, and forward and backward position normalised to
[0,1]. Its difference from BERT4ItemSeg on non-core items is not statistically significant. Almost
all the gain over regex is **structural** (ordering plus position), not semantic. It is also
CPU-only, so if it ties on this data the full panel runs without GPU scheduling.

Install: `pip install itemseg` (3.4.0, requires-python >=3.8). Source at
`github.com/hsinmin/itemseg`. Annotated dataset at
`im.ntu.edu.tw/~lu/data/itemseg/itemseg10kdata.7z`.

**Implement both `itemseg-bert` and `itemseg-crf` as separate registered extractors.** They are
different arms in the benchmark and have different operational profiles.

### Known limitation

Trained models cannot recognise items they were never trained on. The SEC added Item 1A in 2005
and Item 1C (cybersecurity) for fiscal years ending on or after 15 December 2023. Item 1C is
inside the research window, and the trained model will fail on it silently. Only prompt-based
approaches adapt without retraining. Handle Item 1C via the healing layer or exclude it
explicitly.

## own-regex

**Method: title-bearing patterns plus a comma-list negative lookbehind.**

Two independent precision filters, and they must be **separate ablation flags**, not one bundle:

1. `title_keyword` — start patterns require the item title immediately after the number, e.g.
   Item 7 must be followed by `management's discussion|md&a`. Prevents matching bare
   cross-references that edgar-crawler relies on line anchoring to reject.
2. `comma_lookbehind` — `(?<!,\s)` on start patterns, suppressing matches inside comma-separated
   cross-reference lists ("as set forth in Items 1, 1A, and 7").

If both ship as one arm and the number moves, the source is unattributable; if they move in
opposite directions, nothing is visible.

### Fixes required before use

Carried over from the prototype notebook. These are bugs, not preferences:

- **`ITEM1A_END` second alternative terminates on Item 1**, the *preceding* item. It can only
  fire on a later stray reference and truncates or corrupts the span. Remove it; let the Item 2
  branch carry the terminator, or extend to Item 3.
- **No line anchoring.** Add `\n[^\S\r\n]*` to start patterns. This only works once the
  HTML-to-text step preserves block boundaries, which inscriptis does and naive tag stripping
  does not.
- **No monotonic ordering.** Items are extracted independently, discarding the strict ordering
  constraint. Thread a position list as edgar-crawler does.
- **Terminators too narrow.** `ITEM7A_END` accepts only `item 8 ... Finan`; a deviant Item 8
  heading returns `None` for 7A even when the start matched cleanly. Fall through to 9, 9A, 9B.
- **Table removal must happen on the DOM, before stripping.** Stripping tags and then looking
  for delimiter-heavy lines finds nothing on post-2001 HTML. Superseded by using inscriptis,
  which handles this.

Keep the longest-span-wins pairing from the prototype. It is a genuine TOC defence: a TOC start
pairs with an adjacent TOC end and produces a short span that loses.

## Execution model

All registered extractors in `config.tools` run on **every** filing. No early exit.

Rationale: a filing where the first extractor succeeds never gets seen by the others, so the
agreement signal is lost exactly where it would be informative. Extraction is cheap relative to
embedding.

Selection happens afterwards, at the firm level. See `validation-and-selection.md`.

## Performance

Wall-clock per filing is a selection criterion, not a nuisance. It multiplies by tens of
thousands of filings times however many rebuilds.

- `edgartools`: slowest. Full DOM plus style parsing plus cascade.
- `inscriptis` (itemseg, own-regex): comparable order on large filings. Layout-aware, pure Python.
- `edgar-crawler`: fastest in the median. Occasional catastrophic tail if a generated pattern
  backtracks; wrap regex execution in a timeout.

Measure this. Time all extractors on a stratified sample of ~200 filings spanning the size and
era range before committing to a priority order.
