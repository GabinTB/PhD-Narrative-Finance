# Extraction benchmark

## Purpose

Determine the extractor priority order empirically. It is currently unknown.

The only published number covers `itemseg`. `edgartools` has never been benchmarked against
anything by anyone; the claim that it is strongest on HTML is an inference from reading its
source (anchors and typography are strictly more information than line-start regex), not a
measurement. `edgar-crawler` has not been benchmarked directly either; the itemseg paper built
its own regex baseline and does not ship it, and `edgar-crawler` is more engineered than a naive
baseline (line anchoring, monotonic ordering), so it plausibly scores above 0.9048.

**Do not build a corpus on an unmeasured ordering.** Running this is roughly an afternoon and
settles the question permanently, with a sentence in the paper that cannot be argued with.

## Gold data

Dataset: `im.ntu.edu.tw/~lu/data/itemseg/itemseg10kdata.7z`. 3,737 manually annotated 10-Ks,
FY2001-2019, with a 373-filing held-out test split.

## The blocking problem: labels are representation-scoped

The gold labels are line-level BIO tags over **inscriptis** output. `edgar-crawler`'s line breaks
come from a different HTML-to-text path; `edgartools` does not produce lines at all. The itemseg
paper states outright that its dataset cannot be directly applied to other tools because the
preprocessing routines differ.

**Therefore: do not compare labels. Compare extracted text.**

1. Reconstruct the gold item text from the BIO labels.
2. Normalise both gold and candidate: casefold, strip non-alphanumeric, collapse whitespace.
3. Score with token-level IoU.

Representation-agnostic, and the only fair way to put a DOM-based extractor and three text-based
ones on the same axis. Boundary precision is lost, which does not matter: what matters is whether
the text being embedded is the right text.

**Build this harness first.** Everything else in the benchmark is downstream of it.

## Arms

Seven, not four. The own-regex additions must be separate flags.

| Arm | Tests |
|---|---|
| `edgar-crawler` upstream | baseline |
| `edgar-crawler` + `title_keyword` | precision on bare cross-references |
| `edgar-crawler` + `comma_lookbehind` | precision on reference lists |
| `edgar-crawler` + both | interaction |
| `edgartools` | DOM, anchors, typography |
| `itemseg-bert` | published SOTA |
| `itemseg-crf` | whether the GPU can be skipped |

Include the CRF. It is shipped, CPU-only, and was statistically tied with BERT on non-core items.
If it ties on this data too, the full 25-year panel runs without GPU scheduling, which is a real
operational difference.

## Metrics

**Report** line-level macro-F1 so the numbers sit next to the published table.

**Decide on** per-filing IoU above a threshold, stratified. That is what determines whether a
firm-year is usable.

**Additionally compute** the metric nobody in this literature reports and which actually
determines panel size:

> The fraction of firm-year **pairs** where both years segment correctly.

One bad year kills the pair. Compute it per extractor on the gold set before picking.

## Expect the additions to be invisible in aggregate

The test split is 373 filings. The published gap between BERT (0.9826) and CRF (0.9788) was
already not significant on non-core items at that sample size. Two regex tweaks moving aggregate
performance by a fraction of a point will not be detectable.

Looking only at the aggregate produces "no difference" and teaches nothing.

### Stratify

The additions target one specific failure, cross-reference contamination, which affects a
minority of filings. Slice by:

- format regime (pre-2001 text vs HTML vs post-2019 inline XBRL)
- filing agent
- decade
- TOC present or absent
- filer category

Report the aggregate for comparability; **decide on the strata**. An addition that fixes 40% of
failures in a 5% subpopulation is worth keeping even though it moves the aggregate by 0.002.

### Confidence intervals

Bootstrap CIs on the differences, matching the published methodology. Without them, noise gets
over-read.

## Timing

Wall-clock per filing is a selection criterion, not a footnote. It multiplies by tens of
thousands of filings times however many rebuilds.

Time every arm on a stratified sample of ~200 filings spanning the size and era range. If one
extractor is 10x slower for a fraction of a point of accuracy that is not even measurable,
that changes the cost-benefit of putting it first.

## Output

A benchmark report written to `dev/benchmark/` containing:

- the aggregate table, comparable to the published one
- per-stratum tables
- the pair-correctness metric per extractor
- bootstrap CIs on all pairwise differences
- timing distribution per extractor
- the agreement matrix between extractors on filings where multiple pass

The resulting priority order goes into config, not into code.
