"""Narrative scoring: taxonomy-based headline scoring pipeline.

Canonical scorer (spec: python/doc/narratives.md, owner rulings on top of it):

  - config.py        ScoringConfig / RunMetadata, pooling, aggregation and sentiment enums.
  - primitives.py    Primitive table (CSV + JSONL joined on the sha1 path hash),
                     primitive-text embeddings, scoring matrix, pooled scores.
  - corrections.py   RAW / R1 / R2 (apply_mode: the same transform on both sides).
  - f0.py            F0: per-headline trim, t-digest + Welford pool, N_eff, tau formulas.
  - warmup.py        N_eff / spectrum computation (run inside the monthly tau job).
  - partitions.py    Monthly null partitions (f0_monthly_partitions artifact).
  - tau_asof.py      Point-in-time tau series (tau_asof artifact) and its reader.
  - calibration.py   Point-in-time records, lookahead guard, mu_asof lookup.
  - selection.py     q-candidates on the full row -> tau -> optional jump cut.
  - aggregation.py   primitive -> narrative per headline; day accumulators (ddof=0).
  - streaming.py     Bounded-memory per-day access to headline embeddings, with one SENT_*
                     column of a headline_sentiment artifact joined in for split runs.
  - pipeline.py      score_dates(): the single live/historical entry point.
  - artifacts.py     Datalake registration; score_range_to_datalake(): the one scoring
                     path (chronological replay of the live loop).
  - jobs.py          CLI: score / tau-asof / mark-temp (--split sign --sentiment-source
                     --sentiment-column --neutral-eps for the sentiment split).

Sentiment scores are produced outside this package, as headline_sentiment artifacts
(ravenpack/headlines/sentiment*.py: RavenPack CSS/ESS, RavenBERT, FinBERT).
  - validation.py    Dense reference implementations, derived views, summaries.
  - _kernels/        Optional compiled fused kernel (asserted equal to numpy).

Suspended (Ray/GPU track): gpu_scoring.py, ray_hybrid.py, _kernels/fused_gate.pyx;
the two modules raise NotImplementedError on import and their tests are skipped.
"""
