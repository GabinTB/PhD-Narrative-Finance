"""Narrative scoring: taxonomy-based headline scoring pipeline.

Canonical scorer (spec: python/doc/narratives.md, owner rulings on top of it):

  - config.py        ScoringConfig / RunMetadata, pooling and aggregation enums.
  - tags.py          Sentiment tags: labels as disjoint intervals of the headline score.
  - assets.py        Asset layer: universe entities, membership as of a day, asset tables.
  - primitives.py    Primitive table (CSV + JSONL joined on the sha1 path hash),
                     primitive-text embeddings, scoring matrix, pooled scores.

Everything before scoring is NLP and lives in the ``nlp`` package: embedding
backends, RAW / R1 / R2 corrections (``nlp.corrections``), sentiment.
  - f0.py            F0: per-headline trim, t-digest + Welford pool, N_eff, tau formulas.
  - warmup.py        N_eff / spectrum computation (run inside the monthly tau job).
  - partitions.py    Monthly null partitions (f0_monthly_partitions artifact).
  - tau_asof.py      Point-in-time tau series (tau_asof artifact) and its reader.
  - calibration.py   Point-in-time records, lookahead guard, mu_asof lookup.
  - selection.py     q-candidates on the full row -> tau -> optional jump cut.
  - aggregation.py   primitive -> narrative per headline; day accumulators (ddof=0).
  - streaming.py     Day access to headlines: each partition read once (embeddings, tag
                     codes, assets), served day by day in bounded chunks.
  - pipeline.py      score_dates(): the single live/historical entry point.
  - artifacts.py     Datalake registration; score_range_to_datalake(): the one scoring
                     path (chronological replay of the live loop).
  - jobs.py          CLI: score / tau-asof / mark-temp (--tag NAME=CONDITION
                     --sentiment-source --sentiment-rule --assets-universe --min-relevance).

Sentiment is produced outside this package, as headline_sentiment artifacts
(ravenpack/headlines/sentiment*.py: RavenPack CSS/ESS, RavenBERT / FinBERT 41-grid tables).
  - validation.py    Dense reference implementations, derived views, summaries,
                     sum_tags (a run's labels -> one panel), merge_poles.
  - _kernels/        Optional compiled fused kernel (asserted equal to numpy).

"""
