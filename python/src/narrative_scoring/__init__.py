"""Narrative scoring: taxonomy-based headline scoring pipeline.

Scores RavenPack headline embeddings against a versioned taxonomy of
"primitive" narrative descriptions, with:

  - taxonomy.py      Canonical 4-level hierarchy over evergreen/ravenpack CSVs
                      plus their paraphrase JSONLs (taxonomy.py).
  - descriptions.py   Embeds taxonomy descriptions (centroid/max/median pooling).
  - corrections.py    RAW / R1 (mean-centering) / R2 (mean-direction removal).
  - null_model.py      The F0 rejection floor (per-headline trim, pooled null,
                      effective-N, empirical quantile).
  - garbage.py         The garbage-catcher rejection layer.
  - scoring.py          Per-day/per-month scoring driver + datalake entry point.
  - aggregate.py        Roll primitive-level daily scores up the hierarchy.
  - schema.py            Polars schemas for every artifact this package writes.

See each module's docstring for the exact formulas -- they are specified
precisely and must not be approximated.
"""
