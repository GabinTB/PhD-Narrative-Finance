"""NLP layer: everything that turns text into corrected vectors and sentiment scores.

Two layers:

  - backends/        Inference engines (local, TEI, embedx). Canonical model
                     operations only: embed(texts), classify(texts), info().
  - NLP objects      Pre/post-processing on top of a backend:
      corrections.py     RAW / R1 / R2 embedding corrections.

``narrative_scoring`` consumes these objects and does scoring only.
"""
