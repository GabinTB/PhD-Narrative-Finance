"""NLP layer: everything that turns text into corrected vectors and sentiment scores.

Two layers:

  - backends/        Inference engines (local, TEI, embedx). Canonical model
                     operations only: embed(texts), classify(texts), info().
  - llm.py           Generative models behind OpenAI-compatible endpoints
                     (Ollama, OpenAI, Anthropic, Perplexity, Google, Mistral):
                     schema-validated JSON completions, served-model checks.
  - batched_job/     The same completions through vendor batch APIs (OpenAI,
                     Anthropic, Google, Mistral; 50% off): one BatchChatJob
                     contract, one child per vendor, live fallback via llm.py.
  - NLP objects      Pre/post-processing on top of a backend:
      corrections.py     RAW / R1 / R2 embedding corrections.

``narrative_scoring`` consumes these objects and does scoring only.
"""
