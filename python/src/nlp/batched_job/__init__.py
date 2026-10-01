"""Asynchronous batch jobs of structured chat completions, one interface for every vendor.

    provider     batch API                          transport         discount
    openai       Batch API (JSONL file)             openai SDK        50%
    anthropic    Message Batches (inline)           anthropic SDK     50%
    google       batchGenerateContent (inline)      httpx, native     50%
    mistral      /v1/batch/jobs (JSONL file)        httpx, native     50%
    perplexity   none: live requests only (nlp.llm.ChatBackend)
    ollama       none: live requests only

Same endpoint and key variables as ``nlp.llm.PROVIDERS``. Usage::

    job = batch_job("mistral", "mistral-medium-2604", schema=SCHEMA, name="ner")
    results = job.run([ChatRequest(id, system, user) for ...], fallback=chat_backend)

``run`` submits, polls, validates every reply and redoes the failures live through
``fallback``. For long batches, ``submit`` returns a ``BatchHandle`` to persist
(``to_dict()``) and re-attach later with ``wait`` + ``results``.
"""
from __future__ import annotations

from typing import Any

from nlp.batched_job.base import (
    BatchChatJob,
    BatchHandle,
    BatchProgress,
    BatchStatus,
    ChatRequest,
    RawReply,
)

SUPPORTED: tuple[str, ...] = ("openai", "anthropic", "google", "mistral")


def job_class(provider: str) -> type[BatchChatJob] | None:
    """The batch job class of ``provider``, or None when it has no batch API."""
    if provider == "openai":
        from nlp.batched_job.openai import OpenAIBatchJob
        return OpenAIBatchJob
    if provider == "anthropic":
        from nlp.batched_job.anthropic import AnthropicBatchJob
        return AnthropicBatchJob
    if provider == "google":
        from nlp.batched_job.google import GoogleBatchJob
        return GoogleBatchJob
    if provider == "mistral":
        from nlp.batched_job.mistral import MistralBatchJob
        return MistralBatchJob
    return None


def batch_job(provider: str, model: str, **kwargs: Any) -> BatchChatJob | None:
    """A batch job for ``provider`` / ``model``, or None when the provider has no batch API
    (callers then stay on ``nlp.llm.ChatBackend``)."""
    cls = job_class(provider)
    return cls(model, **kwargs) if cls else None


__all__ = ["SUPPORTED", "BatchChatJob", "BatchHandle", "BatchProgress", "BatchStatus",
           "ChatRequest", "RawReply", "batch_job", "job_class"]
