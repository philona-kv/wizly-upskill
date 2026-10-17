"""Centralized LLM/embedding usage tracking (no prompt content logged)."""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_records: contextvars.ContextVar[list[LlmUsageRecord]] = contextvars.ContextVar(
    "llm_usage_records", default=[]
)


@dataclass(slots=True)
class UsageContext:
    request_id: str | None = None
    conversation_id: str | None = None
    workflow_id: str | None = None


_ctx: contextvars.ContextVar[UsageContext | None] = contextvars.ContextVar(
    "llm_usage_context", default=None
)


@dataclass(slots=True)
class LlmUsageRecord:
    operation: str
    model: str
    duration_ms: int
    success: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    is_token_estimate: bool = False
    retry_count: int = 0
    provider_request_id: str | None = None
    estimated_cost_usd: float | None = None


@contextmanager
def usage_context(ctx: UsageContext):
    token = _ctx.set(ctx)
    rec_token = _records.set([])
    try:
        yield
    finally:
        _ctx.reset(token)
        _records.reset(rec_token)


def clear_usage_records() -> None:
    _records.set([])


def get_usage_records() -> list[LlmUsageRecord]:
    return list(_records.get())


def _append(record: LlmUsageRecord) -> None:
    records = _records.get()
    records.append(record)
    _records.set(records)
    logger.info(
        "llm_usage",
        extra={
            "operation": record.operation,
            "model": record.model,
            "duration_ms": record.duration_ms,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "success": record.success,
            "is_token_estimate": record.is_token_estimate,
        },
    )


def record_usage(
    *,
    operation: str,
    model: str,
    duration_ms: int,
    success: bool,
    usage: Any | None = None,
    is_token_estimate: bool = False,
    retry_count: int = 0,
    provider_request_id: str | None = None,
) -> LlmUsageRecord:
    input_tokens = output_tokens = cached_tokens = None
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(
            usage, "completion_tokens", None
        )
        cached_tokens = getattr(usage, "cached_tokens", None)
    from app.observability.pricing import estimate_cost_usd

    cost = estimate_cost_usd(model, input_tokens, output_tokens, cached_tokens)
    record = LlmUsageRecord(
        operation=operation,
        model=model,
        duration_ms=duration_ms,
        success=success,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        is_token_estimate=is_token_estimate,
        retry_count=retry_count,
        provider_request_id=provider_request_id,
        estimated_cost_usd=cost,
    )
    _append(record)
    return record


class track_llm_call:
    def __init__(self, operation: str, model: str) -> None:
        self._operation = operation
        self._model = model
        self._started = 0.0

    async def __aenter__(self):
        self._started = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        duration_ms = int((time.perf_counter() - self._started) * 1000)
        record_usage(
            operation=self._operation,
            model=self._model,
            duration_ms=duration_ms,
            success=exc_type is None,
        )
        return False
