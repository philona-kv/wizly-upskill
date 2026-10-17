"""Structured OpenAI outputs with Pydantic validation and safe fallbacks."""

from __future__ import annotations

import json
import logging

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


async def parse_model[T: BaseModel](
    client: AsyncOpenAI,
    model: str,
    *,
    instructions: str,
    user_input: str,
    schema: type[T],
    temperature: float | None = 0.0,
) -> T:
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_input},
    ]
    kwargs: dict = {"model": model, "messages": messages, "response_format": schema}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        import time

        from app.observability.llm_usage import record_usage

        started = time.perf_counter()
        completion = await client.beta.chat.completions.parse(**kwargs)
        record_usage(
            operation="structured_llm",
            model=model,
            duration_ms=int((time.perf_counter() - started) * 1000),
            success=True,
            usage=getattr(completion, "usage", None),
        )
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            return parsed
    except (openai.OpenAIError, ValidationError, IndexError, AttributeError) as exc:
        logger.info("structured_parse_fallback", extra={"exc_type": type(exc).__name__})
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **({"temperature": temperature} if temperature is not None else {}),
    )
    raw = completion.choices[0].message.content or "{}"
    data = json.loads(raw)
    return schema.model_validate(data)
