"""OpenAI embeddings with batching, bounded retries, and result validation."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

import openai
from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)
_MAX_BACKOFF_SECONDS = 30.0


class Embedder(Protocol):
    model: str
    dimensions: int

    async def embed(
        self, texts: Sequence[str], token_counts: Sequence[int] | None = None
    ) -> list[list[float]]: ...


def _sanitize(exc: openai.OpenAIError) -> str:
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return "Embedding provider rejected the credentials."
    if isinstance(exc, openai.RateLimitError):
        return "Embedding provider rate limit or quota exceeded; try again later."
    if isinstance(exc, openai.APITimeoutError):
        return "Embedding provider timed out."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not connect to the embedding provider."
    if isinstance(exc, openai.BadRequestError):
        return "Embedding provider rejected the request."
    if isinstance(exc, openai.InternalServerError):
        return "Embedding provider is temporarily unavailable."
    return "Embedding generation failed."


class OpenAIEmbeddingService:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        dimensions: int,
        *,
        batch_size: int = 96,
        max_batch_tokens: int = 250_000,
        max_retries: int = 4,
        base_backoff: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self.model = model
        self.dimensions = dimensions
        self._batch_size = batch_size
        self._max_batch_tokens = max_batch_tokens
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAIEmbeddingService:
        client_kwargs: dict = {
            "api_key": settings.require_openai(),
            "timeout": settings.openai_timeout_seconds,
            "max_retries": 0,  # retries are handled (and bounded) here
        }
        base_url = settings.embedding_api_base()
        if base_url is not None:
            client_kwargs["base_url"] = base_url
            logger.info(
                "embedding_client_litellm",
                extra={"base_url": base_url, "model": settings.openai_embedding_model},
            )
        client = AsyncOpenAI(**client_kwargs)
        return cls(
            client,
            settings.openai_embedding_model,
            settings.openai_embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            max_batch_tokens=settings.embedding_max_batch_tokens,
            max_retries=settings.openai_max_retries,
        )

    async def aclose(self) -> None:
        await self._client.close()

    def _batches(
        self, texts: Sequence[str], token_counts: Sequence[int] | None
    ) -> list[tuple[int, int]]:
        """Return [start, end) index ranges respecting item and token limits."""
        ranges: list[tuple[int, int]] = []
        start, tokens = 0, 0
        for i in range(len(texts)):
            item_tokens = token_counts[i] if token_counts else 0
            if i > start and (
                i - start >= self._batch_size or tokens + item_tokens > self._max_batch_tokens
            ):
                ranges.append((start, i))
                start, tokens = i, 0
            tokens += item_tokens
        if start < len(texts):
            ranges.append((start, len(texts)))
        return ranges

    def _backoff(self, attempt: int, exc: Exception) -> float:
        retry_after: float | None = None
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                retry_after = float(response.headers.get("retry-after", ""))
            except (TypeError, ValueError):
                retry_after = None
        delay = retry_after if retry_after is not None else self._base_backoff * 2**attempt
        return min(delay, _MAX_BACKOFF_SECONDS) + random.uniform(0, self._base_backoff)  # noqa: S311

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                import time

                from app.observability.llm_usage import record_usage

                started = time.perf_counter()
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
                record_usage(
                    operation="embedding",
                    model=self.model,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    success=True,
                    usage=getattr(response, "usage", None),
                )
                break
            except _RETRYABLE as exc:
                if attempt >= self._max_retries:
                    logger.warning(
                        "embedding_retries_exhausted",
                        extra={"attempts": attempt + 1, "exc_type": type(exc).__name__},
                    )
                    raise EmbeddingError(_sanitize(exc)) from exc
                delay = self._backoff(attempt, exc)
                logger.info(
                    "embedding_retry",
                    extra={
                        "attempt": attempt + 1,
                        "delay_s": round(delay, 2),
                        "exc_type": type(exc).__name__,
                    },
                )
                await self._sleep(delay)
                attempt += 1
            except openai.OpenAIError as exc:
                logger.warning("embedding_failed", extra={"exc_type": type(exc).__name__})
                raise EmbeddingError(_sanitize(exc)) from exc

        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(batch) or [d.index for d in data] != list(range(len(batch))):
            raise EmbeddingError("Embedding provider returned an unexpected number of vectors.")
        vectors = [list(d.embedding) for d in data]
        if any(len(v) != self.dimensions for v in vectors):
            raise EmbeddingError(
                f"Embedding provider returned vectors with unexpected dimensions "
                f"(expected {self.dimensions})."
            )
        return vectors

    async def embed(
        self, texts: Sequence[str], token_counts: Sequence[int] | None = None
    ) -> list[list[float]]:
        if token_counts is not None and len(token_counts) != len(texts):
            raise ValueError("token_counts must match texts")
        if any(not t.strip() for t in texts):
            raise ValueError("Cannot embed empty text")
        # Identical texts (e.g. repeated boilerplate pages) are embedded once.
        position: dict[str, int] = {}
        unique: list[str] = []
        unique_tokens: list[int] = []
        for i, text in enumerate(texts):
            if text not in position:
                position[text] = len(unique)
                unique.append(text)
                if token_counts is not None:
                    unique_tokens.append(token_counts[i])

        vectors: list[list[float]] = []
        # Sequential batches keep ordering trivial and stay friendly to rate limits.
        for start, end in self._batches(unique, unique_tokens if token_counts else None):
            vectors.extend(await self._embed_batch(unique[start:end]))
        return [vectors[position[text]] for text in texts]
