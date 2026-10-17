from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from app.core.exceptions import EmbeddingError
from app.services.embedding_service import OpenAIEmbeddingService

DIMS = 4


def _vector(text: str) -> list[float]:
    return [float(len(text)), 0.0, 0.0, 1.0]


class FakeEmbeddings:
    def __init__(
        self, failures: list[Exception] | None = None, *, shuffle: bool = False, dims: int = DIMS
    ) -> None:
        self.failures = list(failures or [])
        self.calls: list[list[str]] = []
        self.shuffle = shuffle
        self.dims = dims

    async def create(self, *, model: str, input: list[str], dimensions: int, **_: Any):
        self.calls.append(list(input))
        if self.failures:
            raise self.failures.pop(0)
        data = [
            SimpleNamespace(index=i, embedding=_vector(t)[: self.dims] + [0.0] * (self.dims - DIMS))
            for i, t in enumerate(input)
        ]
        if self.shuffle:
            data.reverse()
        return SimpleNamespace(data=data)


def _service(fake: FakeEmbeddings, **kwargs: Any) -> tuple[OpenAIEmbeddingService, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = SimpleNamespace(embeddings=fake)
    service = OpenAIEmbeddingService(
        client, "text-embedding-3-small", DIMS, sleep=fake_sleep, base_backoff=0.01, **kwargs
    )
    return service, sleeps


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def _rate_limit() -> openai.RateLimitError:
    response = httpx.Response(429, request=_request(), headers={"retry-after": "0"})
    return openai.RateLimitError("rate limited", response=response, body=None)


async def test_preserves_order_across_batches() -> None:
    fake = FakeEmbeddings(shuffle=True)
    service, _ = _service(fake, batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vectors = await service.embed(texts)
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [len(c) for c in fake.calls] == [2, 2, 1]


async def test_batches_respect_token_budget() -> None:
    fake = FakeEmbeddings()
    service, _ = _service(fake, batch_size=100, max_batch_tokens=10)
    await service.embed(["a", "b", "c"], token_counts=[6, 6, 3])
    assert fake.calls == [["a"], ["b", "c"]]


async def test_identical_texts_embedded_once() -> None:
    fake = FakeEmbeddings()
    service, _ = _service(fake)
    vectors = await service.embed(["same", "other", "same"])
    assert fake.calls == [["same", "other"]]
    assert vectors[0] == vectors[2]


async def test_retries_transient_errors_then_succeeds() -> None:
    fake = FakeEmbeddings([_rate_limit(), openai.APITimeoutError(_request())])
    service, sleeps = _service(fake, max_retries=3)
    vectors = await service.embed(["hello"])
    assert len(vectors) == 1
    assert len(sleeps) == 2
    assert len(fake.calls) == 3


async def test_retries_are_bounded() -> None:
    fake = FakeEmbeddings([_rate_limit() for _ in range(10)])
    service, sleeps = _service(fake, max_retries=2)
    with pytest.raises(EmbeddingError, match="rate limit"):
        await service.embed(["hello"])
    assert len(fake.calls) == 3
    assert len(sleeps) == 2


async def test_non_retryable_error_fails_immediately() -> None:
    response = httpx.Response(401, request=_request())
    fake = FakeEmbeddings(
        [openai.AuthenticationError("bad key sk-secret", response=response, body=None)]
    )
    service, sleeps = _service(fake)
    with pytest.raises(EmbeddingError) as info:
        await service.embed(["hello"])
    assert "sk-secret" not in info.value.message
    assert sleeps == []


async def test_dimension_mismatch_is_rejected() -> None:
    fake = FakeEmbeddings(dims=DIMS + 2)
    service, _ = _service(fake)
    with pytest.raises(EmbeddingError, match="dimensions"):
        await service.embed(["hello"])


async def test_empty_text_rejected() -> None:
    service, _ = _service(FakeEmbeddings())
    with pytest.raises(ValueError):
        await service.embed(["ok", "  "])
