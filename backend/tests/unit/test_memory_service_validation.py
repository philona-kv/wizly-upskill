import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.memory import ProposedMemory
from app.services.memory_service import MemoryService


@pytest.mark.asyncio
async def test_persist_rejects_secrets() -> None:
    session_factory = MagicMock()
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[[0.0] * 1536])
    service = MemoryService(
        session_factory,
        embedder,
        embedding_model="text-embedding-3-small",
        top_k=5,
        min_importance=0.0,
        max_records=10,
        enabled=True,
    )
    prop = ProposedMemory(
        memory_type="user_provided_fact",
        content="My api_key=sk-123456789012345678901234",
    )
    result = await service.persist_proposed(uuid.uuid4(), uuid.uuid4(), prop)
    assert result is None
    embedder.embed.assert_not_called()
