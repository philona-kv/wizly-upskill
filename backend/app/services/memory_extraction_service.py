"""Extract durable memories from conversation segments (structured LLM)."""

from __future__ import annotations

import logging
import uuid

from openai import AsyncOpenAI

from app.db.models import Message, MessageRole
from app.schemas.memory import MemoryExtractionResult
from app.services.memory_service import MemoryService
from app.services.structured_llm import parse_model

logger = logging.getLogger(__name__)

EXTRACT_INSTRUCTIONS = """Extract durable facts for future conversations.
Only user-stated preferences, projects, goals, decisions, unresolved questions.
Do NOT extract greetings, secrets, API keys, or verbatim document passages.
Use source_type user_statement for explicit user facts only.
Return JSON matching the schema with a memories array."""


class MemoryExtractionService:
    def __init__(
        self,
        client: AsyncOpenAI | None,
        model: str,
        memory_service: MemoryService,
        *,
        enabled: bool,
    ) -> None:
        self._client = client
        self._model = model
        self._memory = memory_service
        self._enabled = enabled

    async def extract_and_store(
        self,
        owner_id: uuid.UUID,
        conversation_id: uuid.UUID,
        messages: list[Message],
    ) -> int:
        if not self._enabled or self._client is None:
            return 0
        user_lines = []
        for msg in messages:
            if msg.role != MessageRole.USER.value:
                continue
            if len(msg.content) > 2000:
                continue
            user_lines.append(f"[{msg.id}] {msg.content[:1500]}")
        if not user_lines:
            return 0
        try:
            result = await parse_model(
                self._client,
                self._model,
                instructions=EXTRACT_INSTRUCTIONS,
                user_input="\n".join(user_lines),
                schema=MemoryExtractionResult,
                temperature=None,
            )
        except Exception as exc:
            logger.warning("memory_extraction_failed", extra={"exc_type": type(exc).__name__})
            return 0
        saved = 0
        for prop in result.memories:
            if await self._memory.persist_proposed(owner_id, conversation_id, prop):
                saved += 1
        logger.info(
            "memory_extraction_complete",
            extra={"conversation_id": str(conversation_id), "saved": saved},
        )
        return saved
