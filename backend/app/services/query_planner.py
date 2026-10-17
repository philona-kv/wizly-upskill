"""Structured query planning (does not answer the user question)."""

from __future__ import annotations

import logging
from typing import Protocol

from openai import AsyncOpenAI

from app.services.structured_llm import parse_model
from app.workflows.schemas import CompactMessage, QueryPlan, QuerySource

logger = logging.getLogger(__name__)

PLANNER_INSTRUCTIONS = """You plan retrieval for a document-grounded assistant.
Do NOT answer the user's question.

Return JSON matching the schema:
- standalone_query: resolve follow-ups using conversation context
- sources: include "documents" when uploaded files may help; include
  "conversation_history" when prior chat turns are needed; include
  "persistent_memory" when prior user preferences/projects from memory context apply
- needs_clarification: true only when the question cannot be resolved safely
- clarification_prompt: user-facing clarification question when needs_clarification
- routing_summary: one short sentence for operators (no hidden chain-of-thought)
- search_hints: optional keywords for search
"""


class QueryPlanner(Protocol):
    async def plan(
        self,
        *,
        question: str,
        recent_messages: list[CompactMessage],
        context_hint: str = "",
    ) -> QueryPlan: ...


def fallback_plan(question: str) -> QueryPlan:
    return QueryPlan(
        standalone_query=question.strip(),
        sources=[QuerySource.DOCUMENTS.value],
        routing_summary="Fallback: search eligible documents with the original question.",
    )


def _history_block(messages: list[CompactMessage], limit: int = 8) -> str:
    lines = []
    for msg in messages[-limit:]:
        lines.append(f"{msg.role}: {msg.content[:500]}")
    return "\n".join(lines)


class OpenAIQueryPlanner:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def plan(
        self,
        *,
        question: str,
        recent_messages: list[CompactMessage],
        context_hint: str = "",
    ) -> QueryPlan:
        history = _history_block(recent_messages)
        hint = f"\n\nContext bundle:\n{context_hint[:4000]}" if context_hint else ""
        user_input = f"Conversation:\n{history}\n\nCurrent question:\n{question}{hint}"
        try:
            plan = await parse_model(
                self._client,
                self._model,
                instructions=PLANNER_INSTRUCTIONS,
                user_input=user_input,
                schema=QueryPlan,
                temperature=None,
            )
            if not plan.sources:
                plan = plan.model_copy(update={"sources": [QuerySource.DOCUMENTS.value]})
            return plan
        except Exception as exc:
            logger.warning("query_plan_failed", extra={"exc_type": type(exc).__name__})
            return fallback_plan(question)


class HeuristicQueryPlanner:
    """Deterministic planner for tests and API-failure fallback."""

    async def plan(
        self,
        *,
        question: str,
        recent_messages: list[CompactMessage],
        context_hint: str = "",
    ) -> QueryPlan:
        del recent_messages, context_hint
        return fallback_plan(question)
