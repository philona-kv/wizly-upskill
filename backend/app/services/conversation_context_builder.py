"""Assemble bounded non-document context for planning and generation."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.chunking_service import get_encoding
from app.services.memory_service import RetrievedMemory
from app.workflows.schemas import CompactMessage


@dataclass(frozen=True, slots=True)
class AssembledContext:
    recent_block: str
    summary_block: str
    memory_block: str
    combined_untrusted: str
    token_estimate: int


class ConversationContextBuilder:
    def __init__(self, *, token_budget: int, embedding_model: str) -> None:
        self._budget = token_budget
        self._enc = get_encoding(embedding_model)

    def build(
        self,
        *,
        question: str,
        recent_messages: list[CompactMessage],
        summary_text: str | None,
        memories: list[RetrievedMemory],
        exclude_message_id: str | None = None,
    ) -> AssembledContext:
        del question
        recent_lines = []
        for msg in recent_messages[-6:]:
            if exclude_message_id and str(msg.id) == exclude_message_id:
                continue
            recent_lines.append(f"{msg.role}: {msg.content[:600]}")
        recent_block = ""
        if recent_lines:
            recent_block = (
                "Recent conversation (not document evidence):\n" + "\n".join(recent_lines)
            )
        summary_block = ""
        if summary_text and summary_text.strip():
            summary_block = (
                "Conversation summary (compressed context, not proof of document contents):\n"
                + summary_text.strip()[:3000]
            )
        mem_lines = []
        for mem in memories[:8]:
            label = mem.source_type.replace("_", " ")
            mem_lines.append(f"- [{mem.memory_type}/{label}] {mem.content[:400]}")
        memory_block = ""
        if mem_lines:
            memory_block = (
                "Persistent memories from earlier conversations "
                "(user context — not document evidence):\n"
                + "\n".join(mem_lines)
            )
        parts = [p for p in (summary_block, memory_block, recent_block) if p]
        combined = "\n\n".join(parts)
        tokens = len(self._enc.encode(combined, disallowed_special=())) if combined else 0
        if tokens > self._budget and combined:
            combined = self._enc.decode(
                self._enc.encode(combined, disallowed_special=())[: self._budget]
            )
            tokens = self._budget
        return AssembledContext(
            recent_block=recent_block,
            summary_block=summary_block,
            memory_block=memory_block,
            combined_untrusted=combined,
            token_estimate=tokens,
        )
