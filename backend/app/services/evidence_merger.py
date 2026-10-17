"""Normalize and merge document + conversation evidence for the workflow."""

from __future__ import annotations

import uuid

from app.services.context_builder import BuiltContext, ContextBuilder
from app.services.context_builder import EvidenceItem as DocEvidence
from app.services.retrieval_service import RetrievedChunk
from app.workflows.schemas import CompactMessage, EvidenceItem, EvidenceSourceType


def chunks_to_evidence(
    chunks: list[RetrievedChunk],
    *,
    retrieval_attempt: int = 1,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for chunk in chunks:
        items.append(
            EvidenceItem(
                evidence_id=str(chunk.chunk_id),
                source_type=EvidenceSourceType.DOCUMENT,
                content=chunk.content,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                document_title=chunk.document_title,
                original_filename=chunk.original_filename,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                retrieval_score=chunk.distance,
                retrieval_attempt=retrieval_attempt,
            )
        )
    return items


def messages_to_evidence(
    messages: list[CompactMessage],
    *,
    conversation_id: uuid.UUID,
    max_messages: int = 6,
    max_chars: int = 800,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for message in messages[-max_messages:]:
        text = message.content.strip()
        if not text:
            continue
        excerpt = text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"
        items.append(
            EvidenceItem(
                evidence_id=str(message.id),
                source_type=EvidenceSourceType.CONVERSATION,
                content=excerpt,
                conversation_id=conversation_id,
                message_id=message.id,
                message_role=message.role,
                excerpt=excerpt,
            )
        )
    return items


def merge_evidence_lists(
    document_items: list[EvidenceItem],
    conversation_items: list[EvidenceItem],
    *,
    max_items: int,
) -> list[EvidenceItem]:
    seen_chunks: set[tuple[uuid.UUID, uuid.UUID]] = set()
    seen_messages: set[uuid.UUID] = set()
    merged: list[EvidenceItem] = []
    for item in document_items:
        if item.chunk_id and item.document_id:
            key = (item.document_id, item.chunk_id)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
        merged.append(item)
        if len(merged) >= max_items:
            return merged
    for item in conversation_items:
        if item.message_id:
            if item.message_id in seen_messages:
                continue
            seen_messages.add(item.message_id)
        merged.append(item)
        if len(merged) >= max_items:
            break
    return merged


def assign_citation_ids(document_items: list[EvidenceItem]) -> list[EvidenceItem]:
    out: list[EvidenceItem] = []
    n = 0
    for item in document_items:
        if item.source_type != EvidenceSourceType.DOCUMENT:
            continue
        n += 1
        out.append(item.model_copy(update={"citation_id": f"S{n}"}))
    return out


def build_prompt_blocks(
    context_builder: ContextBuilder,
    merged: list[EvidenceItem],
) -> tuple[str, str, list[DocEvidence], BuiltContext]:
    doc_items = [i for i in merged if i.source_type == EvidenceSourceType.DOCUMENT]
    conv_items = [i for i in merged if i.source_type == EvidenceSourceType.CONVERSATION]
    chunks = [
        RetrievedChunk(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            document_title=item.document_title or "",
            original_filename=item.original_filename or "",
            content=item.content,
            page_start=item.page_start or 0,
            page_end=item.page_end or 0,
            chunk_index=0,
            distance=item.retrieval_score or 0.0,
        )
        for item in doc_items
        if item.chunk_id and item.document_id
    ]
    built = context_builder.build(chunks)
    conv_lines = []
    for item in conv_items:
        role = item.message_role or "user"
        conv_lines.append(f"{role.capitalize()}: {item.content}")
    conv_block = ""
    if conv_lines:
        conv_block = (
            "Recent conversation excerpts (for context only; not proof of document content):\n"
            + "\n".join(conv_lines)
        )
    return built.prompt_block, conv_block, built.evidence, built
