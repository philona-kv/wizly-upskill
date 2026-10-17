"""Pydantic models for the Twin LangGraph workflow (serializable, no secrets)."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.conversation import Citation


class EvidenceSourceType(StrEnum):
    DOCUMENT = "document"
    CONVERSATION = "conversation"


class RetrievalStrategy(StrEnum):
    SEMANTIC = "semantic"
    BROADEN_SEMANTIC = "broaden_semantic"
    FULLTEXT = "fulltext"
    NEIGHBOR_EXPAND = "neighbor_expand"
    SECTION_FOCUS = "section_focus"


class QuerySource(StrEnum):
    DOCUMENTS = "documents"
    CONVERSATION_HISTORY = "conversation_history"
    PERSISTENT_MEMORY = "persistent_memory"


class QueryPlan(BaseModel):
    standalone_query: str = Field(min_length=1, max_length=4000)
    sources: list[
        Literal["documents", "conversation_history", "persistent_memory"]
    ] = Field(min_length=1)
    requires_multiple_sources: bool = False
    requires_multi_hop: bool = False
    search_hints: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    routing_summary: str = ""


class EvidenceItem(BaseModel):
    evidence_id: str
    source_type: EvidenceSourceType
    content: str
    document_id: uuid.UUID | None = None
    chunk_id: uuid.UUID | None = None
    conversation_id: uuid.UUID | None = None
    message_id: uuid.UUID | None = None
    document_title: str | None = None
    original_filename: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    retrieval_score: float | None = None
    retrieval_attempt: int = 1
    citation_id: str | None = None
    excerpt: str | None = None
    message_role: str | None = None
    created_at: datetime | None = None


class EvidenceAssessment(BaseModel):
    sufficient: bool
    missing_information: list[str] = Field(default_factory=list)
    supported_aspects: list[str] = Field(default_factory=list)
    suggested_strategy: Literal[
        "none",
        "rewrite_query",
        "broaden_semantic_search",
        "use_hybrid_search",
        "expand_neighboring_chunks",
        "retrieve_document_section",
        "clarify_question",
    ] = "none"
    explanation: str = ""


class AnswerValidation(BaseModel):
    valid: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    correction_required: bool = False
    explanation: str = ""


class WorkflowStatus(StrEnum):
    SUCCESS = "success"
    ABSTAIN = "abstain"
    CLARIFY = "clarify"
    ERROR = "error"


class WorkflowDebugMetadata(BaseModel):
    selected_sources: list[str] = Field(default_factory=list)
    retrieval_attempts: int = 0
    retrieval_strategy: str = ""
    document_chunks: int = 0
    conversation_messages: int = 0
    memory_count: int = 0
    summary_version: int | None = None
    used_persistent_context: bool = False
    evidence_sufficient: bool | None = None
    answer_repaired: bool = False
    duration_ms: int = 0
    routing_summary: str = ""
    node_timings: list[dict[str, Any]] = Field(default_factory=list)
    llm_call_count: int = 0


class CompactMessage(BaseModel):
    id: uuid.UUID
    role: str
    content: str


class WorkflowInput(BaseModel):
    request_id: uuid.UUID
    owner_id: uuid.UUID
    conversation_id: uuid.UUID
    current_user_message_id: uuid.UUID
    original_question: str
    recent_messages: list[CompactMessage]
    client_message_id: uuid.UUID | None = None
    graph_version: str = "twin-v4"


class WorkflowOutput(BaseModel):
    status: WorkflowStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    used_llm: bool = True
    model_name: str | None = None
    error_code: str | None = None
    operational_metadata: WorkflowDebugMetadata | None = None

    model_config = {"arbitrary_types_allowed": True}


def assessment_to_strategy(assessment: EvidenceAssessment) -> RetrievalStrategy | None:
    mapping: dict[str, RetrievalStrategy] = {
        "rewrite_query": RetrievalStrategy.SEMANTIC,
        "broaden_semantic_search": RetrievalStrategy.BROADEN_SEMANTIC,
        "use_hybrid_search": RetrievalStrategy.FULLTEXT,
        "expand_neighboring_chunks": RetrievalStrategy.NEIGHBOR_EXPAND,
        "retrieve_document_section": RetrievalStrategy.SECTION_FOCUS,
    }
    return mapping.get(assessment.suggested_strategy)
