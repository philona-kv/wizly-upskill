"""LangGraph state for the Twin chat workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class TwinGraphState(TypedDict, total=False):
    request_id: str
    owner_id: str
    conversation_id: str
    current_user_message_id: str
    original_question: str
    recent_messages: list[dict[str, Any]]
    query_plan: dict[str, Any]
    standalone_query: str
    selected_sources: list[str]
    retrieval_strategy: str
    retrieval_attempt: int
    max_retrieval_attempts: int
    max_answer_repairs: int
    document_evidence: list[dict[str, Any]]
    conversation_evidence: list[dict[str, Any]]
    merged_evidence: list[dict[str, Any]]
    document_retrieval_error: str | None
    conversation_retrieval_error: str | None
    evidence_assessment: dict[str, Any]
    answer_draft: str
    validation_result: dict[str, Any]
    answer_repair_count: int
    final_answer: str
    final_citations: list[dict[str, Any]]
    used_llm: bool
    model_name: str | None
    status: str
    error_code: str | None
    operational_metadata: dict[str, Any]
    evidence_prompt_block: str
    conversation_context_block: str
    enable_debug: bool
    focus_chunk_ids: list[str]
    rewritten_query: str | None
    conversation_summary_text: str
    conversation_summary_version: int | None
    retrieved_memories: list[dict[str, Any]]
    context_bundle: str
    graph_version: str
    node_timings: list[dict[str, Any]]
    playground_mode: bool
    retrieval_scope: str
    selected_document_ids: list[str]
    generation_instructions: str | None
    include_conversation_history: bool
    prompt_template_text: str | None
