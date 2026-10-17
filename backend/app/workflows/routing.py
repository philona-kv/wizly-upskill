"""Pure routing helpers for LangGraph conditional edges."""

from __future__ import annotations

from app.workflows.schemas import EvidenceSourceType
from app.workflows.state import TwinGraphState


def _has_document_evidence(state: TwinGraphState) -> bool:
    if state.get("document_evidence"):
        return True
    for item in state.get("merged_evidence") or []:
        if item.get("source_type") == EvidenceSourceType.DOCUMENT.value:
            return True
    return False


def after_plan(state: TwinGraphState) -> str:
    plan = state.get("query_plan") or {}
    if plan.get("needs_clarification"):
        return "finalize_clarification"
    return "parallel_retrieval"


def after_assess(state: TwinGraphState) -> str:
    sources = state.get("selected_sources") or []
    doc_only = sources == ["documents"]
    if (
        state.get("document_retrieval_error")
        and not state.get("document_evidence")
        and doc_only
    ):
        return "finalize_error"
    assessment = state.get("evidence_assessment") or {}
    if assessment.get("sufficient"):
        return "generate_answer"
    attempt = state.get("retrieval_attempt") or 1
    max_attempts = state.get("max_retrieval_attempts") or 1
    strategy = assessment.get("suggested_strategy") or "none"
    if attempt < max_attempts and strategy not in ("none", "clarify_question"):
        return "plan_retrieval_retry"
    if strategy == "clarify_question":
        return "finalize_clarification"
    # Retrieved authorized chunks: let the generator answer or explain gaps (Subtask 2 behavior).
    if _has_document_evidence(state):
        return "generate_answer"
    return "finalize_abstain"


def after_validate(state: TwinGraphState) -> str:
    validation = state.get("validation_result") or {}
    if validation.get("valid"):
        return "finalize_success"
    repairs = state.get("answer_repair_count") or 0
    max_repairs = state.get("max_answer_repairs") or 0
    if repairs < max_repairs and validation.get("correction_required"):
        return "repair_answer"
    return "finalize_safe_fallback"
