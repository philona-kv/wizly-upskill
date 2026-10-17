"""Deterministic offline evaluation (no live LLM)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from app.services.evidence_merger import chunks_to_evidence
from app.services.generation_service import INSUFFICIENT_EVIDENCE
from app.services.retrieval_service import RetrievedChunk
from app.workflows.schemas import WorkflowStatus
from evaluation.fixtures.offline_registry import OfflineRegistry, build_default_registry
from evaluation.metrics.answers import score_answer_metrics
from evaluation.metrics.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k
from evaluation.metrics.sources import source_selection_match
from evaluation.schemas.models import EvalCase, EvalResult, ExecutionStatus, MetricScores


@dataclass
class OfflineRunConfig:
    top_k: int = 8
    owner_id: uuid.UUID | None = None


def _chunk_to_retrieved(chunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.UUID(chunk.chunk_id),
        document_id=uuid.UUID(chunk.document_id),
        document_title=chunk.document_title,
        original_filename=chunk.original_filename,
        content=chunk.content,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_index=0,
        distance=0.1,
    )


def _offline_answer(question: str, evidence_block: str, should_abstain: bool) -> str:
    if should_abstain or not evidence_block.strip():
        return INSUFFICIENT_EVIDENCE
    if "48 hours" in question.lower() and "48 hours" in evidence_block:
        return "Schedule changes require 48 hours notice. [S1]"
    if "deadline" in question.lower() and "March 31" in evidence_block:
        return "Compliance deadline is March 31. [S1]"
    if "technologies" in question.lower() or "technology" in question.lower():
        return "Technologies include PostgreSQL, Python, and LangGraph. [S1]"
    if "rate limit" in question.lower():
        return "The API rate limit is 100 requests per minute. [S1]"
    if "codename" in question.lower() or "aurora" in question.lower():
        return "The project codename is Aurora. [S1]"
    if "bullet" in question.lower():
        return "The team prefers bullet-point summaries. [S1]"
    if "encrypt" in question.lower():
        return "Data must be encrypted at rest. [S1]"
    if "compare" in question.lower() or "stricter" in question.lower():
        return "Security standards require MFA; remote work policy focuses on notice periods. [S1]"
    if "ignore previous instructions" in evidence_block.lower():
        return "The document mentions instructions but they are not authoritative. [S1]"
    return "Answer grounded in retrieved evidence. [S1]"


async def run_case_offline(
    case: EvalCase,
    registry: OfflineRegistry | None = None,
    *,
    config: OfflineRunConfig | None = None,
    mode: str = "baseline",
) -> EvalResult:
    config = config or OfflineRunConfig()
    registry = registry or build_default_registry()
    owner = config.owner_id or registry.owner_id
    started = time.perf_counter()
    try:
        hits = registry.retrieve(case.question, top_k=config.top_k, owner_scope=owner)
        retrieved_ids = [c.chunk_id for c in hits]
        relevant = set(case.expected_chunk_ids)
        scores = MetricScores(
            recall_at_k=recall_at_k(relevant, retrieved_ids, config.top_k),
            precision_at_k=precision_at_k(relevant, retrieved_ids, config.top_k),
            mrr=mrr(relevant, retrieved_ids),
            ndcg_at_k=ndcg_at_k(case.chunk_relevance_grades, retrieved_ids, config.top_k),
            document_id_hit=bool(set(case.expected_document_ids) & {c.document_id for c in hits})
            if case.expected_document_ids
            else None,
            page_hit=bool(case.expected_pages)
            and any(c.page_start in case.expected_pages for c in hits),
        )
        chunks = [_chunk_to_retrieved(c) for c in hits]
        evidence = chunks_to_evidence(chunks)
        evidence_block = "\n".join(item.content for item in evidence)
        answer = _offline_answer(case.question, evidence_block, case.should_abstain)
        actual_abstain = answer.strip() == INSUFFICIENT_EVIDENCE or "could not find enough" in answer.lower()
        answer_scores = score_answer_metrics(
            case,
            answer=answer,
            actual_abstention=actual_abstain,
            citation_valid=True if "[S1]" in answer and evidence else None,
            evidence_texts=[item.content for item in evidence],
        )
        scores = scores.model_copy(update=answer_scores.model_dump(exclude_unset=True))
        if mode == "agentic":
            selected = ["documents"]
            if case.conversation_context:
                selected.append("conversation_history")
            if case.relevant_memory_facts:
                selected.append("persistent_memory")
            scores.source_selection_correct = source_selection_match(
                case.required_retrieval_sources, selected
            )
        status = WorkflowStatus.ABSTAIN if actual_abstain else WorkflowStatus.SUCCESS
        latency = int((time.perf_counter() - started) * 1000)
        return EvalResult(
            case_id=case.case_id,
            execution_status=ExecutionStatus.SUCCESS,
            runner_mode=f"offline-{mode}",
            generated_answer=answer,
            retrieved_chunk_ids=retrieved_ids,
            retrieved_document_ids=[c.document_id for c in hits],
            retrieved_pages=[c.page_start for c in hits],
            expected_abstention=case.should_abstain,
            actual_abstention=actual_abstain,
            selected_sources=case.required_retrieval_sources if mode == "agentic" else [],
            workflow_status=status.value,
            latency_ms_total=latency,
            llm_calls=0,
            scores=scores,
            metadata={"category": case.category, "split": case.split.value},
        )
    except Exception as exc:
        return EvalResult(
            case_id=case.case_id,
            execution_status=ExecutionStatus.FAILED,
            runner_mode=f"offline-{mode}",
            error=str(exc)[:500],
            expected_abstention=case.should_abstain,
            metadata={"category": case.category},
        )
