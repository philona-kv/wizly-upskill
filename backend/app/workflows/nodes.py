"""LangGraph node implementations."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace

from app.core.exceptions import GenerationError
from app.services.answer_repair import AnswerRepairService
from app.services.answer_validator import DeterministicAnswerValidator
from app.services.citation_validator import validate_answer
from app.services.context_builder import ContextBuilder
from app.services.conversation_context_builder import ConversationContextBuilder
from app.services.conversation_summary_service import ConversationSummaryService
from app.services.evidence_assessor import EvidenceAssessor
from app.services.evidence_merger import (
    assign_citation_ids,
    build_prompt_blocks,
    merge_evidence_lists,
)
from app.schemas.playground import PlaygroundRetrievalScope
from app.services.generation_service import (
    INSUFFICIENT_EVIDENCE,
    PLAYGROUND_SAFETY_PREFIX,
    Generator,
)
from app.services.prompt_renderer import render_prompt_template
from app.services.retrieval_service import RetrievalContext
from app.services.memory_service import MemoryService
from app.services.query_planner import QueryPlanner
from app.workflows.schemas import (
    AnswerValidation,
    CompactMessage,
    EvidenceAssessment,
    EvidenceItem,
    EvidenceSourceType,
    QueryPlan,
    QuerySource,
    RetrievalStrategy,
    WorkflowStatus,
    assessment_to_strategy,
)
from app.workflows.state import TwinGraphState
from app.workflows.tools import ConversationHistoryTool, DocumentSearchTool, parallel_retrieve

logger = logging.getLogger(__name__)


@dataclass
class WorkflowDeps:
    planner: QueryPlanner
    assessor: EvidenceAssessor
    generator: Generator
    document_tool: DocumentSearchTool
    conversation_tool: ConversationHistoryTool
    context_builder: ContextBuilder
    answer_validator: DeterministicAnswerValidator = field(
        default_factory=DeterministicAnswerValidator
    )
    answer_repair: AnswerRepairService | None = None
    max_evidence_items: int = 24
    memory_service: MemoryService | None = None
    summary_service: ConversationSummaryService | None = None
    conversation_context_builder: ConversationContextBuilder | None = None


def _retrieval_context_from_state(state: TwinGraphState) -> RetrievalContext:
    if not state.get("playground_mode"):
        return RetrievalContext.twin()
    scope_raw = state.get("retrieval_scope") or PlaygroundRetrievalScope.COMBINED.value
    scope = PlaygroundRetrievalScope(scope_raw)
    doc_ids = [uuid.UUID(x) for x in state.get("selected_document_ids") or []]
    return RetrievalContext.playground(scope, document_ids=doc_ids)


def _log_node(state: TwinGraphState, node: str) -> None:
    logger.info(
        "workflow_node",
        extra={
            "graph_node": node,
            "request_id": state.get("request_id"),
            "conversation_id": state.get("conversation_id"),
            "retrieval_attempt": state.get("retrieval_attempt"),
        },
    )


def make_nodes(deps: WorkflowDeps):
    async def load_conversation_context(state: TwinGraphState) -> dict:
        _log_node(state, "load_conversation_context")
        return {}

    async def load_conversation_summary(state: TwinGraphState) -> dict:
        _log_node(state, "load_conversation_summary")
        if deps.summary_service is None:
            return {"conversation_summary_text": "", "conversation_summary_version": None}
        owner_id = uuid.UUID(state["owner_id"])
        conversation_id = uuid.UUID(state["conversation_id"])
        summary = await deps.summary_service.get_summary(owner_id, conversation_id)
        if summary is None:
            return {"conversation_summary_text": "", "conversation_summary_version": None}
        return {
            "conversation_summary_text": summary.summary_text,
            "conversation_summary_version": summary.version,
        }

    async def retrieve_relevant_memories(state: TwinGraphState) -> dict:
        _log_node(state, "retrieve_relevant_memories")
        if deps.memory_service is None:
            return {"retrieved_memories": []}
        owner_id = uuid.UUID(state["owner_id"])
        query = state.get("standalone_query") or state["original_question"]
        hits = await deps.memory_service.search(owner_id, query)
        return {
            "retrieved_memories": [
                {
                    "id": str(m.id),
                    "memory_type": m.memory_type,
                    "content": m.content[:500],
                    "source_type": m.source_type,
                    "source_conversation_id": str(m.source_conversation_id)
                    if m.source_conversation_id
                    else None,
                }
                for m in hits
            ]
        }

    async def resolve_contextual_query(state: TwinGraphState) -> dict:
        _log_node(state, "resolve_contextual_query")
        if state.get("playground_mode") and not state.get("include_conversation_history"):
            return {"context_bundle": "", "standalone_query": state["original_question"]}
        messages = [_compact(m) for m in state.get("recent_messages") or []]
        bundle = ""
        if deps.conversation_context_builder is not None:
            from app.services.memory_service import RetrievedMemory

            mems = [
                RetrievedMemory(
                    id=uuid.UUID(m["id"]),
                    memory_type=m["memory_type"],
                    content=m["content"],
                    source_type=m["source_type"],
                    source_conversation_id=uuid.UUID(m["source_conversation_id"])
                    if m.get("source_conversation_id")
                    else None,
                )
                for m in state.get("retrieved_memories") or []
            ]
            assembled = deps.conversation_context_builder.build(
                question=state["original_question"],
                recent_messages=messages,
                summary_text=state.get("conversation_summary_text") or None,
                memories=mems,
                exclude_message_id=state.get("current_user_message_id"),
            )
            bundle = assembled.combined_untrusted
        return {"context_bundle": bundle, "standalone_query": state["original_question"]}

    async def plan_query(state: TwinGraphState) -> dict:
        _log_node(state, "plan_query")
        messages = [_compact(m) for m in state.get("recent_messages") or []]
        plan = await deps.planner.plan(
            question=state["original_question"],
            recent_messages=messages,
            context_hint=state.get("context_bundle") or "",
        )
        sources = list(plan.sources)
        if state.get("playground_mode"):
            if not state.get("include_conversation_history"):
                sources = [s for s in sources if s == QuerySource.DOCUMENTS.value]
                if QuerySource.DOCUMENTS.value not in sources:
                    sources = [QuerySource.DOCUMENTS.value]
            scope_raw = state.get("retrieval_scope") or ""
            if scope_raw == PlaygroundRetrievalScope.SELECTED.value:
                sources = [QuerySource.DOCUMENTS.value]
        return {
            "query_plan": plan.model_dump(mode="json"),
            "standalone_query": plan.standalone_query,
            "selected_sources": sources,
            "retrieval_strategy": RetrievalStrategy.HYBRID.value,
            "retrieval_attempt": state.get("retrieval_attempt") or 1,
        }

    async def parallel_retrieval(state: TwinGraphState) -> dict:
        _log_node(state, "parallel_retrieval")
        owner_id = uuid.UUID(state["owner_id"])
        conversation_id = uuid.UUID(state["conversation_id"])
        query = (
            state.get("rewritten_query")
            or state.get("standalone_query")
            or state["original_question"]
        )
        strat_raw = state.get("retrieval_strategy") or RetrievalStrategy.SEMANTIC.value
        strategy = RetrievalStrategy(strat_raw)
        sources = state.get("selected_sources") or [QuerySource.DOCUMENTS.value]
        messages = [_compact(m) for m in state.get("recent_messages") or []]
        prior = [EvidenceItem.model_validate(x) for x in state.get("document_evidence") or []]
        focus = [uuid.UUID(x) for x in state.get("focus_chunk_ids") or []]
        doc_items, conv_items, doc_err, conv_err = await parallel_retrieve(
            sources=sources,
            document_tool=deps.document_tool,
            conversation_tool=deps.conversation_tool,
            owner_id=owner_id,
            conversation_id=conversation_id,
            query=query,
            strategy=strategy,
            retrieval_attempt=state.get("retrieval_attempt") or 1,
            recent_messages=messages,
            focus_chunk_ids=focus,
            prior_document_evidence=prior,
            retrieval_context=_retrieval_context_from_state(state),
        )
        return {
            "document_evidence": [i.model_dump(mode="json") for i in doc_items],
            "conversation_evidence": [i.model_dump(mode="json") for i in conv_items],
            "document_retrieval_error": doc_err,
            "conversation_retrieval_error": conv_err,
        }

    async def merge_evidence(state: TwinGraphState) -> dict:
        _log_node(state, "merge_evidence")
        doc = [EvidenceItem.model_validate(x) for x in state.get("document_evidence") or []]
        conv = [EvidenceItem.model_validate(x) for x in state.get("conversation_evidence") or []]
        doc = assign_citation_ids([i for i in doc if i.source_type == EvidenceSourceType.DOCUMENT])
        conv = [i for i in conv if i.source_type == EvidenceSourceType.CONVERSATION]
        merged = merge_evidence_lists(doc, conv, max_items=deps.max_evidence_items)
        prompt, conv_block, _ctx, _built = build_prompt_blocks(deps.context_builder, merged)
        return {
            "merged_evidence": [i.model_dump(mode="json") for i in merged],
            "evidence_prompt_block": prompt,
            "conversation_context_block": conv_block,
        }

    async def assess_evidence(state: TwinGraphState) -> dict:
        _log_node(state, "assess_evidence")
        merged = [EvidenceItem.model_validate(x) for x in state.get("merged_evidence") or []]
        assessment = await deps.assessor.assess(
            question=state["original_question"],
            standalone_query=state.get("standalone_query") or state["original_question"],
            evidence=merged,
        )
        return {"evidence_assessment": assessment.model_dump(mode="json")}

    async def plan_retrieval_retry(state: TwinGraphState) -> dict:
        _log_node(state, "plan_retrieval_retry")
        assessment = EvidenceAssessment.model_validate(state.get("evidence_assessment") or {})
        strategy = assessment_to_strategy(assessment) or RetrievalStrategy.BROADEN_SEMANTIC
        attempt = (state.get("retrieval_attempt") or 1) + 1
        update: dict = {
            "retrieval_attempt": attempt,
            "retrieval_strategy": strategy.value,
            "document_evidence": [],
            "conversation_evidence": [],
        }
        if assessment.suggested_strategy == "rewrite_query":
            update["rewritten_query"] = state.get("standalone_query") or state["original_question"]
        merged = [EvidenceItem.model_validate(x) for x in state.get("merged_evidence") or []]
        focus = [str(i.chunk_id) for i in merged if i.chunk_id][:8]
        if strategy in (RetrievalStrategy.NEIGHBOR_EXPAND, RetrievalStrategy.SECTION_FOCUS):
            update["focus_chunk_ids"] = focus
        return update

    async def generate_answer(state: TwinGraphState) -> dict:
        _log_node(state, "generate_answer")
        prompt = state.get("evidence_prompt_block") or ""
        conv = state.get("conversation_context_block") or ""
        ctx = state.get("context_bundle") or ""
        parts = [p for p in (ctx, conv, prompt) if p.strip()]
        combined = "\n\n".join(parts)
        if not combined.strip():
            return {
                "answer_draft": INSUFFICIENT_EVIDENCE,
                "used_llm": False,
                "model_name": None,
            }
        history = _history_from_state(state)
        instructions = state.get("generation_instructions")
        template_text = state.get("prompt_template_text")
        if template_text:
            task = render_prompt_template(
                template_text,
                question=state["original_question"],
                retrieved_context=combined,
                conversation_context=conv,
                citation_instructions="Cite supporting sources using exact identifiers from the evidence (e.g. [S1]).",
            )
            instructions = PLAYGROUND_SAFETY_PREFIX + task
        try:
            text = await deps.generator.generate(
                question=state["original_question"],
                evidence_block=combined,
                history=history,
                instructions=instructions,
            )
        except GenerationError:
            raise
        return {
            "answer_draft": text,
            "used_llm": True,
            "model_name": deps.generator.model,
        }

    async def validate_answer_node(state: TwinGraphState) -> dict:
        _log_node(state, "validate_answer")
        draft = state.get("answer_draft") or ""
        merged = [EvidenceItem.model_validate(x) for x in state.get("merged_evidence") or []]
        doc_items = [i for i in merged if i.source_type == EvidenceSourceType.DOCUMENT]
        _, _, ctx_evidence, _ = build_prompt_blocks(deps.context_builder, merged)
        result = deps.answer_validator.validate(
            answer=draft,
            document_evidence=doc_items,
            doc_context_evidence=ctx_evidence,
        )
        return {"validation_result": result.model_dump(mode="json")}

    async def repair_answer(state: TwinGraphState) -> dict:
        _log_node(state, "repair_answer")
        if deps.answer_repair is None:
            return {"answer_repair_count": (state.get("answer_repair_count") or 0) + 1}
        validation = AnswerValidation.model_validate(state.get("validation_result") or {})
        history = _history_from_state(state)
        text = await deps.answer_repair.repair(
            question=state["original_question"],
            evidence_block=state.get("evidence_prompt_block") or "",
            conversation_block=state.get("conversation_context_block") or "",
            draft=state.get("answer_draft") or "",
            validation=validation,
            history=history,
        )
        return {
            "answer_draft": text,
            "answer_repair_count": (state.get("answer_repair_count") or 0) + 1,
        }

    async def finalize_success(state: TwinGraphState) -> dict:
        _log_node(state, "finalize_success")
        draft = state.get("answer_draft") or ""
        merged = [EvidenceItem.model_validate(x) for x in state.get("merged_evidence") or []]
        _, _, ctx_evidence, _ = build_prompt_blocks(deps.context_builder, merged)
        validated = validate_answer(draft, ctx_evidence)
        citations = [c.model_dump(mode="json") for c in validated.citations]
        return {
            "status": WorkflowStatus.SUCCESS.value,
            "final_answer": validated.content,
            "final_citations": citations,
            "used_llm": state.get("used_llm", True),
            "model_name": state.get("model_name"),
        }

    async def finalize_abstain(state: TwinGraphState) -> dict:
        _log_node(state, "finalize_abstain")
        return {
            "status": WorkflowStatus.ABSTAIN.value,
            "final_answer": INSUFFICIENT_EVIDENCE,
            "final_citations": [],
            "used_llm": False,
            "model_name": None,
        }

    async def finalize_clarification(state: TwinGraphState) -> dict:
        _log_node(state, "finalize_clarification")
        plan = QueryPlan.model_validate(state.get("query_plan") or {})
        prompt = plan.clarification_prompt or "Could you clarify your question?"
        return {
            "status": WorkflowStatus.CLARIFY.value,
            "final_answer": prompt,
            "final_citations": [],
            "used_llm": False,
            "model_name": None,
        }

    async def finalize_error(state: TwinGraphState) -> dict:
        _log_node(state, "finalize_error")
        code = "retrieval_error"
        msg = state.get("document_retrieval_error") or "Document retrieval failed."
        return {
            "status": WorkflowStatus.ERROR.value,
            "error_code": code,
            "final_answer": msg,
            "final_citations": [],
            "used_llm": False,
        }

    async def finalize_safe_fallback(state: TwinGraphState) -> dict:
        _log_node(state, "finalize_safe_fallback")
        draft = state.get("answer_draft") or ""
        merged = [EvidenceItem.model_validate(x) for x in state.get("merged_evidence") or []]
        _, _, ctx_evidence, _ = build_prompt_blocks(deps.context_builder, merged)
        validated = validate_answer(draft, ctx_evidence)
        return {
            "status": WorkflowStatus.SUCCESS.value,
            "final_answer": validated.content or INSUFFICIENT_EVIDENCE,
            "final_citations": [c.model_dump(mode="json") for c in validated.citations],
            "used_llm": state.get("used_llm", True),
            "model_name": state.get("model_name"),
        }

    raw = {
        "load_conversation_context": load_conversation_context,
        "load_conversation_summary": load_conversation_summary,
        "retrieve_relevant_memories": retrieve_relevant_memories,
        "resolve_contextual_query": resolve_contextual_query,
        "plan_query": plan_query,
        "parallel_retrieval": parallel_retrieval,
        "merge_evidence": merge_evidence,
        "assess_evidence": assess_evidence,
        "plan_retrieval_retry": plan_retrieval_retry,
        "generate_answer": generate_answer,
        "validate_answer": validate_answer_node,
        "repair_answer": repair_answer,
        "finalize_success": finalize_success,
        "finalize_abstain": finalize_abstain,
        "finalize_clarification": finalize_clarification,
        "finalize_error": finalize_error,
        "finalize_safe_fallback": finalize_safe_fallback,
    }

    def _timed(name: str, fn):
        async def wrapped(state: TwinGraphState) -> dict:
            started = time.perf_counter()
            result = await fn(state)
            ms = int((time.perf_counter() - started) * 1000)
            timings = list(state.get("node_timings") or [])
            timings.append({"node": name, "duration_ms": ms})
            merged = dict(result or {})
            merged["node_timings"] = timings
            return merged

        return wrapped

    return {name: _timed(name, fn) for name, fn in raw.items()}


def _compact(raw: dict) -> CompactMessage:
    return CompactMessage(id=uuid.UUID(raw["id"]), role=raw["role"], content=raw["content"])


def _history_from_state(state: TwinGraphState) -> list[SimpleNamespace]:
    history: list[SimpleNamespace] = []
    current_id = state.get("current_user_message_id")
    for raw in state.get("recent_messages") or []:
        if raw.get("id") == current_id:
            continue
        history.append(SimpleNamespace(role=raw["role"], content=raw["content"]))
    return history
