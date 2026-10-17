"""LangGraph runner for Playground queries (no chat checkpoint recovery)."""

from __future__ import annotations

import logging
import time
import uuid

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.core.exceptions import EmbeddingError, GenerationError, RetrievalError
from app.observability.llm_usage import LlmUsageRecord, get_usage_records
from app.schemas.conversation import Citation
from app.schemas.playground import PlaygroundRetrievalScope
from app.workflows.schemas import (
    CompactMessage,
    QueryPlan,
    WorkflowDebugMetadata,
    WorkflowOutput,
    WorkflowStatus,
)
from app.workflows.state import TwinGraphState

logger = logging.getLogger(__name__)


def _summarize_usage(records: list[LlmUsageRecord]) -> dict:
    return {
        "input_tokens": sum(r.input_tokens or 0 for r in records),
        "output_tokens": sum(r.output_tokens or 0 for r in records),
        "estimated_cost_usd": sum(r.estimated_cost_usd or 0.0 for r in records),
        "llm_call_count": len(records),
    }


class PlaygroundWorkflowInput(BaseModel):
    request_id: uuid.UUID
    owner_id: uuid.UUID
    original_question: str
    scope: PlaygroundRetrievalScope = PlaygroundRetrievalScope.COMBINED
    document_ids: list[uuid.UUID] = Field(default_factory=list)
    prompt_template_text: str | None = None
    include_conversation_history: bool = False
    conversation_id: uuid.UUID | None = None
    recent_messages: list[CompactMessage] = Field(default_factory=list)
    graph_version: str = "playground-v1"


class PlaygroundWorkflowRunner:
    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        max_retrieval_attempts: int,
        max_answer_repairs: int,
        enable_debug: bool,
        graph_version: str,
    ) -> None:
        self._graph = graph
        self._max_retrieval_attempts = max_retrieval_attempts
        self._max_answer_repairs = max_answer_repairs
        self._enable_debug = enable_debug
        self._graph_version = graph_version

    def _initial_state(self, workflow_input: PlaygroundWorkflowInput) -> TwinGraphState:
        conv_id = workflow_input.conversation_id or uuid.uuid4()
        return {
            "request_id": str(workflow_input.request_id),
            "owner_id": str(workflow_input.owner_id),
            "conversation_id": str(conv_id),
            "current_user_message_id": str(uuid.uuid4()),
            "original_question": workflow_input.original_question,
            "recent_messages": [
                {"id": str(m.id), "role": m.role, "content": m.content[:4000]}
                for m in workflow_input.recent_messages
            ],
            "retrieval_attempt": 1,
            "max_retrieval_attempts": self._max_retrieval_attempts,
            "max_answer_repairs": self._max_answer_repairs,
            "answer_repair_count": 0,
            "enable_debug": self._enable_debug,
            "graph_version": workflow_input.graph_version or self._graph_version,
            "document_evidence": [],
            "conversation_evidence": [],
            "merged_evidence": [],
            "retrieved_memories": [],
            "context_bundle": "",
            "conversation_summary_text": "",
            "playground_mode": True,
            "retrieval_scope": workflow_input.scope.value,
            "selected_document_ids": [str(d) for d in workflow_input.document_ids],
            "generation_instructions": None,
            "include_conversation_history": workflow_input.include_conversation_history,
            "prompt_template_text": workflow_input.prompt_template_text,
        }

    async def run(
        self, workflow_input: PlaygroundWorkflowInput
    ) -> tuple[WorkflowOutput, dict, dict]:
        started = time.perf_counter()
        initial = self._initial_state(workflow_input)
        config = {
            "configurable": {
                "thread_id": f"playground:{workflow_input.owner_id}:{workflow_input.request_id}",
            }
        }
        try:
            final = await self._graph.ainvoke(initial, config=config)
        except GenerationError:
            raise
        except (RetrievalError, EmbeddingError) as exc:
            raise exc if isinstance(exc, RetrievalError) else RetrievalError(str(exc.message)) from exc
        except Exception as exc:
            logger.warning("playground_workflow_failed", extra={"exc_type": type(exc).__name__})
            raise GenerationError("Answer generation failed.") from exc

        duration_ms = int((time.perf_counter() - started) * 1000)
        status = WorkflowStatus(final.get("status") or WorkflowStatus.ERROR.value)
        if status == WorkflowStatus.ERROR:
            code = final.get("error_code") or "retrieval_error"
            if code == "retrieval_error":
                raise RetrievalError(final.get("final_answer") or "Document retrieval failed.")
            raise GenerationError(final.get("final_answer") or "Workflow failed.")

        citations = [Citation.model_validate(c) for c in final.get("final_citations") or []]
        metadata: WorkflowDebugMetadata | None = None
        if self._enable_debug:
            plan = QueryPlan.model_validate(final.get("query_plan") or {})
            assessment = final.get("evidence_assessment") or {}
            metadata = WorkflowDebugMetadata(
                selected_sources=final.get("selected_sources") or [],
                retrieval_attempts=final.get("retrieval_attempt") or 1,
                retrieval_strategy=final.get("retrieval_strategy") or "",
                document_chunks=len(final.get("document_evidence") or []),
                conversation_messages=len(final.get("conversation_evidence") or []),
                memory_count=0,
                summary_version=None,
                used_persistent_context=False,
                evidence_sufficient=assessment.get("sufficient"),
                answer_repaired=(final.get("answer_repair_count") or 0) > 0,
                duration_ms=duration_ms,
                routing_summary=plan.routing_summary,
                node_timings=final.get("node_timings") or [],
                llm_call_count=len(get_usage_records()),
            )

        return WorkflowOutput(
            status=status,
            answer=final.get("final_answer") or "",
            citations=citations,
            used_llm=bool(final.get("used_llm", True)),
            model_name=final.get("model_name"),
            error_code=final.get("error_code"),
            operational_metadata=metadata,
        ), final, _summarize_usage(get_usage_records())
