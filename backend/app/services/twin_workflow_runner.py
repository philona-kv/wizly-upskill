"""Invoke the compiled LangGraph workflow for a chat turn."""

from __future__ import annotations

import asyncio
import logging
import time

from langgraph.graph.state import CompiledStateGraph

from app.core.exceptions import EmbeddingError, GenerationError, RetrievalError
from app.schemas.conversation import Citation
from app.workflows.schemas import (
    QueryPlan,
    WorkflowDebugMetadata,
    WorkflowInput,
    WorkflowOutput,
    WorkflowStatus,
)
from app.observability.llm_usage import get_usage_records
from app.workflows.state import TwinGraphState

logger = logging.getLogger(__name__)


class TwinWorkflowRunner:
    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        max_retrieval_attempts: int,
        max_answer_repairs: int,
        enable_debug: bool,
        graph_version: str,
        recovery_enabled: bool = False,
        max_recovery_attempts: int = 0,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._graph = graph
        self._max_retrieval_attempts = max_retrieval_attempts
        self._max_answer_repairs = max_answer_repairs
        self._enable_debug = enable_debug
        self._graph_version = graph_version
        self._recovery_enabled = recovery_enabled
        self._max_recovery_attempts = max_recovery_attempts
        self._retry_backoff_seconds = retry_backoff_seconds

    def _thread_config(self, workflow_input: WorkflowInput) -> dict:
        thread_id = f"{workflow_input.owner_id}:{workflow_input.conversation_id}"
        return {"configurable": {"thread_id": thread_id}}

    def _initial_state(self, workflow_input: WorkflowInput) -> TwinGraphState:
        return {
            "request_id": str(workflow_input.request_id),
            "owner_id": str(workflow_input.owner_id),
            "conversation_id": str(workflow_input.conversation_id),
            "current_user_message_id": str(workflow_input.current_user_message_id),
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
        }

    async def _maybe_resume(self, config: dict, graph_version: str) -> bool:
        try:
            snapshot = await self._graph.aget_state(config)
        except Exception:
            return False
        if not snapshot or not snapshot.next:
            return False
        values = snapshot.values or {}
        return values.get("graph_version") == graph_version

    async def _invoke_once(
        self, initial: TwinGraphState, config: dict, *, resume: bool
    ) -> dict:
        if resume:
            return await self._graph.ainvoke(None, config=config)
        return await self._graph.ainvoke(initial, config=config)

    async def run(self, workflow_input: WorkflowInput) -> WorkflowOutput:
        started = time.perf_counter()
        initial = self._initial_state(workflow_input)
        config = self._thread_config(workflow_input)
        graph_version = initial["graph_version"] or self._graph_version
        attempts = 1 + (
            self._max_recovery_attempts if self._recovery_enabled else 0
        )
        resume = False
        if self._recovery_enabled:
            resume = await self._maybe_resume(config, graph_version)

        final: dict | None = None
        last_exc: Exception | None = None
        for attempt in range(attempts):
            if attempt > 0:
                await asyncio.sleep(self._retry_backoff_seconds * attempt)
                resume = await self._maybe_resume(config, graph_version)
            try:
                final = await self._invoke_once(initial, config, resume=resume)
                break
            except GenerationError:
                raise
            except (RetrievalError, EmbeddingError) as exc:
                last_exc = exc
                resume = False
            except Exception as exc:
                logger.warning("workflow_failed", extra={"exc_type": type(exc).__name__})
                last_exc = exc
                resume = False
        if final is None:
            if isinstance(last_exc, RetrievalError):
                raise last_exc
            if isinstance(last_exc, EmbeddingError):
                raise RetrievalError(str(last_exc.message)) from last_exc
            raise GenerationError("Answer generation failed.") from last_exc

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
            memories = final.get("retrieved_memories") or []
            ctx = final.get("context_bundle") or ""
            metadata = WorkflowDebugMetadata(
                selected_sources=final.get("selected_sources") or [],
                retrieval_attempts=final.get("retrieval_attempt") or 1,
                retrieval_strategy=final.get("retrieval_strategy") or "",
                document_chunks=len(final.get("document_evidence") or []),
                conversation_messages=len(final.get("conversation_evidence") or []),
                memory_count=len(memories),
                summary_version=final.get("conversation_summary_version"),
                used_persistent_context=bool(ctx.strip() or memories),
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
        )
