import uuid
from dataclasses import dataclass, field

import pytest

from app.services.answer_repair import AnswerRepairService
from app.services.answer_validator import DeterministicAnswerValidator
from app.services.context_builder import ContextBuilder
from app.services.evidence_assessor import HeuristicEvidenceAssessor
from app.services.generation_service import INSUFFICIENT_EVIDENCE
from app.services.query_planner import HeuristicQueryPlanner
from app.services.twin_workflow_runner import TwinWorkflowRunner
from app.workflows.graph import build_twin_graph
from app.workflows.nodes import WorkflowDeps
from app.workflows.schemas import WorkflowInput, WorkflowStatus


@dataclass
class FakeGenerator:
    model: str = "test-model"
    calls: int = 0

    async def generate(
        self, *, question: str, evidence_block: str, history, instructions=None
    ) -> str:
        self.calls += 1
        if evidence_block.strip():
            return "Answer from docs [S1]."
        return INSUFFICIENT_EVIDENCE


@dataclass
class FakeDocumentTool:
    items: list = field(default_factory=list)
    fail: bool = False

    async def search(self, *args, **kwargs):
        if self.fail:
            from app.core.exceptions import RetrievalError

            raise RetrievalError("Document retrieval failed.")
        from app.services.evidence_merger import chunks_to_evidence
        from app.services.retrieval_service import RetrievedChunk

        if not self.items:
            return []
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title="T",
            original_filename="f.pdf",
            content="48 hours notice required",
            page_start=1,
            page_end=1,
            chunk_index=0,
            distance=0.1,
        )
        return chunks_to_evidence([chunk])


@dataclass
class FakeConversationTool:
    async def search(self, *args, **kwargs):
        return []


def _runner(doc_tool: FakeDocumentTool) -> TwinWorkflowRunner:
    gen = FakeGenerator()
    deps = WorkflowDeps(
        planner=HeuristicQueryPlanner(),
        assessor=HeuristicEvidenceAssessor(),
        generator=gen,
        document_tool=doc_tool,  # type: ignore[arg-type]
        conversation_tool=FakeConversationTool(),  # type: ignore[arg-type]
        context_builder=ContextBuilder(max_tokens=1000, embedding_model="text-embedding-3-small"),
        answer_validator=DeterministicAnswerValidator(),
        answer_repair=AnswerRepairService(gen),  # type: ignore[arg-type]
        max_evidence_items=10,
    )
    graph = build_twin_graph(deps, checkpointer=None)
    return TwinWorkflowRunner(
        graph,
        max_retrieval_attempts=2,
        max_answer_repairs=1,
        enable_debug=True,
        graph_version="twin-v4",
    )


@pytest.mark.asyncio
async def test_graph_abstain_without_documents() -> None:
    runner = _runner(FakeDocumentTool())
    out = await runner.run(
        WorkflowInput(
            request_id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            current_user_message_id=uuid.uuid4(),
            original_question="What is the cancellation policy?",
            recent_messages=[],
        )
    )
    assert out.status == WorkflowStatus.ABSTAIN


@pytest.mark.asyncio
async def test_graph_success_with_evidence() -> None:
    runner = _runner(FakeDocumentTool(items=["x"]))
    out = await runner.run(
        WorkflowInput(
            request_id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            current_user_message_id=uuid.uuid4(),
            original_question="What notice is required for cancellation?",
            recent_messages=[],
        )
    )
    assert out.status == WorkflowStatus.SUCCESS
    assert out.operational_metadata is not None
