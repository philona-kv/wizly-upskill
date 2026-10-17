import uuid

import pytest

from app.core.exceptions import RetrievalError
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
from tests.unit.test_twin_graph import FakeConversationTool, FakeDocumentTool, FakeGenerator


@pytest.mark.asyncio
async def test_document_retrieval_failure_surfaces_error() -> None:
    gen = FakeGenerator()
    deps = WorkflowDeps(
        planner=HeuristicQueryPlanner(),
        assessor=HeuristicEvidenceAssessor(),
        generator=gen,
        document_tool=FakeDocumentTool(fail=True),  # type: ignore[arg-type]
        conversation_tool=FakeConversationTool(),  # type: ignore[arg-type]
        context_builder=ContextBuilder(max_tokens=1000, embedding_model="text-embedding-3-small"),
        answer_validator=DeterministicAnswerValidator(),
        answer_repair=AnswerRepairService(gen),  # type: ignore[arg-type]
        max_evidence_items=10,
    )
    runner = TwinWorkflowRunner(
        build_twin_graph(deps, checkpointer=None),
        max_retrieval_attempts=1,
        max_answer_repairs=0,
        enable_debug=True,
        graph_version="twin-v4",
        recovery_enabled=False,
        max_recovery_attempts=0,
    )
    with pytest.raises(RetrievalError):
        await runner.run(
            WorkflowInput(
                request_id=uuid.uuid4(),
                owner_id=uuid.uuid4(),
                conversation_id=uuid.uuid4(),
                current_user_message_id=uuid.uuid4(),
                original_question="What is the policy?",
                recent_messages=[],
            )
        )


@pytest.mark.asyncio
async def test_abstain_without_evidence() -> None:
    runner = TwinWorkflowRunner(
        build_twin_graph(
            WorkflowDeps(
                planner=HeuristicQueryPlanner(),
                assessor=HeuristicEvidenceAssessor(),
                generator=FakeGenerator(),
                document_tool=FakeDocumentTool(),  # type: ignore[arg-type]
                conversation_tool=FakeConversationTool(),  # type: ignore[arg-type]
                context_builder=ContextBuilder(
                    max_tokens=1000, embedding_model="text-embedding-3-small"
                ),
                answer_validator=DeterministicAnswerValidator(),
                answer_repair=AnswerRepairService(FakeGenerator()),  # type: ignore[arg-type]
                max_evidence_items=10,
            ),
            checkpointer=None,
        ),
        max_retrieval_attempts=2,
        max_answer_repairs=1,
        enable_debug=True,
        graph_version="twin-v4",
    )
    out = await runner.run(
        WorkflowInput(
            request_id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            current_user_message_id=uuid.uuid4(),
            original_question="Unanswerable?",
            recent_messages=[],
        )
    )
    assert out.status == WorkflowStatus.ABSTAIN
    assert INSUFFICIENT_EVIDENCE in out.answer or "enough evidence" in out.answer.lower()
