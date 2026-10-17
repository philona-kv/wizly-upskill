"""Single-shot RAG baseline for evaluation (production retrieval path)."""

from __future__ import annotations

import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Message, MessageRole
from app.services.context_builder import ContextBuilder
from app.services.generation_service import Generator, INSUFFICIENT_EVIDENCE
from app.services.retrieval_service import RetrievalService
from app.workflows.schemas import RetrievalStrategy
from evaluation.metrics.answers import score_answer_metrics
from evaluation.metrics.retrieval import mrr, precision_at_k, recall_at_k
from evaluation.schemas.models import EvalCase, EvalResult, ExecutionStatus, MetricScores
from app.services.citation_validator import validate_answer


async def run_case_baseline(
    case: EvalCase,
    *,
    session_factory: async_sessionmaker[AsyncSession],
    retrieval: RetrievalService,
    generator: Generator,
    context_builder: ContextBuilder,
    owner_id: uuid.UUID,
    top_k: int = 8,
) -> EvalResult:
    started = time.perf_counter()
    try:
        async with session_factory() as session:
            chunks = await retrieval.retrieve(
                session,
                owner_id,
                case.question,
                strategy=RetrievalStrategy.SEMANTIC,
                top_k=top_k,
            )
        retrieved_ids = [str(c.chunk_id) for c in chunks]
        relevant = set(case.expected_chunk_ids)
        scores = MetricScores(
            recall_at_k=recall_at_k(relevant, retrieved_ids, top_k),
            precision_at_k=precision_at_k(relevant, retrieved_ids, top_k),
            mrr=mrr(relevant, retrieved_ids),
        )
        built = context_builder.build(chunks)
        history = []
        for turn in case.conversation_context:
            role = MessageRole.USER.value if turn.role == "user" else MessageRole.ASSISTANT.value
            history.append(
                Message(role=role, content=turn.content, conversation_id=uuid.uuid4())
            )
        answer = await generator.generate(
            question=case.question,
            evidence_block=built.prompt_block,
            history=history,
        )
        validated = validate_answer(answer, built.evidence)
        import re

        cites = re.findall(r"\[(S\d+)\]", answer)
        allowed = {item.citation_id for item in built.evidence}
        citation_valid = all(c in allowed for c in cites) if cites else not case.should_abstain
        actual_abstain = answer.strip() == INSUFFICIENT_EVIDENCE
        answer_scores = score_answer_metrics(
            case,
            answer=answer,
            actual_abstention=actual_abstain,
            citation_valid=citation_valid,
            evidence_texts=[item.excerpt for item in built.evidence],
        )
        scores = scores.model_copy(update=answer_scores.model_dump(exclude_unset=True))
        return EvalResult(
            case_id=case.case_id,
            execution_status=ExecutionStatus.SUCCESS,
            runner_mode="baseline-live",
            generated_answer=answer,
            retrieved_chunk_ids=retrieved_ids,
            citation_validation_valid=validated.valid,
            expected_abstention=case.should_abstain,
            actual_abstention=actual_abstain,
            latency_ms_total=int((time.perf_counter() - started) * 1000),
            scores=scores,
            metadata={"category": case.category},
        )
    except Exception as exc:
        return EvalResult(
            case_id=case.case_id,
            execution_status=ExecutionStatus.FAILED,
            runner_mode="baseline-live",
            error=str(exc)[:500],
            metadata={"category": case.category},
        )
