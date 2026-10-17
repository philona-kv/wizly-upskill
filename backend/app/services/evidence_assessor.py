"""Evidence sufficiency assessment."""

from __future__ import annotations

import logging
from typing import Protocol

from openai import AsyncOpenAI

from app.services.structured_llm import parse_model
from app.workflows.schemas import EvidenceAssessment, EvidenceItem, EvidenceSourceType

logger = logging.getLogger(__name__)

ASSESSOR_INSTRUCTIONS = """Decide whether retrieved evidence can support answering the question.
High vector similarity alone is not sufficient.

Return JSON with sufficient, missing_information, supported_aspects,
suggested_strategy (none when sufficient), and a brief internal explanation.
Do not answer the user question."""


class EvidenceAssessor(Protocol):
    async def assess(
        self,
        *,
        question: str,
        standalone_query: str,
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment: ...


def heuristic_assessment(
    question: str,
    evidence: list[EvidenceItem],
) -> EvidenceAssessment:
    doc_items = [e for e in evidence if e.source_type == EvidenceSourceType.DOCUMENT]
    if not doc_items:
        return EvidenceAssessment(
            sufficient=False,
            missing_information=["No document passages retrieved."],
            suggested_strategy="broaden_semantic_search",
            explanation="No document evidence.",
        )
    combined = " ".join(item.content.lower() for item in doc_items)
    tokens = [t for t in question.lower().split() if len(t) > 3][:6]
    hits = sum(1 for t in tokens if t in combined)
    if hits >= max(1, len(tokens) // 3):
        return EvidenceAssessment(
            sufficient=True,
            supported_aspects=["Retrieved passages appear relevant."],
            suggested_strategy="none",
            explanation="Heuristic relevance match.",
        )
    return EvidenceAssessment(
        sufficient=False,
        missing_information=["Retrieved passages may not cover the question."],
        suggested_strategy="rewrite_query",
        explanation="Weak term overlap.",
    )


class OpenAIEvidenceAssessor:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def assess(
        self,
        *,
        question: str,
        standalone_query: str,
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment:
        snippets = []
        for item in evidence[:12]:
            label = item.source_type.value
            snippets.append(f"[{label}] {item.content[:400]}")
        user_input = (
            f"Question: {question}\nStandalone query: {standalone_query}\n\nEvidence:\n"
            + "\n".join(snippets)
        )
        try:
            return await parse_model(
                self._client,
                self._model,
                instructions=ASSESSOR_INSTRUCTIONS,
                user_input=user_input,
                schema=EvidenceAssessment,
                temperature=None,
            )
        except Exception as exc:
            logger.warning("evidence_assess_failed", extra={"exc_type": type(exc).__name__})
            return heuristic_assessment(question, evidence)


class HeuristicEvidenceAssessor:
    async def assess(
        self,
        *,
        question: str,
        standalone_query: str,
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment:
        del standalone_query
        return heuristic_assessment(question, evidence)
