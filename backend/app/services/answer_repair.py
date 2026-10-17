"""Bounded answer repair when validation fails."""

from __future__ import annotations

from app.services.generation_service import Generator
from app.workflows.schemas import AnswerValidation


class AnswerRepairService:
    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    async def repair(
        self,
        *,
        question: str,
        evidence_block: str,
        conversation_block: str,
        draft: str,
        validation: AnswerValidation,
        history,
    ) -> str:
        errors = validation.invalid_citations + validation.unsupported_claims
        err_text = "; ".join(errors) if errors else validation.explanation
        combined = evidence_block
        if conversation_block:
            combined = conversation_block + "\n\n" + combined
        repair_prompt = (
            f"{combined}\n\n"
            f"The previous answer had validation errors: {err_text}\n"
            f"Previous answer:\n{draft}\n\n"
            "Provide a corrected answer using ONLY the evidence above. "
            "Use valid citation IDs only. Do not invent facts."
        )
        return await self._generator.generate(
            question=f"{question}\n\n{repair_prompt}",
            evidence_block="",
            history=history,
        )
