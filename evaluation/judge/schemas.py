"""Optional LLM-as-judge structured output (--live-llm only)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeScores(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    citation_support: float = Field(ge=0.0, le=1.0)
    explanation: str = ""
