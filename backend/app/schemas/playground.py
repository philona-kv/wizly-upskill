"""Playground API schemas and retrieval scope."""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class PlaygroundRetrievalScope(StrEnum):
    GLOBAL_ONLY = "global_only"
    PRIVATE_ONLY = "private_only"
    COMBINED = "combined"
    SELECTED = "selected"


class PlaygroundQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    scope: PlaygroundRetrievalScope = PlaygroundRetrievalScope.COMBINED
    document_ids: list[uuid.UUID] = Field(default_factory=list)
    prompt_version_id: uuid.UUID | None = None
    include_conversation_history: bool = False
    conversation_id: uuid.UUID | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    @field_validator("document_ids")
    @classmethod
    def selected_requires_ids(cls, v: list[uuid.UUID], info) -> list[uuid.UUID]:
        scope = info.data.get("scope")
        if scope == PlaygroundRetrievalScope.SELECTED and not v:
            raise ValueError("document_ids is required when scope is 'selected'.")
        return v


class RetrievedEvidenceRead(BaseModel):
    evidence_id: str
    document_id: uuid.UUID | None = None
    chunk_id: uuid.UUID | None = None
    document_title: str | None = None
    original_filename: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    retrieval_score: float | None = None
    excerpt: str | None = None


class PlaygroundQueryResponse(BaseModel):
    status: str
    answer: str
    citations: list[dict] = Field(default_factory=list)
    retrieval_scope: PlaygroundRetrievalScope
    document_evidence: list[RetrievedEvidenceRead] = Field(default_factory=list)
    operational_metadata: dict | None = None
    token_usage: dict = Field(default_factory=dict)
    model_name: str | None = None


class PromptTemplateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    template_text: str = Field(min_length=1)


class PromptTemplateVersionCreate(BaseModel):
    template_text: str = Field(min_length=1)


class PromptTemplateRead(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    current_version: int
    is_system: bool
    owner_id: uuid.UUID | None = None

    model_config = {"from_attributes": True}


class PromptTemplateVersionRead(BaseModel):
    id: uuid.UUID
    template_id: uuid.UUID
    version: int
    template_text: str
    config: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class PromptTemplateList(BaseModel):
    items: list[PromptTemplateRead]
    total: int
    limit: int
    offset: int


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class ExperimentRunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    scope: PlaygroundRetrievalScope = PlaygroundRetrievalScope.COMBINED
    document_ids: list[uuid.UUID] = Field(default_factory=list)
    prompt_version_ids: list[uuid.UUID] = Field(min_length=1)


class ExperimentRunRead(BaseModel):
    id: uuid.UUID
    experiment_id: uuid.UUID
    prompt_version_id: uuid.UUID | None
    retrieval_scope: str
    input_question: str
    answer: str | None
    citations: list = Field(default_factory=list)
    latency_ms: int | None
    token_usage: dict = Field(default_factory=dict)
    estimated_cost_usd: float | None
    status: str
    metrics: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class EvaluationRunRequest(BaseModel):
    dataset_path: str = Field(default="evaluation/datasets/core.jsonl")
    scope: PlaygroundRetrievalScope = PlaygroundRetrievalScope.GLOBAL_ONLY
    document_ids: list[uuid.UUID] = Field(default_factory=list)
    prompt_version_id: uuid.UUID | None = None
    limit: int | None = Field(default=10, ge=1, le=100)


class EvaluationRunResponse(BaseModel):
    run_id: uuid.UUID
    case_count: int
    aggregate_metrics: dict = Field(default_factory=dict)
    results: list[dict] = Field(default_factory=list)
