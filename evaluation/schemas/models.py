"""Pydantic models for evaluation cases, results, and reports."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EvalSplit(StrEnum):
    DEV = "dev"
    HOLDOUT = "holdout"


class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class MemoryFactLabel(BaseModel):
    memory_type: str = ""
    content_substring: str = Field(min_length=1, max_length=500)
    source_type: str | None = None


class EvalCase(BaseModel):
    case_id: str = Field(min_length=1, max_length=128)
    category: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=4000)
    expected_answer: str | None = None
    expected_key_facts: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)
    chunk_relevance_grades: dict[str, float] = Field(default_factory=dict)
    required_retrieval_sources: list[str] = Field(default_factory=list)
    should_abstain: bool = False
    relevant_memory_facts: list[MemoryFactLabel] = Field(default_factory=list)
    conversation_context: list[ConversationTurn] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    split: EvalSplit = EvalSplit.DEV
    fixture_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    offline_only: bool = False


class MetricScores(BaseModel):
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    ndcg_at_k: float | None = None
    document_id_hit: bool | None = None
    page_hit: bool | None = None
    key_fact_recall: float | None = None
    abstention_correct: bool | None = None
    citation_valid: bool | None = None
    faithfulness_offline: float | None = None
    source_selection_correct: bool | None = None
    memory_recall: float | None = None
    judge_relevance: float | None = None
    judge_faithfulness: float | None = None
    judge_citation_support: float | None = None


class LlmUsageSnapshot(BaseModel):
    operation: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    duration_ms: int = 0
    success: bool = True
    is_token_estimate: bool = False
    estimated_cost_usd: float | None = None


class EvalResult(BaseModel):
    case_id: str
    execution_status: ExecutionStatus
    runner_mode: str = "offline"
    generated_answer: str = ""
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    retrieved_pages: list[int] = Field(default_factory=list)
    generated_citations: list[dict[str, Any]] = Field(default_factory=list)
    citation_validation_valid: bool | None = None
    expected_abstention: bool = False
    actual_abstention: bool = False
    selected_sources: list[str] = Field(default_factory=list)
    retrieval_attempts: int = 0
    workflow_status: str | None = None
    latency_ms_total: int = 0
    node_timings_ms: dict[str, int] = Field(default_factory=dict)
    llm_calls: int = 0
    llm_usage: list[LlmUsageSnapshot] = Field(default_factory=list)
    estimated_cost_usd: float | None = None
    cost_available: bool = False
    scores: MetricScores = Field(default_factory=MetricScores)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    run_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_commit: str | None = None
    dataset_path: str = ""
    dataset_version: str = "core-v1"
    mode: str = "offline"
    live_llm: bool = False
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    evaluator_prompt_version: str | None = None
    k_values: list[int] = Field(default_factory=lambda: [8])
    case_count: int = 0
    success_count: int = 0
    failed_count: int = 0


class CategoryAggregate(BaseModel):
    category: str
    count: int
    metrics: dict[str, float | None] = Field(default_factory=dict)


class RunReport(BaseModel):
    metadata: RunMetadata
    results: list[EvalResult] = Field(default_factory=list)
    aggregates: dict[str, float | None] = Field(default_factory=dict)
    aggregates_by_category: list[CategoryAggregate] = Field(default_factory=list)
    latency_ms: dict[str, float | None] = Field(default_factory=dict)


class CompareReport(BaseModel):
    baseline_path: str
    candidate_path: str
    metric_deltas: dict[str, float | None] = Field(default_factory=dict)
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
