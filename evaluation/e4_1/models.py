"""Typed E4.1 diagnostic records and summaries."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from evaluation.e4.models import E4QuestionStatus


E4_1ReviewCategory = Literal[
    "EXACT_OR_ANLS_CORRECT",
    "FORMAT_MISMATCH",
    "WRONG_ANSWER",
    "VALID_CITATION_WRONG_ANSWER",
    "NO_CITATION",
    "GROUNDING_REJECTED",
    "EMPTY_ANSWER",
    "OTHER",
    "UNCLASSIFIED",
]


class E4_1QuestionRecord(BaseModel):
    """One selected-question diagnostic result with raw and metric answers."""

    model_config = ConfigDict(extra="forbid")

    question_key: str = Field(min_length=1)
    evaluation_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    accepted_answers: list[str] = Field(min_length=1)
    ground_truth_status: str = Field(min_length=1)
    target_document_id: UUID | None = None
    relevant_chunk_ids: list[UUID] = Field(default_factory=list)
    configuration: str = Field(min_length=1)
    status: E4QuestionStatus
    reason_code: str | None = None
    raw_response: str | None = None
    metric_answer: str | None = None
    model: str | None = None
    citations: list[str] = Field(default_factory=list)
    citations_valid: bool = False
    sources: list[dict[str, Any]] = Field(default_factory=list)
    raw_anls: float = Field(default=0, ge=0, le=1)
    raw_exact_match: bool = False
    metric_anls: float = Field(default=0, ge=0, le=1)
    metric_exact_match: bool = False
    citation_labels_emitted: int = Field(default=0, ge=0)
    citation_labels_valid: int = Field(default=0, ge=0)
    citation_document_hit: bool = False
    all_citations_true_document: bool | None = None
    gold_evidence_citation_hit: bool = False
    gold_evidence_citation_count: int = Field(default=0, ge=0)
    gold_evidence_citation_precision: float | None = Field(default=None, ge=0, le=1)
    answer_supported_by_cited_evidence: bool | None = None
    retrieval_time_ms: float | None = Field(default=None, ge=0)
    reranking_time_ms: float | None = Field(default=None, ge=0)
    generation_time_ms: float | None = Field(default=None, ge=0)
    total_pipeline_time_ms: float | None = Field(default=None, ge=0)
    error: str | None = None
    review_category: E4_1ReviewCategory = "UNCLASSIFIED"


class E4_1AnswerMetrics(BaseModel):
    """Mean answer metric values at one denominator level."""

    anls: float | None = Field(default=None, ge=0, le=1)
    exact_match: float | None = Field(default=None, ge=0, le=1)


class E4_1ConfigurationSummary(BaseModel):
    """Aggregate diagnostic result for one retrieval configuration."""

    model_config = ConfigDict(extra="forbid")

    configuration: str = Field(min_length=1)
    search_mode: str = Field(min_length=1)
    rerank: bool
    top_k: int = Field(gt=0)
    questions_attempted: int = Field(ge=0)
    questions_scorable: int = Field(ge=0)
    questions_answered: int = Field(ge=0)
    questions_failed: int = Field(ge=0)
    completion_rate_scorable: float | None = Field(default=None, ge=0, le=1)
    raw_end_to_end: E4_1AnswerMetrics
    raw_scorable: E4_1AnswerMetrics
    metric_end_to_end: E4_1AnswerMetrics
    metric_scorable: E4_1AnswerMetrics
    citation: dict[str, float | None] = Field(default_factory=dict)
    latency: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    failures: dict[str, int] = Field(default_factory=dict)
    review_categories: dict[str, int] = Field(default_factory=dict)


class E4_1RunSummary(BaseModel):
    """Top-level E4.1 artifact summary."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "e4_1.v1"
    status: Literal["completed", "CONTROLLED_FAILURE"]
    dataset: str = "docvqa"
    split: str
    documents_requested: int = Field(ge=0)
    documents_prepared: int = Field(ge=0)
    documents_indexed: int = Field(ge=0)
    questions_available: int = Field(ge=0)
    questions_scorable_available: int = Field(ge=0)
    questions_selected: int = Field(ge=0)
    configurations: dict[str, E4_1ConfigurationSummary] = Field(default_factory=dict)
    production_timeout_baseline: dict[str, Any] = Field(default_factory=dict)
    warmups: dict[str, Any] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)

