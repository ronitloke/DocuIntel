"""Typed E4 per-question records, summaries, and run metadata."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


E4QuestionStatus = Literal[
    "ANSWERED",
    "ABSTAINED",
    "ANSWER_NOT_INDEXED",
    "DOCUMENT_PROCESSING_FAILED",
    "RETRIEVAL_NO_RELEVANT_EVIDENCE",
    "GENERATION_FAILED",
    "GROUNDING_REJECTED",
    "INVALID_GROUND_TRUTH",
    "OTHER_CONTROLLED_REASON",
]


class E4LatencyStats(BaseModel):
    """Mean, median, and inclusive interpolated P95 for one stage."""

    mean_ms: float | None = Field(default=None, ge=0)
    median_ms: float | None = Field(default=None, ge=0)
    p95_ms: float | None = Field(default=None, ge=0)
    samples: int = Field(default=0, ge=0)


class E4AnswerMetrics(BaseModel):
    """Answer quality at one denominator level."""

    anls: float | None = Field(default=None, ge=0, le=1)
    exact_match: float | None = Field(default=None, ge=0, le=1)


class E4CitationMetrics(BaseModel):
    """Deterministic citation and evidence metrics."""

    citation_presence_rate: float | None = Field(default=None, ge=0, le=1)
    citation_reference_validity_rate: float | None = Field(default=None, ge=0, le=1)
    citation_document_hit_rate: float | None = Field(default=None, ge=0, le=1)
    all_citations_true_document_rate: float | None = Field(default=None, ge=0, le=1)
    gold_evidence_citation_hit_rate: float | None = Field(default=None, ge=0, le=1)
    gold_evidence_citation_precision: float | None = Field(default=None, ge=0, le=1)
    answer_supported_by_cited_evidence_rate: float | None = Field(default=None, ge=0, le=1)


class E4QuestionRecord(BaseModel):
    """One complete, bounded evaluation outcome for one configuration."""

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
    answer: str | None = None
    model: str | None = None
    citations: list[str] = Field(default_factory=list)
    citations_valid: bool = False
    sources: list[dict[str, Any]] = Field(default_factory=list)
    anls: float = Field(default=0, ge=0, le=1)
    exact_match: bool = False
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
    context_build_time_ms: float | None = Field(default=None, ge=0)
    generation_time_ms: float | None = Field(default=None, ge=0)
    grounding_verification_time_ms: float | None = Field(default=None, ge=0)
    total_pipeline_time_ms: float | None = Field(default=None, ge=0)
    error: str | None = None


class E4ConfigurationSummary(BaseModel):
    """Aggregate result for one primary RAG configuration."""

    model_config = ConfigDict(extra="forbid")

    configuration: str = Field(min_length=1)
    search_mode: str = Field(min_length=1)
    rerank: bool
    top_k: int = Field(gt=0)
    questions_attempted: int = Field(ge=0)
    questions_scorable: int = Field(ge=0)
    questions_unscorable: int = Field(ge=0)
    questions_answered: int = Field(ge=0)
    questions_abstained: int = Field(ge=0)
    questions_failed: int = Field(ge=0)
    answer_coverage_rate: float | None = Field(default=None, ge=0, le=1)
    end_to_end: E4AnswerMetrics
    scorable: E4AnswerMetrics
    citation: E4CitationMetrics
    abstention_rate: float | None = Field(default=None, ge=0, le=1)
    correct_abstention_count: int = Field(default=0, ge=0)
    latency: dict[str, E4LatencyStats]
    failures: dict[str, int] = Field(default_factory=dict)


class E4RunSummary(BaseModel):
    """Top-level E4 artifact summary."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "e4.v1"
    status: Literal["completed", "DOCVQA_DATA_REQUIRED", "CONTROLLED_FAILURE"]
    reason_code: str | None = None
    dataset: str = "docvqa"
    split: str
    documents_requested: int = Field(ge=0)
    documents_prepared: int = Field(ge=0)
    documents_indexed: int = Field(ge=0)
    questions_attempted: int = Field(ge=0)
    questions_scorable: int = Field(ge=0)
    questions_unscorable: int = Field(ge=0)
    answer_indexability_rate: float | None = Field(default=None, ge=0, le=1)
    configurations: dict[str, E4ConfigurationSummary] = Field(default_factory=dict)
    deltas: dict[str, Any] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
