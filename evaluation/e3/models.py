"""Typed E3 question, corpus-mapping, and artifact schemas."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


QuestionStatus = Literal[
    "SCORABLE",
    "ANSWER_NOT_INDEXED",
    "DOCUMENT_PROCESSING_FAILED",
    "INVALID_GROUND_TRUTH",
    "OTHER_CONTROLLED_REASON",
]


class E3Question(BaseModel):
    """One bounded DocVQA question selected for every method."""

    model_config = ConfigDict(extra="forbid")

    question_key: str = Field(min_length=1)
    evaluation_id: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    source_document_id: str | None = None
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    accepted_answers: list[str] = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    local_pdf_path: str = Field(min_length=1)


class IndexedChunk(BaseModel):
    """Detached indexed chunk projection used to build answer ground truth."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: UUID
    document_id: UUID
    sequence_number: int = Field(ge=1)
    text: str
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    section_heading: str | None = None


class E3QuestionGroundTruth(BaseModel):
    """Deterministic answer-bearing chunk classification for one question."""

    model_config = ConfigDict(extra="forbid")

    question_key: str = Field(min_length=1)
    status: QuestionStatus
    reason_code: str | None = None
    target_document_id: UUID | None = None
    relevant_chunk_ids: list[UUID] = Field(default_factory=list)
    answer_matches: dict[str, list[UUID]] = Field(default_factory=dict)
    normalized_answers: list[str] = Field(default_factory=list)


class CorpusMapping(BaseModel):
    """Run-owned mapping from E1 source truth to indexed DocuIntel data."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    source_document_id: str | None = None
    local_pdf_path: str = Field(min_length=1)
    original_filename: str = Field(min_length=1)
    document_id: UUID | None = None
    stored_filename: str | None = None
    checksum_sha256: str | None = None
    indexed_chunk_ids: list[UUID] = Field(default_factory=list)
    indexed_chunk_count: int = Field(default=0, ge=0)
    processing_status: Literal["pending", "indexed", "failed", "skipped"] = "pending"
    failure_reason: str | None = None
    cleaned_up: bool = False


class E3ResultItem(BaseModel):
    """Bounded retrieval result metadata written for debugging and audit."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    chunk_id: UUID
    document_id: UUID
    filename: str
    sequence_number: int = Field(ge=1)
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    excerpt: str
    semantic_score: float | None = None
    keyword_score: float | None = None
    hybrid_score: float | None = None
    base_rank: int | None = Field(default=None, ge=1)
    rerank_score: float | None = None


class E3QuestionMethodResult(BaseModel):
    """One method's result for one question."""

    model_config = ConfigDict(extra="forbid")

    question_key: str = Field(min_length=1)
    method: str = Field(min_length=1)
    result_items: list[E3ResultItem] = Field(default_factory=list)
    relevant_chunk_ids: list[UUID] = Field(default_factory=list)
    first_relevant_rank: int | None = Field(default=None, ge=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    hit_at_k: dict[str, bool] = Field(default_factory=dict)
    document_hit_at_k: dict[str, bool] = Field(default_factory=dict)
    retrieval_time_ms: float | None = Field(default=None, ge=0)
    reranking_time_ms: float | None = Field(default=None, ge=0)
    total_retrieval_pipeline_ms: float | None = Field(default=None, ge=0)
    wall_clock_time_ms: float | None = Field(default=None, ge=0)
    error: str | None = None


class E3MetricStats(BaseModel):
    """Mean, median, and P95 for one latency measure."""

    mean_ms: float | None = None
    median_ms: float | None = None
    p95_ms: float | None = None
    samples: int = Field(ge=0)


class E3MethodSummary(BaseModel):
    """Aggregate retrieval quality and latency for one configuration."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(min_length=1)
    questions_attempted: int = Field(ge=0)
    scorable_questions: int = Field(ge=0)
    search_errors: int = Field(ge=0)
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    hit_at_k: dict[str, float] = Field(default_factory=dict)
    mrr: float | None = Field(default=None, ge=0, le=1)
    document_hit_at_k: dict[str, float] = Field(default_factory=dict)
    retrieval_latency: E3MetricStats
    reranking_latency: E3MetricStats
    total_pipeline_latency: E3MetricStats
    wall_clock_latency: E3MetricStats
    candidate_count: int | None = Field(default=None, ge=0)


class E3RunSummary(BaseModel):
    """Top-level E3 summary, including controlled blocked states."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "e3.v1"
    status: Literal["completed", "DOCVQA_DATA_REQUIRED", "CONTROLLED_FAILURE"]
    reason_code: str | None = None
    dataset: str = "docvqa"
    split: str
    documents_requested: int
    documents_prepared: int
    documents_indexed: int
    questions_attempted: int
    questions_scorable: int
    questions_unscorable: int
    answer_indexability_rate: float | None = None
    methods: dict[str, E3MethodSummary] = Field(default_factory=dict)
    deltas: dict[str, Any] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)

