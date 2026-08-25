"""Pydantic schemas for human-editable evaluation data and reports."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.models.search import SearchFilters, SearchMode


class EvaluationCase(BaseModel):
    """One retrieval or grounded-answer evaluation case."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    description: str | None = None
    expected_document: str | None = None
    expected_documents: list[str] = Field(default_factory=list)
    expected_page: int | None = Field(default=None, ge=1)
    expected_pages: list[int] = Field(default_factory=list, min_length=0)
    expected_chunk: str | None = None
    expected_chunks: list[str] = Field(default_factory=list)
    expected_answer: str | None = None
    expected_facts: list[str] = Field(default_factory=list)
    filters: SearchFilters | None = None
    tags: list[str] = Field(default_factory=list)
    expect_no_evidence: bool = False

    @model_validator(mode="after")
    def normalize_expected_items(self) -> "EvaluationCase":
        """Support singular convenience fields without losing plural labels."""

        if self.expected_document and self.expected_document not in self.expected_documents:
            self.expected_documents.insert(0, self.expected_document)
        if self.expected_chunk and self.expected_chunk not in self.expected_chunks:
            self.expected_chunks.insert(0, self.expected_chunk)
        if self.expected_page and self.expected_page not in self.expected_pages:
            self.expected_pages.insert(0, self.expected_page)
        if self.expect_no_evidence and (self.expected_documents or self.expected_chunks):
            raise ValueError("expect_no_evidence cases cannot define expected documents or chunks.")
        return self


class EvaluationDataset(BaseModel):
    """A validated collection of cases loaded from JSON or JSONL."""

    name: str = Field(min_length=1)
    description: str | None = None
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> "EvaluationDataset":
        """Reject ambiguous duplicate case identifiers."""

        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation case ids must be unique.")
        return self


class EvaluationConfiguration(BaseModel):
    """One existing search configuration under evaluation."""

    mode: SearchMode
    rerank: bool = False
    top_k: int = Field(default=5, ge=1)

    @property
    def label(self) -> str:
        """Return a stable human-readable configuration label."""

        suffix = " + rerank" if self.rerank else ""
        return f"{self.mode.value}{suffix}"


class EvaluationResultItem(BaseModel):
    """Bounded result metadata retained for per-case debugging."""

    source_id: str | None = None
    rank: int = Field(ge=1)
    base_rank: int | None = Field(default=None, ge=1)
    rank_delta: int | None = None
    chunk_id: UUID
    document_id: UUID
    filename: str
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    excerpt: str
    semantic_score: float | None = None
    keyword_score: float | None = None
    hybrid_score: float | None = None
    reranker_score: float | None = None


class EvaluationDocumentCheck(BaseModel):
    """Preflight proof that an expected document is present and indexed."""

    case_id: str
    question: str
    expected_document: str
    exists: bool
    indexed_chunks: int = Field(ge=0)


class RetrievalCaseResult(BaseModel):
    """Measured retrieval outcome for one evaluation case."""

    case_id: str
    question: str
    expected_documents: list[str]
    expected_chunks: list[str]
    expect_no_evidence: bool
    results: list[EvaluationResultItem]
    success_at_k: dict[str, bool]
    recall_at_k: dict[str, float]
    reciprocal_rank: float
    no_evidence_correct: bool | None = None
    base_rank: int | None = None
    final_rank: int | None = None
    rank_delta: int | None = None
    retrieval_time_ms: float | None = Field(default=None, ge=0)
    rerank_time_ms: float | None = Field(default=None, ge=0)
    total_search_time_ms: float | None = Field(default=None, ge=0)
    error: str | None = None


class RetrievalSummary(BaseModel):
    """Aggregate retrieval quality and latency metrics."""

    cases: int
    positive_cases: int
    no_evidence_cases: int
    success_at_k: dict[str, float]
    recall_at_k: dict[str, float]
    mrr: float
    no_evidence_correct_rate: float | None = None
    mean_retrieval_latency_ms: float | None = None
    median_retrieval_latency_ms: float | None = None
    mean_rerank_latency_ms: float | None = None
    median_rerank_latency_ms: float | None = None
    mean_total_search_latency_ms: float | None = None
    median_total_search_latency_ms: float | None = None
    rerank_impact: dict[str, float | int | None] = Field(default_factory=dict)


class RetrievalEvaluationReport(BaseModel):
    """Machine-readable report for one retrieval configuration."""

    evaluation_type: Literal["retrieval"] = "retrieval"
    dataset: str
    configuration: EvaluationConfiguration
    generated_at: datetime
    corpus_checks: list[EvaluationDocumentCheck] = Field(default_factory=list)
    summary: RetrievalSummary
    cases: list[RetrievalCaseResult]
    baseline_comparison: dict[str, Any] | None = None
    quality_gates: dict[str, Any] | None = None


class ComparisonReport(BaseModel):
    """Machine-readable report comparing multiple retrieval configurations."""

    evaluation_type: Literal["comparison"] = "comparison"
    dataset: str
    generated_at: datetime
    corpus_checks: list[EvaluationDocumentCheck] = Field(default_factory=list)
    reports: list[RetrievalEvaluationReport]
    baseline_comparison: dict[str, Any] | None = None
    quality_gates: dict[str, Any] | None = None


class FactEvaluation(BaseModel):
    """Deterministic normalized key-fact match result."""

    fact: str
    matched: bool


class RAGCaseResult(BaseModel):
    """Measured grounded-answer outcome for one evaluation case."""

    case_id: str
    question: str
    answer: str | None = None
    sources: list[EvaluationResultItem] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    citations_present: bool = False
    citations_valid: bool = False
    expected_document_cited: bool | None = None
    facts: list[FactEvaluation] = Field(default_factory=list)
    key_fact_coverage: float = 0.0
    evidence_support: bool = False
    no_evidence_correct: bool | None = None
    generation_time_ms: float | None = Field(default=None, ge=0)
    total_time_ms: float | None = Field(default=None, ge=0)
    error: str | None = None


class RAGSummary(BaseModel):
    """Aggregate deterministic RAG quality and latency metrics."""

    cases: int
    key_fact_coverage: float
    citation_presence_rate: float
    citation_validity_rate: float
    expected_document_citation_rate: float | None = None
    evidence_support_rate: float
    no_evidence_cases: int
    no_evidence_correct_rate: float | None = None
    mean_generation_latency_ms: float | None = None
    median_generation_latency_ms: float | None = None
    mean_total_rag_latency_ms: float | None = None
    median_total_rag_latency_ms: float | None = None


class RAGEvaluationReport(BaseModel):
    """Machine-readable report for grounded-answer evaluation."""

    evaluation_type: Literal["rag"] = "rag"
    dataset: str
    configuration: EvaluationConfiguration
    generated_at: datetime
    summary: RAGSummary
    cases: list[RAGCaseResult]


class QualityGateConfig(BaseModel):
    """Optional thresholds supplied by a developer or CI job."""

    minimum_success_at_3: float | None = Field(default=None, ge=0, le=1)
    minimum_mrr: float | None = Field(default=None, ge=0, le=1)
    maximum_mean_search_latency_ms: float | None = Field(default=None, ge=0)


class QualityGateResult(BaseModel):
    """Quality-gate outcome without universal hard-coded thresholds."""

    passed: bool
    failures: list[str] = Field(default_factory=list)
