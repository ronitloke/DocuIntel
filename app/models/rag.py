"""Request and response models for grounded single-question RAG."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.models.search import SearchFilters, SearchMode


class AskRequest(BaseModel):
    """A grounded question over all or an explicit bounded document scope."""

    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    search_mode: SearchMode = SearchMode.HYBRID
    rerank: bool = True
    filters: SearchFilters | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject whitespace-only questions while normalizing outer whitespace."""

        question = value.strip()
        if not question:
            raise ValueError("question must contain non-whitespace text.")
        return question


class RAGSource(BaseModel):
    """Evidence metadata corresponding to one context source label."""

    source_id: str
    document_id: UUID
    chunk_id: UUID
    filename: str
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    section_heading: str | None = None
    content_type: str | None = None
    contains_ocr: bool
    excerpt: str
    final_rank: int = Field(ge=1)
    base_rank: int | None = Field(default=None, ge=1)
    reranker_score: float | None = None


class AskResponse(BaseModel):
    """Generated answer, validated citations, evidence, and stage timings."""

    answer: str
    model: str
    sources: list[RAGSource]
    citations: list[str]
    citations_valid: bool
    retrieval_time_ms: float = Field(ge=0)
    rerank_time_ms: float | None = Field(default=None, ge=0)
    generation_time_ms: float = Field(ge=0)
    total_time_ms: float = Field(ge=0)
    document_scope: str = "all"
    selected_document_ids: list[UUID] = Field(default_factory=list)
    retrieved_document_ids: list[UUID] = Field(default_factory=list)
    source_document_ids: list[UUID] = Field(default_factory=list)
