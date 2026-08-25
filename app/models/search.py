"""Request and response models for Modules 5–6 search APIs."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchMode(str, Enum):
    """Supported retrieval strategies."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchFilters(BaseModel):
    """Optional SQL-side scope constraints for retrieval."""

    document_ids: list[UUID] | None = None
    content_types: list[str] | None = None
    contains_ocr: bool | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @field_validator("content_types")
    @classmethod
    def validate_content_types(cls, value: list[str] | None) -> list[str] | None:
        """Restrict content filters to persisted Module 4 content types."""

        if value is None:
            return None
        allowed = {"text", "table", "list", "mixed"}
        if any(item not in allowed for item in value):
            raise ValueError("content_types must contain only text, table, list, or mixed.")
        return value

    @model_validator(mode="after")
    def validate_page_range(self) -> "SearchFilters":
        """Reject an inverted page range before it reaches SQL."""

        if self.page_start is not None and self.page_end is not None and self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start.")
        return self


class SearchRequest(BaseModel):
    """Validated search request body."""

    query: str = Field(min_length=1)
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=5, ge=1)
    filters: SearchFilters | None = None
    rerank: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject whitespace-only queries while normalizing outer whitespace."""

        query = value.strip()
        if not query:
            raise ValueError("query must contain non-whitespace text.")
        return query


class SearchResult(BaseModel):
    """One ranked chunk result without its raw embedding vector."""

    rank: int = Field(ge=1)
    chunk_id: UUID
    document_id: UUID
    original_filename: str
    sequence_number: int = Field(ge=1)
    text: str
    section_heading: str | None = None
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    content_type: str | None = None
    contains_ocr: bool
    semantic_score: float | None = None
    keyword_score: float | None = None
    hybrid_score: float | None = None
    retrieval_method: SearchMode
    base_rank: int | None = Field(default=None, ge=1)
    rerank_score: float | None = None
    reranked: bool = False


class SearchResponse(BaseModel):
    """Search response with measured request timing."""

    query: str
    mode: SearchMode
    results: list[SearchResult]
    total_results: int = Field(ge=0)
    search_time_ms: float = Field(ge=0)
    reranked: bool = False
    retrieval_time_ms: float = Field(ge=0)
    rerank_time_ms: float | None = Field(default=None, ge=0)
    total_search_time_ms: float = Field(ge=0)
