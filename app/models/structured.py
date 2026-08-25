"""Schemas for evidence-grounded extraction and safe table querying."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.documents import DocumentStatus


class ExtractionFieldType(str, Enum):
    """Small, safe set of caller-requested extraction types."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    LIST_STRING = "list[string]"


class ExtractionStatus(str, Enum):
    """Evidence status for one requested field."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


class ExtractionFieldDefinition(BaseModel):
    """One caller-defined field with no executable expressions."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    description: str | None = Field(default=None, max_length=500)
    type: ExtractionFieldType = ExtractionFieldType.STRING

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Normalize a field identifier without changing its caller meaning."""

        return value.strip()

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        """Turn blank optional descriptions into absent descriptions."""

        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class StructuredExtractionRequest(BaseModel):
    """Request bounded extraction from exactly one indexed document."""

    model_config = ConfigDict(extra="forbid")

    fields: list[ExtractionFieldDefinition] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_unique_fields(self) -> "StructuredExtractionRequest":
        """Reject duplicate names before any provider call."""

        names = [field.name.casefold() for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("extraction field names must be unique.")
        return self


class ExtractionCandidate(BaseModel):
    """One evidence-supported alternative for an ambiguous field."""

    model_config = ConfigDict(extra="forbid")

    value: Any
    sources: list[str] = Field(default_factory=list, max_length=20)


class StructuredExtractionFieldResult(BaseModel):
    """Validated result for one requested field."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1, max_length=100)
    value: Any = None
    status: ExtractionStatus
    sources: list[str] = Field(default_factory=list, max_length=20)
    candidates: list[ExtractionCandidate] = Field(default_factory=list, max_length=20)


class StructuredExtractionSource(BaseModel):
    """Chunk provenance corresponding to an extraction source label."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    document_id: UUID
    chunk_id: UUID
    filename: str
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    section_heading: str | None = None
    excerpt: str


class StructuredExtractionResponse(BaseModel):
    """Safe structured values, provenance, and stage timings."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    filename: str
    model: str
    fields: list[StructuredExtractionFieldResult]
    sources: list[StructuredExtractionSource]
    evidence_loading_time_ms: float = Field(ge=0)
    generation_time_ms: float = Field(ge=0)
    validation_time_ms: float = Field(ge=0)
    total_time_ms: float = Field(ge=0)


class StructuredExtractionLLMResponse(BaseModel):
    """Strict provider contract before safe API projection."""

    model_config = ConfigDict(extra="forbid")

    fields: list[StructuredExtractionFieldResult] = Field(min_length=1, max_length=50)


class TableInventoryItem(BaseModel):
    """Small table identity/provenance projection."""

    model_config = ConfigDict(extra="forbid")

    table_id: UUID
    document_id: UUID
    filename: str
    page_number: int = Field(ge=1)
    table_index: int = Field(ge=1)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    headers: list[str]
    status: DocumentStatus = DocumentStatus.READY


class TableInventoryResponse(BaseModel):
    """All detected structured tables for one document."""

    document_id: UUID
    filename: str
    tables: list[TableInventoryItem]


class TablePreviewResponse(TableInventoryItem):
    """Bounded table preview for the presentation layer."""

    rows: list[list[str]]
    preview_row_count: int = Field(ge=0)
    truncated: bool


class TableQueryOperation(str, Enum):
    """Allowed deterministic table operations."""

    SELECT = "select"
    FILTER = "filter"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    SORT = "sort"
    TOP_N = "top_n"


class TableFilterOperator(str, Enum):
    """Safe comparison operators for row filtering."""

    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"


class TableQueryPlan(BaseModel):
    """Constrained plan; it contains no SQL, Python, or executable expression."""

    model_config = ConfigDict(extra="forbid")

    operation: TableQueryOperation
    target_column: str | None = Field(default=None, max_length=200)
    return_columns: list[str] = Field(default_factory=list, max_length=50)
    filter_column: str | None = Field(default=None, max_length=200)
    filter_operator: TableFilterOperator | None = None
    filter_value: Any = None
    sort_direction: str = Field(default="desc", pattern=r"^(asc|desc)$")
    limit: int | None = Field(default=10, ge=1, le=100)

    @field_validator("target_column", "filter_column")
    @classmethod
    def normalize_column(cls, value: str | None) -> str | None:
        """Normalize column labels without accepting expressions."""

        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("return_columns")
    @classmethod
    def normalize_return_columns(cls, value: list[str]) -> list[str]:
        """Normalize and deduplicate requested display columns."""

        normalized: list[str] = []
        seen: set[str] = set()
        for column in value:
            cleaned = column.strip()
            if not cleaned:
                raise ValueError("return_columns must not contain empty names.")
            key = cleaned.casefold()
            if key not in seen:
                normalized.append(cleaned)
                seen.add(key)
        return normalized


class TableQueryRequest(BaseModel):
    """Natural-language question with an optional already constrained plan."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    plan: TableQueryPlan | None = None

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """Reject whitespace-only table questions."""

        question = value.strip()
        if not question:
            raise ValueError("question must contain non-whitespace text.")
        return question


class TableQuerySource(BaseModel):
    """Table provenance for a human answer and its selected rows."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=r"^T[1-9][0-9]*$")
    document_id: UUID
    filename: str
    page_number: int = Field(ge=1)
    table_id: UUID
    table_index: int = Field(ge=1)
    row_indices: list[int] = Field(default_factory=list)


class TableStructuredResult(BaseModel):
    """Machine-readable deterministic result of a validated table plan."""

    operation: TableQueryOperation
    column: str | None = None
    value: Any = None
    rows: list[dict[str, str]] = Field(default_factory=list)
    row_count: int = Field(ge=0)


class TableQueryResponse(BaseModel):
    """Human answer plus the exact structured result and table provenance."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    filename: str
    table: TableInventoryItem
    question: str
    plan: TableQueryPlan
    result: TableStructuredResult
    answer: str
    sources: list[TableQuerySource]
    table_loading_time_ms: float = Field(ge=0)
    plan_generation_time_ms: float = Field(ge=0)
    execution_time_ms: float = Field(ge=0)
    total_time_ms: float = Field(ge=0)
