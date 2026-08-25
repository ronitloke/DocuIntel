"""Typed contracts for deterministic document and version comparison."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.documents import DocumentStatus


class ComparisonMode(str, Enum):
    """Interpretation of the explicitly selected document pair."""

    DOCUMENT = "document"
    VERSION = "version"


class ComparisonChangeType(str, Enum):
    """Deterministic change categories exposed by the comparison API."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class ComparisonScope(str, Enum):
    """Evidence scope of one change item."""

    TEXT = "text"
    TABLE = "table"
    METADATA = "metadata"


class ComparisonDocument(BaseModel):
    """Safe identity and readiness metadata for one compared document."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    filename: str
    title: str | None = None
    page_count: int = Field(ge=0)
    status: DocumentStatus
    is_indexed: bool


class ComparisonEvidence(BaseModel):
    """A source label that resolves to persisted document evidence."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=r"^[AB][1-9][0-9]*$")
    document_id: UUID
    filename: str
    page_number: int | None = Field(default=None, ge=1)
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    chunk_id: UUID | None = None
    sequence_number: int | None = Field(default=None, ge=1)
    section_heading: str | None = None
    table_id: UUID | None = None
    table_index: int | None = Field(default=None, ge=1)
    row_indices: list[int] = Field(default_factory=list)
    column: str | None = None


class ComparisonTableDetail(BaseModel):
    """Structured details for one deterministic table change."""

    model_config = ConfigDict(extra="forbid")

    table_change_type: str = Field(
        pattern=r"^(table_added|table_removed|table_unchanged|header_added|header_removed|row_added|row_removed|cell_modified)$"
    )
    row_key: str | None = None
    column: str | None = None
    before: str | None = None
    after: str | None = None
    row_values: dict[str, str] = Field(default_factory=dict)


class ComparisonChange(BaseModel):
    """One added, removed, modified, or optionally unchanged evidence item."""

    model_config = ConfigDict(extra="forbid")

    change_id: str = Field(pattern=r"^C[1-9][0-9]*$")
    change_type: ComparisonChangeType
    scope: ComparisonScope
    base_text: str | None = None
    target_text: str | None = None
    base_provenance: list[ComparisonEvidence] = Field(default_factory=list)
    target_provenance: list[ComparisonEvidence] = Field(default_factory=list)
    section: str | None = None
    similarity: float | None = Field(default=None, ge=0, le=1)
    table_detail: ComparisonTableDetail | None = None


class ComparisonStatistics(BaseModel):
    """Deterministic counts for the complete comparison, including hidden unchanged items."""

    model_config = ConfigDict(extra="forbid")

    added_count: int = Field(ge=0)
    removed_count: int = Field(ge=0)
    modified_count: int = Field(ge=0)
    unchanged_count: int = Field(ge=0)
    table_change_count: int = Field(ge=0)


class ComparisonRequest(BaseModel):
    """Request exactly one explicitly ordered document pair."""

    model_config = ConfigDict(extra="forbid")

    base_document_id: UUID
    target_document_id: UUID
    mode: ComparisonMode = ComparisonMode.DOCUMENT
    include_tables: bool = True
    include_unchanged: bool = False
    generate_summary: bool = True

    @model_validator(mode="after")
    def reject_same_document(self) -> "ComparisonRequest":
        """Prevent meaningless self-comparisons before repository access."""

        if self.base_document_id == self.target_document_id:
            raise ValueError("base_document_id and target_document_id must be different.")
        return self


class ComparisonResponse(BaseModel):
    """Machine-readable comparison result with bounded optional summary metadata."""

    model_config = ConfigDict(extra="forbid")

    base_document: ComparisonDocument
    target_document: ComparisonDocument
    mode: ComparisonMode
    changes: list[ComparisonChange]
    statistics: ComparisonStatistics
    summary: str
    summary_model: str | None = None
    summary_source_labels: list[str] = Field(default_factory=list)
    content_loading_time_ms: float = Field(ge=0)
    alignment_time_ms: float = Field(ge=0)
    table_comparison_time_ms: float = Field(ge=0)
    summary_generation_time_ms: float = Field(ge=0)
    total_time_ms: float = Field(ge=0)
