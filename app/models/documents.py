"""Request and response models for document ingestion and management."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_serializer


class DocumentStatus(str, Enum):
    """Persisted lifecycle states for an ingested document."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class PDFMetadata(BaseModel):
    """Metadata exposed from the PDF document when available."""

    title: str | None
    author: str | None
    subject: str | None
    keywords: str | None
    creator: str | None
    producer: str | None
    creation_date: str | None
    modification_date: str | None


class LayoutElement(BaseModel):
    """A heuristically classified native PDF layout element."""

    element_type: Literal["heading", "paragraph", "list_item", "table", "other"]
    text: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    font_size: float | None = None
    is_bold: bool | None = None
    table_index: int | None = None


class ExtractedTable(BaseModel):
    """A table detected by PyMuPDF's native table finder."""

    table_index: int = Field(ge=1)
    page_number: int = Field(ge=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class PageExtraction(BaseModel):
    """Native or OCR extraction and structure details for one PDF page."""

    page_number: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)
    has_native_text: bool
    needs_ocr: bool
    extraction_method: Literal["native", "ocr"]
    ocr_applied: bool = False
    ocr_success: bool | None = None
    ocr_confidence: float | None = Field(default=None, ge=0, le=100)
    layout_elements: list[LayoutElement] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)

    @model_serializer(mode="plain")
    def serialize_compatibly(self) -> dict[str, Any]:
        """Keep Module 1 native-page JSON stable while exposing Module 2 data."""

        payload: dict[str, Any] = {
            "page_number": self.page_number,
            "text": self.text,
            "character_count": self.character_count,
            "has_native_text": self.has_native_text,
            "needs_ocr": self.needs_ocr,
            "extraction_method": self.extraction_method,
        }
        if self.ocr_applied:
            payload["ocr_applied"] = True
        if self.ocr_success is not None:
            payload["ocr_success"] = self.ocr_success
        if self.ocr_confidence is not None:
            payload["ocr_confidence"] = self.ocr_confidence
        if self.layout_elements:
            payload["layout_elements"] = self.layout_elements
        if self.tables:
            payload["tables"] = self.tables
        return payload


class DocumentIngestionResponse(BaseModel):
    """Public result returned after successful PDF ingestion."""

    document_id: UUID
    original_filename: str
    stored_filename: str
    mime_type: str
    file_size_bytes: int = Field(ge=1)
    checksum_sha256: str
    page_count: int = Field(ge=0)
    pages_with_native_text: int = Field(ge=0)
    pages_requiring_ocr: int = Field(ge=0)
    pages_processed_by_ocr: int = Field(default=0, ge=0)
    ocr_failed_pages: int = Field(default=0, ge=0)
    unresolved_ocr_pages: int = Field(default=0, ge=0)
    heading_count: int = Field(default=0, ge=0)
    table_count: int = Field(default=0, ge=0)
    layout_element_count: int = Field(default=0, ge=0)
    status: DocumentStatus
    metadata: PDFMetadata
    pages: list[PageExtraction]


class DocumentListItem(BaseModel):
    """A compact document representation suitable for paginated lists."""

    id: UUID
    original_filename: str
    stored_filename: str
    mime_type: str
    file_size_bytes: int = Field(ge=1)
    page_count: int = Field(ge=0)
    status: DocumentStatus
    checksum_sha256: str
    created_at: datetime
    updated_at: datetime
    is_indexed: bool
    indexed_at: datetime | None = None
    chunk_count: int = Field(ge=0)
    embedding_model: str | None = None
    embedding_dimension: int | None = Field(default=None, gt=0)


class DocumentListResponse(BaseModel):
    """A SQL-paginated collection of persisted documents."""

    items: list[DocumentListItem]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class DocumentSummary(BaseModel):
    """Processing counts returned without embedding every page's text."""

    page_count: int = Field(ge=0)
    pages_with_native_text: int = Field(ge=0)
    pages_requiring_ocr: int = Field(ge=0)
    pages_processed_by_ocr: int = Field(ge=0)
    ocr_failed_pages: int = Field(ge=0)
    unresolved_ocr_pages: int = Field(ge=0)
    heading_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    layout_element_count: int = Field(ge=0)


class DocumentDetailResponse(DocumentListItem):
    """Document metadata and processing summary."""

    metadata: PDFMetadata
    summary: DocumentSummary


class PageListItem(BaseModel):
    """Compact persisted page information for the page collection endpoint."""

    id: UUID
    document_id: UUID
    page_number: int = Field(ge=1)
    character_count: int = Field(ge=0)
    has_native_text: bool
    needs_ocr: bool
    extraction_method: Literal["native", "ocr"]
    ocr_applied: bool
    ocr_success: bool | None = None
    ocr_confidence: float | None = Field(default=None, ge=0, le=100)
    created_at: datetime


class PageListResponse(BaseModel):
    """A SQL-paginated collection of pages for one document."""

    items: list[PageListItem]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class PersistedPageResponse(BaseModel):
    """One complete persisted page including structure details."""

    id: UUID
    document_id: UUID
    page_number: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)
    has_native_text: bool
    needs_ocr: bool
    extraction_method: Literal["native", "ocr"]
    ocr_applied: bool
    ocr_success: bool | None = None
    ocr_confidence: float | None = Field(default=None, ge=0, le=100)
    created_at: datetime
    layout_elements: list[LayoutElement] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)


class ChunkListItem(BaseModel):
    """A public chunk projection that deliberately excludes the vector array."""

    id: UUID
    document_id: UUID
    sequence_number: int = Field(ge=1)
    text: str
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    section_heading: str | None = None
    content_type: Literal["text", "table", "list", "mixed"] | None = None
    contains_ocr: bool
    character_count: int = Field(ge=0)
    token_count: int | None = Field(default=None, ge=0)
    embedding_model: str | None = None
    embedding_dimension: int | None = Field(default=None, gt=0)
    fingerprint_sha256: str | None = None
    created_at: datetime
    updated_at: datetime


class ChunkListResponse(BaseModel):
    """A SQL-paginated collection of public chunk projections."""

    items: list[ChunkListItem]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class DocumentIndexingResponse(BaseModel):
    """Safe summary returned after a document index replacement."""

    document_id: UUID
    chunks_created: int = Field(ge=0)
    embeddings_created: int = Field(ge=0)
    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    status: Literal["indexed"]
