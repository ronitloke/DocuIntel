"""Typed contracts for deterministic high-confidence PII workflows."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PIIType(str, Enum):
    """Structured PII categories supported by Module 12.4."""

    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    IBAN = "iban"
    CREDIT_CARD = "credit_card"


SUPPORTED_PII_TYPES = tuple(PIIType)


class PIIDetectionSource(str, Enum):
    """Where the detector obtained the reviewed text."""

    NATIVE_PDF_TEXT = "native_pdf_text"
    OCR_TEXT = "ocr_text"


class PIICoordinates(BaseModel):
    """A verified PDF page rectangle in PyMuPDF coordinates."""

    model_config = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float


class PIIDocumentInfo(BaseModel):
    """Minimal document identity returned by privacy endpoints."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    filename: str
    page_count: int = Field(ge=0)


class PIIDetectRequest(BaseModel):
    """Optional allow-list of deterministic PII detectors."""

    model_config = ConfigDict(extra="forbid")

    pii_types: list[PIIType] | None = None

    @field_validator("pii_types")
    @classmethod
    def normalize_types(cls, value: list[PIIType] | None) -> list[PIIType] | None:
        """Reject an empty allow-list and preserve first-seen unique types."""

        if value is None:
            return None
        if not value:
            raise ValueError("pii_types must contain at least one supported type.")
        return list(dict.fromkeys(value))


class PIIDetection(BaseModel):
    """One reviewable, deterministic PII candidate."""

    model_config = ConfigDict(extra="forbid")

    detection_id: str = Field(pattern=r"^pii-[0-9a-f]{24}$")
    pii_type: PIIType
    matched_text: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    source: PIIDetectionSource
    content_type: str = Field(min_length=1)
    coordinates: PIICoordinates | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=1)
    detector: str = Field(min_length=1)
    validation_status: str = Field(min_length=1)
    confidence_category: str = Field(default="high", pattern=r"^high$")
    redactable: bool
    coordinate_source: str | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> "PIIDetection":
        """Require a non-empty, ordered character span."""

        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset.")
        return self


class PIIDetectionResponse(BaseModel):
    """Detection results and bounded operational timings."""

    model_config = ConfigDict(extra="forbid")

    document: PIIDocumentInfo
    detections: list[PIIDetection]
    detection_count: int = Field(ge=0)
    counts_by_type: dict[PIIType, int]
    detection_time_ms: float = Field(ge=0)
    total_time_ms: float = Field(ge=0)


class PIIRedactRequest(BaseModel):
    """Explicit caller selection for irreversible redaction."""

    model_config = ConfigDict(extra="forbid")

    detection_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("detection_ids")
    @classmethod
    def validate_detection_ids(cls, value: list[str]) -> list[str]:
        """Require non-empty unique server-issued detection identifiers."""

        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("detection_ids must not contain blank values.")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("detection_ids must be unique.")
        return cleaned


class PIIRedactedArtifact(BaseModel):
    """Safe metadata for a generated redacted PDF."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: UUID
    document_id: UUID
    filename: str = Field(pattern=r"^redacted-[0-9a-f-]{36}\.pdf$")
    media_type: str = "application/pdf"
    page_count: int = Field(ge=0)
    size_bytes: int = Field(gt=0)
    download_url: str = Field(pattern=r"^/api/v1/documents/[0-9a-f-]+/pii/artifacts/[0-9a-f-]+$")


class PIIRedactResponse(BaseModel):
    """Result of one explicit, verified redaction operation."""

    model_config = ConfigDict(extra="forbid")

    document: PIIDocumentInfo
    redacted_detection_ids: list[str]
    redacted_count: int = Field(ge=1)
    redacted_pages: list[int]
    artifact: PIIRedactedArtifact
    redaction_time_ms: float = Field(ge=0)
    total_time_ms: float = Field(ge=0)
