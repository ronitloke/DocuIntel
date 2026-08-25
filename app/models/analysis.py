"""Request and response models for Module 11 document analysis."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class SummaryStyle(str, Enum):
    """Supported bounded summary presentation styles."""

    BRIEF = "brief"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"


class DocumentSummaryRequest(BaseModel):
    """Request one grounded summary of an indexed document."""

    style: SummaryStyle = SummaryStyle.BRIEF


class DocumentClassificationRequest(BaseModel):
    """Request classification against a caller-supplied finite label set."""

    labels: list[str] = Field(min_length=2, max_length=20)

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, value: list[str]) -> list[str]:
        """Normalize labels and reject empty, duplicate, or oversized values."""

        normalized: list[str] = []
        seen: set[str] = set()
        for label in value:
            cleaned = label.strip()
            if not cleaned:
                raise ValueError("classification labels must not be empty.")
            if len(cleaned) > 100:
                raise ValueError("classification labels must be 100 characters or fewer.")
            key = cleaned.casefold()
            if key in seen:
                raise ValueError("classification labels must be unique after normalization.")
            seen.add(key)
            normalized.append(cleaned)
        return normalized


class AnalysisSource(BaseModel):
    """Deterministic provenance for one chunk supplied to analysis."""

    source_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    document_id: UUID
    chunk_id: UUID
    sequence_number: int = Field(ge=1)
    filename: str
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    section_heading: str | None = None
    excerpt: str


class DocumentSummaryResponse(BaseModel):
    """Grounded summary plus document and generation provenance."""

    document_id: UUID
    filename: str
    title: str | None = None
    summary: str
    style: SummaryStyle
    model: str
    pages_represented: list[int]
    chunks_represented: int = Field(ge=0)
    sources: list[AnalysisSource]
    content_loading_time_ms: float = Field(ge=0)
    partial_generation_time_ms: float = Field(ge=0)
    final_synthesis_time_ms: float = Field(ge=0)
    generation_time_ms: float = Field(ge=0)
    total_time_ms: float = Field(ge=0)
    grounding_verification_time_ms: float = Field(default=0, ge=0)
    grounding_repair_time_ms: float = Field(default=0, ge=0)
    grounding_verification_passes: int = Field(default=0, ge=0)


class DocumentClassificationResponse(BaseModel):
    """Constrained classification result with evidence and timings."""

    document_id: UUID
    filename: str
    title: str | None = None
    selected_label: str
    rationale: str
    model: str
    sources: list[AnalysisSource]
    content_loading_time_ms: float = Field(ge=0)
    generation_time_ms: float = Field(ge=0)
    total_time_ms: float = Field(ge=0)


class ClassificationLLMResponse(BaseModel):
    """Machine-readable provider result validated before API projection."""

    selected_label: str = Field(min_length=1, max_length=100)
    rationale: str = Field(min_length=1, max_length=4000)

    @field_validator("selected_label", "rationale")
    @classmethod
    def strip_text(cls, value: str) -> str:
        """Normalize provider strings before constrained validation."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("provider output must contain non-empty text.")
        return cleaned


class GroundingClaim(BaseModel):
    """One factual claim assessed against supplied document evidence."""

    claim: str = Field(min_length=1, max_length=2000)
    supported: bool
    source_labels: list[str] = Field(default_factory=list, max_length=20)
    supporting_evidence: str = Field(default="", max_length=2000)
    reason: str = Field(default="", max_length=1000)

    @field_validator("claim", "supporting_evidence", "reason")
    @classmethod
    def strip_claim_text(cls, value: str) -> str:
        """Normalize short verifier diagnostics without exposing long reasoning."""

        return value.strip()

    @field_validator("source_labels")
    @classmethod
    def validate_source_labels(cls, value: list[str]) -> list[str]:
        """Require deterministic S-number labels in verifier output."""

        normalized: list[str] = []
        seen: set[str] = set()
        for label in value:
            cleaned = label.strip().upper()
            if (
                not cleaned.startswith("S")
                or not cleaned[1:].isdigit()
                or int(cleaned[1:]) < 1
                or (len(cleaned) > 2 and cleaned[1] == "0")
            ):
                raise ValueError("source labels must use the S1, S2, ... format.")
            if cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        return normalized


class GroundingVerificationResponse(BaseModel):
    """Structured local-model assessment of one generated summary draft."""

    claims: list[GroundingClaim] = Field(min_length=1, max_length=200)
    has_unsupported_claims: bool
    repaired_summary: str = Field(default="", max_length=20000)

    @model_validator(mode="after")
    def require_consistent_unsupported_flag(self) -> "GroundingVerificationResponse":
        """Prevent a provider from marking an explicitly unsupported claim as clean."""

        if any(not claim.supported for claim in self.claims) and not self.has_unsupported_claims:
            raise ValueError("has_unsupported_claims must be true when a claim is unsupported.")
        return self


class GroundingRepairResponse(BaseModel):
    """Structured repaired summary returned by the local verifier/repair call."""

    repaired_summary: str = Field(min_length=1, max_length=20000)

    @field_validator("repaired_summary")
    @classmethod
    def strip_repaired_summary(cls, value: str) -> str:
        """Reject empty repair text while preserving its requested formatting."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("repaired_summary must contain text.")
        return cleaned
