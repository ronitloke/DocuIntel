"""Normalized, dataset-independent schemas for evaluation corpora."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationBoundingBox(BaseModel):
    """A normalized left/top/right/bottom bounding box."""

    model_config = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_geometry(self) -> "EvaluationBoundingBox":
        """Reject inverted or zero-area boxes without changing source labels."""

        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("evaluation bounding boxes must have positive area.")
        return self


class EvaluationPage(BaseModel):
    """One source page or image associated with an evaluation document."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    source_image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationLayoutRegion(BaseModel):
    """Ground-truth layout annotation supplied by a source dataset."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    label: str = Field(min_length=1)
    bounding_box: EvaluationBoundingBox
    text: str | None = None
    source_annotation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationEntity(BaseModel):
    """Ground-truth text/entity annotation, commonly used for FUNSD."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    bounding_box: EvaluationBoundingBox | None = None
    bounding_boxes: list[EvaluationBoundingBox] = Field(default_factory=list)
    source_annotation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationQAPair(BaseModel):
    """A normalized question and the answers explicitly supplied by DocVQA."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    accepted_answers: list[str] = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    source_document_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationDocument(BaseModel):
    """One normalized document record written to a JSONL manifest."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    source_document_id: str | None = None
    local_pdf_path: str = Field(min_length=1)
    page_count: int = Field(ge=1)
    pages: list[EvaluationPage] = Field(min_length=1)
    layout_regions: list[EvaluationLayoutRegion] = Field(default_factory=list)
    entities: list[EvaluationEntity] = Field(default_factory=list)
    qa_pairs: list[EvaluationQAPair] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pages(self) -> "EvaluationDocument":
        """Keep the declared page count consistent with normalized pages."""

        page_numbers = {page.page_number for page in self.pages}
        if len(page_numbers) != len(self.pages):
            raise ValueError("evaluation document pages must have unique page numbers.")
        if self.page_count != len(self.pages):
            raise ValueError("page_count must match the number of normalized pages.")
        return self


class PreparationMetadata(BaseModel):
    """Reproducibility metadata written beside every preparation manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "e1.v1"
    dataset: str
    source: str
    split: str
    requested_limit: int
    prepared: int = Field(ge=0)
    skipped: int = Field(ge=0)
    failed: int = Field(ge=0)
    prepared_at: str
    source_revision: str | None = None
    command: str
    options: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class PreparationResult(BaseModel):
    """Bounded preparation outcome used by both CLIs and tests."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    split: str
    requested: int = Field(gt=0)
    prepared: int = Field(ge=0)
    skipped: int = Field(ge=0)
    failed: int = Field(ge=0)
    manifest_path: str
    output_directory: str
    metadata_path: str
    errors: list[str] = Field(default_factory=list)

