"""Validation and statistics for prepared evaluation manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf as fitz

from evaluation.schemas import EvaluationDocument

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ManifestValidationError(ValueError):
    """A manifest is malformed or references unavailable local data."""


@dataclass(frozen=True)
class ManifestStatistics:
    """Counts that apply to the structures actually present in a manifest."""

    documents: int
    pages: int
    layout_regions: int
    entities: int
    qa_pairs: int


@dataclass
class ManifestValidationResult:
    """Validated records, statistics, and non-fatal inspection warnings."""

    records: list[EvaluationDocument]
    statistics: ManifestStatistics
    dataset: str
    split: str
    warnings: list[str] = field(default_factory=list)


def load_manifest(path: Path) -> list[EvaluationDocument]:
    """Parse a JSONL manifest and validate every normalized record."""

    if not path.is_file():
        raise ManifestValidationError(f"Manifest does not exist: {path}")
    records: list[EvaluationDocument] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            records.append(EvaluationDocument.model_validate(payload))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ManifestValidationError(f"Invalid manifest record on line {line_number}: {exc}") from exc
    if not records:
        return []
    ids = [record.evaluation_id for record in records]
    if len(ids) != len(set(ids)):
        raise ManifestValidationError("Duplicate evaluation_id values exist in the manifest.")
    datasets = {record.dataset for record in records}
    splits = {record.split for record in records}
    if len(datasets) != 1 or len(splits) != 1:
        raise ManifestValidationError("All manifest records must use one dataset and one split.")
    return records


def resolve_local_path(manifest_path: Path, stored_path: str) -> Path:
    """Resolve repository-relative paths without accepting arbitrary traversal."""

    candidate = Path(stored_path)
    if candidate.is_absolute():
        return candidate.resolve()
    repository_candidate = (PROJECT_ROOT / candidate).resolve()
    if repository_candidate.exists():
        return repository_candidate
    return (manifest_path.parent / candidate).resolve()


def validate_manifest(path: Path) -> ManifestValidationResult:
    """Validate referenced PDFs, page counts, IDs, and normalized ground truth."""

    records = load_manifest(path)
    if not records:
        return ManifestValidationResult(
            records=[],
            statistics=ManifestStatistics(0, 0, 0, 0, 0),
            dataset="",
            split="",
        )
    for record in records:
        pdf_path = resolve_local_path(path, record.local_pdf_path)
        if not pdf_path.is_file():
            raise ManifestValidationError(
                f"Referenced PDF does not exist for {record.evaluation_id}: {record.local_pdf_path}"
            )
        try:
            with fitz.open(pdf_path) as pdf:
                if pdf.page_count != record.page_count:
                    raise ManifestValidationError(
                        f"Page count mismatch for {record.evaluation_id}: "
                        f"manifest={record.page_count}, PDF={pdf.page_count}"
                    )
        except ManifestValidationError:
            raise
        except (RuntimeError, fitz.FileDataError) as exc:
            raise ManifestValidationError(
                f"Referenced PDF cannot be opened for {record.evaluation_id}: {pdf_path}"
            ) from exc
    statistics = ManifestStatistics(
        documents=len(records),
        pages=sum(len(record.pages) for record in records),
        layout_regions=sum(len(record.layout_regions) for record in records),
        entities=sum(len(record.entities) for record in records),
        qa_pairs=sum(len(record.qa_pairs) for record in records),
    )
    return ManifestValidationResult(
        records=records,
        statistics=statistics,
        dataset=records[0].dataset,
        split=records[0].split,
    )

