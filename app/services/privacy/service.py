"""Local, deterministic PII detection and irreversible PDF redaction."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

import pymupdf as fitz

from app.core.config import PROJECT_ROOT
from app.core.exceptions import (
    DatabaseNotConfiguredError,
    DocumentNotFoundError,
    PIIArtifactNotFoundError,
    PIIRedactionError,
    PIIValidationError,
)
from app.models.documents import DocumentStatus
from app.models.privacy import (
    PIICoordinates,
    PIIDetectRequest,
    PIIDetection,
    PIIDetectionResponse,
    PIIDetectionSource,
    PIIDocumentInfo,
    PIIRedactRequest,
    PIIRedactResponse,
    PIIRedactedArtifact,
    PIIType,
    SUPPORTED_PII_TYPES,
)
from app.services.privacy.coordinates import resolve_word_coordinates
from app.services.privacy.detectors import PIICandidate, detect_pii

logger = logging.getLogger(__name__)

DEFAULT_REDACTION_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "redacted"


class PrivacyRepository(Protocol):
    """Repository surface needed by the privacy service."""

    def get_document(self, document_id: UUID):
        """Load a document with pages, layout, and tables."""


class PrivacyService:
    """Coordinate safe document loading, detection, and PDF redaction."""

    def __init__(
        self,
        repository: PrivacyRepository | None,
        storage_directory: Path,
        artifact_directory: Path | None = None,
    ) -> None:
        self.repository = repository
        self.storage_directory = storage_directory
        self.artifact_directory = artifact_directory or DEFAULT_REDACTION_DIRECTORY

    def detect(
        self,
        document_id: UUID,
        request: PIIDetectRequest,
    ) -> PIIDetectionResponse:
        """Detect requested high-confidence PII without requiring an LLM."""

        started = time.perf_counter()
        document, source_path = self._load_ready_document(document_id)
        requested_types = list(request.pii_types or SUPPORTED_PII_TYPES)
        detection_started = time.perf_counter()
        detections = self._scan(document, source_path, requested_types)
        detection_ms = _elapsed_ms(detection_started)
        counts = Counter(detection.pii_type for detection in detections)
        counts_by_type = {pii_type: counts.get(pii_type, 0) for pii_type in requested_types}
        logger.info(
            "PII scan completed document_id=%s detection_count=%s counts=%s",
            document_id,
            len(detections),
            {key.value: value for key, value in counts_by_type.items()},
        )
        return PIIDetectionResponse(
            document=self._document_info(document),
            detections=detections,
            detection_count=len(detections),
            counts_by_type=counts_by_type,
            detection_time_ms=detection_ms,
            total_time_ms=_elapsed_ms(started),
        )

    def redact(
        self,
        document_id: UUID,
        request: PIIRedactRequest,
    ) -> PIIRedactResponse:
        """Re-detect, validate, and irreversibly redact explicit selections."""

        started = time.perf_counter()
        document, source_path = self._load_ready_document(document_id)
        detections = self._scan(document, source_path, list(SUPPORTED_PII_TYPES))
        by_id = {detection.detection_id: detection for detection in detections}
        missing = [detection_id for detection_id in request.detection_ids if detection_id not in by_id]
        if missing:
            raise PIIValidationError("One or more selected detections are no longer valid.")
        selected = [by_id[detection_id] for detection_id in request.detection_ids]
        non_redactable = [detection for detection in selected if not detection.redactable]
        if non_redactable:
            raise PIIRedactionError(
                "At least one selected detection has no verified PDF coordinates and cannot be redacted safely."
            )

        redaction_started = time.perf_counter()
        source_hash = _file_sha256(source_path)
        artifact_id = uuid4()
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        final_path = self._artifact_path(document_id, artifact_id)
        temporary_path = self.artifact_directory / f".{artifact_id}.tmp"
        pages = sorted({detection.page_number for detection in selected})
        try:
            self._write_redacted_pdf(source_path, temporary_path, selected)
            temporary_path.replace(final_path)
            self._verify_redacted_pdf(final_path, document.page_count, selected)
            if _file_sha256(source_path) != source_hash:
                raise PIIRedactionError("The source PDF changed during redaction; no artifact was retained.")
        except PIIRedactionError:
            temporary_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise
        except (OSError, RuntimeError, ValueError, fitz.FileDataError) as exc:
            temporary_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise PIIRedactionError("The redacted PDF could not be generated or verified safely.") from exc

        redaction_ms = _elapsed_ms(redaction_started)
        artifact = PIIRedactedArtifact(
            artifact_id=artifact_id,
            document_id=document_id,
            filename=f"redacted-{artifact_id}.pdf",
            page_count=document.page_count,
            size_bytes=final_path.stat().st_size,
            download_url=f"/api/v1/documents/{document_id}/pii/artifacts/{artifact_id}",
        )
        logger.info(
            "PII redaction completed document_id=%s redacted_count=%s pages=%s artifact_id=%s",
            document_id,
            len(selected),
            pages,
            artifact_id,
        )
        return PIIRedactResponse(
            document=self._document_info(document),
            redacted_detection_ids=request.detection_ids,
            redacted_count=len(selected),
            redacted_pages=pages,
            artifact=artifact,
            redaction_time_ms=redaction_ms,
            total_time_ms=_elapsed_ms(started),
        )

    def artifact_path(self, document_id: UUID, artifact_id: UUID) -> tuple[Path, str]:
        """Resolve only a generated artifact with a server-controlled name."""

        if self.repository is None:
            raise DatabaseNotConfiguredError("PostgreSQL is required for privacy artifacts.")
        document = self.repository.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError("The requested document was not found.")
        path = self._artifact_path(document_id, artifact_id)
        if not path.is_file():
            raise PIIArtifactNotFoundError("The generated redacted PDF was not found.")
        return path, f"redacted-{artifact_id}.pdf"

    def _load_ready_document(self, document_id: UUID):
        if self.repository is None:
            raise DatabaseNotConfiguredError("PostgreSQL is required for privacy operations.")
        document = self.repository.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError("The requested document was not found.")
        if document.status != DocumentStatus.READY:
            raise PIIValidationError("The document must be ready before privacy processing.")
        root = self.storage_directory.resolve()
        stored_filename = str(document.stored_filename)
        source_path = (root / stored_filename).resolve()
        if source_path.parent != root or source_path.name != stored_filename or not source_path.is_file():
            raise DocumentNotFoundError("The original source PDF is not available.")
        return document, source_path

    def _scan(self, document, source_path: Path, pii_types: list[PIIType]) -> list[PIIDetection]:
        """Scan persisted OCR/native page content and resolve native coordinates."""

        detections: list[PIIDetection] = []
        try:
            pdf = fitz.open(source_path)
        except (RuntimeError, fitz.FileDataError) as exc:
            raise PIIRedactionError("The original source PDF could not be opened safely.") from exc
        try:
            for page in sorted(document.pages, key=lambda item: item.page_number):
                native = str(page.extraction_method).casefold() == "native"
                pdf_page = pdf.load_page(page.page_number - 1) if native and page.page_number <= pdf.page_count else None
                page_text = (pdf_page.get_text("text") if pdf_page is not None else None) or str(page.extracted_text or "")
                source = PIIDetectionSource.NATIVE_PDF_TEXT if native else PIIDetectionSource.OCR_TEXT
                for candidate in detect_pii(page_text, pii_types):
                    coordinates = resolve_word_coordinates(pdf_page, candidate.matched_text) if pdf_page is not None else None
                    coordinate_model = PIICoordinates(**dict(zip(("x0", "y0", "x1", "y1"), coordinates))) if coordinates else None
                    detections.append(
                        self._project_detection(
                            document.id,
                            page.page_number,
                            candidate,
                            source,
                            coordinate_model,
                        )
                    )
        finally:
            pdf.close()
        return sorted(
            detections,
            key=lambda item: (item.page_number, item.start_offset, item.pii_type.value, item.detection_id),
        )

    @staticmethod
    def _project_detection(
        document_id: UUID,
        page_number: int,
        candidate: PIICandidate,
        source: PIIDetectionSource,
        coordinates: PIICoordinates | None,
    ) -> PIIDetection:
        digest = hashlib.sha256(
            (
                f"{document_id}:{page_number}:{candidate.pii_type.value}:"
                f"{candidate.start_offset}:{candidate.end_offset}:{candidate.matched_text.casefold()}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        return PIIDetection(
            detection_id=f"pii-{digest}",
            pii_type=candidate.pii_type,
            matched_text=candidate.matched_text,
            page_number=page_number,
            source=source,
            content_type="native_text" if source is PIIDetectionSource.NATIVE_PDF_TEXT else "ocr_text",
            coordinates=coordinates,
            start_offset=candidate.start_offset,
            end_offset=candidate.end_offset,
            detector=candidate.detector,
            validation_status=candidate.validation_status,
            redactable=coordinates is not None,
            coordinate_source="pymupdf_words" if coordinates is not None else None,
        )

    def _write_redacted_pdf(
        self,
        source_path: Path,
        output_path: Path,
        detections: list[PIIDetection],
    ) -> None:
        pdf = fitz.open(source_path)
        try:
            for detection in detections:
                if detection.coordinates is None:
                    raise PIIRedactionError("A selected detection has no verified coordinates.")
                page = pdf.load_page(detection.page_number - 1)
                rect = fitz.Rect(
                    detection.coordinates.x0,
                    detection.coordinates.y0,
                    detection.coordinates.x1,
                    detection.coordinates.y1,
                )
                if not rect.is_valid or rect.width <= 0 or rect.height <= 0 or not page.rect.contains(rect):
                    raise PIIRedactionError("A selected detection resolved outside its PDF page.")
                page.add_redact_annot(rect, fill=(0, 0, 0))
            for page_number in sorted({detection.page_number for detection in detections}):
                pdf.load_page(page_number - 1).apply_redactions()
            pdf.save(output_path, garbage=4, deflate=True)
        finally:
            pdf.close()

    @staticmethod
    def _verify_redacted_pdf(
        output_path: Path,
        expected_page_count: int,
        detections: list[PIIDetection],
    ) -> None:
        try:
            with fitz.open(output_path) as pdf:
                if pdf.page_count != expected_page_count:
                    raise PIIRedactionError("The redacted PDF changed page count during verification.")
                for detection in detections:
                    text = pdf.load_page(detection.page_number - 1).get_text("text") or ""
                    if _normalize_for_check(detection.matched_text) in _normalize_for_check(text):
                        raise PIIRedactionError(
                            "Verification found selected PII still extractable from the redacted PDF."
                        )
        except PIIRedactionError:
            raise
        except (RuntimeError, fitz.FileDataError) as exc:
            raise PIIRedactionError("The generated redacted PDF could not be reopened safely.") from exc

    def _artifact_path(self, document_id: UUID, artifact_id: UUID) -> Path:
        root = self.artifact_directory.resolve()
        filename = f"{document_id}-{artifact_id}.pdf"
        path = (root / filename).resolve()
        if path.parent != root or path.name != filename:
            raise PIIArtifactNotFoundError("The generated artifact path is invalid.")
        return path

    @staticmethod
    def _document_info(document) -> PIIDocumentInfo:
        return PIIDocumentInfo(
            document_id=document.id,
            filename=document.original_filename,
            page_count=document.page_count,
        )


def _normalize_for_check(value: str) -> str:
    return " ".join(value.split()).casefold()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
