"""Deterministic Module 12.4 detection, redaction, security, and API tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from app.api.routes.privacy import get_privacy_service
from app.core.exceptions import PIIValidationError, PIIRedactionError
from app.main import create_app
from app.models.documents import DocumentStatus
from app.models.privacy import (
    PIIDetectRequest,
    PIIDetectionResponse,
    PIIDocumentInfo,
    PIIRedactRequest,
    PIIRedactResponse,
    PIIRedactedArtifact,
    PIIType,
)
from app.services.privacy.detectors import detect_pii
from app.services.privacy.service import PrivacyService


DOCUMENT_ID = uuid4()


def make_pdf(path: Path, text: str, *, page_count: int = 1) -> None:
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    for index, line in enumerate(text.splitlines(), start=1):
        page.insert_text((72, 72 + index * 28), line, fontsize=12)
    for _ in range(1, page_count):
        document.new_page(width=612, height=792)
    document.save(path)
    document.close()


def native_document(text: str, *, document_id: UUID = DOCUMENT_ID):
    return SimpleNamespace(
        id=document_id,
        original_filename="module12_4_pii.pdf",
        stored_filename=f"{document_id}.pdf",
        page_count=1,
        status=DocumentStatus.READY,
        is_indexed=True,
        pages=[
            SimpleNamespace(
                page_number=1,
                extraction_method="native",
                extracted_text=text,
            )
        ],
    )


class FakeRepository:
    def __init__(self, document):
        self.document = document

    def get_document(self, document_id):
        return self.document if document_id == self.document.id else None


def test_detectors_validate_structured_pii_and_ignore_numeric_controls() -> None:
    text = (
        "Email privacy.test@example.com; Phone +1 (202) 555-0147; "
        "IBAN GB82 WEST 1234 5698 7654 32; Card 4111 1111 1111 1111; "
        "Invoice INV-2026-0043 Qty 3 Price 2400 Date 2026-08-19."
    )
    detections = detect_pii(text, list(PIIType))

    assert [item.pii_type for item in detections] == list(PIIType)
    assert {item.matched_text for item in detections} == {
        "privacy.test@example.com",
        "+1 (202) 555-0147",
        "GB82 WEST 1234 5698 7654 32",
        "4111 1111 1111 1111",
    }
    assert all(item.validation_status for item in detections)


@pytest.mark.parametrize(
    "text",
    [
        "Invoice INV-2026-0043",
        "Notice period 30 days",
        "Qty 3 and price 2400",
        "Date 2026-08-19",
        "ID 1234-5678-9012",
    ],
)
def test_false_positive_controls_do_not_become_pii(text: str) -> None:
    assert detect_pii(text, list(PIIType)) == []


def test_invalid_iban_and_credit_card_checksums_are_rejected() -> None:
    text = "IBAN GB82 WEST 1234 5698 7654 31 Card 4111 1111 1111 1112"
    assert detect_pii(text, [PIIType.IBAN, PIIType.CREDIT_CARD]) == []


def test_request_validation_rejects_empty_and_duplicate_selection() -> None:
    with pytest.raises(ValueError):
        PIIDetectRequest(pii_types=[])
    with pytest.raises(ValueError):
        PIIRedactRequest(detection_ids=["pii-1", "pii-1"])
    with pytest.raises(ValueError):
        PIIRedactRequest(detection_ids=[])


def test_native_coordinates_are_verified_and_ocr_is_non_redactable(tmp_path: Path) -> None:
    text = "Email privacy.test@example.com"
    source_path = tmp_path / f"{DOCUMENT_ID}.pdf"
    make_pdf(source_path, text)
    service = PrivacyService(FakeRepository(native_document(text)), tmp_path, tmp_path / "redacted")
    result = service.detect(DOCUMENT_ID, PIIDetectRequest(pii_types=[PIIType.EMAIL]))
    assert result.detection_count == 1
    assert result.detections[0].redactable is True
    assert result.detections[0].coordinates is not None
    assert result.detections[0].coordinate_source == "pymupdf_words"

    ocr_doc = native_document(text)
    ocr_doc.pages[0].extraction_method = "ocr"
    ocr_service = PrivacyService(FakeRepository(ocr_doc), tmp_path, tmp_path / "ocr-redacted")
    ocr_result = ocr_service.detect(DOCUMENT_ID, PIIDetectRequest(pii_types=[PIIType.EMAIL]))
    assert ocr_result.detections[0].redactable is False
    with pytest.raises(PIIRedactionError):
        ocr_service.redact(DOCUMENT_ID, PIIRedactRequest(detection_ids=[ocr_result.detections[0].detection_id]))


def test_selective_redaction_is_real_and_preserves_original(tmp_path: Path) -> None:
    text = "Email privacy.test@example.com\nCard 4111 1111 1111 1111\nInvoice INV-2026-0043"
    source_path = tmp_path / f"{DOCUMENT_ID}.pdf"
    make_pdf(source_path, text, page_count=2)
    original_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    document = native_document(text)
    document.page_count = 2
    service = PrivacyService(FakeRepository(document), tmp_path, tmp_path / "redacted")

    scan = service.detect(DOCUMENT_ID, PIIDetectRequest())
    email = next(item for item in scan.detections if item.pii_type is PIIType.EMAIL)
    card = next(item for item in scan.detections if item.pii_type is PIIType.CREDIT_CARD)
    response = service.redact(DOCUMENT_ID, PIIRedactRequest(detection_ids=[email.detection_id]))

    artifact_path = tmp_path / "redacted" / f"{DOCUMENT_ID}-{response.artifact.artifact_id}.pdf"
    assert artifact_path.is_file()
    assert response.artifact.page_count == 2
    assert source_path.read_bytes() and hashlib.sha256(source_path.read_bytes()).hexdigest() == original_hash
    with fitz.open(artifact_path) as redacted:
        redacted_text = "\n".join(page.get_text("text") for page in redacted)
        assert "privacy.test@example.com" not in redacted_text
        assert "4111 1111 1111 1111" in redacted_text
        assert "INV-2026-0043" in redacted_text


def test_redaction_does_not_trust_client_coordinates_or_fake_ids(tmp_path: Path) -> None:
    text = "Email privacy.test@example.com"
    source_path = tmp_path / f"{DOCUMENT_ID}.pdf"
    make_pdf(source_path, text)
    service = PrivacyService(FakeRepository(native_document(text)), tmp_path, tmp_path / "redacted")
    with pytest.raises(PIIValidationError):
        service.redact(DOCUMENT_ID, PIIRedactRequest(detection_ids=["pii-000000000000000000000000"]))

    application = create_app(storage_directory=tmp_path / "uploads")
    application.dependency_overrides[get_privacy_service] = lambda: service
    with TestClient(application) as client:
        response = client.post(
            f"/api/v1/documents/{DOCUMENT_ID}/pii/redact",
            json={
                "detection_ids": ["pii-000000000000000000000000"],
                "coordinates": [0, 0, 600, 600],
            },
        )
    assert response.status_code == 422


class FixedPrivacyService:
    def detect(self, document_id, request):
        return PIIDetectionResponse(
            document=PIIDocumentInfo(document_id=document_id, filename="fixture.pdf", page_count=1),
            detections=[],
            detection_count=0,
            counts_by_type={pii_type: 0 for pii_type in (request.pii_types or list(PIIType))},
            detection_time_ms=1,
            total_time_ms=1,
        )

    def redact(self, document_id, request):
        artifact_id = uuid4()
        return PIIRedactResponse(
            document=PIIDocumentInfo(document_id=document_id, filename="fixture.pdf", page_count=1),
            redacted_detection_ids=request.detection_ids,
            redacted_count=len(request.detection_ids),
            redacted_pages=[1],
            artifact=PIIRedactedArtifact(
                artifact_id=artifact_id,
                document_id=document_id,
                filename=f"redacted-{artifact_id}.pdf",
                page_count=1,
                size_bytes=10,
                download_url=f"/api/v1/documents/{document_id}/pii/artifacts/{artifact_id}",
            ),
            redaction_time_ms=1,
            total_time_ms=2,
        )


def test_privacy_api_contract_and_validation(tmp_path: Path) -> None:
    document_id = uuid4()
    application = create_app(storage_directory=tmp_path / "uploads")
    application.dependency_overrides[get_privacy_service] = lambda: FixedPrivacyService()
    with TestClient(application) as client:
        detection = client.post(
            f"/api/v1/documents/{document_id}/pii/detect",
            json={"pii_types": ["email", "email"]},
        )
        invalid_type = client.post(
            f"/api/v1/documents/{document_id}/pii/detect",
            json={"pii_types": ["name"]},
        )
        invalid_uuid = client.post(
            "/api/v1/documents/not-a-uuid/pii/detect",
            json={},
        )
    assert detection.status_code == 200
    assert detection.json()["detection_count"] == 0
    assert invalid_type.status_code == 422
    assert invalid_uuid.status_code == 422
