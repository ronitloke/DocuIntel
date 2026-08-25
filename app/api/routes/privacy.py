"""Thin HTTP routes for local deterministic PII review and redaction."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from app.core.config import PROJECT_ROOT, Settings
from app.core.exceptions import DocumentIngestionError
from app.db.repository import DocumentRepository
from app.db.session import Database
from app.models.privacy import (
    PIIDetectRequest,
    PIIDetectionResponse,
    PIIRedactRequest,
    PIIRedactResponse,
)
from app.services.documents.pdf_ingestion import DEFAULT_UPLOAD_DIRECTORY
from app.services.privacy.service import PrivacyService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_privacy_service(request: Request) -> PrivacyService:
    """Build the privacy service over the existing repository and upload root."""

    database: Database | None = request.app.state.database
    storage_directory = request.app.state.pdf_storage_directory or DEFAULT_UPLOAD_DIRECTORY
    return PrivacyService(
        repository=DocumentRepository(database) if database is not None else None,
        storage_directory=Path(storage_directory),
        artifact_directory=PROJECT_ROOT / "data" / "processed" / "redacted",
    )


@router.post(
    "/documents/{document_id}/pii/detect",
    response_model=PIIDetectionResponse,
    tags=["privacy"],
    summary="Detect high-confidence structured PII",
)
def detect_document_pii(
    document_id: UUID,
    request_body: PIIDetectRequest,
    service: PrivacyService = Depends(get_privacy_service),
) -> PIIDetectionResponse:
    """Scan one ready document without invoking an LLM or retrieval model."""

    try:
        return service.detect(document_id, request_body)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except Exception as exc:
        logger.exception("Unexpected PII detection failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="PII detection could not be completed safely.") from exc


@router.post(
    "/documents/{document_id}/pii/redact",
    response_model=PIIRedactResponse,
    tags=["privacy"],
    summary="Create a new PDF with explicitly selected PII redacted",
)
def redact_document_pii(
    document_id: UUID,
    request_body: PIIRedactRequest,
    service: PrivacyService = Depends(get_privacy_service),
) -> PIIRedactResponse:
    """Revalidate server-issued detection IDs before applying PDF redactions."""

    try:
        return service.redact(document_id, request_body)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except Exception as exc:
        logger.exception("Unexpected PII redaction failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="The redacted PDF could not be created safely.") from exc


@router.get(
    "/documents/{document_id}/pii/artifacts/{artifact_id}",
    response_class=FileResponse,
    tags=["privacy"],
    summary="Download one generated redacted PDF",
)
def download_redacted_artifact(
    document_id: UUID,
    artifact_id: UUID,
    service: PrivacyService = Depends(get_privacy_service),
) -> FileResponse:
    """Serve only a server-generated artifact under the controlled redaction root."""

    try:
        path, filename = service.artifact_path(document_id, artifact_id)
        return FileResponse(path, media_type="application/pdf", filename=filename)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected redacted artifact download failure document_id=%s artifact_id=%s",
            document_id,
            artifact_id,
        )
        raise HTTPException(status_code=500, detail="The redacted artifact could not be downloaded safely.") from exc
