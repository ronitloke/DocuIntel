"""Thin FastAPI routes for Module 11 document analysis."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings
from app.core.exceptions import DocumentIngestionError
from app.db.repository import DocumentRepository
from app.db.session import Database
from app.models.analysis import (
    DocumentClassificationRequest,
    DocumentClassificationResponse,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
)
from app.services.analysis.classifier import DocumentClassifier
from app.services.analysis.grounding import GroundingVerifier
from app.services.analysis.service import AnalysisService
from app.services.analysis.summarizer import DocumentSummarizer

logger = logging.getLogger(__name__)
router = APIRouter()


def get_analysis_service(request: Request) -> AnalysisService:
    """Build analysis orchestration from application-scoped resources."""

    settings: Settings = request.app.state.settings
    database: Database | None = request.app.state.database
    repository = DocumentRepository(database) if database is not None else None
    grounding_verifier = GroundingVerifier(request.app.state.ollama_client)
    summarizer = DocumentSummarizer(
        request.app.state.ollama_client,
        grounding_verifier=grounding_verifier,
        grounding_enabled=settings.summary_grounding_enabled,
        grounding_max_passes=settings.summary_grounding_max_passes,
    )
    classifier = DocumentClassifier(request.app.state.ollama_client)
    return AnalysisService(repository, summarizer, classifier, settings)


@router.post(
    "/documents/{document_id}/summary",
    response_model=DocumentSummaryResponse,
    tags=["analysis"],
    summary="Generate a grounded document summary",
)
async def summarize_document(
    document_id: UUID,
    request_body: DocumentSummaryRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> DocumentSummaryResponse:
    """Summarize ordered indexed chunks without persisting generated text."""

    try:
        return await service.summarize(document_id, request_body)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected document summary failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="The document could not be summarized.") from exc


@router.post(
    "/documents/{document_id}/classify",
    response_model=DocumentClassificationResponse,
    tags=["analysis"],
    summary="Classify a document against supplied labels",
)
async def classify_document(
    document_id: UUID,
    request_body: DocumentClassificationRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> DocumentClassificationResponse:
    """Classify ordered indexed chunks using only caller-supplied labels."""

    try:
        return await service.classify(document_id, request_body)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected document classification failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="The document could not be classified.") from exc
