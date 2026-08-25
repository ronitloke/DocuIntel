"""Thin API routes for Module 12.2 extraction and structured table querying."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.config import Settings
from app.core.exceptions import DocumentIngestionError
from app.db.repository import DocumentRepository
from app.db.session import Database
from app.models.structured import (
    StructuredExtractionRequest,
    StructuredExtractionResponse,
    TableInventoryResponse,
    TablePreviewResponse,
    TableQueryRequest,
    TableQueryResponse,
)
from app.services.analysis.structured_extraction import StructuredExtractionService
from app.services.tables.query import TableQueryService

logger = logging.getLogger(__name__)
router = APIRouter()


def _repository(request: Request) -> DocumentRepository | None:
    """Build the existing repository adapter from application state."""

    database: Database | None = request.app.state.database
    return DocumentRepository(database) if database is not None else None


def get_structured_extraction_service(request: Request) -> StructuredExtractionService:
    """Build one transient extraction orchestrator."""

    settings: Settings = request.app.state.settings
    return StructuredExtractionService(
        repository=_repository(request),
        ollama_client=request.app.state.ollama_client,
        settings=settings,
    )


def get_table_query_service(request: Request) -> TableQueryService:
    """Build one transient deterministic table-query orchestrator."""

    settings: Settings = request.app.state.settings
    return TableQueryService(
        repository=_repository(request),
        ollama_client=request.app.state.ollama_client,
        settings=settings,
    )


@router.post(
    "/documents/{document_id}/extract",
    response_model=StructuredExtractionResponse,
    tags=["analysis"],
    summary="Extract caller-defined structured fields",
)
async def extract_document_fields(
    document_id: UUID,
    request_body: StructuredExtractionRequest,
    service: StructuredExtractionService = Depends(get_structured_extraction_service),
) -> StructuredExtractionResponse:
    """Extract typed values from one explicitly selected indexed document."""

    try:
        return await service.extract(document_id, request_body)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected structured extraction failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="The document could not be extracted safely.") from exc


@router.get(
    "/documents/{document_id}/tables",
    response_model=TableInventoryResponse,
    tags=["tables"],
    summary="List structured tables in one document",
)
def list_document_tables(
    document_id: UUID,
    service: TableQueryService = Depends(get_table_query_service),
) -> TableInventoryResponse:
    """Return table IDs, dimensions, headers, and page provenance."""

    try:
        return service.inventory(document_id)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except Exception as exc:
        logger.exception("Unexpected table inventory failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Tables could not be listed safely.") from exc


@router.get(
    "/documents/{document_id}/tables/{table_id}",
    response_model=TablePreviewResponse,
    tags=["tables"],
    summary="Preview one structured table",
)
def preview_document_table(
    document_id: UUID,
    table_id: UUID,
    preview_rows: int = Query(default=20, ge=0, le=100),
    service: TableQueryService = Depends(get_table_query_service),
) -> TablePreviewResponse:
    """Return a small table preview without sending unbounded rows to the UI."""

    try:
        return service.preview(document_id, table_id, preview_rows=preview_rows)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except Exception as exc:
        logger.exception("Unexpected table preview failure table_id=%s", table_id)
        raise HTTPException(status_code=500, detail="The table preview could not be loaded.") from exc


@router.post(
    "/documents/{document_id}/tables/{table_id}/query",
    response_model=TableQueryResponse,
    tags=["tables"],
    summary="Run a safe deterministic table query",
)
async def query_document_table(
    document_id: UUID,
    table_id: UUID,
    request_body: TableQueryRequest,
    service: TableQueryService = Depends(get_table_query_service),
) -> TableQueryResponse:
    """Translate an optional natural-language question into a finite table plan."""

    try:
        return await service.query(document_id, table_id, request_body)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected table query failure table_id=%s", table_id)
        raise HTTPException(status_code=500, detail="The table query could not be completed.") from exc
