"""Thin document-management API routes."""

from __future__ import annotations

import logging
from pathlib import Path as FileSystemPath
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import Response

from app.core.config import Settings
from app.core.exceptions import DocumentIngestionError
from app.db.repository import DocumentRepository
from app.db.session import Database
from app.models.documents import (
    DocumentDetailResponse,
    DocumentIngestionResponse,
    DocumentIndexingResponse,
    DocumentListResponse,
    DocumentStatus,
    ChunkListResponse,
    ChunkListItem,
    PageListResponse,
    PersistedPageResponse,
)
from app.services.documents.document_management import DocumentManagementService
from app.services.documents.indexing import DocumentIndexingService
from app.services.chunking.structure_aware import StructureAwareChunker
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.documents.pdf_ingestion import DEFAULT_UPLOAD_DIRECTORY, PDFIngestionService
from app.services.ocr.tesseract_ocr import OCRService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_document_management_service(request: Request) -> DocumentManagementService:
    """Build the management service from application-scoped resources."""

    settings: Settings = request.app.state.settings
    storage_directory: FileSystemPath = (
        request.app.state.pdf_storage_directory or DEFAULT_UPLOAD_DIRECTORY
    )
    ocr_service: OCRService | None = request.app.state.ocr_service
    database: Database | None = request.app.state.database
    repository = DocumentRepository(database) if database is not None else None
    return DocumentManagementService(
        ingestion_service=PDFIngestionService(
            settings=settings,
            storage_directory=storage_directory,
            ocr_service=ocr_service,
        ),
        repository=repository,
        storage_directory=storage_directory,
        persistence_required=request.app.state.persistence_required,
    )


def _raise_document_http_error(exc: DocumentIngestionError) -> None:
    """Translate a safe application error into an HTTP error."""

    raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc


def get_document_indexing_service(request: Request) -> DocumentIndexingService:
    """Build the indexing workflow from application-scoped resources."""

    database: Database | None = request.app.state.database
    embedding_service: EmbeddingService = request.app.state.embedding_service
    return DocumentIndexingService(
        repository=DocumentRepository(database) if database is not None else None,
        chunker=StructureAwareChunker(settings=request.app.state.settings),
        embedding_service=embedding_service,
    )


@router.post(
    "/upload",
    response_model=DocumentIngestionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    summary="Upload and persist one PDF",
    description=(
        "Validate, store, extract, and persist one PDF. Native text, selective "
        "Tesseract OCR, heuristic layout, and native tables are recorded in PostgreSQL."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="A PDF document to ingest."),
    service: DocumentManagementService = Depends(get_document_management_service),
) -> DocumentIngestionResponse:
    """Ingest one PDF and commit its metadata and structure atomically."""

    try:
        return await service.ingest(file)
    except DocumentIngestionError as exc:
        logger.warning(
            "PDF ingestion rejected filename=%s reason=%s",
            file.filename,
            exc.public_message,
        )
        _raise_document_http_error(exc)
    except Exception as exc:
        logger.exception("Unexpected upload endpoint failure filename=%s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The PDF could not be processed.",
        ) from exc
    finally:
        await file.close()


@router.get("", response_model=DocumentListResponse, tags=["documents"])
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    document_status: DocumentStatus | None = Query(default=None, alias="status"),
    service: DocumentManagementService = Depends(get_document_management_service),
) -> DocumentListResponse:
    """Return a SQL-paginated list of persisted documents."""

    try:
        return service.list_documents(page, page_size, document_status)
    except DocumentIngestionError as exc:
        _raise_document_http_error(exc)
    except Exception as exc:
        logger.exception("Unexpected document-list endpoint failure")
        raise HTTPException(status_code=500, detail="Documents could not be listed.") from exc


@router.get("/{document_id}", response_model=DocumentDetailResponse, tags=["documents"])
def get_document(
    document_id: UUID,
    service: DocumentManagementService = Depends(get_document_management_service),
) -> DocumentDetailResponse:
    """Return document metadata and persisted processing statistics."""

    try:
        return service.get_document(document_id)
    except DocumentIngestionError as exc:
        _raise_document_http_error(exc)
    except Exception as exc:
        logger.exception("Unexpected document detail endpoint failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="The document could not be loaded.") from exc


@router.post(
    "/{document_id}/index",
    response_model=DocumentIndexingResponse,
    status_code=status.HTTP_200_OK,
    tags=["documents"],
    summary="Generate chunks and local embeddings for one document",
)
def index_document(
    document_id: UUID,
    service: DocumentIndexingService = Depends(get_document_indexing_service),
) -> DocumentIndexingResponse:
    """Chunk and embed a previously persisted document without adding search."""

    try:
        result = service.index_document(document_id)
        return DocumentIndexingResponse(
            document_id=result.document_id,
            chunks_created=result.chunks_created,
            embeddings_created=result.embeddings_created,
            embedding_model=result.embedding_model,
            embedding_dimension=result.embedding_dimension,
            status=result.status,
        )
    except DocumentIngestionError as exc:
        _raise_document_http_error(exc)
    except Exception as exc:
        logger.exception("Unexpected document-index endpoint failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="The document could not be indexed.") from exc


@router.get("/{document_id}/pages", response_model=PageListResponse, tags=["documents"])
def list_document_pages(
    document_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: DocumentManagementService = Depends(get_document_management_service),
) -> PageListResponse:
    """Return paginated persisted page metadata."""

    try:
        return service.list_pages(document_id, page, page_size)
    except DocumentIngestionError as exc:
        _raise_document_http_error(exc)
    except Exception as exc:
        logger.exception("Unexpected page-list endpoint failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Pages could not be listed.") from exc


@router.get("/{document_id}/chunks", response_model=ChunkListResponse, tags=["documents"])
def list_document_chunks(
    document_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: DocumentManagementService = Depends(get_document_management_service),
) -> ChunkListResponse:
    """Return paginated chunk text and metadata without raw vectors."""

    try:
        return service.list_chunks(document_id, page, page_size)
    except DocumentIngestionError as exc:
        _raise_document_http_error(exc)
    except Exception as exc:
        logger.exception("Unexpected chunk-list endpoint failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="Chunks could not be listed.") from exc


@router.get(
    "/{document_id}/chunks/{chunk_id}",
    response_model=ChunkListItem,
    tags=["documents"],
)
def get_document_chunk(
    document_id: UUID,
    chunk_id: UUID,
    service: DocumentManagementService = Depends(get_document_management_service),
) -> ChunkListItem:
    """Return one chunk's text and metadata without its vector."""

    try:
        return service.get_chunk(document_id, chunk_id)
    except DocumentIngestionError as exc:
        _raise_document_http_error(exc)
    except Exception as exc:
        logger.exception(
            "Unexpected chunk-detail endpoint failure document_id=%s chunk_id=%s",
            document_id,
            chunk_id,
        )
        raise HTTPException(status_code=500, detail="The chunk could not be loaded.") from exc


@router.get(
    "/{document_id}/pages/{page_number}",
    response_model=PersistedPageResponse,
    tags=["documents"],
)
def get_document_page(
    document_id: UUID,
    page_number: int,
    service: DocumentManagementService = Depends(get_document_management_service),
) -> PersistedPageResponse:
    """Return one page's text, OCR state, layout elements, and tables."""

    if page_number < 1:
        raise HTTPException(status_code=422, detail="page_number must be at least 1.")
    try:
        return service.get_page(document_id, page_number)
    except DocumentIngestionError as exc:
        _raise_document_http_error(exc)
    except Exception as exc:
        logger.exception(
            "Unexpected page detail endpoint failure document_id=%s page=%s",
            document_id,
            page_number,
        )
        raise HTTPException(status_code=500, detail="The page could not be loaded.") from exc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"])
def delete_document(
    document_id: UUID,
    service: DocumentManagementService = Depends(get_document_management_service),
) -> Response:
    """Delete one document, its dependent rows, and its generated local PDF."""

    try:
        service.delete_document(document_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except DocumentIngestionError as exc:
        _raise_document_http_error(exc)
    except Exception as exc:
        logger.exception("Unexpected document-delete endpoint failure document_id=%s", document_id)
        raise HTTPException(status_code=500, detail="The document could not be deleted.") from exc
