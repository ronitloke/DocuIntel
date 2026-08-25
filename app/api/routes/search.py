"""Thin search API route for Module 5 retrieval and Module 6 reranking."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings
from app.core.exceptions import DocumentIngestionError, SearchValidationError
from app.db.repository import DocumentRepository
from app.db.session import Database
from app.models.search import SearchRequest, SearchResponse
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.reranking.cross_encoder import CrossEncoderReranker
from app.services.retrieval.search import SearchService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_search_service(request: Request) -> SearchService:
    """Build the search service from application-scoped database/model resources."""

    database: Database | None = request.app.state.database
    settings: Settings = request.app.state.settings
    return SearchService(
        repository=DocumentRepository(database) if database is not None else None,
        embedding_service=request.app.state.embedding_service,
        settings=settings,
        reranker=request.app.state.reranker,
    )


@router.post("/search", response_model=SearchResponse, tags=["search"])
def search_documents(
    request_body: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Search indexed chunks with semantic, keyword, or hybrid retrieval."""

    try:
        return service.search(request_body)
    except SearchValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected search endpoint failure mode=%s", request_body.mode.value)
        raise HTTPException(status_code=500, detail="The search could not be completed.") from exc
