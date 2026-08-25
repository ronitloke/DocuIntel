"""Thin API route for grounded single-question RAG."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings
from app.core.exceptions import DocumentIngestionError
from app.db.repository import DocumentRepository
from app.db.session import Database
from app.models.rag import AskRequest, AskResponse
from app.services.llm.ollama import OllamaClient
from app.services.rag.service import RAGService
from app.services.retrieval.search import SearchService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    """Build RAG orchestration from application-scoped dependencies."""

    database: Database | None = request.app.state.database
    settings: Settings = request.app.state.settings
    search_service = SearchService(
        repository=DocumentRepository(database) if database is not None else None,
        embedding_service=request.app.state.embedding_service,
        settings=settings,
        reranker=request.app.state.reranker,
    )
    return RAGService(
        search_service=search_service,
        ollama_client=request.app.state.ollama_client,
        settings=settings,
    )


@router.post("/ask", response_model=AskResponse, tags=["rag"])
async def ask_documents(
    request_body: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    """Answer one question using retrieved document evidence and Ollama."""

    try:
        return await service.ask(request_body)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected RAG endpoint failure mode=%s", request_body.search_mode.value)
        raise HTTPException(status_code=500, detail="The question could not be answered.") from exc
