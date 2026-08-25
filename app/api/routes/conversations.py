"""Thin API routes for persisted multi-turn RAG conversations."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from app.api.routes.rag import get_rag_service
from app.core.config import Settings
from app.core.exceptions import DatabaseNotConfiguredError, DocumentIngestionError
from app.db.repository import ConversationRepository
from app.db.session import Database
from app.models.conversations import (
    ConversationAskRequest,
    ConversationAskResponse,
    ConversationCreate,
    ConversationResponse,
    MessageResponse,
)
from app.services.conversations.query_rewriter import ConversationQueryRewriter
from app.services.conversations.service import ConversationService
from app.services.rag.service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_conversation_service(
    request: Request,
    rag_service: RAGService = Depends(get_rag_service),
) -> ConversationService:
    """Build the conversation orchestration from application-scoped dependencies."""

    database: Database | None = request.app.state.database
    if database is None:
        raise DatabaseNotConfiguredError(
            "PostgreSQL is required for conversation persistence but is not configured."
        )
    settings: Settings = request.app.state.settings
    return ConversationService(
        repository=ConversationRepository(database),
        rag_service=rag_service,
        query_rewriter=ConversationQueryRewriter(request.app.state.ollama_client),
        settings=settings,
    )


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["conversations"],
)
def create_conversation(
    request_body: ConversationCreate,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Create an empty persisted conversation session."""

    try:
        return service.create(request_body.title)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc


@router.get("/conversations", response_model=list[ConversationResponse], tags=["conversations"])
def list_conversations(
    limit: int = Query(default=100, ge=1, le=100),
    service: ConversationService = Depends(get_conversation_service),
) -> list[ConversationResponse]:
    """List recent conversations without returning their message bodies."""

    try:
        return service.list(limit)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    tags=["conversations"],
)
def get_conversation(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Return one conversation's metadata."""

    try:
        return service.get(conversation_id)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
    tags=["conversations"],
)
def get_messages(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
) -> list[MessageResponse]:
    """Return the complete persisted message history in sequence order."""

    try:
        return service.messages(conversation_id)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    tags=["conversations"],
)
def delete_conversation(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
) -> Response:
    """Delete a conversation and its messages transactionally."""

    try:
        service.delete(conversation_id)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/conversations/{conversation_id}/ask",
    response_model=ConversationAskResponse,
    tags=["conversations"],
)
async def ask_in_conversation(
    conversation_id: UUID,
    request_body: ConversationAskRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationAskResponse:
    """Answer one question with bounded prior history and durable messages."""

    try:
        return await service.ask(conversation_id, request_body)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected conversational RAG failure conversation_id=%s", conversation_id)
        raise HTTPException(
            status_code=500,
            detail="The conversational question could not be answered.",
        ) from exc
