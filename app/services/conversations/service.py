"""Conversation orchestration over persisted history and the existing RAG service."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import UUID

from app.core.config import Settings
from app.db.models import Conversation, ConversationMessage
from app.db.repository import ConversationRepository
from app.models.conversations import (
    ConversationAskRequest,
    ConversationAskResponse,
    ConversationResponse,
    MessageResponse,
    MessageRole,
)
from app.models.search import SearchFilters
from app.services.conversations.history import ConversationHistoryBuilder
from app.services.conversations.query_rewriter import ConversationQueryRewriter
from app.services.conversations.scope import (
    encode_message_scope,
    extract_message_scope,
    strip_message_scope,
)
from app.services.rag.service import RAGService

logger = logging.getLogger(__name__)


class ConversationService:
    """Coordinate durable messages, bounded rewrite context, and conversational RAG."""

    def __init__(
        self,
        repository: ConversationRepository,
        rag_service: RAGService,
        query_rewriter: ConversationQueryRewriter,
        settings: Settings,
        history_builder: ConversationHistoryBuilder | None = None,
    ) -> None:
        self.repository = repository
        self.rag_service = rag_service
        self.query_rewriter = query_rewriter
        self.settings = settings
        self.history_builder = history_builder or ConversationHistoryBuilder()

    def create(self, title: str | None) -> ConversationResponse:
        """Create a session without creating a system prompt or synthetic message."""

        return self._conversation_response(self.repository.create_conversation(title))

    def list(self, limit: int = 100) -> list[ConversationResponse]:
        """List recent sessions without loading message content."""

        return [self._conversation_response(item) for item in self.repository.list_conversations(limit)]

    def get(self, conversation_id: UUID) -> ConversationResponse:
        """Return one session or the repository's controlled 404."""

        return self._conversation_response(self.repository.require_conversation(conversation_id))

    def messages(self, conversation_id: UUID) -> list[MessageResponse]:
        """Return all persisted messages in chronological order."""

        self.repository.require_conversation(conversation_id)
        return [self._message_response(item) for item in self.repository.list_messages(conversation_id)]

    def delete(self, conversation_id: UUID) -> None:
        """Delete a session and its database-cascaded messages."""

        if not self.repository.delete_conversation(conversation_id):
            from app.core.exceptions import ConversationNotFoundError

            raise ConversationNotFoundError("The requested conversation was not found.")

    async def ask(
        self,
        conversation_id: UUID,
        request: ConversationAskRequest,
    ) -> ConversationAskResponse:
        """Persist the question, rewrite only when useful history exists, then answer."""

        total_started = perf_counter()
        self.repository.require_conversation(conversation_id)
        prior_scope_messages = self.repository.list_recent_messages(
            conversation_id,
            max_messages=max(self.settings.rag_history_max_messages + 2, 20),
            max_chars=max(self.settings.rag_history_max_chars * 2, 12000),
        )
        previous_scope = self._latest_scope(prior_scope_messages)
        effective_request = self._preserve_scope(request, previous_scope)
        selected_document_ids = self._request_document_ids(effective_request)
        user_message = self.repository.append_message(
            conversation_id,
            MessageRole.USER,
            encode_message_scope(request.question, selected_document_ids),
        )
        self.repository.set_title_if_empty(conversation_id, request.question[:512])

        history_started = perf_counter()
        recent_messages = self.repository.list_recent_messages(
            conversation_id,
            max_messages=self.settings.rag_history_max_messages + 1,
            max_chars=self.settings.rag_history_max_chars + len(request.question),
        )
        prior_messages = [message for message in recent_messages if message.id != user_message.id]
        history = self.history_builder.build(
            prior_messages,
            max_messages=self.settings.rag_history_max_messages,
            max_chars=self.settings.rag_history_max_chars,
        )
        history_load_time_ms = round((perf_counter() - history_started) * 1000, 3)

        rewrite_started = perf_counter()
        retrieval_query = await self.query_rewriter.rewrite(
            question=request.question,
            history_text=history.text,
        )
        query_rewrite_time_ms = round((perf_counter() - rewrite_started) * 1000, 3)
        logger.info(
            "Conversation RAG retrieval prepared conversation_id=%s history_messages=%s "
            "retrieval_query_length=%s history_ms=%.3f rewrite_ms=%.3f",
            conversation_id,
            len(history.messages),
            len(retrieval_query),
            history_load_time_ms,
            query_rewrite_time_ms,
        )

        try:
            rag_response = await self.rag_service.ask_conversational(
                effective_request,
                retrieval_query=retrieval_query,
                history_text=history.text,
            )
            assistant_message = self.repository.append_message(
                conversation_id,
                MessageRole.ASSISTANT,
                rag_response.answer,
            )
        except Exception:
            logger.exception(
                "Conversation answer failed conversation_id=%s; user message remains persisted",
                conversation_id,
            )
            raise

        total_time_ms = round((perf_counter() - total_started) * 1000, 3)
        logger.info(
            "Conversation RAG request completed conversation_id=%s message_count=%s "
            "generation_ms=%.3f total_ms=%.3f",
            conversation_id,
            len(history.messages) + 2,
            rag_response.generation_time_ms,
            total_time_ms,
        )
        response_data = rag_response.model_dump()
        response_data["total_time_ms"] = total_time_ms
        return ConversationAskResponse(
            **response_data,
            conversation_id=conversation_id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            retrieval_query=retrieval_query,
            history_load_time_ms=history_load_time_ms,
            query_rewrite_time_ms=query_rewrite_time_ms,
        )

    @staticmethod
    def _conversation_response(conversation: Conversation) -> ConversationResponse:
        """Project a detached SQLAlchemy conversation to its API contract."""

        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    @staticmethod
    def _message_response(message: ConversationMessage) -> MessageResponse:
        """Project a detached SQLAlchemy message to its API contract."""

        return MessageResponse(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role,
            content=strip_message_scope(message.content),
            sequence_number=message.sequence_number,
            created_at=message.created_at,
        )

    @staticmethod
    def _latest_scope(messages: list[ConversationMessage]) -> list[UUID]:
        """Find the most recent selected scope without treating assistant text as metadata."""

        for message in reversed(messages):
            if message.role is MessageRole.USER:
                return extract_message_scope(message.content)
        return []

    @staticmethod
    def _request_document_ids(request: ConversationAskRequest) -> list[UUID]:
        """Return explicit selected IDs for safe persistence on the current turn."""

        if request.filters is None or request.filters.document_ids is None:
            return []
        return list(dict.fromkeys(request.filters.document_ids))

    @staticmethod
    def _preserve_scope(
        request: ConversationAskRequest,
        previous_scope: list[UUID],
    ) -> ConversationAskRequest:
        """Keep a prior selected scope unless the caller explicitly sends filters null."""

        if previous_scope and "filters" not in request.model_fields_set:
            return request.model_copy(
                update={"filters": SearchFilters(document_ids=previous_scope)}
            )
        if previous_scope and request.filters is not None:
            if (
                request.filters.document_ids is None
                and "document_ids" not in request.filters.model_fields_set
            ):
                return request.model_copy(
                    update={
                        "filters": request.filters.model_copy(
                            update={"document_ids": previous_scope}
                        )
                    }
                )
        return request
