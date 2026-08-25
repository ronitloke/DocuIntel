"""Pydantic contracts for persisted conversation sessions and messages."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.models.rag import AskRequest, AskResponse


class MessageRole(str, Enum):
    """Roles that can be persisted in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"


class ConversationCreate(BaseModel):
    """Optional metadata for creating a new conversation."""

    title: str | None = Field(default=None, max_length=512)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        """Treat blank titles as omitted while preserving meaningful text."""

        if value is None:
            return None
        title = value.strip()
        return title or None


class ConversationResponse(BaseModel):
    """Persisted conversation metadata."""

    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    """One persisted user or assistant message."""

    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    sequence_number: int = Field(ge=1)
    created_at: datetime


class ConversationAskRequest(AskRequest):
    """A conversational question using the existing RAG request options."""


class ConversationAskResponse(AskResponse):
    """Generated answer plus persistence and conversational retrieval metadata."""

    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID | None = None
    retrieval_query: str
    history_load_time_ms: float = Field(ge=0)
    query_rewrite_time_ms: float = Field(ge=0)
