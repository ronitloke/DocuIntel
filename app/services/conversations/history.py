"""Bounded, deterministic conversation history selection for RAG prompts."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.db.models import ConversationMessage
from app.services.conversations.scope import strip_message_scope


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    """Safe prompt projection of one persisted message."""

    id: UUID
    role: str
    content: str
    sequence_number: int


@dataclass(frozen=True, slots=True)
class ConversationHistory:
    """Selected messages and their inspectable prompt representation."""

    messages: list[HistoryMessage]
    text: str


class ConversationHistoryBuilder:
    """Keep only the newest useful messages under count and character budgets."""

    def build(
        self,
        messages: list[ConversationMessage],
        *,
        max_messages: int,
        max_chars: int,
    ) -> ConversationHistory:
        """Select newest messages, then restore chronological order for the model."""

        if max_messages <= 0 or max_chars <= 0:
            return ConversationHistory(messages=[], text="")

        ordered = sorted(messages, key=lambda message: (message.sequence_number, message.id))
        selected: list[HistoryMessage] = []
        used_chars = 0
        for message in reversed(ordered[-max_messages:]):
            role = message.role.value
            prefix = f"{role}: "
            content = strip_message_scope(message.content)
            formatted_length = len(prefix) + len(content)
            separator_length = 2 if selected else 0
            available = max_chars - used_chars - separator_length
            if available <= len(prefix):
                break
            if formatted_length > available:
                content = content[: available - len(prefix)].rstrip()
            if not content:
                break
            selected.append(
                HistoryMessage(
                    id=message.id,
                    role=role,
                    content=content,
                    sequence_number=message.sequence_number,
                )
            )
            used_chars += separator_length + len(prefix) + len(content)
            if len(selected) >= max_messages:
                break

        selected.reverse()
        text = "\n\n".join(f"{item.role}: {item.content}" for item in selected)
        return ConversationHistory(messages=selected, text=text)
