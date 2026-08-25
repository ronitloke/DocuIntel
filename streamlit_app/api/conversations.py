"""Conversation API adapters for persistent multi-turn RAG."""

from __future__ import annotations

from typing import Any

from .client import ApiClient


def list_conversations(client: ApiClient, *, limit: int = 100) -> list[dict[str, Any]]:
    return client.get("/api/v1/conversations", params={"limit": limit})


def create_conversation(client: ApiClient, title: str | None = None) -> dict[str, Any]:
    return client.post_json("/api/v1/conversations", {"title": title or None})


def get_conversation(client: ApiClient, conversation_id: str) -> dict[str, Any]:
    return client.get(f"/api/v1/conversations/{conversation_id}")


def list_messages(client: ApiClient, conversation_id: str) -> list[dict[str, Any]]:
    return client.get(f"/api/v1/conversations/{conversation_id}/messages")


def ask_in_conversation(
    client: ApiClient,
    conversation_id: str,
    *,
    question: str,
    top_k: int,
    search_mode: str,
    rerank: bool,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return client.post_json(
        f"/api/v1/conversations/{conversation_id}/ask",
        {
            "question": question,
            "top_k": top_k,
            "search_mode": search_mode,
            "rerank": rerank,
            "filters": filters,
        },
    )


def delete_conversation(client: ApiClient, conversation_id: str) -> None:
    client.delete(f"/api/v1/conversations/{conversation_id}")

