"""Grounded single-question RAG API adapter."""

from __future__ import annotations

from typing import Any

from .client import ApiClient


def ask(
    client: ApiClient,
    *,
    question: str,
    top_k: int,
    search_mode: str,
    rerank: bool,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return client.post_json(
        "/api/v1/ask",
        {
            "question": question,
            "top_k": top_k,
            "search_mode": search_mode,
            "rerank": rerank,
            "filters": filters,
        },
    )

