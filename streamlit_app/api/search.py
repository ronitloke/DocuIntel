"""Search API adapter that preserves Module 5/6 request semantics."""

from __future__ import annotations

from typing import Any

from .client import ApiClient


def search(
    client: ApiClient,
    *,
    query: str,
    mode: str,
    top_k: int,
    rerank: bool,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return client.post_json(
        "/api/v1/search",
        {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "rerank": rerank,
            "filters": filters,
        },
    )

