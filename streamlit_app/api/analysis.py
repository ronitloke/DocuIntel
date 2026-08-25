"""HTTP adapters for the Module 11 document-analysis endpoints."""

from __future__ import annotations

from typing import Any

from .client import ApiClient


def summarize(client: ApiClient, document_id: str, *, style: str) -> dict[str, Any]:
    """Request one grounded summary through FastAPI."""

    return client.post_json(
        f"/api/v1/documents/{document_id}/summary",
        {"style": style},
    )


def classify(client: ApiClient, document_id: str, *, labels: list[str]) -> dict[str, Any]:
    """Request one constrained classification through FastAPI."""

    return client.post_json(
        f"/api/v1/documents/{document_id}/classify",
        {"labels": labels},
    )


def extract(
    client: ApiClient,
    document_id: str,
    *,
    fields: list[dict[str, str | None]],
) -> dict[str, Any]:
    """Request evidence-grounded structured extraction through FastAPI."""

    return client.post_json(
        f"/api/v1/documents/{document_id}/extract",
        {"fields": fields},
    )
