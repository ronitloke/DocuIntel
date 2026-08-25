"""HTTP adapters for Module 12.2 table inventory and querying."""

from __future__ import annotations

from typing import Any

from .client import ApiClient


def list_tables(client: ApiClient, document_id: str) -> dict[str, Any]:
    """List structured tables for one document."""

    return client.get(f"/api/v1/documents/{document_id}/tables")


def preview_table(
    client: ApiClient,
    document_id: str,
    table_id: str,
    *,
    preview_rows: int = 20,
) -> dict[str, Any]:
    """Load a bounded table preview."""

    return client.get(
        f"/api/v1/documents/{document_id}/tables/{table_id}",
        params={"preview_rows": preview_rows},
    )


def query_table(
    client: ApiClient,
    document_id: str,
    table_id: str,
    *,
    question: str,
) -> dict[str, Any]:
    """Ask one table question through the public API."""

    return client.post_json(
        f"/api/v1/documents/{document_id}/tables/{table_id}/query",
        {"question": question},
    )
