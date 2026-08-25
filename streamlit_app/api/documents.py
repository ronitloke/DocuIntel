"""Document-management API adapters."""

from __future__ import annotations

from typing import Any

from .client import ApiClient


def list_documents(
    client: ApiClient,
    *,
    page: int = 1,
    page_size: int = 100,
    status: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"page": page, "page_size": page_size}
    if status:
        params["status"] = status
    return client.get("/api/v1/documents", params=params)


def upload_document(client: ApiClient, filename: str, content: bytes) -> dict[str, Any]:
    return client.upload_pdf("/api/v1/documents/upload", filename, content)


def get_document(client: ApiClient, document_id: str) -> dict[str, Any]:
    return client.get(f"/api/v1/documents/{document_id}")


def index_document(client: ApiClient, document_id: str) -> dict[str, Any]:
    return client.post_json(f"/api/v1/documents/{document_id}/index", {})


def list_pages(client: ApiClient, document_id: str, *, page_size: int = 100) -> dict[str, Any]:
    return client.get(
        f"/api/v1/documents/{document_id}/pages",
        params={"page": 1, "page_size": page_size},
    )


def get_page(client: ApiClient, document_id: str, page_number: int) -> dict[str, Any]:
    return client.get(f"/api/v1/documents/{document_id}/pages/{page_number}")


def list_chunks(client: ApiClient, document_id: str, *, page_size: int = 100) -> dict[str, Any]:
    return client.get(
        f"/api/v1/documents/{document_id}/chunks",
        params={"page": 1, "page_size": page_size},
    )


def delete_document(client: ApiClient, document_id: str) -> None:
    client.delete(f"/api/v1/documents/{document_id}")

