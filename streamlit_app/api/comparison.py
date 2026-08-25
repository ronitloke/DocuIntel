"""HTTP adapter for the Module 12.3 comparison endpoint."""

from __future__ import annotations

from typing import Any

from .client import ApiClient


def compare_documents(
    client: ApiClient,
    *,
    base_document_id: str,
    target_document_id: str,
    mode: str = "document",
    include_tables: bool = True,
    include_unchanged: bool = False,
    generate_summary: bool = True,
) -> dict[str, Any]:
    """Call the public comparison API with the exact typed request contract."""

    return client.post_json(
        "/api/v1/compare",
        {
            "base_document_id": base_document_id,
            "target_document_id": target_document_id,
            "mode": mode,
            "include_tables": include_tables,
            "include_unchanged": include_unchanged,
            "generate_summary": generate_summary,
        },
    )
