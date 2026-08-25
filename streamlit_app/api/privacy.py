"""HTTP adapters for the Module 12.4 privacy workflow."""

from __future__ import annotations

from typing import Any

from .client import ApiClient


def detect_pii(
    client: ApiClient,
    document_id: str,
    pii_types: list[str],
) -> dict[str, Any]:
    """Call the deterministic PII detection endpoint."""

    return client.post_json(
        f"/api/v1/documents/{document_id}/pii/detect",
        {"pii_types": pii_types},
    )


def redact_pii(
    client: ApiClient,
    document_id: str,
    detection_ids: list[str],
) -> dict[str, Any]:
    """Call the explicit selection redaction endpoint."""

    return client.post_json(
        f"/api/v1/documents/{document_id}/pii/redact",
        {"detection_ids": detection_ids},
    )


def download_redacted_artifact(client: ApiClient, download_url: str) -> bytes:
    """Download only the server-generated artifact URL returned by FastAPI."""

    return client.get_bytes(download_url)
