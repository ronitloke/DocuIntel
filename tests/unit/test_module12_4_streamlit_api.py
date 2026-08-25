"""HTTP adapter and safe UI error tests for Module 12.4."""

from __future__ import annotations

import json

import httpx

from streamlit_app.api.client import ApiClient, ApiError
from streamlit_app.api.privacy import detect_pii, download_redacted_artifact, redact_pii
from streamlit_app.app import format_module_12_4_api_error


def test_privacy_adapters_preserve_routes_payloads_and_binary_download() -> None:
    seen: list[tuple[str, str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=b"%PDF-test")
        seen.append((request.method, request.url.path, json.loads(request.content)))
        if request.url.path.endswith("/detect"):
            return httpx.Response(200, json={"detection_count": 1})
        return httpx.Response(200, json={"redacted_count": 1})

    client = ApiClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        scan = detect_pii(client, "document-id", ["email", "credit_card"])
        redaction = redact_pii(client, "document-id", ["pii-123"])
        content = download_redacted_artifact(client, "/api/v1/documents/document-id/pii/artifacts/artifact-id")
    finally:
        client.close()

    assert scan == {"detection_count": 1}
    assert redaction == {"redacted_count": 1}
    assert content == b"%PDF-test"
    assert seen == [
        (
            "POST",
            "/api/v1/documents/document-id/pii/detect",
            {"pii_types": ["email", "credit_card"]},
        ),
        (
            "POST",
            "/api/v1/documents/document-id/pii/redact",
            {"detection_ids": ["pii-123"]},
        ),
    ]


def test_privacy_ui_error_mapping_does_not_expose_tracebacks() -> None:
    assert format_module_12_4_api_error(
        ApiError("At least one selected detection has no verified PDF coordinates.", status_code=422),
        action="redaction",
    ) == (
        "PDF redaction was rejected: At least one selected detection has no verified PDF coordinates."
    )
    assert format_module_12_4_api_error(ApiError("connection failed"), action="scan") == "connection failed"
