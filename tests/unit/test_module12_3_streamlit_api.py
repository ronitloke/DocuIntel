"""HTTP contract tests for the minimal Module 12.3 Streamlit comparison flow."""

from __future__ import annotations

import json

import httpx

from streamlit_app.api.client import ApiClient, ApiError
from streamlit_app.api.comparison import compare_documents
from streamlit_app.app import format_module_12_3_api_error


def test_comparison_adapter_preserves_route_payload_and_response() -> None:
    seen: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"summary": "Changed [A1][B1].", "changes": []})

    client = ApiClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        response = compare_documents(
            client,
            base_document_id="base-id",
            target_document_id="target-id",
            mode="version",
            include_tables=False,
            include_unchanged=True,
            generate_summary=False,
        )
    finally:
        client.close()

    assert response["summary"] == "Changed [A1][B1]."
    assert seen == [
        (
            "POST",
            "/api/v1/compare",
            {
                "base_document_id": "base-id",
                "target_document_id": "target-id",
                "mode": "version",
                "include_tables": False,
                "include_unchanged": True,
                "generate_summary": False,
            },
        )
    ]


def test_comparison_adapter_errors_are_user_facing_without_traceback() -> None:
    error = ApiError("Both comparison documents must be ready and indexed.", status_code=422)
    assert format_module_12_3_api_error(error) == (
        "Comparison request was rejected: Both comparison documents must be ready and indexed."
    )

    unavailable = ApiError("connection failed", status_code=None)
    assert format_module_12_3_api_error(unavailable) == "connection failed"
