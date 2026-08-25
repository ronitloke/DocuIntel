"""Unit coverage for the Module 11 Streamlit HTTP adapters."""

from __future__ import annotations

import json

import httpx

from streamlit_app.api import analysis
from streamlit_app.api.client import ApiClient


def test_analysis_adapters_use_public_fastapi_routes() -> None:
    """Analyze requests are HTTP-only and preserve backend payload contracts."""

    seen: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append((request.url.path, payload))
        return httpx.Response(200, json={"document_id": "doc-1"})

    client = ApiClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        assert analysis.summarize(client, "doc-1", style="brief")["document_id"] == "doc-1"
        assert analysis.classify(
            client,
            "doc-1",
            labels=["Employment Policy", "Other"],
        )["document_id"] == "doc-1"
    finally:
        client.close()

    assert seen == [
        ("/api/v1/documents/doc-1/summary", {"style": "brief"}),
        (
            "/api/v1/documents/doc-1/classify",
            {"labels": ["Employment Policy", "Other"]},
        ),
    ]
