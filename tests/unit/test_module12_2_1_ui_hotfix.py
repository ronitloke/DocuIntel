"""Regression tests for the Module 12.2.1 Streamlit adapter/UI hotfix."""

from __future__ import annotations

import json

import httpx

from streamlit_app.api import analysis, tables
from streamlit_app.api.client import ApiClient, ApiError
from streamlit_app.app import format_module_12_2_api_error, structured_table_empty_state


def test_extraction_adapter_exists_and_deserializes_structured_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/documents/doc-1/extract"
        assert json.loads(request.content) == {
            "fields": [
                {
                    "name": "notice_period",
                    "type": "string",
                    "description": "Required notice",
                }
            ]
        }
        return httpx.Response(
            200,
            json={
                "document_id": "doc-1",
                "fields": [
                    {
                        "field": "notice_period",
                        "value": "thirty days",
                        "status": "found",
                        "sources": ["S1"],
                        "candidates": [],
                    }
                ],
                "sources": [],
            },
        )

    client = ApiClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        response = analysis.extract(
            client,
            "doc-1",
            fields=[
                {
                    "name": "notice_period",
                    "type": "string",
                    "description": "Required notice",
                }
            ],
        )
    finally:
        client.close()

    assert response["fields"][0]["value"] == "thirty days"
    assert response["fields"][0]["sources"] == ["S1"]


def test_table_adapters_use_exact_paths_and_preview_query_parameter() -> None:
    seen: list[tuple[str, str, dict[str, str], object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, dict(request.url.params), payload))
        return httpx.Response(200, json={"tables": []} if request.url.path.endswith("/tables") else {})

    client = ApiClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        tables.list_tables(client, "doc-1")
        tables.preview_table(client, "doc-1", "table-1", preview_rows=20)
        tables.query_table(client, "doc-1", "table-1", question="Which is highest?")
    finally:
        client.close()

    assert seen == [
        ("GET", "/api/v1/documents/doc-1/tables", {}, None),
        ("GET", "/api/v1/documents/doc-1/tables/table-1", {"preview_rows": "20"}, None),
        (
            "POST",
            "/api/v1/documents/doc-1/tables/table-1/query",
            {},
            {"question": "Which is highest?"},
        ),
    ]


def test_no_table_inventory_is_a_normal_empty_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"document_id": "doc-1", "filename": "manual.pdf", "tables": []})

    client = ApiClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        response = tables.list_tables(client, "doc-1")
    finally:
        client.close()

    assert response["tables"] == []
    assert structured_table_empty_state(response["tables"]) == (
        "No structured tables were detected for this document."
    )
    assert structured_table_empty_state([{"table_id": "table-1"}]) is None


def test_validation_and_missing_endpoint_errors_are_user_facing() -> None:
    validation = ApiError("Column Product contains non-numeric values.", status_code=422)
    missing = ApiError("Not Found", status_code=404)

    assert format_module_12_2_api_error(validation, action="table_query") == (
        "Table query request was rejected: Column Product contains non-numeric values."
    )
    assert "Not Found" not in format_module_12_2_api_error(missing, action="table_inventory")
    assert "Restart FastAPI" in format_module_12_2_api_error(missing, action="table_inventory")


def test_extraction_validation_error_is_not_a_traceback() -> None:
    error = ApiError("Field definitions are invalid.", status_code=422)

    assert format_module_12_2_api_error(error, action="extraction") == (
        "Structured extraction request was rejected: Field definitions are invalid."
    )
