"""Tests for the Module 12.2 Streamlit HTTP-only adapters and field UX parser."""

from __future__ import annotations

import json

import httpx

from streamlit_app.api import analysis, tables
from streamlit_app.api.client import ApiClient
from streamlit_app.app import parse_extraction_field_lines


def test_extraction_field_lines_parse_bounded_definitions_and_reject_malformed_lines() -> None:
    fields, errors = parse_extraction_field_lines(
        "notice_period | string | Required notice\n"
        "invoice_reference | string\n"
        "bad-name | string\n"
        "employee_name | python_eval\n"
    )

    assert fields == [
        {"name": "notice_period", "type": "string", "description": "Required notice"},
        {"name": "invoice_reference", "type": "string", "description": None},
    ]
    assert len(errors) == 2


def test_extraction_and_table_adapters_preserve_public_routes_and_payloads() -> None:
    seen: list[tuple[str, str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, payload))
        if request.url.path.endswith("/tables"):
            return httpx.Response(200, json={"tables": []})
        if request.method == "GET":
            return httpx.Response(200, json={"table_id": "table-1", "rows": []})
        return httpx.Response(200, json={"answer": "A fact. [T1]"})

    client = ApiClient("http://testserver", transport=httpx.MockTransport(handler))
    try:
        analysis.extract(
            client,
            "doc-1",
            fields=[{"name": "notice_period", "type": "string", "description": None}],
        )
        tables.list_tables(client, "doc-1")
        tables.preview_table(client, "doc-1", "table-1")
        tables.query_table(client, "doc-1", "table-1", question="Which is highest?")
    finally:
        client.close()

    assert seen == [
        (
            "POST",
            "/api/v1/documents/doc-1/extract",
            {"fields": [{"name": "notice_period", "type": "string", "description": None}]},
        ),
        ("GET", "/api/v1/documents/doc-1/tables", None),
        ("GET", "/api/v1/documents/doc-1/tables/table-1", None),
        (
            "POST",
            "/api/v1/documents/doc-1/tables/table-1/query",
            {"question": "Which is highest?"},
        ),
    ]
