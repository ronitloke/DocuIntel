"""Unit tests for the Module 10 HTTP-only presentation adapters."""

from __future__ import annotations

import json

import httpx
import pytest

from streamlit_app.api import conversations, documents, rag, search
from streamlit_app.api.client import ApiClient, ApiError


def make_client(handler):
    return ApiClient("http://testserver", transport=httpx.MockTransport(handler))


def test_client_gets_json_and_uses_relative_api_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    client = make_client(handler)
    try:
        assert client.get("/health") == {"status": "ok"}
    finally:
        client.close()


def test_client_surfaces_detail_without_stack_trace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "Database is not ready."})

    client = make_client(handler)
    try:
        with pytest.raises(ApiError, match="Database is not ready") as raised:
            client.get("/ready")
        assert raised.value.status_code == 503
    finally:
        client.close()


def test_client_handles_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = make_client(handler)
    try:
        with pytest.raises(ApiError, match="timed out"):
            client.get("/health")
    finally:
        client.close()


def test_client_handles_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(handler)
    try:
        with pytest.raises(ApiError, match="unavailable"):
            client.get("/health")
    finally:
        client.close()


def test_upload_uses_pdf_multipart() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/documents/upload"
        assert request.headers["content-type"].startswith("multipart/form-data")
        assert b"sample.pdf" in request.content
        assert b"%PDF" in request.content
        return httpx.Response(201, json={"status": "uploaded"})

    client = make_client(handler)
    try:
        assert documents.upload_document(client, "sample.pdf", b"%PDF-1.7") == {"status": "uploaded"}
    finally:
        client.close()


def test_search_payload_preserves_mode_rerank_and_filters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/v1/search"
        assert payload == {
            "query": "leave policy",
            "mode": "hybrid",
            "top_k": 5,
            "rerank": True,
            "filters": {"document_ids": ["doc-1"]},
        }
        return httpx.Response(200, json={"results": []})

    client = make_client(handler)
    try:
        assert search.search(
            client,
            query="leave policy",
            mode="hybrid",
            top_k=5,
            rerank=True,
            filters={"document_ids": ["doc-1"]},
        ) == {"results": []}
    finally:
        client.close()


def test_rag_payload_uses_ask_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/ask"
        payload = json.loads(request.content)
        assert payload["question"] == "What is the policy?"
        assert payload["search_mode"] == "hybrid"
        assert payload["rerank"] is True
        return httpx.Response(200, json={"answer": "The policy says ... [S1]"})

    client = make_client(handler)
    try:
        response = rag.ask(
            client,
            question="What is the policy?",
            top_k=5,
            search_mode="hybrid",
            rerank=True,
        )
        assert response["answer"].endswith("[S1]")
    finally:
        client.close()


def test_document_index_and_delete_use_existing_endpoints() -> None:
    paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append((request.method, request.url.path))
        return httpx.Response(200 if request.method == "POST" else 204)

    client = make_client(handler)
    try:
        documents.index_document(client, "doc-1")
        documents.delete_document(client, "doc-1")
    finally:
        client.close()
    assert paths == [
        ("POST", "/api/v1/documents/doc-1/index"),
        ("DELETE", "/api/v1/documents/doc-1"),
    ]


def test_conversation_adapter_preserves_persistent_turn_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/conversations/conversation-1/ask"
        payload = json.loads(request.content)
        assert payload["question"] == "And what about exceptions?"
        assert payload["filters"] is None
        return httpx.Response(200, json={"conversation_id": "conversation-1", "sources": []})

    client = make_client(handler)
    try:
        response = conversations.ask_in_conversation(
            client,
            "conversation-1",
            question="And what about exceptions?",
            top_k=5,
            search_mode="hybrid",
            rerank=True,
        )
        assert response["conversation_id"] == "conversation-1"
    finally:
        client.close()


def test_conversation_list_and_messages_use_public_routes() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/messages"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[])

    client = make_client(handler)
    try:
        assert conversations.list_conversations(client) == []
        assert conversations.list_messages(client, "conversation-1") == []
    finally:
        client.close()
    assert seen == ["/api/v1/conversations", "/api/v1/conversations/conversation-1/messages"]

