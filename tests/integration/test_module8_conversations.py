"""PostgreSQL-backed Module 8 persistence and optional real conversational RAG tests."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import fitz
import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.db.repository import ConversationRepository
from app.db.session import Database, create_database
from app.main import create_app
from app.models.conversations import MessageRole
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.llm.ollama import OllamaClient
from app.services.reranking.cross_encoder import CrossEncoderReranker

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.module8,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="Set TEST_DATABASE_URL to an isolated PostgreSQL/pgvector test database.",
    ),
]


def make_pdf() -> bytes:
    """Create a small deterministic policy document for indexed retrieval."""

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 80), "Employment Notice Policy", fontsize=20)
    page.insert_text(
        (72, 120),
        "Employees must give thirty days written notice before resignation.",
        fontsize=12,
    )
    page.insert_text((72, 220), "Product Catalog", fontsize=20)
    page.insert_text(
        (72, 260),
        "The company sells monitors and keyboards to business customers.",
        fontsize=12,
    )
    try:
        return document.tobytes()
    finally:
        document.close()


class FakeEmbeddingModel:
    """Deterministic 384-dimensional vectors for database-backed tests."""

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        vectors: list[list[float]] = []
        for value in texts:
            lowered = value.lower()
            notice = float(any(term in lowered for term in ("notice", "resignation", "employee")))
            product = float(any(term in lowered for term in ("monitor", "keyboard", "product")))
            vectors.append([notice, product] + [0.0] * 382)
        return vectors


class FakeCrossEncoderModel:
    """Prefer the notice section while preserving the CrossEncoder contract."""

    device = "cpu"

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        return [4.0 if "notice" in candidate.lower() else 0.1 for _, candidate in pairs]


@pytest.fixture(scope="module")
def database() -> Iterator[Database]:
    """Apply all migrations to the isolated database for this module."""

    assert TEST_DATABASE_URL is not None
    original_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    get_settings.cache_clear()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(config, "head")
    database = create_database(Settings(database_url=TEST_DATABASE_URL))
    assert database is not None
    try:
        yield database
    finally:
        command.downgrade(config, "base")
        database.engine.dispose()
        if original_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = original_url
        get_settings.cache_clear()


def upload_and_index(client: TestClient, filename: str) -> UUID:
    """Upload and index the controlled policy fixture."""

    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, BytesIO(make_pdf()), "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    document_id = UUID(upload.json()["document_id"])
    indexed = client.post(f"/api/v1/documents/{document_id}/index")
    assert indexed.status_code == 200, indexed.text
    return document_id


def test_conversation_repository_orders_messages_and_cascades(
    database: Database,
) -> None:
    """Messages receive deterministic sequence numbers and delete with their session."""

    repository = ConversationRepository(database)
    conversation = repository.create_conversation()
    first = repository.append_message(conversation.id, MessageRole.USER, "first question")
    second = repository.append_message(conversation.id, MessageRole.ASSISTANT, "first answer")
    third = repository.append_message(conversation.id, MessageRole.USER, "follow up")

    assert [item.sequence_number for item in repository.list_messages(conversation.id)] == [1, 2, 3]
    recent = repository.list_recent_messages(
        conversation.id,
        max_messages=2,
        max_chars=100,
    )
    assert [item.id for item in recent] == [second.id, third.id]
    assert repository.delete_conversation(conversation.id) is True
    assert repository.get_conversation(conversation.id) is None
    assert repository.list_messages(conversation.id) == []


def test_conversational_api_persists_turns_rewrites_follow_up_and_honors_filter(
    database: Database,
    tmp_path: Path,
) -> None:
    """The API reuses filtered Module 5/6 retrieval and persists both turns."""

    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        payloads.append(payload)
        system = str(payload.get("system", ""))
        if "rewrite a user's latest" in system.lower():
            answer = "What employee notice period is required before resignation?"
        else:
            answer = "Employees must give thirty days written notice. [S1]"
        return httpx.Response(200, json={"response": answer})

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="test/module8-embedding",
        embedding_dimension=384,
        ollama_model="test/module8-model",
        rag_history_max_messages=10,
        rag_history_max_chars=6000,
    )
    client = TestClient(
        create_app(
            settings=settings,
            database=database,
            storage_directory=tmp_path / "uploads",
            embedding_service=EmbeddingService(settings, model=FakeEmbeddingModel()),
            reranker=CrossEncoderReranker(settings, model=FakeCrossEncoderModel()),
            ollama_client=OllamaClient(settings, transport=httpx.MockTransport(handler)),
        )
    )
    with client:
        document_id = upload_and_index(client, "module8-conversation.pdf")
        created = client.post("/api/v1/conversations", json={})
        assert created.status_code == 201, created.text
        conversation_id = created.json()["id"]

        first = client.post(
            f"/api/v1/conversations/{conversation_id}/ask",
            json={
                "question": "What notice is required before resignation?",
                "top_k": 1,
                "search_mode": "hybrid",
                "rerank": True,
                "filters": {"document_ids": [str(document_id)]},
            },
        )
        assert first.status_code == 200, first.text
        assert first.json()["retrieval_query"] == "What notice is required before resignation?"

        second = client.post(
            f"/api/v1/conversations/{conversation_id}/ask",
            json={
                "question": "And how many days is that?",
                "top_k": 1,
                "search_mode": "hybrid",
                "rerank": True,
                "filters": {"document_ids": [str(document_id)]},
            },
        )
        assert second.status_code == 200, second.text
        body = second.json()

        messages = client.get(f"/api/v1/conversations/{conversation_id}/messages")
        assert messages.status_code == 200, messages.text
        assert [item["role"] for item in messages.json()] == ["user", "assistant", "user", "assistant"]
        assert [item["sequence_number"] for item in messages.json()] == [1, 2, 3, 4]
        assert body["retrieval_query"] == "What employee notice period is required before resignation?"
        assert body["sources"][0]["document_id"] == str(document_id)
        assert body["citations_valid"] is True
        assert len(payloads) == 3
        assert "And how many days is that?" in str(payloads[-1]["prompt"])
        assert "thirty days" in str(payloads[-1]["prompt"])
        assert payloads[-1]["stream"] is False


def test_conversation_api_returns_no_result_without_generation_call(
    database: Database,
    tmp_path: Path,
) -> None:
    """A first-turn empty retrieval is answered without calling Ollama."""

    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"response": "should not be called"})

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="test/module8-empty-embedding",
        embedding_dimension=384,
        ollama_model="test/module8-empty-model",
    )
    client = TestClient(
        create_app(
            settings=settings,
            database=database,
            storage_directory=tmp_path / "empty-uploads",
            embedding_service=EmbeddingService(settings, model=FakeEmbeddingModel()),
            reranker=CrossEncoderReranker(settings, model=FakeCrossEncoderModel()),
            ollama_client=OllamaClient(settings, transport=httpx.MockTransport(handler)),
        )
    )
    with client:
        conversation = client.post("/api/v1/conversations", json={}).json()
        response = client.post(
            f"/api/v1/conversations/{conversation['id']}/ask",
            json={
                "question": "What is in this unrelated document?",
                "top_k": 1,
                "search_mode": "keyword",
                "rerank": False,
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["sources"] == []
    assert calls == 0


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_CONVERSATION_TEST") != "1",
    reason="Set RUN_REAL_CONVERSATION_TEST=1 for the local Ollama conversation acceptance test.",
)
def test_real_postgresql_ollama_conversation_cold_and_warm(
    database: Database,
    tmp_path: Path,
) -> None:
    """Run two real turns against PostgreSQL, CrossEncoder, and CPU-mode Ollama."""

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3.2:3b",
    )
    with TestClient(
        create_app(
            settings=settings,
            database=database,
            storage_directory=tmp_path / "real-uploads",
            embedding_service=EmbeddingService(settings),
            reranker=CrossEncoderReranker(settings),
            ollama_client=OllamaClient(settings),
        )
    ) as client:
        document_id = upload_and_index(client, "module8-real-conversation.pdf")
        conversation = client.post("/api/v1/conversations", json={}).json()
        conversation_id = conversation["id"]
        first = client.post(
            f"/api/v1/conversations/{conversation_id}/ask",
            json={
                "question": "What notice period is required before resignation?",
                "top_k": 1,
                "search_mode": "hybrid",
                "rerank": True,
                "filters": {"document_ids": [str(document_id)]},
            },
        )
        second = client.post(
            f"/api/v1/conversations/{conversation_id}/ask",
            json={
                "question": "How many days notice are required?",
                "top_k": 1,
                "search_mode": "hybrid",
                "rerank": True,
                "filters": {"document_ids": [str(document_id)]},
            },
        )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_body = first.json()
    second_body = second.json()
    assert first_body["answer"]
    assert second_body["answer"]
    assert first_body["sources"] and second_body["sources"]
    assert first_body["citations_valid"] is True
    assert second_body["citations_valid"] is True
    assert second_body["retrieval_query"]
    print(
        json.dumps(
            {
                "cold": {
                    "question": first_body["question"] if "question" in first_body else "notice period",
                    "answer": first_body["answer"],
                    "sources": first_body["sources"],
                    "timings": first_body,
                },
                "warm": {
                    "retrieval_query": second_body["retrieval_query"],
                    "answer": second_body["answer"],
                    "sources": second_body["sources"],
                    "timings": second_body,
                },
            }
        )
    )
