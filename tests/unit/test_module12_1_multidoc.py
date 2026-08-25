"""Deterministic unit coverage for explicit multi-document RAG scope."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.core.config import Settings
from app.core.exceptions import SearchValidationError
from app.db.repository import SearchCandidate
from app.models.rag import AskRequest
from app.models.conversations import ConversationAskRequest, MessageRole
from app.models.search import SearchFilters, SearchMode, SearchRequest
from app.services.conversations.service import ConversationService
from app.services.rag.prompts import GROUNDED_SYSTEM_PROMPT, build_user_prompt
from app.services.rag.service import RAGService
from app.services.reranking.cross_encoder import RerankedCandidate
from app.services.retrieval.search import SearchService


class ScopeRepository:
    """Repository double exposing only the selected-scope search contract."""

    def __init__(self, document_ids: list[UUID]) -> None:
        self.document_ids = set(document_ids)

    def indexed_document_ids(self, document_ids: Iterable[UUID]) -> set[UUID]:
        return set(document_ids).intersection(self.document_ids)

    def semantic_search(self, *, query_embedding, limit, filters, min_similarity):
        return self._candidates(filters)

    def keyword_search(self, *, query, limit, text_search_config, filters):
        return self._candidates(filters)

    @staticmethod
    def _candidates(filters: SearchFilters | None) -> list[SearchCandidate]:
        assert filters is not None and filters.document_ids is not None
        document_id = filters.document_ids[0]
        return [
            SearchCandidate(
                chunk_id=UUID(int=document_id.int ^ 1),
                document_id=document_id,
                original_filename=f"document-{document_id.hex[:6]}.pdf",
                sequence_number=1,
                text=f"Document {document_id} states the notice period is thirty days.",
                section_heading="Notice",
                start_page=1,
                end_page=1,
                content_type="text",
                contains_ocr=False,
                score=0.9,
            )
        ]


class FixedEmbedding:
    """Embedding double for the existing SearchService abstraction."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]


class FixedReranker:
    """Reranker double proving that selected candidates share one rerank pass."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, candidates) -> list[RerankedCandidate]:
        self.calls += 1
        return [
            RerankedCandidate(item.candidate_id, rank, float(-rank))
            for rank, item in reversed(list(enumerate(candidates, start=1)))
        ]


class FixedOllama:
    """Generation double that exposes the final multi-document prompt."""

    model = "test-model"

    def __init__(self) -> None:
        self.user_prompt = ""

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.user_prompt = user_prompt
        return "The sources report a thirty-day notice period. [S1] [S2]"


class ConversationRepositoryDouble:
    """Small persistence double proving scope survives a follow-up request."""

    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.conversation = SimpleNamespace(
            id=uuid4(),
            title=None,
            created_at=now,
            updated_at=now,
        )
        self.messages: list[SimpleNamespace] = []

    def require_conversation(self, conversation_id: UUID):
        assert conversation_id == self.conversation.id
        return self.conversation

    def list_recent_messages(self, conversation_id: UUID, *, max_messages: int, max_chars: int):
        return self.messages[-max_messages:]

    def list_messages(self, conversation_id: UUID):
        return list(self.messages)

    def append_message(self, conversation_id: UUID, role: MessageRole, content: str):
        message = SimpleNamespace(
            id=uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            sequence_number=len(self.messages) + 1,
            created_at=datetime.now(UTC),
        )
        self.messages.append(message)
        return message

    def set_title_if_empty(self, conversation_id: UUID, title: str) -> None:
        self.conversation.title = self.conversation.title or title


class RecordingConversationRAG:
    """RAG double recording the effective scope and rewritten query."""

    model = "test-model"

    def __init__(self) -> None:
        self.requests: list[ConversationAskRequest] = []
        self.queries: list[str] = []

    async def ask_conversational(self, request, *, retrieval_query: str, history_text: str):
        self.requests.append(request)
        self.queries.append(retrieval_query)
        from app.models.rag import AskResponse

        return AskResponse(
            answer="Supported answer.",
            model=self.model,
            sources=[],
            citations=[],
            citations_valid=False,
            retrieval_time_ms=1,
            generation_time_ms=1,
            total_time_ms=2,
        )


class FixedConversationRewriter:
    """Rewrite double that keeps follow-up behavior deterministic."""

    async def rewrite(self, *, question: str, history_text: str) -> str:
        return "standalone notice comparison" if history_text else question


def make_settings() -> Settings:
    return Settings(
        search_candidate_multiplier=1,
        rag_selected_document_candidate_multiplier=1,
        rag_max_selected_documents=2,
        rag_max_selected_candidates=10,
        rerank_max_candidates=10,
    )


def test_selected_scope_deduplicates_ids_and_merges_per_document_candidates() -> None:
    """Each selected document gets retrieval opportunity before one shared result boundary."""

    first, second = uuid4(), uuid4()
    search = SearchService(
        ScopeRepository([first, second]),
        FixedEmbedding(),
        make_settings(),
    )
    response = search.search_selected_documents(
        SearchRequest(
            query="notice period",
            mode=SearchMode.HYBRID,
            top_k=2,
            rerank=False,
            filters=SearchFilters(document_ids=[first, second, first]),
        )
    )

    assert {result.document_id for result in response.results} == {first, second}
    assert response.reranked is False
    assert response.retrieval_time_ms >= 0


def test_selected_scope_reranks_combined_candidates_and_projects_provenance() -> None:
    """Reranking is applied once after per-document retrieval and source IDs remain exact."""

    first, second = uuid4(), uuid4()
    reranker = FixedReranker()
    search = SearchService(
        ScopeRepository([first, second]),
        FixedEmbedding(),
        make_settings(),
        reranker=reranker,
    )
    ollama = FixedOllama()
    response = asyncio.run(
        RAGService(search, ollama, make_settings()).ask(
            AskRequest(
                question="What is the notice period?",
                top_k=2,
                search_mode=SearchMode.HYBRID,
                rerank=True,
                filters=SearchFilters(document_ids=[first, second, first]),
            )
        )
    )

    assert reranker.calls == 1
    assert response.document_scope == "selected"
    assert response.selected_document_ids == [first, second]
    assert set(response.retrieved_document_ids) == {first, second}
    assert set(response.source_document_ids) == {first, second}
    assert [source.source_id for source in response.sources] == ["S1", "S2"]
    assert "Analyze each supplied document independently" in ollama.user_prompt
    assert response.citations_valid is True


def test_selected_scope_rejects_empty_oversized_and_unindexed_sets() -> None:
    """Invalid explicit scopes fail safely before any provider call."""

    first, second, missing = uuid4(), uuid4(), uuid4()
    search = SearchService(ScopeRepository([first, second]), FixedEmbedding(), make_settings())

    with pytest.raises(SearchValidationError, match="at least one"):
        search.search_selected_documents(
            SearchRequest(
                query="notice",
                filters=SearchFilters(document_ids=[]),
            )
        )
    with pytest.raises(SearchValidationError, match="maximum"):
        search.search_selected_documents(
            SearchRequest(
                query="notice",
                filters=SearchFilters(document_ids=[first, second, missing]),
            )
        )

    small_scope_settings = make_settings()
    small_scope_settings.rag_max_selected_documents = 3
    with pytest.raises(SearchValidationError, match="ready"):
        SearchService(
            ScopeRepository([first, second]),
            FixedEmbedding(),
            small_scope_settings,
        ).search_selected_documents(
            SearchRequest(
                query="notice",
                filters=SearchFilters(document_ids=[first, missing]),
            )
        )


def test_multi_document_prompt_defines_conflict_agreement_and_untrusted_data_rules() -> None:
    """The model receives explicit multi-document grounding and injection boundaries."""

    assert "do not assume that they agree" in GROUNDED_SYSTEM_PROMPT
    assert "state the disagreement" in GROUNDED_SYSTEM_PROMPT
    assert "selected documents support" in GROUNDED_SYSTEM_PROMPT
    prompt = build_user_prompt(
        question="Compare the notice periods.",
        context="[S1] Document: a.pdf\nContent: thirty days",
        scope="The question was restricted to documents A and B.",
    )
    assert "documents A and B" in prompt
    assert "<retrieval_scope>" in prompt
    assert "<retrieved_docuintel_sources>" in prompt


def test_conversation_follow_up_preserves_selected_scope_when_filters_are_omitted() -> None:
    """A persisted selected scope cannot silently broaden on an omitted follow-up filter."""

    first, second = uuid4(), uuid4()
    repository = ConversationRepositoryDouble()
    rag = RecordingConversationRAG()
    service = ConversationService(
        repository=repository,
        rag_service=rag,
        query_rewriter=FixedConversationRewriter(),
        settings=Settings(rag_history_max_messages=10, rag_history_max_chars=6000),
    )
    conversation_id = repository.conversation.id

    asyncio.run(
        service.ask(
            conversation_id,
            ConversationAskRequest(
                question="What notice periods are described?",
                filters=SearchFilters(document_ids=[first, second]),
            ),
        )
    )
    asyncio.run(
        service.ask(
            conversation_id,
            ConversationAskRequest(question="Which one is longer?"),
        )
    )

    assert rag.requests[1].filters is not None
    assert rag.requests[1].filters.document_ids == [first, second]
    assert rag.queries[1] == "standalone notice comparison"
    assert [message.content for message in service.messages(conversation_id)] == [
        "What notice periods are described?",
        "Supported answer.",
        "Which one is longer?",
        "Supported answer.",
    ]
