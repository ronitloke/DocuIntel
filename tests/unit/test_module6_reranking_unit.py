"""Unit coverage for Module 6 cross-encoder reranking."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.core.config import Settings
from app.core.exceptions import RerankerServiceError
from app.db.repository import SearchCandidate
from app.models.search import SearchRequest
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.reranking.cross_encoder import CrossEncoderReranker, RerankInput
from app.services.retrieval.search import SearchService


class FakeCrossEncoder:
    """Small deterministic model double with the CrossEncoder predict contract."""

    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        self.calls.append((pairs, kwargs))
        return [5.4 if "notice" in candidate.lower() else 0.1 for _, candidate in pairs]


def test_cross_encoder_is_lazy_batched_and_deterministic() -> None:
    """The model loads once, receives headings, and sorts by raw score."""

    model = FakeCrossEncoder()
    loads: list[tuple[str, int]] = []

    def loader(model_name: str, max_length: int) -> FakeCrossEncoder:
        loads.append((model_name, max_length))
        return model

    service = CrossEncoderReranker(
        Settings(
            reranker_model="test/cross-encoder",
            reranker_batch_size=2,
            reranker_max_length=16,
        ),
        model_loader=loader,
    )
    candidates = [
        RerankInput(uuid4(), "Computer monitors are available."),
        RerankInput(uuid4(), "Employees give thirty days written notice.", "Termination"),
    ]

    first = service.rerank("How much notice?", candidates)
    second = service.rerank("How much notice?", candidates)

    assert len(loads) == 1
    assert loads == [("test/cross-encoder", 16)]
    assert model.calls[0][1] == {
        "batch_size": 2,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    assert "Termination" in model.calls[0][0][1][1]
    assert len(model.calls[0][0][1][1]) <= 128
    assert first[0].candidate_id == candidates[1].candidate_id
    assert first[0].base_rank == 2
    assert first[0].score == pytest.approx(5.4)
    assert [item.candidate_id for item in first] == [item.candidate_id for item in second]


def test_empty_candidates_do_not_load_the_model() -> None:
    """An empty retrieval result is a successful no-op."""

    service = CrossEncoderReranker(
        Settings(reranker_model="test/cross-encoder"),
        model_loader=lambda *_: pytest.fail("model should not load"),
    )

    assert service.rerank("notice", []) == []
    assert not service.model_loaded


def test_invalid_query_and_model_failure_are_controlled_errors() -> None:
    """Bad input and unavailable local models never become silent fallbacks."""

    candidate = RerankInput(uuid4(), "some text")
    service = CrossEncoderReranker(Settings(), model=FakeCrossEncoder())
    with pytest.raises(RerankerServiceError):
        service.rerank("   ", [candidate])

    def failing_loader(*_: object) -> object:
        raise RuntimeError("model unavailable")

    unavailable = CrossEncoderReranker(Settings(), model_loader=failing_loader)
    with pytest.raises(RerankerServiceError):
        unavailable.rerank("notice", [candidate])


class FakeEmbeddingModel:
    """One-dimensional semantic test model with the configured 384d shape."""

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        return [[1.0] + [0.0] * 383 for _ in texts]


class FakeRepository:
    """Repository double that returns a deliberately weak base order."""

    def __init__(self, candidates: list[SearchCandidate]) -> None:
        self.candidates = candidates
        self.semantic_limit: int | None = None

    def semantic_search(self, **kwargs: object) -> list[SearchCandidate]:
        self.semantic_limit = int(kwargs["limit"])
        return self.candidates

    def keyword_search(self, **_: object) -> list[SearchCandidate]:
        return []


def make_candidate(sequence: int, text: str) -> SearchCandidate:
    """Create a detached candidate for search-service reranking tests."""

    return SearchCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        original_filename=f"document-{sequence}.pdf",
        sequence_number=sequence,
        text=text,
        section_heading=None,
        start_page=1,
        end_page=1,
        content_type="text",
        contains_ocr=False,
        score=float(10 - sequence),
    )


def test_search_service_preserves_base_metadata_and_rank_movement() -> None:
    """Second-stage ranking returns top-k while retaining the original rank."""

    candidates = [
        make_candidate(1, "Unrelated warehouse inventory."),
        make_candidate(2, "Another unrelated policy."),
        make_candidate(3, "Employees must give thirty days written notice."),
    ]
    repository = FakeRepository(candidates)
    reranker = CrossEncoderReranker(Settings(), model=FakeCrossEncoder())
    service = SearchService(
        repository=repository,  # type: ignore[arg-type]
        embedding_service=EmbeddingService(model=FakeEmbeddingModel()),
        settings=Settings(rerank_candidate_count=20, rerank_candidate_multiplier=4),
        reranker=reranker,
    )

    response = service.search(
        SearchRequest(query="What is the employee notice period?", mode="semantic", top_k=1, rerank=True)
    )

    assert repository.semantic_limit == 20
    assert response.reranked
    assert response.rerank_time_ms is not None
    assert response.results[0].rank == 1
    assert response.results[0].base_rank == 3
    assert response.results[0].reranked
    assert response.results[0].rerank_score == pytest.approx(5.4)
    assert response.results[0].text == candidates[2].text


def test_search_service_rejects_requested_reranking_without_service() -> None:
    """Explicit reranking never silently degrades to Module 5 ranking."""

    candidate = make_candidate(1, "Employees give notice.")
    service = SearchService(
        repository=FakeRepository([candidate]),  # type: ignore[arg-type]
        embedding_service=EmbeddingService(model=FakeEmbeddingModel()),
        settings=Settings(),
    )

    with pytest.raises(RerankerServiceError):
        service.search(SearchRequest(query="notice", mode="semantic", rerank=True))
