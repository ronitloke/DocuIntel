"""Fast Module 5 search validation and fusion tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.core.config import Settings
from app.db.repository import SearchCandidate
from app.models.search import SearchRequest
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.retrieval.search import SearchService


def candidate(sequence: int, score: float) -> SearchCandidate:
    """Create a minimal detached repository candidate for fusion tests."""

    return SearchCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        original_filename=f"document-{sequence}.pdf",
        sequence_number=sequence,
        text=f"chunk {sequence}",
        section_heading=None,
        start_page=1,
        end_page=1,
        content_type="text",
        contains_ocr=False,
        score=score,
    )


def test_search_request_rejects_blank_and_inverted_page_filters() -> None:
    """Pydantic validation rejects invalid user input before database access."""

    with pytest.raises(ValueError):
        SearchRequest(query="   ")
    with pytest.raises(ValueError):
        SearchRequest(
            query="notice",
            filters={"page_start": 4, "page_end": 2},
        )


def test_rrf_fusion_deduplicates_and_uses_deterministic_scores() -> None:
    """A chunk in both lists receives two rank contributions exactly once."""

    shared = candidate(1, 0.9)
    semantic_only = candidate(2, 0.8)
    keyword_only = candidate(3, 0.7)
    service = SearchService(
        repository=None,
        embedding_service=EmbeddingService(model=object()),
        settings=Settings(hybrid_rrf_k=60),
    )

    # Reuse the same identity in both lists while preserving each subsystem's score.
    keyword_shared = SearchCandidate(
        chunk_id=shared.chunk_id,
        document_id=shared.document_id,
        original_filename=shared.original_filename,
        sequence_number=shared.sequence_number,
        text=shared.text,
        section_heading=shared.section_heading,
        start_page=shared.start_page,
        end_page=shared.end_page,
        content_type=shared.content_type,
        contains_ocr=shared.contains_ocr,
        score=0.4,
    )
    results = service._fuse([shared, semantic_only], [keyword_shared, keyword_only], top_k=3)

    assert len(results) == 3
    assert len({result.chunk_id for result in results}) == 3
    assert results[0].chunk_id == shared.chunk_id
    assert results[0].semantic_score == 0.9
    assert results[0].keyword_score == 0.4
    assert results[0].hybrid_score == pytest.approx(1 / 61 + 1 / 61)
