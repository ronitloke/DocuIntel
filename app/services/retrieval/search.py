"""Semantic, PostgreSQL full-text, hybrid retrieval, and second-stage reranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID

from app.core.config import Settings
from app.core.exceptions import (
    DatabaseNotConfiguredError,
    RerankerServiceError,
    SearchValidationError,
)
from app.db.repository import DocumentRepository, SearchCandidate
from app.models.search import SearchMode, SearchRequest, SearchResponse, SearchResult
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.reranking.cross_encoder import CrossEncoderReranker, RerankInput

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _MergedCandidate:
    """One unique chunk assembled from semantic and lexical result lists."""

    candidate: SearchCandidate
    semantic_score: float | None = None
    keyword_score: float | None = None
    hybrid_score: float = 0.0


class SearchService:
    """Orchestrate SQL retrieval and deterministic result projection."""

    def __init__(
        self,
        repository: DocumentRepository | None,
        embedding_service: EmbeddingService,
        settings: Settings,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.repository = repository
        self.embedding_service = embedding_service
        self.settings = settings
        self.reranker = reranker

    def search(self, request: SearchRequest) -> SearchResponse:
        """Execute one configured search mode and return ranked chunk metadata."""

        self._validate_request(request)
        repository = self._require_repository()
        started = perf_counter()
        candidate_limit = self._candidate_limit(request)
        retrieval_started = perf_counter()
        semantic_candidates: list[SearchCandidate] = []
        keyword_candidates: list[SearchCandidate] = []

        if request.mode in {SearchMode.SEMANTIC, SearchMode.HYBRID}:
            query_embedding = self.embedding_service.embed_texts([request.query])[0]
            semantic_candidates = repository.semantic_search(
                query_embedding=query_embedding,
                limit=candidate_limit,
                filters=request.filters,
                min_similarity=self.settings.semantic_min_similarity,
            )
        if request.mode in {SearchMode.KEYWORD, SearchMode.HYBRID}:
            keyword_candidates = repository.keyword_search(
                query=request.query,
                limit=candidate_limit,
                text_search_config=self.settings.postgres_text_search_config,
                filters=request.filters,
            )

        if request.mode is SearchMode.HYBRID:
            results = self._fuse(semantic_candidates, keyword_candidates, candidate_limit)
        elif request.mode is SearchMode.SEMANTIC:
            results = [
                self._to_result(
                    candidate,
                    request.mode,
                    rank,
                    semantic_score=candidate.score,
                )
                for rank, candidate in enumerate(semantic_candidates, start=1)
            ]
        else:
            results = [
                self._to_result(
                    candidate,
                    request.mode,
                    rank,
                    keyword_score=candidate.score,
                )
                for rank, candidate in enumerate(keyword_candidates, start=1)
            ]

        retrieval_time_ms = round((perf_counter() - retrieval_started) * 1000, 3)
        rerank_time_ms: float | None = None
        if request.rerank and results:
            rerank_started = perf_counter()
            results = self._rerank(request.query, results, request.top_k)
            rerank_time_ms = round((perf_counter() - rerank_started) * 1000, 3)
        elif not request.rerank:
            results = results[: request.top_k]

        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        logger.info(
            "Search completed mode=%s query_length=%s top_k=%s filters=%s "
            "rerank_requested=%s candidate_limit=%s semantic_candidates=%s "
            "keyword_candidates=%s results=%s retrieval_ms=%.3f rerank_ms=%s total_ms=%.3f",
            request.mode.value,
            len(request.query),
            request.top_k,
            bool(request.filters),
            request.rerank,
            candidate_limit,
            len(semantic_candidates),
            len(keyword_candidates),
            len(results),
            retrieval_time_ms,
            rerank_time_ms,
            elapsed_ms,
        )
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            results=results,
            total_results=len(results),
            search_time_ms=elapsed_ms,
            reranked=request.rerank,
            retrieval_time_ms=retrieval_time_ms,
            rerank_time_ms=rerank_time_ms,
            total_search_time_ms=elapsed_ms,
        )

    def search_selected_documents(self, request: SearchRequest) -> SearchResponse:
        """Retrieve per-document candidates, then optionally rerank them together.

        This keeps document scope orchestration above the repository while still
        reusing the normal semantic, keyword, hybrid, and CrossEncoder paths.
        Each selected document gets a bounded opportunity to contribute evidence;
        irrelevant documents are not added to the final result unless their
        retrieved chunks survive the shared ranking boundary.
        """

        self._validate_request(request)
        if request.filters is None or request.filters.document_ids is None:
            raise SearchValidationError(
                "search_selected_documents requires an explicit document_ids scope."
            )
        document_ids = list(dict.fromkeys(request.filters.document_ids))
        if not document_ids:
            raise SearchValidationError(
                "document_ids must contain at least one document ID when a scope is selected."
            )
        if len(document_ids) > self.settings.rag_max_selected_documents:
            raise SearchValidationError(
                "The selected document scope exceeds the configured maximum of "
                f"{self.settings.rag_max_selected_documents} documents."
            )

        repository = self._require_repository()
        indexed_ids = repository.indexed_document_ids(document_ids)
        invalid_ids = [document_id for document_id in document_ids if document_id not in indexed_ids]
        if invalid_ids:
            raise SearchValidationError(
                "Every selected document must exist, be ready, and have indexed chunks."
            )

        started = perf_counter()
        per_document_top_k = min(
            max(
                request.top_k,
                request.top_k * self.settings.rag_selected_document_candidate_multiplier,
            ),
            self.settings.search_max_top_k,
        )
        candidates: dict[UUID, SearchResult] = {}
        retrieval_time_ms = 0.0
        for document_id in document_ids:
            scoped_filters = request.filters.model_copy(update={"document_ids": [document_id]})
            scoped_response = self.search(
                request.model_copy(
                    update={
                        "top_k": per_document_top_k,
                        "rerank": False,
                        "filters": scoped_filters,
                    }
                )
            )
            retrieval_time_ms += scoped_response.retrieval_time_ms
            for result in scoped_response.results:
                candidates.setdefault(result.chunk_id, result)

        ordered = sorted(candidates.values(), key=self._selected_scope_sort_key)
        max_candidates = min(
            self.settings.rag_max_selected_candidates,
            self.settings.rerank_max_candidates if request.rerank else self.settings.rag_max_selected_candidates,
        )
        ordered = ordered[:max_candidates]
        combined = [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(ordered, start=1)
        ]

        rerank_time_ms: float | None = None
        if request.rerank and combined:
            rerank_started = perf_counter()
            combined = self._rerank(request.query, combined, request.top_k)
            rerank_time_ms = round((perf_counter() - rerank_started) * 1000, 3)
        else:
            combined = combined[: request.top_k]
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        retrieval_time_ms = round(retrieval_time_ms, 3)
        logger.info(
            "Selected-document search completed document_count=%s candidate_count=%s "
            "results=%s mode=%s rerank=%s retrieval_ms=%.3f rerank_ms=%s total_ms=%.3f",
            len(document_ids),
            len(candidates),
            len(combined),
            request.mode.value,
            request.rerank,
            retrieval_time_ms,
            rerank_time_ms,
            elapsed_ms,
        )
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            results=combined,
            total_results=len(combined),
            search_time_ms=elapsed_ms,
            reranked=request.rerank,
            retrieval_time_ms=retrieval_time_ms,
            rerank_time_ms=rerank_time_ms,
            total_search_time_ms=elapsed_ms,
        )

    @staticmethod
    def _selected_scope_sort_key(result: SearchResult) -> tuple[float, int, str, int, str]:
        """Order per-document results deterministically before shared reranking."""

        score = result.hybrid_score
        if result.retrieval_method is SearchMode.SEMANTIC:
            score = result.semantic_score
        elif result.retrieval_method is SearchMode.KEYWORD:
            score = result.keyword_score
        return (
            -(score if score is not None else 0.0),
            result.rank,
            str(result.document_id),
            result.sequence_number,
            str(result.chunk_id),
        )

    def _candidate_limit(self, request: SearchRequest) -> int:
        """Choose a bounded retrieval pool for base or second-stage search."""

        if not request.rerank:
            if request.mode is SearchMode.HYBRID:
                return request.top_k * self.settings.search_candidate_multiplier
            return request.top_k
        return min(
            max(
                self.settings.rerank_candidate_count,
                request.top_k * self.settings.rerank_candidate_multiplier,
            ),
            self.settings.rerank_max_candidates,
        )

    def _rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Apply the dedicated reranker while preserving base result metadata."""

        if self.reranker is None:
            raise RerankerServiceError(
                "Reranking was requested, but the local reranking service is unavailable."
            )
        base_by_id = {result.chunk_id: result for result in results}
        reranked = self.reranker.rerank(
            query,
            [
                RerankInput(
                    candidate_id=result.chunk_id,
                    text=result.text,
                    section_heading=result.section_heading,
                )
                for result in results
            ],
        )
        reranked_ids = [item.candidate_id for item in reranked]
        if (
            len(reranked_ids) != len(base_by_id)
            or len(set(reranked_ids)) != len(reranked_ids)
            or set(reranked_ids) != set(base_by_id)
        ):
            raise RerankerServiceError(
                "The reranker returned an invalid candidate set and results were not reranked."
            )
        final_results: list[SearchResult] = []
        for final_rank, item in enumerate(reranked[:top_k], start=1):
            base_result = base_by_id[item.candidate_id]
            final_results.append(
                base_result.model_copy(
                    update={
                        "rank": final_rank,
                        "base_rank": item.base_rank,
                        "rerank_score": item.score,
                        "reranked": True,
                    }
                )
            )
        return final_results

    def _fuse(
        self,
        semantic_candidates: list[SearchCandidate],
        keyword_candidates: list[SearchCandidate],
        top_k: int,
    ) -> list[SearchResult]:
        """Fuse ranked lists with configurable Reciprocal Rank Fusion."""

        merged: dict[UUID, _MergedCandidate] = {}
        for rank, candidate in enumerate(semantic_candidates, start=1):
            item = merged.setdefault(candidate.chunk_id, _MergedCandidate(candidate=candidate))
            item.semantic_score = candidate.score
            item.hybrid_score += 1 / (self.settings.hybrid_rrf_k + rank)
        for rank, candidate in enumerate(keyword_candidates, start=1):
            item = merged.setdefault(candidate.chunk_id, _MergedCandidate(candidate=candidate))
            item.keyword_score = candidate.score
            item.hybrid_score += 1 / (self.settings.hybrid_rrf_k + rank)

        ordered = sorted(
            merged.values(),
            key=lambda item: (
                -item.hybrid_score,
                str(item.candidate.document_id),
                item.candidate.sequence_number,
                str(item.candidate.chunk_id),
            ),
        )[:top_k]
        return [
            self._to_result(
                item.candidate,
                SearchMode.HYBRID,
                rank,
                semantic_score=item.semantic_score,
                keyword_score=item.keyword_score,
                hybrid_score=item.hybrid_score,
            )
            for rank, item in enumerate(ordered, start=1)
        ]

    @staticmethod
    def _to_result(
        candidate: SearchCandidate,
        mode: SearchMode,
        rank: int,
        *,
        semantic_score: float | None = None,
        keyword_score: float | None = None,
        hybrid_score: float | None = None,
    ) -> SearchResult:
        """Project a repository candidate without a vector field."""

        return SearchResult(
            rank=rank,
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            original_filename=candidate.original_filename,
            sequence_number=candidate.sequence_number,
            text=candidate.text,
            section_heading=candidate.section_heading,
            start_page=candidate.start_page,
            end_page=candidate.end_page,
            content_type=candidate.content_type,
            contains_ocr=candidate.contains_ocr,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            hybrid_score=hybrid_score,
            retrieval_method=mode,
        )

    def _validate_request(self, request: SearchRequest) -> None:
        """Apply settings-backed validation not expressible in the shared schema."""

        if len(request.query) > self.settings.search_max_query_chars:
            raise SearchValidationError(
                f"query must be at most {self.settings.search_max_query_chars} characters."
            )
        if request.top_k > self.settings.search_max_top_k:
            raise SearchValidationError(f"top_k must be at most {self.settings.search_max_top_k}.")
        if request.rerank and self.settings.rerank_max_candidates < request.top_k:
            raise SearchValidationError(
                "rerank_max_candidates must be at least top_k when reranking is enabled."
            )

    def _require_repository(self) -> DocumentRepository:
        """Require PostgreSQL before executing any retrieval query."""

        if self.repository is None:
            raise DatabaseNotConfiguredError(
                "PostgreSQL is required for search but is not configured."
            )
        return self.repository
