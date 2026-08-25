"""Grounded single-question RAG orchestration over existing search services."""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Sequence
from time import perf_counter
from typing import Iterable
from uuid import UUID

from app.core.config import Settings
from app.core.exceptions import RAGServiceError, SearchValidationError
from app.models.rag import AskRequest, AskResponse, RAGSource
from app.models.search import SearchRequest
from app.services.llm.ollama import OllamaClient
from app.services.rag.context import ContextSource, RAGContextBuilder
from app.services.rag.prompts import (
    CONVERSATIONAL_SYSTEM_PROMPT,
    GROUNDED_SYSTEM_PROMPT,
    build_conversational_user_prompt,
    build_user_prompt,
)
from app.services.retrieval.search import SearchService

logger = logging.getLogger(__name__)

NO_RESULTS_ANSWER = "I couldn't find relevant information in the indexed documents to answer that question."
_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def validate_citation_labels(
    answer: str,
    available_source_ids: Collection[str],
) -> tuple[list[str], bool]:
    """Extract stable source labels and report whether every label is available."""

    cited = list(dict.fromkeys(f"S{match}" for match in _CITATION_PATTERN.findall(answer)))
    available = set(available_source_ids)
    return cited, bool(cited) and all(source_id in available for source_id in cited)


class RAGService:
    """Run retrieval, bounded context construction, and one Ollama generation."""

    def __init__(
        self,
        search_service: SearchService,
        ollama_client: OllamaClient,
        settings: Settings,
        context_builder: RAGContextBuilder | None = None,
    ) -> None:
        self.search_service = search_service
        self.ollama_client = ollama_client
        self.settings = settings
        self.context_builder = context_builder or RAGContextBuilder()

    async def ask(self, request: AskRequest) -> AskResponse:
        """Answer one question from filtered, retrieved, and optionally reranked chunks."""

        return await self._ask(
            request=request,
            retrieval_query=request.question,
            history_text=None,
        )

    async def ask_conversational(
        self,
        request: AskRequest,
        *,
        retrieval_query: str,
        history_text: str,
    ) -> AskResponse:
        """Answer the current question while searching with its rewritten query."""

        return await self._ask(
            request=request,
            retrieval_query=retrieval_query,
            history_text=history_text,
        )

    async def _ask(
        self,
        *,
        request: AskRequest,
        retrieval_query: str,
        history_text: str | None,
    ) -> AskResponse:
        """Share the stateless retrieval/generation implementation with Module 8."""

        started = perf_counter()
        top_k = request.top_k or self.settings.rag_default_top_k
        selected_document_ids = self._selected_document_ids(request)
        normalized_request = request
        if selected_document_ids:
            normalized_request = request.model_copy(
                update={
                    "filters": request.filters.model_copy(
                        update={"document_ids": selected_document_ids}
                    )
                }
            )
        logger.info(
            "RAG request started question_length=%s mode=%s rerank=%s top_k=%s filters=%s",
            len(retrieval_query),
            request.search_mode.value,
            request.rerank,
            top_k,
            bool(request.filters),
        )
        search_request = SearchRequest(
            query=retrieval_query,
            mode=normalized_request.search_mode,
            top_k=top_k,
            rerank=normalized_request.rerank,
            filters=normalized_request.filters,
        )
        selected_search = getattr(self.search_service, "search_selected_documents", None)
        if selected_document_ids and callable(selected_search):
            search_response = selected_search(search_request)
        else:
            search_response = self.search_service.search(search_request)
        retrieval_time_ms = search_response.retrieval_time_ms
        rerank_time_ms = search_response.rerank_time_ms
        logger.info(
            "RAG retrieval completed result_count=%s retrieval_ms=%.3f rerank_ms=%s",
            len(search_response.results),
            retrieval_time_ms,
            rerank_time_ms,
        )

        if not search_response.results:
            total_time_ms = round((perf_counter() - started) * 1000, 3)
            return AskResponse(
                answer=NO_RESULTS_ANSWER,
                model=self.ollama_client.model,
                sources=[],
                citations=[],
                citations_valid=False,
                retrieval_time_ms=retrieval_time_ms,
                rerank_time_ms=rerank_time_ms,
                generation_time_ms=0.0,
                total_time_ms=total_time_ms,
                document_scope="selected" if selected_document_ids else "all",
                selected_document_ids=selected_document_ids,
            )

        context = self.context_builder.build(
            search_response.results,
            max_chars=self.settings.rag_max_context_chars,
        )
        if not context.sources:
            raise RAGServiceError("The configured RAG context budget cannot contain a source.")

        provider_started = perf_counter()
        if history_text and history_text.strip():
            system_prompt = CONVERSATIONAL_SYSTEM_PROMPT
            user_prompt = build_conversational_user_prompt(
                question=request.question,
                history=history_text,
                context=context.text,
                scope=self._scope_prompt(selected_document_ids),
            )
        else:
            system_prompt = GROUNDED_SYSTEM_PROMPT
            user_prompt = build_user_prompt(
                question=request.question,
                context=context.text,
                scope=self._scope_prompt(selected_document_ids),
            )
        answer = await self.ollama_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        generation_time_ms = round((perf_counter() - provider_started) * 1000, 3)
        citations, citations_valid = self._validate_citations(answer, context.sources)
        total_time_ms = round((perf_counter() - started) * 1000, 3)
        retrieved_document_ids = self._unique_document_ids(
            result.document_id for result in search_response.results
        )
        source_document_ids = self._unique_document_ids(
            source.result.document_id for source in context.sources
        )
        logger.info(
            "RAG generation completed model=%s context_sources=%s citations=%s "
            "generation_ms=%.3f total_ms=%.3f",
            self.ollama_client.model,
            len(context.sources),
            len(citations),
            generation_time_ms,
            total_time_ms,
        )
        return AskResponse(
            answer=answer,
            model=self.ollama_client.model,
            sources=[self._source_model(source) for source in context.sources],
            citations=citations,
            citations_valid=citations_valid,
            retrieval_time_ms=retrieval_time_ms,
            rerank_time_ms=rerank_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            document_scope="selected" if selected_document_ids else "all",
            selected_document_ids=selected_document_ids,
            retrieved_document_ids=retrieved_document_ids,
            source_document_ids=source_document_ids,
        )

    def _selected_document_ids(self, request: AskRequest) -> list[UUID]:
        """Normalize and validate the explicit RAG document scope."""

        if request.filters is None or request.filters.document_ids is None:
            return []
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
        return document_ids

    @staticmethod
    def _unique_document_ids(document_ids: Iterable[UUID]) -> list[UUID]:
        """Return UUID-like IDs in first-seen order without duplicates."""

        return list(dict.fromkeys(document_ids))

    @staticmethod
    def _scope_prompt(selected_document_ids: list[UUID]) -> str:
        """Describe scope without treating selected documents as mutually consistent."""

        if not selected_document_ids:
            return "All indexed documents were eligible for retrieval; only the supplied source blocks are evidence."
        ids = ", ".join(str(document_id) for document_id in selected_document_ids)
        return (
            "The question was explicitly restricted to these document IDs: "
            f"{ids}. Analyze each supplied document independently; a selected document "
            "may be irrelevant or absent from the final evidence."
        )

    @staticmethod
    def _validate_citations(
        answer: str,
        sources: Sequence[ContextSource],
    ) -> tuple[list[str], bool]:
        """Reject unknown labels and flag answers that omitted all citations."""

        cited, valid = validate_citation_labels(
            answer,
            [source.source_id for source in sources],
        )
        available = {source.source_id for source in sources}
        unknown = [source_id for source_id in cited if source_id not in available]
        if unknown:
            raise RAGServiceError(
                "The local model returned a citation for a source that was not provided."
            )
        return cited, valid

    @staticmethod
    def _source_model(source: ContextSource) -> RAGSource:
        """Project context metadata without exposing raw vectors or full chunks."""

        result = source.result
        return RAGSource(
            source_id=source.source_id,
            document_id=result.document_id,
            chunk_id=result.chunk_id,
            filename=result.original_filename,
            start_page=result.start_page,
            end_page=result.end_page,
            section_heading=result.section_heading,
            content_type=result.content_type,
            contains_ocr=result.contains_ocr,
            excerpt=source.excerpt,
            final_rank=result.rank,
            base_rank=result.base_rank,
            reranker_score=result.rerank_score,
        )
