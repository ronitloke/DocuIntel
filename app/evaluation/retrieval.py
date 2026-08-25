"""Retrieval-only evaluation over the existing SearchService abstraction."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from collections.abc import Sequence

from app.core.exceptions import DocumentIngestionError
from app.evaluation.metrics import DEFAULT_K_VALUES, aggregate_retrieval, build_retrieval_case_result
from app.evaluation.models import (
    ComparisonReport,
    EvaluationCase,
    EvaluationConfiguration,
    EvaluationDocumentCheck,
    EvaluationDataset,
    RetrievalCaseResult,
    RetrievalEvaluationReport,
)
from app.models.search import SearchRequest
from app.services.retrieval.search import SearchService

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Run comparable cases through semantic, keyword, hybrid, and reranked search."""

    def __init__(self, search_service: SearchService) -> None:
        self.search_service = search_service

    def evaluate(
        self,
        dataset: EvaluationDataset,
        configuration: EvaluationConfiguration,
        *,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
    ) -> RetrievalEvaluationReport:
        """Evaluate one configuration without modifying documents or conversations."""

        corpus_checks = self.validate_dataset(dataset)
        case_results = [self._evaluate_case(case, configuration, k_values) for case in dataset.cases]
        return RetrievalEvaluationReport(
            dataset=dataset.name,
            configuration=configuration,
            generated_at=datetime.now(UTC),
            corpus_checks=corpus_checks,
            summary=aggregate_retrieval(case_results, k_values=k_values),
            cases=case_results,
        )

    def compare(
        self,
        dataset: EvaluationDataset,
        configurations: Sequence[EvaluationConfiguration],
        *,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
    ) -> ComparisonReport:
        """Run identical cases through multiple existing search configurations."""

        corpus_checks = self.validate_dataset(dataset)
        reports = [
            self._evaluate_with_checks(dataset, configuration, k_values, corpus_checks)
            for configuration in configurations
        ]
        return ComparisonReport(
            dataset=dataset.name,
            generated_at=datetime.now(UTC),
            corpus_checks=corpus_checks,
            reports=reports,
        )

    def validate_dataset(self, dataset: EvaluationDataset) -> list[EvaluationDocumentCheck]:
        """Fail before scoring when filename/UUID labels do not match indexed data."""

        labeled_cases = [
            (case, expected_document)
            for case in dataset.cases
            for expected_document in case.expected_documents
        ]
        if not labeled_cases:
            return []

        repository = self.search_service.repository
        if repository is None:
            raise ValueError(
                "Evaluation dataset contains expected documents, but PostgreSQL is not configured."
            )
        inventory = repository.evaluation_document_inventory()
        checks: list[EvaluationDocumentCheck] = []
        missing: list[str] = []
        unindexed: list[str] = []
        for case, expected_document in labeled_cases:
            normalized = expected_document.casefold().strip()
            matches = [
                record
                for record in inventory
                if record.original_filename.casefold().strip() == normalized
                or str(record.document_id).casefold() == normalized
            ]
            indexed_chunks = max(
                (record.chunk_count for record in matches if record.is_indexed),
                default=0,
            )
            exists = bool(matches)
            checks.append(
                EvaluationDocumentCheck(
                    case_id=case.id,
                    question=case.question,
                    expected_document=expected_document,
                    exists=exists,
                    indexed_chunks=indexed_chunks,
                )
            )
            if not exists:
                missing.append(expected_document)
            elif indexed_chunks == 0:
                unindexed.append(expected_document)

        if missing:
            labels = ", ".join(sorted(set(missing)))
            raise ValueError(
                f"Evaluation dataset does not match indexed corpus: expected document(s) not found: {labels}."
            )
        if unindexed:
            labels = ", ".join(sorted(set(unindexed)))
            raise ValueError(
                f"Evaluation dataset does not match indexed corpus: expected document(s) have no indexed chunks: {labels}."
            )
        return checks

    def _evaluate_with_checks(
        self,
        dataset: EvaluationDataset,
        configuration: EvaluationConfiguration,
        k_values: Sequence[int],
        corpus_checks: list[EvaluationDocumentCheck],
    ) -> RetrievalEvaluationReport:
        """Evaluate one configuration after one shared corpus preflight."""

        case_results = [self._evaluate_case(case, configuration, k_values) for case in dataset.cases]
        return RetrievalEvaluationReport(
            dataset=dataset.name,
            configuration=configuration,
            generated_at=datetime.now(UTC),
            corpus_checks=corpus_checks,
            summary=aggregate_retrieval(case_results, k_values=k_values),
            cases=case_results,
        )

    def _evaluate_case(
        self,
        case: EvaluationCase,
        configuration: EvaluationConfiguration,
        k_values: Sequence[int],
    ) -> RetrievalCaseResult:
        """Execute one SearchService request and retain only bounded evaluation evidence."""

        try:
            response = self.search_service.search(
                SearchRequest(
                    query=case.question,
                    mode=configuration.mode,
                    top_k=configuration.top_k,
                    rerank=configuration.rerank,
                    filters=case.filters,
                )
            )
            return build_retrieval_case_result(
                case,
                response.results,
                retrieval_time_ms=response.retrieval_time_ms,
                rerank_time_ms=response.rerank_time_ms,
                total_search_time_ms=response.total_search_time_ms,
                k_values=k_values,
            )
        except Exception as exc:
            message = exc.public_message if isinstance(exc, DocumentIngestionError) else str(exc)
            logger.warning("Evaluation retrieval case failed case_id=%s error=%s", case.id, message)
            return build_retrieval_case_result(
                case,
                [],
                retrieval_time_ms=None,
                rerank_time_ms=None,
                total_search_time_ms=None,
                k_values=k_values,
                error=message,
            )
