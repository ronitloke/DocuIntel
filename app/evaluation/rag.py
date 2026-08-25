"""Deterministic grounded-answer evaluation over the existing RAGService."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.core.exceptions import DocumentIngestionError
from app.evaluation.metrics import aggregate_rag, evaluate_rag_response
from app.evaluation.models import (
    EvaluationConfiguration,
    EvaluationDataset,
    RAGEvaluationReport,
    RAGCaseResult,
)
from app.models.rag import AskRequest
from app.services.rag.service import RAGService

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate facts, citations, source matching, evidence support, and latency."""

    def __init__(self, rag_service: RAGService) -> None:
        self.rag_service = rag_service

    async def evaluate(
        self,
        dataset: EvaluationDataset,
        configuration: EvaluationConfiguration,
    ) -> RAGEvaluationReport:
        """Run each case through RAG without creating persistent conversation messages."""

        results: list[RAGCaseResult] = []
        for case in dataset.cases:
            try:
                response = await self.rag_service.ask(
                    AskRequest(
                        question=case.question,
                        top_k=configuration.top_k,
                        search_mode=configuration.mode,
                        rerank=configuration.rerank,
                        filters=case.filters,
                    )
                )
                results.append(evaluate_rag_response(case, response))
            except Exception as exc:
                message = exc.public_message if isinstance(exc, DocumentIngestionError) else str(exc)
                logger.warning("Evaluation RAG case failed case_id=%s error=%s", case.id, message)
                results.append(
                    RAGCaseResult(
                        case_id=case.id,
                        question=case.question,
                        citations_valid=False,
                        evidence_support=False,
                        error=message,
                    )
                )
        return RAGEvaluationReport(
            dataset=dataset.name,
            configuration=configuration,
            generated_at=datetime.now(UTC),
            summary=aggregate_rag(results),
            cases=results,
        )
