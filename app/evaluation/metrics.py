"""Pure deterministic retrieval, answer, citation, and baseline metrics."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from statistics import mean, median
from typing import Any

from app.evaluation.models import (
    EvaluationCase,
    EvaluationResultItem,
    FactEvaluation,
    QualityGateConfig,
    QualityGateResult,
    RAGCaseResult,
    RAGSummary,
    RetrievalCaseResult,
    RetrievalSummary,
)
from app.models.rag import AskResponse
from app.models.search import SearchResult

DEFAULT_K_VALUES = (1, 3, 5, 10)
_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize case and unnecessary whitespace for conservative matching."""

    return _WHITESPACE.sub(" ", value.casefold()).strip()


def reciprocal_rank(rank: int | None) -> float:
    """Return reciprocal rank, or zero when no relevant result was found."""

    return 1 / rank if rank is not None and rank > 0 else 0.0


def success_at_k(ranks: Iterable[int], k: int) -> bool:
    """Return one when any relevant result occurs at rank K or better."""

    return any(rank <= k for rank in ranks)


def recall_at_k(relevant_keys: set[str], retrieved_keys: set[str], k: int, total_relevant: int) -> float:
    """Return known relevant items retrieved in the first K divided by known relevant items."""

    if total_relevant <= 0:
        return 0.0
    return len(relevant_keys.intersection(retrieved_keys)) / total_relevant


def _document_key(value: str) -> str:
    """Normalize a filename or UUID-like document identifier."""

    return normalize_text(value)


def matching_relevance_keys(result: SearchResult, case: EvaluationCase) -> set[str]:
    """Return known relevance labels matched by one result."""

    if case.expected_chunks:
        chunk_id = str(result.chunk_id)
        return {f"chunk:{expected}" for expected in case.expected_chunks if expected == chunk_id}

    result_document_values = {_document_key(str(result.document_id)), _document_key(result.original_filename)}
    expected_documents = {_document_key(value) for value in case.expected_documents}
    if not result_document_values.intersection(expected_documents):
        return set()
    if case.expected_pages:
        if result.start_page is None:
            return set()
        pages = set(case.expected_pages)
        result_pages = {
            page
            for page in range(
                result.start_page or min(pages),
                (result.end_page or result.start_page or min(pages)) + 1,
            )
        }
        if not result_pages.intersection(pages):
            return set()
    return {
        f"document:{expected}"
        for expected in case.expected_documents
        if _document_key(expected) in result_document_values
    }


def known_relevant_keys(case: EvaluationCase) -> set[str]:
    """Return the denominator labels represented by the dataset case."""

    if case.expected_chunks:
        return {f"chunk:{value}" for value in case.expected_chunks}
    return {f"document:{_document_key(value)}" for value in case.expected_documents}


def project_result(result: SearchResult) -> EvaluationResultItem:
    """Keep bounded debug metadata without copying entire document chunks."""

    base_rank = result.base_rank or (result.rank if not result.reranked else None)
    rank_delta = base_rank - result.rank if base_rank is not None else None
    return EvaluationResultItem(
        rank=result.rank,
        base_rank=base_rank,
        rank_delta=rank_delta,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        filename=result.original_filename,
        start_page=result.start_page,
        end_page=result.end_page,
        excerpt=result.text[:500],
        semantic_score=result.semantic_score,
        keyword_score=result.keyword_score,
        hybrid_score=result.hybrid_score,
        reranker_score=result.rerank_score,
    )


def build_retrieval_case_result(
    case: EvaluationCase,
    results: list[SearchResult],
    *,
    retrieval_time_ms: float | None,
    rerank_time_ms: float | None,
    total_search_time_ms: float | None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    error: str | None = None,
) -> RetrievalCaseResult:
    """Calculate per-case success, recall, MRR, ranks, and bounded debug data."""

    expected_keys = known_relevant_keys(case)
    ranked_matches = [(result.rank, matching_relevance_keys(result, case)) for result in results]
    first_rank = next((rank for rank, keys in ranked_matches if keys), None)
    matching_results = [(result, keys) for result, (_, keys) in zip(results, ranked_matches, strict=True) if keys]
    final_rank = min((result.rank for result, _ in matching_results), default=None)
    base_rank = min(
        (
            result.base_rank or (result.rank if not result.reranked else result.rank)
            for result, _ in matching_results
        ),
        default=None,
    )
    rank_delta = base_rank - final_rank if base_rank is not None and final_rank is not None else None
    success: dict[str, bool] = {}
    recall: dict[str, float] = {}
    for k in k_values:
        top_results = results[:k]
        top_keys = set().union(*(matching_relevance_keys(result, case) for result in top_results))
        success[str(k)] = success_at_k(
            [result.rank for result, keys in matching_results],
            k,
        )
        recall[str(k)] = recall_at_k(expected_keys, top_keys, k, len(expected_keys))

    no_evidence_correct = len(results) == 0 if case.expect_no_evidence else None
    return RetrievalCaseResult(
        case_id=case.id,
        question=case.question,
        expected_documents=case.expected_documents,
        expected_chunks=case.expected_chunks,
        expect_no_evidence=case.expect_no_evidence,
        results=[project_result(result) for result in results],
        success_at_k=success,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank(first_rank),
        no_evidence_correct=no_evidence_correct,
        base_rank=base_rank,
        final_rank=final_rank,
        rank_delta=rank_delta,
        retrieval_time_ms=retrieval_time_ms,
        rerank_time_ms=rerank_time_ms,
        total_search_time_ms=total_search_time_ms,
        error=error,
    )


def aggregate_retrieval(
    cases: Sequence[RetrievalCaseResult],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalSummary:
    """Aggregate binary success, known-label recall, MRR, rank movement, and latency."""

    positive = [case for case in cases if not case.expect_no_evidence]
    no_evidence = [case for case in cases if case.expect_no_evidence]
    success = {
        str(k): mean([case.success_at_k[str(k)] for case in positive]) if positive else 0.0
        for k in k_values
    }
    recall = {
        str(k): mean([case.recall_at_k[str(k)] for case in positive]) if positive else 0.0
        for k in k_values
    }
    retrieval_values = [case.retrieval_time_ms for case in cases if case.retrieval_time_ms is not None]
    rerank_values = [case.rerank_time_ms for case in cases if case.rerank_time_ms is not None]
    total_values = [case.total_search_time_ms for case in cases if case.total_search_time_ms is not None]
    moved = [case.rank_delta for case in positive if case.rank_delta is not None]
    rerank_impact: dict[str, float | int | None] = {}
    if moved:
        rerank_impact = {
            "cases_with_rank_movement": len(moved),
            "improved_cases": sum(delta > 0 for delta in moved),
            "unchanged_cases": sum(delta == 0 for delta in moved),
            "worsened_cases": sum(delta < 0 for delta in moved),
            "average_base_rank": mean(
                [case.base_rank for case in positive if case.base_rank is not None]
            ),
            "average_final_rank": mean(
                [case.final_rank for case in positive if case.final_rank is not None]
            ),
            "average_rank_delta": mean(moved),
        }
    no_evidence_values = [case.no_evidence_correct for case in no_evidence if case.no_evidence_correct is not None]
    return RetrievalSummary(
        cases=len(cases),
        positive_cases=len(positive),
        no_evidence_cases=len(no_evidence),
        success_at_k=success,
        recall_at_k=recall,
        mrr=mean([case.reciprocal_rank for case in positive]) if positive else 0.0,
        no_evidence_correct_rate=mean(no_evidence_values) if no_evidence_values else None,
        mean_retrieval_latency_ms=mean(retrieval_values) if retrieval_values else None,
        median_retrieval_latency_ms=median(retrieval_values) if retrieval_values else None,
        mean_rerank_latency_ms=mean(rerank_values) if rerank_values else None,
        median_rerank_latency_ms=median(rerank_values) if rerank_values else None,
        mean_total_search_latency_ms=mean(total_values) if total_values else None,
        median_total_search_latency_ms=median(total_values) if total_values else None,
        rerank_impact=rerank_impact,
    )


def evaluate_key_facts(answer: str, expected_facts: Sequence[str]) -> tuple[list[FactEvaluation], float]:
    """Match expected facts by normalized substring and return coverage."""

    normalized_answer = normalize_text(answer)
    matches = [FactEvaluation(fact=fact, matched=normalize_text(fact) in normalized_answer) for fact in expected_facts]
    return matches, (mean([item.matched for item in matches]) if matches else 0.0)


def evaluate_rag_response(case: EvaluationCase, response: AskResponse) -> RAGCaseResult:
    """Evaluate key facts, citations, expected source, and modest evidence support."""

    sources = [
        EvaluationResultItem(
            source_id=source.source_id,
            rank=source.final_rank,
            base_rank=source.base_rank,
            rank_delta=(source.base_rank - source.final_rank) if source.base_rank else None,
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            filename=source.filename,
            start_page=source.start_page,
            end_page=source.end_page,
            excerpt=source.excerpt[:500],
            reranker_score=source.reranker_score,
        )
        for source in response.sources
    ]
    facts, fact_coverage = evaluate_key_facts(response.answer, case.expected_facts)
    cited_sources = [source for source in sources if source.source_id in response.citations]
    expected_documents = {_document_key(value) for value in case.expected_documents}
    expected_document_cited = (
        any(
            _document_key(str(source.document_id)) in expected_documents
            or _document_key(source.filename) in expected_documents
            for source in cited_sources
        )
        if expected_documents
        else None
    )
    cited_text = normalize_text(" ".join(source.excerpt for source in cited_sources))
    evidence_support = (
        all(normalize_text(fact.fact) in cited_text for fact in facts)
        if facts
        else bool(cited_sources) or case.expect_no_evidence
    )
    no_evidence_correct = (
        not response.sources and not response.citations and not response.answer.strip() == ""
        if case.expect_no_evidence
        else None
    )
    if case.expect_no_evidence:
        no_evidence_correct = not response.sources and not response.citations
    return RAGCaseResult(
        case_id=case.id,
        question=case.question,
        answer=response.answer,
        sources=sources,
        citations=response.citations,
        citations_present=bool(response.citations),
        citations_valid=response.citations_valid,
        expected_document_cited=expected_document_cited,
        facts=facts,
        key_fact_coverage=fact_coverage,
        evidence_support=evidence_support,
        no_evidence_correct=no_evidence_correct,
        generation_time_ms=response.generation_time_ms,
        total_time_ms=response.total_time_ms,
    )


def aggregate_rag(cases: Sequence[RAGCaseResult]) -> RAGSummary:
    """Aggregate deterministic answer-quality and latency metrics."""

    no_evidence = [case for case in cases if case.no_evidence_correct is not None]
    evidence_cases = [case for case in cases if case.no_evidence_correct is None]
    fact_cases = [case for case in evidence_cases if case.facts]
    cited_expected = [case.expected_document_cited for case in cases if case.expected_document_cited is not None]
    generation = [case.generation_time_ms for case in cases if case.generation_time_ms is not None]
    total = [case.total_time_ms for case in cases if case.total_time_ms is not None]
    return RAGSummary(
        cases=len(cases),
        key_fact_coverage=mean([case.key_fact_coverage for case in fact_cases]) if fact_cases else 0.0,
        citation_presence_rate=mean([case.citations_present for case in evidence_cases]) if evidence_cases else 0.0,
        citation_validity_rate=mean([case.citations_valid for case in evidence_cases]) if evidence_cases else 0.0,
        expected_document_citation_rate=mean(cited_expected) if cited_expected else None,
        evidence_support_rate=mean([case.evidence_support for case in evidence_cases]) if evidence_cases else 0.0,
        no_evidence_cases=len(no_evidence),
        no_evidence_correct_rate=mean([case.no_evidence_correct for case in no_evidence]) if no_evidence else None,
        mean_generation_latency_ms=mean(generation) if generation else None,
        median_generation_latency_ms=median(generation) if generation else None,
        mean_total_rag_latency_ms=mean(total) if total else None,
        median_total_rag_latency_ms=median(total) if total else None,
    )


def compare_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare selected quality/latency fields without failing on tiny drift by default."""

    current_summary = current.get("summary", {})
    baseline_summary = baseline.get("summary", {})
    fields = (
        "mrr",
        "mean_total_search_latency_ms",
        "mean_retrieval_latency_ms",
        "mean_rerank_latency_ms",
    )
    deltas: dict[str, float | None] = {}
    regressions: list[str] = []
    for field in fields:
        current_value = current_summary.get(field)
        baseline_value = baseline_summary.get(field)
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            delta = float(current_value) - float(baseline_value)
            deltas[field] = delta
            if field == "mrr" and delta < -tolerance:
                regressions.append(f"{field} decreased by {abs(delta):.6f}")
            if field.startswith("mean_") and delta > tolerance and "latency" in field:
                regressions.append(f"{field} increased by {delta:.3f} ms")
        else:
            deltas[field] = None
    return {"deltas": deltas, "regressions": regressions, "tolerance": tolerance}


def apply_quality_gates(summary: dict[str, Any], gates: QualityGateConfig) -> QualityGateResult:
    """Apply only explicitly requested thresholds and return a CLI-friendly result."""

    failures: list[str] = []
    success_at_3 = summary.get("success_at_k", {}).get("3")
    if gates.minimum_success_at_3 is not None and (
        not isinstance(success_at_3, (int, float)) or success_at_3 < gates.minimum_success_at_3
    ):
        failures.append(f"success_at_3 below {gates.minimum_success_at_3:.3f}")
    mrr = summary.get("mrr")
    if gates.minimum_mrr is not None and (not isinstance(mrr, (int, float)) or mrr < gates.minimum_mrr):
        failures.append(f"mrr below {gates.minimum_mrr:.3f}")
    latency = summary.get("mean_total_search_latency_ms")
    if gates.maximum_mean_search_latency_ms is not None and (
        not isinstance(latency, (int, float)) or latency > gates.maximum_mean_search_latency_ms
    ):
        failures.append(f"mean_total_search_latency_ms above {gates.maximum_mean_search_latency_ms:.3f}")
    return QualityGateResult(passed=not failures, failures=failures)
