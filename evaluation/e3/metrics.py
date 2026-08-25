"""Deterministic E3 answer-ground-truth and retrieval metrics."""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any
from uuid import UUID

from evaluation.e3.models import (
    E3MethodSummary,
    E3Question,
    E3QuestionGroundTruth,
    E3QuestionMethodResult,
    E3MetricStats,
    IndexedChunk,
)

DEFAULT_E3_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
ANSWER_NORMALIZATION_RULES = (
    "Unicode NFKC normalization, casefolding, whitespace collapse, and trimming; "
    "matching is literal with word boundaries for alphanumeric answer edges."
)


def normalize_answer(value: str) -> str:
    """Normalize an answer conservatively without semantic or fuzzy matching."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _answer_occurs(answer: str, chunk_text: str) -> bool:
    normalized_answer = normalize_answer(answer)
    normalized_chunk = normalize_answer(chunk_text)
    if not normalized_answer or not normalized_chunk:
        return False
    left_boundary = r"(?<!\w)" if normalized_answer[0].isalnum() else ""
    right_boundary = r"(?!\w)" if normalized_answer[-1].isalnum() else ""
    return re.search(
        f"{left_boundary}{re.escape(normalized_answer)}{right_boundary}",
        normalized_chunk,
    ) is not None


def build_question_ground_truth(
    question: E3Question,
    *,
    target_document_id: UUID | None,
    chunks: Sequence[IndexedChunk] | None,
    document_indexed: bool,
) -> E3QuestionGroundTruth:
    """Classify one question and identify only literal answer-bearing chunks."""

    normalized_answers = [normalize_answer(answer) for answer in question.accepted_answers]
    normalized_answers = list(dict.fromkeys(answer for answer in normalized_answers if answer))
    if not normalized_answers:
        return E3QuestionGroundTruth(
            question_key=question.question_key,
            status="INVALID_GROUND_TRUTH",
            reason_code="NO_ACCEPTED_ANSWER",
            target_document_id=target_document_id,
        )
    if target_document_id is None or not document_indexed or chunks is None:
        return E3QuestionGroundTruth(
            question_key=question.question_key,
            status="DOCUMENT_PROCESSING_FAILED",
            reason_code="DOCUMENT_NOT_INDEXED",
            target_document_id=target_document_id,
            normalized_answers=normalized_answers,
        )

    matches: dict[str, list[UUID]] = {}
    relevant: set[UUID] = set()
    for answer in normalized_answers:
        answer_matches = [
            chunk.chunk_id
            for chunk in chunks
            if chunk.document_id == target_document_id and _answer_occurs(answer, chunk.text)
        ]
        if answer_matches:
            matches[answer] = answer_matches
            relevant.update(answer_matches)
    ordered_relevant = [
        chunk.chunk_id for chunk in chunks if chunk.chunk_id in relevant
    ]
    if not ordered_relevant:
        return E3QuestionGroundTruth(
            question_key=question.question_key,
            status="ANSWER_NOT_INDEXED",
            reason_code="NO_LITERAL_ANSWER_MATCH",
            target_document_id=target_document_id,
            answer_matches=matches,
            normalized_answers=normalized_answers,
        )
    return E3QuestionGroundTruth(
        question_key=question.question_key,
        status="SCORABLE",
        target_document_id=target_document_id,
        relevant_chunk_ids=ordered_relevant,
        answer_matches=matches,
        normalized_answers=normalized_answers,
    )


def evaluate_ranked_ids(
    ranked_chunk_ids: Sequence[UUID],
    relevant_chunk_ids: Iterable[UUID],
    *,
    ranked_document_ids: Sequence[UUID],
    target_document_id: UUID,
    k_values: Sequence[int] = DEFAULT_E3_K_VALUES,
) -> dict[str, Any]:
    """Calculate chunk and document metrics from one deterministic ranked list."""

    relevant = set(relevant_chunk_ids)
    if not relevant:
        raise ValueError("evaluate_ranked_ids requires at least one relevant chunk.")
    ordered_unique = list(dict.fromkeys(ranked_chunk_ids))
    ranks = {chunk_id: rank for rank, chunk_id in enumerate(ordered_unique, start=1)}
    first_rank = min((ranks[chunk_id] for chunk_id in relevant if chunk_id in ranks), default=None)
    recall_at_k: dict[str, float] = {}
    hit_at_k: dict[str, bool] = {}
    document_hit_at_k: dict[str, bool] = {}
    for k in k_values:
        if k <= 0:
            raise ValueError("k_values must contain only positive integers.")
        chunk_slice = ordered_unique[:k]
        recall_at_k[str(k)] = len(set(chunk_slice) & relevant) / len(relevant)
        hit_at_k[str(k)] = bool(set(chunk_slice) & relevant)
        document_hit_at_k[str(k)] = target_document_id in set(ranked_document_ids[:k])
    return {
        "first_relevant_rank": first_rank,
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
        "recall_at_k": recall_at_k,
        "hit_at_k": hit_at_k,
        "document_hit_at_k": document_hit_at_k,
    }


def _stats(values: Sequence[float]) -> E3MetricStats:
    """Calculate latency statistics with an explicit inclusive P95 definition."""

    if not values:
        return E3MetricStats(samples=0)
    ordered = sorted(values)
    if len(ordered) == 1:
        p95 = ordered[0]
    else:
        position = (len(ordered) - 1) * 0.95
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return E3MetricStats(
        mean_ms=statistics.fmean(ordered),
        median_ms=statistics.median(ordered),
        p95_ms=p95,
        samples=len(ordered),
    )


def aggregate_method_results(
    method: str,
    results: Sequence[E3QuestionMethodResult],
    *,
    scorable_questions: int,
    candidate_count: int | None,
) -> E3MethodSummary:
    """Aggregate the same scorable question set for one method."""

    quality_results = [result for result in results if result.error is None]
    denominator = scorable_questions
    recall_at_k = {
        key: sum(result.recall_at_k.get(key, 0.0) for result in results) / denominator
        for key in (str(k) for k in DEFAULT_E3_K_VALUES)
    } if denominator else {}
    hit_at_k = {
        key: sum(bool(result.hit_at_k.get(key)) for result in results) / denominator
        for key in (str(k) for k in DEFAULT_E3_K_VALUES)
    } if denominator else {}
    document_hit_at_k = {
        key: sum(bool(result.document_hit_at_k.get(key)) for result in results) / denominator
        for key in (str(k) for k in DEFAULT_E3_K_VALUES)
    } if denominator else {}
    mrr = sum(result.reciprocal_rank for result in results) / denominator if denominator else None
    return E3MethodSummary(
        method=method,
        questions_attempted=len(results),
        scorable_questions=scorable_questions,
        search_errors=sum(result.error is not None for result in results),
        recall_at_k=recall_at_k,
        hit_at_k=hit_at_k,
        mrr=mrr,
        document_hit_at_k=document_hit_at_k,
        retrieval_latency=_stats(
            [result.retrieval_time_ms for result in quality_results if result.retrieval_time_ms is not None]
        ),
        reranking_latency=_stats(
            [result.reranking_time_ms for result in quality_results if result.reranking_time_ms is not None]
        ),
        total_pipeline_latency=_stats(
            [result.total_retrieval_pipeline_ms for result in quality_results if result.total_retrieval_pipeline_ms is not None]
        ),
        wall_clock_latency=_stats(
            [result.wall_clock_time_ms for result in quality_results if result.wall_clock_time_ms is not None]
        ),
        candidate_count=candidate_count,
    )


def compare_metric_deltas(
    left: E3MethodSummary,
    right: E3MethodSummary,
    *,
    label: str,
) -> dict[str, Any]:
    """Return absolute metric deltas, preserving the direction and units."""

    metrics: dict[str, Any] = {}
    for key in ("1", "3", "5", "10"):
        left_value = left.recall_at_k.get(key)
        right_value = right.recall_at_k.get(key)
        metrics[f"recall_at_{key}"] = {
            "from": left_value,
            "to": right_value,
            "absolute_delta": right_value - left_value
            if left_value is not None and right_value is not None
            else None,
            "percentage_point_delta": (right_value - left_value) * 100
            if left_value is not None and right_value is not None
            else None,
        }
    left_mrr = left.mrr
    right_mrr = right.mrr
    metrics["mrr"] = {
        "from": left_mrr,
        "to": right_mrr,
        "absolute_delta": right_mrr - left_mrr
        if left_mrr is not None and right_mrr is not None
        else None,
    }
    metrics["comparison"] = label
    return metrics

