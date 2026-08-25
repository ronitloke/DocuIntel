"""Deterministic DocVQA answer, citation, and latency metrics for E4."""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any
from uuid import UUID

from evaluation.e3.metrics import normalize_answer
from evaluation.e4.models import (
    E4AnswerMetrics,
    E4CitationMetrics,
    E4ConfigurationSummary,
    E4LatencyStats,
    E4QuestionRecord,
)


ANLS_REFERENCE = (
    "DocVQA-compatible ANLS: normalize both strings with Unicode NFKC and lower-case, "
    "compute Levenshtein distance d, then max(0, 1 - d / max(len(prediction), len(answer))) "
    "when normalized distance is below 0.5, otherwise 0; use the maximum over accepted answers."
)
EM_NORMALIZATION = (
    "Unicode NFKC, casefolding, whitespace collapse, and trimming; exact equality against "
    "any accepted answer. This is the same conservative literal normalization used by E3."
)
_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def _levenshtein(source: str, target: str) -> int:
    """Calculate character Levenshtein distance without a new dependency."""

    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_char in enumerate(target, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[target_index] + 1,
                    previous[target_index - 1] + (source_char != target_char),
                )
            )
        previous = current
    return previous[-1]


def _anls_text(value: str) -> str:
    """Apply the reference evaluator's Unicode-stable lowercase preprocessing."""

    return unicodedata.normalize("NFKC", value).lower()


def anls_similarity(prediction: str, answer: str, *, threshold: float = 0.5) -> float:
    """Return one DocVQA-compatible ANLS similarity value."""

    predicted = _anls_text(prediction)
    target = _anls_text(answer)
    denominator = max(len(predicted), len(target))
    if denominator == 0:
        return 1.0
    normalized_distance = _levenshtein(predicted, target) / denominator
    return round(1.0 - normalized_distance, 12) if normalized_distance < threshold else 0.0


def anls_score(prediction: str | None, accepted_answers: Sequence[str]) -> float:
    """Use the maximum ANLS score over all accepted DocVQA answers."""

    if prediction is None:
        return 0.0
    return max((anls_similarity(prediction, answer) for answer in accepted_answers), default=0.0)


def normalized_exact_match(prediction: str | None, accepted_answers: Sequence[str]) -> bool:
    """Return normalized exact match against any accepted answer."""

    if prediction is None:
        return False
    normalized_prediction = normalize_answer(prediction)
    return any(normalized_prediction == normalize_answer(answer) for answer in accepted_answers)


def extract_citation_labels(answer: str | None) -> list[str]:
    """Extract stable, de-duplicated source labels in answer order."""

    if not answer:
        return []
    return list(dict.fromkeys(f"S{number}" for number in _CITATION_PATTERN.findall(answer)))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _stats(values: Iterable[float]) -> E4LatencyStats:
    """Calculate mean, median, and inclusive interpolated P95."""

    ordered = sorted(values)
    if not ordered:
        return E4LatencyStats(samples=0)
    if len(ordered) == 1:
        p95 = ordered[0]
    else:
        position = (len(ordered) - 1) * 0.95
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return E4LatencyStats(
        mean_ms=statistics.fmean(ordered),
        median_ms=statistics.median(ordered),
        p95_ms=p95,
        samples=len(ordered),
    )


def _answer_metrics(records: Sequence[E4QuestionRecord], *, scorable_only: bool) -> E4AnswerMetrics:
    selected = [record for record in records if not scorable_only or record.ground_truth_status == "SCORABLE"]
    if not selected:
        return E4AnswerMetrics()
    return E4AnswerMetrics(
        anls=statistics.fmean(record.anls for record in selected),
        exact_match=statistics.fmean(record.exact_match for record in selected),
    )


def summarize_configuration(
    records: Sequence[E4QuestionRecord],
    *,
    configuration: str,
    search_mode: str,
    rerank: bool,
    top_k: int,
) -> E4ConfigurationSummary:
    """Aggregate E4 outcomes, scoring failures as zero in end-to-end metrics."""

    attempted = len(records)
    scorable = sum(record.ground_truth_status == "SCORABLE" for record in records)
    answered = sum(record.status == "ANSWERED" for record in records)
    abstained = sum(record.status == "ABSTAINED" for record in records)
    failed = attempted - answered - abstained
    citation_records = [record for record in records if record.status == "ANSWERED"]
    with_citations = [record for record in citation_records if record.citation_labels_emitted]
    emitted = sum(record.citation_labels_emitted for record in citation_records)
    valid = sum(record.citation_labels_valid for record in citation_records)
    document_hits = sum(record.citation_document_hit for record in with_citations)
    gold_hits = sum(record.gold_evidence_citation_hit for record in with_citations)
    support_records = [record for record in with_citations if record.answer_supported_by_cited_evidence is not None]
    true_document_citations = sum(
        record.citation_labels_valid
        for record in citation_records
        if record.all_citations_true_document is True
    )
    # Per-record precision is retained for audit; the aggregate is citation weighted.
    gold_cited = sum(record.gold_evidence_citation_count for record in with_citations)
    latency_fields = (
        "retrieval_time_ms",
        "reranking_time_ms",
        "context_build_time_ms",
        "generation_time_ms",
        "grounding_verification_time_ms",
        "total_pipeline_time_ms",
    )
    latency = {
        field: _stats(
            getattr(record, field)
            for record in records
            if getattr(record, field) is not None
        )
        for field in latency_fields
    }
    failures = Counter(record.reason_code for record in records if record.reason_code)
    return E4ConfigurationSummary(
        configuration=configuration,
        search_mode=search_mode,
        rerank=rerank,
        top_k=top_k,
        questions_attempted=attempted,
        questions_scorable=scorable,
        questions_unscorable=attempted - scorable,
        questions_answered=answered,
        questions_abstained=abstained,
        questions_failed=failed,
        answer_coverage_rate=_rate(answered, attempted),
        end_to_end=_answer_metrics(records, scorable_only=False),
        scorable=_answer_metrics(records, scorable_only=True),
        citation=E4CitationMetrics(
            citation_presence_rate=_rate(len(with_citations), len(citation_records)),
            citation_reference_validity_rate=_rate(valid, emitted),
            citation_document_hit_rate=_rate(document_hits, len(with_citations)),
            all_citations_true_document_rate=_rate(true_document_citations, emitted),
            gold_evidence_citation_hit_rate=_rate(gold_hits, len(with_citations)),
            gold_evidence_citation_precision=_rate(gold_cited, emitted),
            answer_supported_by_cited_evidence_rate=_rate(
                sum(record.answer_supported_by_cited_evidence is True for record in support_records),
                len(support_records),
            ),
        ),
        abstention_rate=_rate(abstained, attempted),
        correct_abstention_count=sum(
            record.status == "ABSTAINED" and record.ground_truth_status != "SCORABLE"
            for record in records
        ),
        latency=latency,
        failures=dict(sorted(failures.items())),
    )


def compare_configurations(
    hybrid: E4ConfigurationSummary,
    reranked: E4ConfigurationSummary,
) -> dict[str, Any]:
    """Report reranking deltas with units explicit."""

    def delta(left: float | None, right: float | None) -> dict[str, float | None]:
        return {
            "hybrid": left,
            "hybrid_reranked": right,
            "absolute_delta": right - left if left is not None and right is not None else None,
            "percentage_point_delta": (right - left) * 100 if left is not None and right is not None else None,
        }

    return {
        "anls_end_to_end": delta(hybrid.end_to_end.anls, reranked.end_to_end.anls),
        "exact_match_end_to_end": delta(hybrid.end_to_end.exact_match, reranked.end_to_end.exact_match),
        "anls_scorable": delta(hybrid.scorable.anls, reranked.scorable.anls),
        "exact_match_scorable": delta(hybrid.scorable.exact_match, reranked.scorable.exact_match),
        "gold_evidence_citation_hit_rate": delta(
            hybrid.citation.gold_evidence_citation_hit_rate,
            reranked.citation.gold_evidence_citation_hit_rate,
        ),
        "abstention_rate": delta(hybrid.abstention_rate, reranked.abstention_rate),
        "mean_total_pipeline_time_ms": delta(
            hybrid.latency["total_pipeline_time_ms"].mean_ms,
            reranked.latency["total_pipeline_time_ms"].mean_ms,
        ),
        "median_total_pipeline_time_ms": delta(
            hybrid.latency["total_pipeline_time_ms"].median_ms,
            reranked.latency["total_pipeline_time_ms"].median_ms,
        ),
    }
