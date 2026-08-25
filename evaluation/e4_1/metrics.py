"""Deterministic answer parsing, review buckets, and E4.1 metrics."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from evaluation.e3.metrics import normalize_answer
from evaluation.e4.metrics import anls_score, normalized_exact_match
from evaluation.e4.models import E4CitationMetrics, E4LatencyStats
from evaluation.e4_1.models import (
    E4_1AnswerMetrics,
    E4_1ConfigurationSummary,
    E4_1QuestionRecord,
    E4_1ReviewCategory,
)


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
_BOLD_PATTERN = re.compile(r"(\*\*|__)(?P<text>.+?)(?:\1)", re.DOTALL)
_ITALIC_PATTERN = re.compile(r"(?<!\*)\*(?P<text>[^*\n]+)\*(?!\*)")
_CODE_PATTERN = re.compile(r"`(?P<text>[^`\n]+)`")


def strip_citation_labels(answer: str | None, available_source_ids: Iterable[str]) -> str:
    """Remove only citation tokens that refer to provided source blocks."""

    if not answer:
        return ""
    available = set(available_source_ids)

    def replace(match: re.Match[str]) -> str:
        label = f"S{match.group(1)}"
        return "" if label in available else match.group(0)

    return _CITATION_PATTERN.sub(replace, answer)


def cleanup_markdown(answer: str) -> str:
    """Remove narrow Markdown presentation markers without rewriting content."""

    cleaned = _BOLD_PATTERN.sub(lambda match: match.group("text"), answer)
    cleaned = _ITALIC_PATTERN.sub(lambda match: match.group("text"), cleaned)
    cleaned = _CODE_PATTERN.sub(lambda match: match.group("text"), cleaned)
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_metric_answer(answer: str | None, available_source_ids: Iterable[str]) -> str | None:
    """Create a conservative metric view while retaining the original answer."""

    if answer is None:
        return None
    return cleanup_markdown(strip_citation_labels(answer, available_source_ids))


def classify_review_case(
    *,
    status: str,
    raw_response: str | None,
    metric_answer: str | None,
    raw_anls: float,
    raw_exact_match: bool,
    metric_anls: float,
    metric_exact_match: bool,
    citations: Sequence[str],
    citations_valid: bool,
    answer_supported_by_cited_evidence: bool | None,
) -> E4_1ReviewCategory:
    """Assign a conservative, deterministic manual-review bucket."""

    if status == "GROUNDING_REJECTED":
        return "GROUNDING_REJECTED"
    if status != "ANSWERED":
        return "UNCLASSIFIED"
    if not raw_response or not metric_answer:
        return "EMPTY_ANSWER"
    if raw_exact_match or raw_anls > 0:
        return "EXACT_OR_ANLS_CORRECT"
    if metric_exact_match or metric_anls > 0:
        return "FORMAT_MISMATCH"
    if not citations:
        return "NO_CITATION"
    if citations_valid and answer_supported_by_cited_evidence is False:
        return "VALID_CITATION_WRONG_ANSWER"
    return "UNCLASSIFIED"


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None, "samples": 0}
    if len(ordered) == 1:
        p95 = ordered[0]
    else:
        position = (len(ordered) - 1) * 0.95
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": p95,
        "samples": len(ordered),
    }


def _answer_metrics(records: Sequence[E4_1QuestionRecord], *, metric_prefix: str, scorable_only: bool) -> E4_1AnswerMetrics:
    selected = [
        record
        for record in records
        if not scorable_only or record.ground_truth_status == "SCORABLE"
    ]
    if not selected:
        return E4_1AnswerMetrics()
    return E4_1AnswerMetrics(
        anls=statistics.fmean(getattr(record, f"{metric_prefix}_anls") for record in selected),
        exact_match=statistics.fmean(
            getattr(record, f"{metric_prefix}_exact_match") for record in selected
        ),
    )


def summarize_configuration(
    records: Sequence[E4_1QuestionRecord],
    *,
    configuration: str,
    search_mode: str,
    rerank: bool,
    top_k: int,
) -> E4_1ConfigurationSummary:
    """Summarize completion, raw/metric answers, citations, and measured latency."""

    attempted = len(records)
    scorable = sum(record.ground_truth_status == "SCORABLE" for record in records)
    answered = sum(record.status == "ANSWERED" for record in records)
    citation_records = [record for record in records if record.status == "ANSWERED"]
    with_citations = [record for record in citation_records if record.citation_labels_emitted]
    emitted = sum(record.citation_labels_emitted for record in citation_records)
    valid = sum(record.citation_labels_valid for record in citation_records)
    support_records = [
        record for record in with_citations if record.answer_supported_by_cited_evidence is not None
    ]
    true_document = sum(
        record.citation_labels_valid
        for record in citation_records
        if record.all_citations_true_document is True
    )
    gold_cited = sum(record.gold_evidence_citation_count for record in with_citations)
    latency_fields = (
        "retrieval_time_ms",
        "reranking_time_ms",
        "generation_time_ms",
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
    reviews = Counter(record.review_category for record in records)
    return E4_1ConfigurationSummary(
        configuration=configuration,
        search_mode=search_mode,
        rerank=rerank,
        top_k=top_k,
        questions_attempted=attempted,
        questions_scorable=scorable,
        questions_answered=answered,
        questions_failed=attempted - answered,
        completion_rate_scorable=_rate(answered, scorable),
        raw_end_to_end=_answer_metrics(records, metric_prefix="raw", scorable_only=False),
        raw_scorable=_answer_metrics(records, metric_prefix="raw", scorable_only=True),
        metric_end_to_end=_answer_metrics(records, metric_prefix="metric", scorable_only=False),
        metric_scorable=_answer_metrics(records, metric_prefix="metric", scorable_only=True),
        citation=E4CitationMetrics(
            citation_presence_rate=_rate(len(with_citations), len(citation_records)),
            citation_reference_validity_rate=_rate(valid, emitted),
            citation_document_hit_rate=_rate(
                sum(record.citation_document_hit for record in with_citations), len(with_citations)
            ),
            all_citations_true_document_rate=_rate(true_document, emitted),
            gold_evidence_citation_hit_rate=_rate(
                sum(record.gold_evidence_citation_hit for record in with_citations), len(with_citations)
            ),
            gold_evidence_citation_precision=_rate(gold_cited, emitted),
            answer_supported_by_cited_evidence_rate=_rate(
                sum(record.answer_supported_by_cited_evidence is True for record in support_records),
                len(support_records),
            ),
        ).model_dump(),
        latency=latency,
        failures=dict(sorted(failures.items())),
        review_categories=dict(sorted(reviews.items())),
    )


def metrics_csv_rows(summaries: dict[str, E4_1ConfigurationSummary]) -> list[dict[str, Any]]:
    """Flatten diagnostic summaries for CSV inspection."""

    rows: list[dict[str, Any]] = []
    for configuration, summary in summaries.items():
        scalar = {
            "questions_attempted": summary.questions_attempted,
            "questions_scorable": summary.questions_scorable,
            "questions_answered": summary.questions_answered,
            "questions_failed": summary.questions_failed,
            "completion_rate_scorable": summary.completion_rate_scorable,
        }
        for scope, metrics in (
            ("raw_end_to_end", summary.raw_end_to_end),
            ("raw_scorable", summary.raw_scorable),
            ("metric_end_to_end", summary.metric_end_to_end),
            ("metric_scorable", summary.metric_scorable),
        ):
            scalar[f"{scope}_anls"] = metrics.anls
            scalar[f"{scope}_exact_match"] = metrics.exact_match
        scalar.update({f"citation_{key}": value for key, value in summary.citation.items()})
        for stage, stats in summary.latency.items():
            scalar.update({f"{stage}_{key}": value for key, value in stats.items()})
        for key, value in summary.review_categories.items():
            scalar[f"review_{key}"] = value
        for metric, value in scalar.items():
            rows.append({"configuration": configuration, "metric": metric, "value": value})
    return rows

