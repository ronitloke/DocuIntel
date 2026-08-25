"""Focused deterministic E4 metric tests; no benchmark claims are made here."""

from uuid import uuid4

import pytest

from evaluation.e4.metrics import (
    anls_score,
    anls_similarity,
    compare_configurations,
    extract_citation_labels,
    normalized_exact_match,
    summarize_configuration,
)
from evaluation.e4.models import E4QuestionRecord


def _record(
    key: str,
    *,
    status: str = "ANSWERED",
    gt_status: str = "SCORABLE",
    answer: str | None = "42 [S1]",
    anls: float = 1.0,
    exact_match: bool = True,
    citations: int = 1,
    valid_citations: int = 1,
    gold_count: int = 1,
    gold_precision: float | None = 1.0,
) -> E4QuestionRecord:
    return E4QuestionRecord(
        question_key=key,
        evaluation_id="doc-1",
        question_id=key,
        question="What is the answer?",
        accepted_answers=["42"],
        ground_truth_status=gt_status,
        configuration="hybrid",
        status=status,  # type: ignore[arg-type]
        answer=answer,
        anls=anls,
        exact_match=exact_match,
        citation_labels_emitted=citations,
        citation_labels_valid=valid_citations,
        gold_evidence_citation_count=gold_count,
        gold_evidence_citation_precision=gold_precision,
        answer_supported_by_cited_evidence=True if citations else None,
    )


def test_anls_matches_reference_formula_and_threshold() -> None:
    assert anls_similarity("Coca cola", "coca cola") == 1.0
    assert anls_similarity("CocaCola", "Coca Cola") == pytest.approx(8 / 9)
    assert anls_similarity("ab", "cd") == 0.0
    assert anls_score("wrong", ["right", "wrong answer"]) == 0.0


def test_multiple_accepted_answers_and_normalized_exact_match() -> None:
    assert anls_score("SECOND", ["first", "second"]) == 1.0
    assert normalized_exact_match("  Café\r\n", ["CAFE\u0301"])
    assert not normalized_exact_match("42", ["43"])


def test_citation_labels_are_deterministic_and_deduplicated() -> None:
    assert extract_citation_labels("Answer [S2], then [S1], again [S2].") == ["S2", "S1"]
    assert extract_citation_labels(None) == []


def test_end_to_end_and_scorable_accounting_keep_failures_visible() -> None:
    records = [
        _record("q-1"),
        _record("q-2", status="ANSWER_NOT_INDEXED", gt_status="ANSWER_NOT_INDEXED", answer=None, anls=0, exact_match=False, citations=0, valid_citations=0, gold_count=0, gold_precision=None),
        _record("q-3", status="ABSTAINED", answer="I don't know", anls=0, exact_match=False, citations=0, valid_citations=0, gold_count=0, gold_precision=None),
    ]
    summary = summarize_configuration(
        records,
        configuration="hybrid",
        search_mode="hybrid",
        rerank=False,
        top_k=5,
    )
    assert summary.questions_attempted == 3
    assert summary.questions_scorable == 2
    assert summary.questions_answered == 1
    assert summary.questions_abstained == 1
    assert summary.questions_failed == 1
    assert summary.end_to_end.anls == pytest.approx(1 / 3)
    assert summary.scorable.anls == pytest.approx(0.5)
    assert summary.answer_coverage_rate == pytest.approx(1 / 3)
    assert summary.citation.citation_presence_rate == 1.0
    assert summary.citation.gold_evidence_citation_precision == 1.0


def test_latency_aggregation_keeps_measured_controlled_failures() -> None:
    failed = _record(
        "q-timeout",
        status="GENERATION_FAILED",
        answer=None,
        anls=0,
        exact_match=False,
        citations=0,
        valid_citations=0,
        gold_count=0,
        gold_precision=None,
    )
    failed.total_pipeline_time_ms = 120000
    summary = summarize_configuration(
        [failed],
        configuration="hybrid",
        search_mode="hybrid",
        rerank=False,
        top_k=5,
    )
    assert summary.latency["total_pipeline_time_ms"].samples == 1
    assert summary.latency["total_pipeline_time_ms"].mean_ms == 120000


def test_reranking_deltas_are_explicitly_absolute_and_percentage_points() -> None:
    left = summarize_configuration(
        [_record("q-1", anls=0.4, exact_match=False)],
        configuration="hybrid",
        search_mode="hybrid",
        rerank=False,
        top_k=5,
    )
    right = summarize_configuration(
        [_record("q-1", anls=0.8)],
        configuration="hybrid_reranked",
        search_mode="hybrid",
        rerank=True,
        top_k=5,
    )
    delta = compare_configurations(left, right)
    assert delta["anls_end_to_end"]["absolute_delta"] == pytest.approx(0.4)
    assert delta["anls_end_to_end"]["percentage_point_delta"] == pytest.approx(40)
