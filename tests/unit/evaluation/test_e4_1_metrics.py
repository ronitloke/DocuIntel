"""Focused E4.1 deterministic diagnostics tests."""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from app.models.rag import AskResponse, RAGSource
from evaluation.e3.models import E3Question, E3QuestionGroundTruth
from evaluation.e4_1.metrics import (
    classify_review_case,
    cleanup_markdown,
    extract_metric_answer,
    strip_citation_labels,
    summarize_configuration,
)
from evaluation.e4_1.models import E4_1QuestionRecord
from evaluation.e4_1.runner import (
    _response_record,
    _run_configuration,
    effective_benchmark_settings,
    write_controlled_state,
)
from app.core.config import Settings


def _question() -> E3Question:
    return E3Question(
        question_key="doc-1:q-1",
        evaluation_id="doc-1",
        source_record_id="source-1",
        source_document_id="source-doc-1",
        question_id="q-1",
        question="What is the answer?",
        accepted_answers=["42"],
        local_pdf_path="doc-1.pdf",
    )


def _ground_truth(document_id, chunk_id) -> E3QuestionGroundTruth:
    return E3QuestionGroundTruth(
        question_key="doc-1:q-1",
        status="SCORABLE",
        target_document_id=document_id,
        relevant_chunk_ids=[chunk_id],
    )


def test_citation_cleanup_is_conservative_and_supports_multiple_labels() -> None:
    assert strip_citation_labels("Answer [S1] and [S2] [S9]", ["S1", "S2"]) == "Answer  and  [S9]"
    assert extract_metric_answer("**42** [S1]", ["S1"]) == "42"


def test_markdown_cleanup_does_not_semantically_rewrite_text() -> None:
    assert cleanup_markdown("# **The answer** is `42`.") == "The answer is 42."


def test_response_projection_preserves_raw_and_creates_metric_view() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    response = AskResponse(
        answer="**42** [S1]",
        model="llama3.2:3b",
        sources=[
            RAGSource(
                source_id="S1",
                document_id=document_id,
                chunk_id=chunk_id,
                filename="doc.pdf",
                excerpt="The answer is 42.",
                contains_ocr=False,
                final_rank=1,
            )
        ],
        citations=["S1"],
        citations_valid=True,
        retrieval_time_ms=10,
        generation_time_ms=20,
        total_time_ms=35,
    )
    record = _response_record(_question(), _ground_truth(document_id, chunk_id), "hybrid", response)
    assert record.raw_response == "**42** [S1]"
    assert record.metric_answer == "42"
    assert record.raw_exact_match is False
    assert record.metric_exact_match is True
    assert record.review_category == "FORMAT_MISMATCH"


def test_review_classification_is_fail_closed_for_ambiguous_completed_answers() -> None:
    assert classify_review_case(
        status="ANSWERED",
        raw_response="An explanation",
        metric_answer="An explanation",
        raw_anls=0,
        raw_exact_match=False,
        metric_anls=0,
        metric_exact_match=False,
        citations=["S1"],
        citations_valid=True,
        answer_supported_by_cited_evidence=True,
    ) == "UNCLASSIFIED"
    assert classify_review_case(
        status="ANSWERED",
        raw_response="Wrong",
        metric_answer="Wrong",
        raw_anls=0,
        raw_exact_match=False,
        metric_anls=0,
        metric_exact_match=False,
        citations=["S1"],
        citations_valid=True,
        answer_supported_by_cited_evidence=False,
    ) == "VALID_CITATION_WRONG_ANSWER"


def test_timeout_override_is_a_copy_and_production_default_is_unchanged() -> None:
    production = Settings(ollama_timeout_seconds=120)
    diagnostic = effective_benchmark_settings(production, 300)
    assert production.ollama_timeout_seconds == 120
    assert diagnostic.ollama_timeout_seconds == 300
    assert effective_benchmark_settings(production, None) is production


def _record(key: str, *, status: str, total_ms: float) -> E4_1QuestionRecord:
    return E4_1QuestionRecord(
        question_key=key,
        evaluation_id="doc-1",
        question_id=key,
        question="What is the answer?",
        accepted_answers=["42"],
        ground_truth_status="SCORABLE",
        configuration="hybrid",
        status=status,  # type: ignore[arg-type]
        total_pipeline_time_ms=total_ms,
        review_category="UNCLASSIFIED",
    )


def test_completed_and_timeout_records_remain_visible_in_accounting() -> None:
    summary = summarize_configuration(
        [_record("q-1", status="ANSWERED", total_ms=100), _record("q-2", status="GENERATION_FAILED", total_ms=120000)],
        configuration="hybrid",
        search_mode="hybrid",
        rerank=False,
        top_k=5,
    )
    assert summary.questions_answered == 1
    assert summary.questions_failed == 1
    assert summary.latency["total_pipeline_time_ms"]["samples"] == 2


def test_warmup_metadata_is_recorded_and_excluded_from_measured_records() -> None:
    document_id = uuid4()
    chunk_id = uuid4()

    class FakeClient:
        timeout_seconds = None

    class FakeRAG:
        ollama_client = FakeClient()

        async def ask(self, _request):
            return AskResponse(
                answer="42 [S1]",
                model="llama3.2:3b",
                sources=[
                    RAGSource(
                        source_id="S1",
                        document_id=document_id,
                        chunk_id=chunk_id,
                        filename="doc.pdf",
                        excerpt="42",
                        contains_ocr=False,
                        final_rank=1,
                    )
                ],
                citations=["S1"],
                citations_valid=True,
                retrieval_time_ms=1,
                generation_time_ms=2,
                total_time_ms=3,
            )

    records, warmup = asyncio.run(
        _run_configuration(
            configuration="hybrid",
            rerank=False,
            questions=[_question()],
            ground_truth={"doc-1:q-1": _ground_truth(document_id, chunk_id)},
            indexed_document_ids=[document_id],
            rag_service=FakeRAG(),
            top_k=5,
        )
    )
    assert warmup["status"] == "completed"
    assert warmup["excluded_from_metrics"] is True
    assert len(records) == 1


def test_controlled_serialization_writes_review_artifact_set(tmp_path: Path) -> None:
    summary = write_controlled_state(
        tmp_path / "e4_1-run",
        split="validation",
        run_id="controlled",
        message="official data missing",
        manifest_path=tmp_path / "manifest.jsonl",
        document_limit=25,
        question_limit=100,
    )
    assert summary.status == "CONTROLLED_FAILURE"
    run_dir = tmp_path / "e4_1-run"
    for filename in (
        "summary.json",
        "per_question.jsonl",
        "answers.jsonl",
        "review_cases.jsonl",
        "metrics.csv",
        "report.md",
        "run_metadata.json",
        "corpus_mapping.jsonl",
    ):
        assert (run_dir / filename).is_file()
