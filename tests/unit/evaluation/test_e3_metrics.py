"""Unit tests for E3 ground truth and retrieval metrics."""

from uuid import uuid4

import pytest

from evaluation.e3.metrics import (
    aggregate_method_results,
    build_question_ground_truth,
    evaluate_ranked_ids,
    normalize_answer,
)
from evaluation.e3.models import E3Question, E3QuestionMethodResult, IndexedChunk


def _question(*answers: str) -> E3Question:
    return E3Question(
        question_key="doc-1:q-1",
        evaluation_id="doc-1",
        source_record_id="source-1",
        source_document_id="source-doc-1",
        question_id="q-1",
        question="What is the total?",
        accepted_answers=list(answers),
        local_pdf_path="data/evaluation/processed/docvqa/doc-1.pdf",
    )


def test_answer_normalization_is_literal_and_conservative() -> None:
    assert normalize_answer("  Total\r\nVALUE ") == "total value"
    assert normalize_answer("Café") == normalize_answer("CAFÉ")


def test_multiple_accepted_answers_build_union_of_relevant_chunks() -> None:
    document_id = uuid4()
    first_chunk = IndexedChunk(
        chunk_id=uuid4(),
        document_id=document_id,
        sequence_number=1,
        text="The total value is 42.",
    )
    second_chunk = IndexedChunk(
        chunk_id=uuid4(),
        document_id=document_id,
        sequence_number=2,
        text="The amount may also be written as forty two.",
    )
    result = build_question_ground_truth(
        _question("42", "forty two"),
        target_document_id=document_id,
        chunks=[first_chunk, second_chunk],
        document_indexed=True,
    )
    assert result.status == "SCORABLE"
    assert result.relevant_chunk_ids == [first_chunk.chunk_id, second_chunk.chunk_id]
    assert set(result.answer_matches) == {"42", "forty two"}


def test_unscorable_states_are_distinct() -> None:
    document_id = uuid4()
    chunk = IndexedChunk(
        chunk_id=uuid4(),
        document_id=document_id,
        sequence_number=1,
        text="No matching evidence.",
    )
    not_indexed = build_question_ground_truth(
        _question("answer"),
        target_document_id=document_id,
        chunks=[chunk],
        document_indexed=True,
    )
    processing_failed = build_question_ground_truth(
        _question("answer"),
        target_document_id=document_id,
        chunks=None,
        document_indexed=False,
    )
    invalid = build_question_ground_truth(
        _question("   "),
        target_document_id=document_id,
        chunks=[chunk],
        document_indexed=True,
    )
    assert not_indexed.status == "ANSWER_NOT_INDEXED"
    assert processing_failed.status == "DOCUMENT_PROCESSING_FAILED"
    assert invalid.status == "INVALID_GROUND_TRUTH"


def test_recall_hit_mrr_and_document_hit_are_distinct() -> None:
    target_document = uuid4()
    other_document = uuid4()
    relevant_one = uuid4()
    relevant_two = uuid4()
    ranked = [uuid4(), relevant_one, relevant_two]
    metrics = evaluate_ranked_ids(
        ranked,
        [relevant_one, relevant_two],
        ranked_document_ids=[other_document, target_document, target_document],
        target_document_id=target_document,
    )
    assert metrics["recall_at_k"]["1"] == 0
    assert metrics["recall_at_k"]["3"] == 1
    assert metrics["hit_at_k"]["3"] is True
    assert metrics["document_hit_at_k"]["1"] is False
    assert metrics["document_hit_at_k"]["3"] is True
    assert metrics["reciprocal_rank"] == 0.5


def test_zero_relevant_result_handling_is_explicit() -> None:
    with pytest.raises(ValueError, match="at least one relevant"):
        evaluate_ranked_ids([], [], ranked_document_ids=[], target_document_id=uuid4())


def test_aggregation_uses_all_scorable_questions_and_inclusive_p95() -> None:
    results = [
        E3QuestionMethodResult(
            question_key=f"q-{index}",
            method="keyword",
            relevant_chunk_ids=[uuid4()],
            reciprocal_rank=1.0 if index == 1 else 0.0,
            recall_at_k={"1": 1.0 if index == 1 else 0.0, "3": 1.0, "5": 1.0, "10": 1.0},
            hit_at_k={"1": index == 1, "3": True, "5": True, "10": True},
            document_hit_at_k={"1": index == 1, "3": True, "5": True, "10": True},
            retrieval_time_ms=float(index),
            total_retrieval_pipeline_ms=float(index + 1),
            wall_clock_time_ms=float(index + 2),
        )
        for index in (1, 2, 3)
    ]
    summary = aggregate_method_results("keyword", results, scorable_questions=3, candidate_count=10)
    assert summary.recall_at_k["1"] == pytest.approx(1 / 3)
    assert summary.mrr == pytest.approx(1 / 3)
    assert summary.retrieval_latency.p95_ms == pytest.approx(2.9)
    assert summary.candidate_count == 10
