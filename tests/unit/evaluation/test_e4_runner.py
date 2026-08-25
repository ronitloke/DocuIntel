"""Focused E4 response-projection and controlled-artifact tests."""

from pathlib import Path
from uuid import uuid4

from app.models.rag import AskResponse, RAGSource
from evaluation.e3.models import E3Question, E3QuestionGroundTruth
from evaluation.e4.runner import _response_record, write_controlled_state


def _question() -> E3Question:
    return E3Question(
        question_key="doc-1:q-1",
        evaluation_id="doc-1",
        source_record_id="source-1",
        source_document_id="docvqa-source-1",
        question_id="q-1",
        question="What is the answer?",
        accepted_answers=["42"],
        local_pdf_path="data/evaluation/processed/docvqa/doc-1.pdf",
    )


def test_response_projection_measures_true_document_and_gold_chunk_citation() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    response = AskResponse(
        answer="The answer is 42. [S1]",
        model="llama3.2:3b",
        sources=[
            RAGSource(
                source_id="S1",
                document_id=document_id,
                chunk_id=chunk_id,
                filename="doc.pdf",
                contains_ocr=False,
                excerpt="The answer is 42.",
                final_rank=1,
            )
        ],
        citations=["S1"],
        citations_valid=True,
        retrieval_time_ms=10,
        rerank_time_ms=None,
        generation_time_ms=20,
        total_time_ms=35,
    )
    record = _response_record(
        _question(),
        E3QuestionGroundTruth(
            question_key="doc-1:q-1",
            status="SCORABLE",
            target_document_id=document_id,
            relevant_chunk_ids=[chunk_id],
        ),
        "hybrid",
        response,
    )
    assert record.status == "ANSWERED"
    assert record.citation_document_hit
    assert record.gold_evidence_citation_hit
    assert record.gold_evidence_citation_precision == 1.0
    assert record.answer_supported_by_cited_evidence
    # The production response includes explanatory text and a citation label;
    # normalized EM intentionally evaluates the complete generated answer.
    assert not record.exact_match


def test_controlled_state_writes_required_artifact_set(tmp_path: Path) -> None:
    summary = write_controlled_state(
        tmp_path / "e4-run",
        split="validation",
        run_id="controlled",
        reason_code="DOCVQA_DATA_REQUIRED",
        message="official data missing",
        manifest_path=tmp_path / "manifest.jsonl",
        document_limit=25,
        question_limit=100,
    )
    assert summary.status == "DOCVQA_DATA_REQUIRED"
    run_dir = tmp_path / "e4-run"
    for filename in ("summary.json", "per_question.jsonl", "answers.jsonl", "metrics.csv", "report.md", "run_metadata.json", "corpus_mapping.jsonl"):
        assert (run_dir / filename).is_file()
