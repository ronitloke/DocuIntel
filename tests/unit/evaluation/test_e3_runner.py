"""Unit tests for E3 bounded question selection, cleanup, and controlled artifacts."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from evaluation.e3.cleanup import CleanupSafetyError, cleanup_run_documents
from evaluation.e3.models import CorpusMapping
from evaluation.e3.runner import METHODS, _method_request, build_questions, write_controlled_state
from evaluation.schemas import EvaluationDocument, EvaluationPage, EvaluationQAPair
from evaluation.e3.models import E3Question


def _record(evaluation_id: str, question_count: int) -> EvaluationDocument:
    return EvaluationDocument(
        evaluation_id=evaluation_id,
        dataset="docvqa",
        split="validation",
        source_record_id=evaluation_id,
        source_document_id=f"source-{evaluation_id}",
        local_pdf_path=f"data/evaluation/processed/docvqa/{evaluation_id}.pdf",
        page_count=1,
        pages=[EvaluationPage(page_number=1, width=100, height=100)],
        qa_pairs=[
            EvaluationQAPair(
                question_id=f"q-{index}",
                question=f"Question {index}",
                accepted_answers=[f"Answer {index}"],
            )
            for index in range(question_count)
        ],
    )


def test_question_limit_is_bounded_and_deterministic() -> None:
    questions = build_questions([_record("doc-b", 2), _record("doc-a", 2)], 3)
    assert [question.question_key for question in questions] == [
        "doc-a:q-0",
        "doc-a:q-1",
        "doc-b:q-0",
    ]


def test_all_methods_receive_the_same_bounded_question_request() -> None:
    document_id = uuid4()
    question = E3Question(
        question_key="doc-a:q-0",
        evaluation_id="doc-a",
        source_record_id="source-a",
        source_document_id="source-doc-a",
        question_id="q-0",
        question="What is the answer?",
        accepted_answers=["Answer"],
        local_pdf_path="doc-a.pdf",
    )
    requests = [
        _method_request(question, mode, rerank, 10, [document_id])
        for _name, mode, rerank in METHODS
    ]
    assert {request.query for request in requests} == {question.question}
    assert {request.top_k for request in requests} == {10}
    assert {tuple(request.filters.document_ids or []) for request in requests} == {(document_id,)}


def test_controlled_state_serializes_without_metrics(tmp_path: Path) -> None:
    summary = write_controlled_state(
        tmp_path,
        split="validation",
        run_id="missing",
        reason_code="DOCVQA_DATA_REQUIRED",
        message="official data missing",
        manifest_path=tmp_path / "manifest.jsonl",
        document_limit=25,
        question_limit=100,
    )
    assert summary.status == "DOCVQA_DATA_REQUIRED"
    assert summary.methods == {}
    assert (tmp_path / "summary.json").is_file()
    assert "No retrieval metrics" in (tmp_path / "report.md").read_text(encoding="utf-8")


class _FakeRepository:
    def __init__(self, document: object) -> None:
        self.document = document
        self.deleted: list[object] = []

    def get_document(self, document_id):
        return self.document if self.document.id == document_id else None

    def delete_document(self, document_id):
        self.deleted.append(document_id)
        return self.document.stored_filename


def test_cleanup_refuses_identity_mismatch_before_delete(tmp_path: Path) -> None:
    document_id = uuid4()
    document = SimpleNamespace(
        id=document_id,
        original_filename="actual.pdf",
        stored_filename="stored.pdf",
        checksum_sha256="a" * 64,
    )
    repository = _FakeRepository(document)
    mapping = CorpusMapping(
        evaluation_id="doc-1",
        source_record_id="source-1",
        local_pdf_path="source.pdf",
        original_filename="different.pdf",
        document_id=document_id,
        stored_filename="stored.pdf",
        checksum_sha256="a" * 64,
        indexed_chunk_count=1,
        processing_status="indexed",
    )
    with pytest.raises(CleanupSafetyError):
        cleanup_run_documents(repository, [mapping], tmp_path)
    assert repository.deleted == []
