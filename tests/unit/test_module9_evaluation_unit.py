"""Deterministic Module 9 dataset, metric, report, and grounding tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import (
    aggregate_retrieval,
    apply_quality_gates,
    build_retrieval_case_result,
    compare_baseline,
    evaluate_key_facts,
    evaluate_rag_response,
    recall_at_k,
    reciprocal_rank,
    success_at_k,
)
from app.evaluation.models import (
    EvaluationCase,
    EvaluationConfiguration,
    EvaluationDataset,
    QualityGateConfig,
    RetrievalEvaluationReport,
    RetrievalSummary,
)
from app.evaluation.reporting import report_payload, write_report
from app.db.repository import EvaluationDocumentRecord
from app.evaluation.retrieval import RetrievalEvaluator
from app.models.rag import AskResponse, RAGSource
from app.models.search import SearchMode, SearchResult
from app.services.rag.service import validate_citation_labels


def make_result(
    rank: int,
    filename: str,
    *,
    base_rank: int | None = None,
    text: str = "irrelevant content",
) -> SearchResult:
    """Create deterministic search results for pure metric tests."""

    return SearchResult(
        rank=rank,
        chunk_id=uuid4(),
        document_id=uuid4(),
        original_filename=filename,
        sequence_number=rank,
        text=text,
        start_page=1,
        end_page=1,
        content_type="text",
        contains_ocr=False,
        retrieval_method=SearchMode.HYBRID,
        base_rank=base_rank,
        rerank_score=3.0 if base_rank is not None else None,
        reranked=base_rank is not None,
    )


def test_success_recall_and_reciprocal_rank_are_mathematically_exact() -> None:
    """The core ranking functions match standard definitions."""

    assert success_at_k([3], 1) is False
    assert success_at_k([3], 3) is True
    assert recall_at_k({"a", "b"}, {"a"}, 3, 2) == 0.5
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(3) == pytest.approx(1 / 3)
    assert reciprocal_rank(None) == 0.0


def test_retrieval_case_calculates_success_recall_mrr_and_rank_movement() -> None:
    """A relevant result at rank three produces the expected metrics."""

    case = EvaluationCase(
        id="case",
        question="policy",
        expected_document="policy.pdf",
    )
    results = [
        make_result(1, "other.pdf"),
        make_result(2, "other.pdf"),
        make_result(3, "policy.pdf", base_rank=5),
    ]

    measured = build_retrieval_case_result(
        case,
        results,
        retrieval_time_ms=4,
        rerank_time_ms=2,
        total_search_time_ms=7,
    )

    assert measured.success_at_k == {"1": False, "3": True, "5": True, "10": True}
    assert measured.recall_at_k["3"] == 1.0
    assert measured.reciprocal_rank == pytest.approx(1 / 3)
    assert measured.base_rank == 5
    assert measured.final_rank == 3
    assert measured.rank_delta == 2


def test_retrieval_aggregate_reports_latency_and_reranker_impact() -> None:
    """Aggregates use means/medians and positive rank delta means improvement."""

    cases = [
        build_retrieval_case_result(
            EvaluationCase(id="a", question="a", expected_document="policy.pdf"),
            [make_result(1, "policy.pdf", base_rank=3)],
            retrieval_time_ms=10,
            rerank_time_ms=5,
            total_search_time_ms=20,
        ),
        build_retrieval_case_result(
            EvaluationCase(id="b", question="b", expect_no_evidence=True),
            [],
            retrieval_time_ms=12,
            rerank_time_ms=None,
            total_search_time_ms=12,
        ),
    ]
    summary = aggregate_retrieval(cases)

    assert summary.mrr == 1.0
    assert summary.mean_retrieval_latency_ms == 11
    assert summary.median_total_search_latency_ms == 16
    assert summary.no_evidence_correct_rate == 1.0
    assert summary.rerank_impact["improved_cases"] == 1


def test_dataset_validation_loads_json_and_rejects_duplicate_ids(tmp_path) -> None:
    """Human-editable JSON is validated before evaluation starts."""

    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "name": "test",
                "cases": [
                    {"id": "one", "question": "one?", "expect_no_evidence": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_dataset(path).cases[0].expect_no_evidence is True

    with pytest.raises(ValidationError):
        EvaluationDataset(
            name="invalid",
            cases=[
                {"id": "same", "question": "one"},
                {"id": "same", "question": "two"},
            ],
        )

    jsonl_path = tmp_path / "dataset.jsonl"
    jsonl_path.write_text(
        '{"id":"jsonl-case","question":"What?","tags":["smoke"]}\n',
        encoding="utf-8",
    )
    jsonl_dataset = load_dataset(jsonl_path)
    assert jsonl_dataset.name == "dataset"
    assert jsonl_dataset.cases[0].tags == ["smoke"]


def test_key_fact_and_citation_matching_are_conservative() -> None:
    """Facts use normalized substring matching and citation labels are reused from RAG."""

    facts, coverage = evaluate_key_facts("Thirty   DAYS of written notice", ["thirty days", "resignation"])
    assert [fact.matched for fact in facts] == [True, False]
    assert coverage == 0.5
    assert validate_citation_labels("Answer [S1] [S3]", ["S1", "S2"]) == (["S1", "S3"], False)


def test_rag_evaluation_checks_facts_citations_source_and_evidence_support() -> None:
    """A grounded response gets full deterministic support when evidence contains its facts."""

    document_id = uuid4()
    response = AskResponse(
        answer="Employees need thirty days written notice. [S1]",
        model="test-model",
        sources=[
            RAGSource(
                source_id="S1",
                document_id=document_id,
                chunk_id=uuid4(),
                filename="policy.pdf",
                start_page=1,
                end_page=1,
                contains_ocr=False,
                excerpt="Employees need thirty days written notice before resignation.",
                final_rank=1,
            )
        ],
        citations=["S1"],
        citations_valid=True,
        retrieval_time_ms=2,
        generation_time_ms=8,
        total_time_ms=11,
    )
    result = evaluate_rag_response(
        EvaluationCase(
            id="policy",
            question="How much notice?",
            expected_document="policy.pdf",
            expected_facts=["thirty days", "written notice"],
        ),
        response,
    )

    assert result.key_fact_coverage == 1.0
    assert result.citations_valid is True
    assert result.expected_document_cited is True
    assert result.evidence_support is True


def test_no_evidence_rag_case_requires_no_sources_or_citations() -> None:
    """Unsupported cases are successful only when the RAG response stays empty of evidence."""

    response = AskResponse(
        answer="I couldn't find relevant information in the indexed documents to answer that question.",
        model="test-model",
        sources=[],
        citations=[],
        citations_valid=False,
        retrieval_time_ms=1,
        generation_time_ms=0,
        total_time_ms=1,
    )
    result = evaluate_rag_response(
        EvaluationCase(id="unsupported", question="unknown?", expect_no_evidence=True),
        response,
    )
    assert result.no_evidence_correct is True
    assert result.evidence_support is True


def test_baseline_comparison_and_quality_gates_are_optional() -> None:
    """Baseline deltas are reported and explicit gates can fail without universal defaults."""

    current = {"summary": {"mrr": 0.75, "mean_total_search_latency_ms": 12.0}}
    baseline = {"summary": {"mrr": 0.80, "mean_total_search_latency_ms": 10.0}}
    comparison = compare_baseline(current, baseline, tolerance=0.001)
    assert comparison["regressions"]
    gate = apply_quality_gates(
        {"success_at_k": {"3": 0.5}, "mrr": 0.5, "mean_total_search_latency_ms": 20},
        QualityGateConfig(minimum_success_at_3=0.9, minimum_mrr=0.8),
    )
    assert gate.passed is False
    assert len(gate.failures) == 2


def test_report_writer_serializes_uuid_and_datetime_fields(tmp_path) -> None:
    """Reports are stable JSON artifacts suitable for CI or human review."""

    report = RetrievalEvaluationReport(
        dataset="report",
        configuration=EvaluationConfiguration(mode="keyword", top_k=1),
        generated_at=datetime.now(UTC),
        summary=RetrievalSummary(
            cases=0,
            positive_cases=0,
            no_evidence_cases=0,
            success_at_k={"1": 0.0},
            recall_at_k={"1": 0.0},
            mrr=0.0,
        ),
        cases=[],
    )
    output = tmp_path / "results" / "report.json"
    write_report(report, output)
    payload = report_payload(report)
    assert payload["configuration"]["mode"] == "keyword"
    assert isinstance(payload["generated_at"], str)
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_retrieval_preflight_confirms_expected_indexed_documents() -> None:
    """Valid filename labels produce case-level existence and chunk checks."""

    repository = SimpleNamespace(
        evaluation_document_inventory=lambda: [
            EvaluationDocumentRecord(
                document_id=uuid4(),
                original_filename="policy.pdf",
                is_indexed=True,
                chunk_count=3,
            )
        ]
    )
    evaluator = RetrievalEvaluator(SimpleNamespace(repository=repository))
    dataset = EvaluationDataset(
        name="preflight",
        cases=[EvaluationCase(id="policy", question="policy?", expected_document="policy.pdf")],
    )

    checks = evaluator.validate_dataset(dataset)

    assert checks[0].exists is True
    assert checks[0].indexed_chunks == 3


def test_retrieval_preflight_fails_for_missing_or_unindexed_documents() -> None:
    """Invalid expected-document labels stop evaluation before zero metrics are reported."""

    repository = SimpleNamespace(
        evaluation_document_inventory=lambda: [
            EvaluationDocumentRecord(
                document_id=uuid4(),
                original_filename="present.pdf",
                is_indexed=True,
                chunk_count=2,
            ),
            EvaluationDocumentRecord(
                document_id=uuid4(),
                original_filename="not-indexed.pdf",
                is_indexed=False,
                chunk_count=0,
            ),
        ]
    )
    evaluator = RetrievalEvaluator(SimpleNamespace(repository=repository))
    dataset = EvaluationDataset(
        name="preflight",
        cases=[
            EvaluationCase(id="present", question="present?", expected_document="present.pdf"),
            EvaluationCase(id="missing", question="missing?", expected_document="missing.pdf"),
        ],
    )

    with pytest.raises(ValueError, match="missing.pdf"):
        evaluator.validate_dataset(dataset)

    unindexed = EvaluationDataset(
        name="preflight",
        cases=[
            EvaluationCase(
                id="unindexed",
                question="unindexed?",
                expected_document="not-indexed.pdf",
            )
        ],
    )
    with pytest.raises(ValueError, match="no indexed chunks"):
        evaluator.validate_dataset(unindexed)


def test_retrieval_preflight_allows_cases_without_document_labels() -> None:
    """Chunk-ID-only or no-label datasets do not require document inventory checks."""

    evaluator = RetrievalEvaluator(SimpleNamespace(repository=None))
    dataset = EvaluationDataset(
        name="unlabeled",
        cases=[EvaluationCase(id="case", question="question?")],
    )

    assert evaluator.validate_dataset(dataset) == []
