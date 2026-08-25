"""Bounded E4.1 diagnostics over the unchanged production RAG path."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

from app.core.config import Settings
from app.core.exceptions import DocumentIngestionError, OllamaServiceError, RAGServiceError
from app.db.repository import DocumentRepository
from app.db.session import create_database
from app.models.documents import DocumentIngestionResponse, DocumentStatus
from app.models.rag import AskRequest, AskResponse
from app.models.search import SearchFilters, SearchMode
from app.services.chunking.structure_aware import StructureAwareChunker
from app.services.documents.document_management import DocumentManagementService
from app.services.documents.indexing import DocumentIndexingService
from app.services.documents.pdf_ingestion import PDFIngestionService
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.llm.ollama import OllamaClient
from app.services.rag.service import NO_RESULTS_ANSWER, RAGService
from app.services.reranking.cross_encoder import CrossEncoderReranker
from app.services.retrieval.search import SearchService
from evaluation.e3.cleanup import cleanup_run_documents
from evaluation.e3.metrics import ANSWER_NORMALIZATION_RULES, build_question_ground_truth
from evaluation.e3.models import CorpusMapping, E3Question, E3QuestionGroundTruth, IndexedChunk
from evaluation.e3.runner import LocalPDFUpload, _chunk_projection, build_questions
from evaluation.e4.metrics import ANLS_REFERENCE, EM_NORMALIZATION
from evaluation.e4.runner import _failure_reason, _question_payload, _safe_database_target, _source_metrics
from evaluation.e4_1.metrics import (
    classify_review_case,
    extract_metric_answer,
    metrics_csv_rows,
    summarize_configuration,
)
from evaluation.e4_1.models import E4_1ConfigurationSummary, E4_1QuestionRecord, E4_1RunSummary
from evaluation.manifests import now_utc_iso
from evaluation.schemas import EvaluationDocument
from evaluation.validation import ManifestValidationError, resolve_local_path, validate_manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class E4_1RunOptions:
    """Bounded E4.1 runtime options."""

    manifest_path: Path
    output_directory: Path
    split: str
    document_limit: int
    question_limit: int
    scorable_question_limit: int = 20
    top_k: int = 5
    generation_timeout_seconds: float | None = None
    keep_indexed: bool = False
    database_url_override: str | None = None
    production_baseline_directory: Path | None = None
    reuse_indexed_corpus: bool = False


CONFIGURATIONS: tuple[tuple[str, bool], ...] = (
    ("hybrid_reranked", True),
    ("hybrid", False),
)


def effective_benchmark_settings(settings: Settings, timeout_seconds: float | None) -> Settings:
    """Return a copied Settings object; never mutate production defaults."""

    if timeout_seconds is None:
        return settings
    if timeout_seconds <= 0:
        raise ValueError("generation_timeout_seconds must be greater than zero")
    return settings.model_copy(update={"ollama_timeout_seconds": timeout_seconds})


def _response_record(
    question: E3Question,
    ground_truth: E3QuestionGroundTruth,
    configuration: str,
    response: AskResponse,
) -> E4_1QuestionRecord:
    """Project one production response into raw and conservative metric views."""

    source_metrics = _source_metrics(
        response,
        target_document_id=ground_truth.target_document_id,
        relevant_chunk_ids=set(ground_truth.relevant_chunk_ids),
        accepted_answers=question.accepted_answers,
    )
    source_ids = [source["source_id"] for source in (source.model_dump(mode="json") for source in response.sources)]
    metric_answer = extract_metric_answer(response.answer, source_ids)
    raw_anls = _anls(response.answer, question.accepted_answers)
    raw_exact = _exact(response.answer, question.accepted_answers)
    metric_anls = _anls(metric_answer, question.accepted_answers)
    metric_exact = _exact(metric_answer, question.accepted_answers)
    status = "ABSTAINED" if not response.sources or response.answer == NO_RESULTS_ANSWER else "ANSWERED"
    review = classify_review_case(
        status=status,
        raw_response=response.answer,
        metric_answer=metric_answer,
        raw_anls=raw_anls,
        raw_exact_match=raw_exact,
        metric_anls=metric_anls,
        metric_exact_match=metric_exact,
        citations=source_metrics["labels"],
        citations_valid=response.citations_valid,
        answer_supported_by_cited_evidence=source_metrics["answer_supported"],
    )
    return E4_1QuestionRecord(
        question_key=question.question_key,
        evaluation_id=question.evaluation_id,
        question_id=question.question_id,
        question=question.question,
        accepted_answers=question.accepted_answers,
        ground_truth_status=ground_truth.status,
        target_document_id=ground_truth.target_document_id,
        relevant_chunk_ids=ground_truth.relevant_chunk_ids,
        configuration=configuration,
        status=status,  # type: ignore[arg-type]
        reason_code="RETRIEVAL_NO_RELEVANT_EVIDENCE" if status == "ABSTAINED" else None,
        raw_response=response.answer,
        metric_answer=metric_answer,
        model=response.model,
        citations=response.citations,
        citations_valid=response.citations_valid,
        sources=[source.model_dump(mode="json") for source in response.sources],
        raw_anls=raw_anls,
        raw_exact_match=raw_exact,
        metric_anls=metric_anls,
        metric_exact_match=metric_exact,
        citation_labels_emitted=len(source_metrics["labels"]),
        citation_labels_valid=source_metrics["valid_labels"],
        citation_document_hit=source_metrics["document_hit"],
        all_citations_true_document=source_metrics["all_true_document"],
        gold_evidence_citation_hit=source_metrics["gold_hit"],
        gold_evidence_citation_count=source_metrics["gold_count"],
        gold_evidence_citation_precision=source_metrics["gold_precision"],
        answer_supported_by_cited_evidence=source_metrics["answer_supported"],
        retrieval_time_ms=response.retrieval_time_ms,
        reranking_time_ms=response.rerank_time_ms,
        generation_time_ms=response.generation_time_ms,
        total_pipeline_time_ms=response.total_time_ms,
        review_category=review,
    )


def _anls(answer: str | None, accepted_answers: list[str]) -> float:
    """Call the existing E4 ANLS implementation without changing its rules."""

    from evaluation.e4.metrics import anls_score

    return anls_score(answer, accepted_answers)


def _exact(answer: str | None, accepted_answers: list[str]) -> bool:
    """Call the existing E4 normalized exact-match implementation."""

    from evaluation.e4.metrics import normalized_exact_match

    return normalized_exact_match(answer, accepted_answers)


def _base_record(
    question: E3Question,
    ground_truth: E3QuestionGroundTruth,
    configuration: str,
    *,
    status: str,
    reason_code: str | None = None,
    elapsed_ms: float | None = None,
    error: str | None = None,
) -> E4_1QuestionRecord:
    """Create a bounded failure or controlled-state record."""

    review = "GROUNDING_REJECTED" if status == "GROUNDING_REJECTED" else "UNCLASSIFIED"
    return E4_1QuestionRecord(
        question_key=question.question_key,
        evaluation_id=question.evaluation_id,
        question_id=question.question_id,
        question=question.question,
        accepted_answers=question.accepted_answers,
        ground_truth_status=ground_truth.status,
        target_document_id=ground_truth.target_document_id,
        relevant_chunk_ids=ground_truth.relevant_chunk_ids,
        configuration=configuration,
        status=status,  # type: ignore[arg-type]
        reason_code=reason_code,
        total_pipeline_time_ms=elapsed_ms,
        error=error,
        review_category=review,  # type: ignore[arg-type]
    )


def _ask_request(question: E3Question, indexed_document_ids: list[UUID], top_k: int, rerank: bool) -> AskRequest:
    """Build the same bounded hybrid request for each diagnostic configuration."""

    return AskRequest(
        question=question.question,
        top_k=top_k,
        search_mode=SearchMode.HYBRID,
        rerank=rerank,
        filters=SearchFilters(document_ids=indexed_document_ids),
    )


async def _run_configuration(
    *,
    configuration: str,
    rerank: bool,
    questions: list[E3Question],
    ground_truth: dict[str, E3QuestionGroundTruth],
    indexed_document_ids: list[UUID],
    rag_service: RAGService,
    top_k: int,
) -> tuple[list[E4_1QuestionRecord], dict[str, Any]]:
    """Warm one configuration, then run the identical selected question set."""

    warmup: dict[str, Any] = {
        "configuration": configuration,
        "excluded_from_metrics": True,
        "warmup_kind": "cold_model_warmup",
    }
    warmup_question = questions[0] if questions else None
    if warmup_question is not None and indexed_document_ids:
        started = time.perf_counter()
        try:
            response = await _benchmark_ask(
                rag_service,
                _ask_request(warmup_question, indexed_document_ids, top_k, rerank),
            )
            warmup.update(
                {
                    "status": "completed",
                    "question_key": warmup_question.question_key,
                    "wall_clock_time_ms": round((time.perf_counter() - started) * 1000, 3),
                    "retrieval_time_ms": response.retrieval_time_ms,
                    "reranking_time_ms": response.rerank_time_ms,
                    "generation_time_ms": response.generation_time_ms,
                    "total_pipeline_time_ms": response.total_time_ms,
                }
            )
        except asyncio.TimeoutError:
            warmup.update(
                {
                    "status": "failed",
                    "question_key": warmup_question.question_key,
                    "wall_clock_time_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error": "BENCHMARK_WARMUP_SAFETY_TIMEOUT",
                }
            )
        except Exception as exc:
            warmup.update(
                {
                    "status": "failed",
                    "question_key": warmup_question.question_key,
                    "wall_clock_time_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error": _failure_reason(exc),
                }
            )

    # A diagnostic that cannot complete its model warm-up is not reliable
    # enough to spend the full question budget.  Preserve the failed warm-up
    # in metadata and let the caller write an honest non-measured result.
    if warmup.get("status") != "completed":
        return [], warmup

    records: list[E4_1QuestionRecord] = []
    for question in questions:
        gt = ground_truth[question.question_key]
        started = time.perf_counter()
        try:
            response = await _benchmark_ask(
                rag_service,
                _ask_request(question, indexed_document_ids, top_k, rerank),
            )
            records.append(_response_record(question, gt, configuration, response))
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            status = "GROUNDING_REJECTED" if isinstance(exc, RAGServiceError) else "GENERATION_FAILED"
            records.append(
                _base_record(
                    question,
                    gt,
                    configuration,
                    status=status,
                    reason_code=_failure_reason(exc),
                    elapsed_ms=elapsed,
                    error=str(exc),
                )
            )
    return records, warmup


async def _benchmark_ask(rag_service: RAGService, request: AskRequest) -> AskResponse:
    """Apply only a diagnostic safety ceiling around a provider call."""

    timeout = getattr(rag_service.ollama_client, "timeout_seconds", None)
    if timeout is None:
        return await rag_service.ask(request)
    # The HTTP client remains responsible for the configured provider timeout.
    # This extra benchmark-only ceiling prevents a host transport stall from
    # making an E4.1 run unbounded; it is never used by FastAPI production code.
    return await asyncio.wait_for(rag_service.ask(request), timeout=float(timeout) + 30.0)


def _safe_database_target(database_url: str | None) -> dict[str, Any]:
    """Record non-secret database location metadata."""

    if not database_url:
        return {"configured": False}
    parsed = urlsplit(database_url.replace("postgresql+psycopg", "postgresql", 1))
    return {
        "configured": True,
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": parsed.port,
        "database": parsed.path.lstrip("/") or None,
    }


def _load_production_baseline(
    directory: Path | None,
    question_keys: set[str],
) -> dict[str, Any]:
    """Project completion facts for the identical subset from the preserved E4 run."""

    if directory is None:
        return {"available": False, "reason": "No preserved E4 baseline directory supplied."}
    path = directory / "answers.jsonl"
    if not path.is_file():
        return {"available": False, "reason": f"Baseline answers file not found: {path}"}
    selected: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("question_key") in question_keys:
            selected.setdefault(payload["configuration"], []).append(payload)
    projected: dict[str, Any] = {"available": True, "directory": str(directory), "configurations": {}}
    for configuration, records in selected.items():
        answered = sum(record.get("status") == "ANSWERED" for record in records)
        projected["configurations"][configuration] = {
            "questions": len(records),
            "answered": answered,
            "completion_rate": answered / len(records) if records else None,
            "timeouts": sum(record.get("reason_code") == "OLLAMA_TIMEOUT" for record in records),
            "failure_counts": {
                key: sum(record.get("reason_code") == key for record in records)
                for key in sorted({record.get("reason_code") for record in records if record.get("reason_code")})
            },
        }
    return projected


def _reuse_existing_corpus(
    records: list[EvaluationDocument],
    repository: DocumentRepository,
) -> tuple[list[CorpusMapping], dict[str, list[IndexedChunk]], list[str]]:
    """Reuse exact manifest-named ready documents without mutating the database."""

    existing, _total = repository.list_documents(1, 500, None)
    by_filename = {document.original_filename: document for document in existing}
    mappings: list[CorpusMapping] = []
    chunks_by_evaluation_id: dict[str, list[IndexedChunk]] = {}
    failures: list[str] = []
    for record in records:
        filename = Path(record.local_pdf_path).name
        mapping = CorpusMapping(
            evaluation_id=record.evaluation_id,
            source_record_id=record.source_record_id,
            source_document_id=record.source_document_id,
            local_pdf_path=record.local_pdf_path,
            original_filename=filename,
        )
        document = by_filename.get(filename)
        if document is None or document.status is not DocumentStatus.READY:
            reason = "REUSED_CORPUS_DOCUMENT_NOT_READY"
            mapping = mapping.model_copy(update={"processing_status": "failed", "failure_reason": reason})
            failures.append(f"{record.evaluation_id}: {reason}")
        else:
            _document, chunks = repository.get_document_with_chunks(document.id)
            projected = _chunk_projection(document.id, chunks)
            if not projected:
                reason = "REUSED_CORPUS_DOCUMENT_NOT_INDEXED"
                mapping = mapping.model_copy(update={"processing_status": "failed", "failure_reason": reason})
                failures.append(f"{record.evaluation_id}: {reason}")
            else:
                mapping = mapping.model_copy(
                    update={
                        "document_id": document.id,
                        "stored_filename": document.stored_filename,
                        "checksum_sha256": document.checksum_sha256,
                        "indexed_chunk_ids": [chunk.chunk_id for chunk in projected],
                        "indexed_chunk_count": len(projected),
                        "processing_status": "indexed",
                    }
                )
                chunks_by_evaluation_id[record.evaluation_id] = projected
        mappings.append(mapping)
    return mappings, chunks_by_evaluation_id, failures


def _review_case(record: E4_1QuestionRecord) -> dict[str, Any]:
    """Create the compact human-review projection without changing scores."""

    return {
        "configuration": record.configuration,
        "question_key": record.question_key,
        "question": record.question,
        "accepted_answers": record.accepted_answers,
        "raw_response": record.raw_response,
        "metric_answer": record.metric_answer,
        "citations": record.citations,
        "retrieved_source_labels": [source.get("source_id") for source in record.sources],
        "raw_anls": record.raw_anls,
        "raw_exact_match": record.raw_exact_match,
        "metric_anls": record.metric_anls,
        "metric_exact_match": record.metric_exact_match,
        "citation_labels_valid": record.citation_labels_valid,
        "answer_supported_by_cited_evidence": record.answer_supported_by_cited_evidence,
        "review_category": record.review_category,
        "status": record.status,
        "failure": record.reason_code,
        "error": record.error,
    }


def _write_artifacts(
    output_directory: Path,
    *,
    summary: E4_1RunSummary,
    records: list[E4_1QuestionRecord],
    mappings: list[CorpusMapping],
    metadata: dict[str, Any],
) -> None:
    """Write the complete E4.1 diagnostic artifact set."""

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    serialized = [record.model_dump(mode="json") for record in records]
    lines = "".join(json.dumps(record, sort_keys=True) + "\n" for record in serialized)
    (output_directory / "per_question.jsonl").write_text(lines, encoding="utf-8")
    (output_directory / "answers.jsonl").write_text(lines, encoding="utf-8")
    review_lines = "".join(json.dumps(_review_case(record), sort_keys=True) + "\n" for record in records)
    (output_directory / "review_cases.jsonl").write_text(review_lines, encoding="utf-8")
    with (output_directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["configuration", "metric", "value"])
        writer.writeheader()
        for row in metrics_csv_rows(summary.configurations):
            writer.writerow(row)
    (output_directory / "corpus_mapping.jsonl").write_text(
        "".join(json.dumps(mapping.model_dump(mode="json"), sort_keys=True) + "\n" for mapping in mappings),
        encoding="utf-8",
    )
    (output_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "report.md").write_text(_report_markdown(summary, metadata), encoding="utf-8")


def _report_markdown(summary: E4_1RunSummary, metadata: dict[str, Any]) -> str:
    """Render the E4.1 report with diagnostic labels and limitations."""

    lines = [
        "# DocuIntel Evaluation E4.1",
        "",
        f"- Status: `{summary.status}`",
        f"- Diagnostic label: **{metadata.get('diagnostic_label')}**",
        f"- Run ID: `{metadata.get('run_id')}`",
        f"- Model: `{metadata.get('models', {}).get('ollama_model')}`",
        f"- Production Ollama HTTP timeout: `{metadata.get('timeout_path', {}).get('production_timeout_seconds')}` seconds",
        f"- Diagnostic timeout: `{metadata.get('timeout_path', {}).get('effective_timeout_seconds')}` seconds",
        "",
        "## Timeout path",
        "",
        "The production timeout is the `httpx.AsyncClient` timeout used by `OllamaClient` for `/api/generate`. "
        "There is no retry, `asyncio.wait_for`, separate Ollama generation timeout, or benchmark runner timeout.",
        "The diagnostic changes only the copied benchmark Settings timeout; model, temperature, prompt, retrieval, reranking, and context remain unchanged.",
        "",
        "## Corpus and subset",
        "",
        f"- Documents prepared/indexed: `{summary.documents_prepared}` / `{summary.documents_indexed}`",
        f"- Questions available/scorable/selected: `{summary.questions_available}` / `{summary.questions_scorable_available}` / `{summary.questions_selected}`",
        "- The selected subset is the first deterministic scorable questions after the existing E3 ordering and ground-truth classification.",
        "",
        "## Configuration results",
        "",
        "| Configuration | Answered | Selected | Completion | Raw ANLS | Metric ANLS | Raw EM | Metric EM | Mean total ms | Median total ms | Mean generation ms | Median generation ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary.configurations.items():
        total = item.latency.get("total_pipeline_time_ms", {})
        generation = item.latency.get("generation_time_ms", {})
        lines.append(
            f"| {name} | {item.questions_answered} | {item.questions_selected if hasattr(item, 'questions_selected') else item.questions_attempted} | "
            f"{item.completion_rate_scorable} | {item.raw_scorable.anls} | {item.metric_scorable.anls} | "
            f"{item.raw_scorable.exact_match} | {item.metric_scorable.exact_match} | "
            f"{total.get('mean_ms')} | {total.get('median_ms')} | {generation.get('mean_ms')} | {generation.get('median_ms')} |"
        )
    lines.extend(
        [
            "",
            "## Answer-format interpretation",
            "",
            "`raw_response` is the unchanged production answer. `metric_answer` removes only provided source labels and narrow Markdown presentation markers, then collapses whitespace. It does not remove explanatory prose, choose substrings, use gold answers, fuzzy-match, or call another model.",
            "If a production answer contains unrestricted explanation, deterministic short-answer extraction is not defensible; the metric view therefore remains conservative.",
            "",
            "## Review buckets",
            "",
        ]
    )
    for name, item in summary.configurations.items():
        lines.append(f"- `{name}`: {item.review_categories}")
    lines.extend(
        [
            "",
            "## Warm-up",
            "",
            "The first request per configuration is an explicit cold/model warm-up and is excluded from measured quality and latency aggregates. Subsequent requests are the warm benchmark loop.",
            "",
            "## Limitations",
            "",
            "- The diagnostic subset is intentionally small and is not a replacement for the official E4 production baseline.",
            "- CPU-mode Ollama latency and small-sample answer quality should not be generalized to other hardware or larger datasets.",
            "",
        ]
    )
    return "\n".join(lines)


def write_controlled_state(
    output_directory: Path,
    *,
    split: str,
    run_id: str,
    message: str,
    manifest_path: Path,
    document_limit: int,
    question_limit: int,
) -> E4_1RunSummary:
    """Write a controlled prerequisite failure without inventing metrics."""

    summary = E4_1RunSummary(
        status="CONTROLLED_FAILURE",
        split=split,
        documents_requested=document_limit,
        documents_prepared=0,
        documents_indexed=0,
        questions_available=0,
        questions_scorable_available=0,
        questions_selected=0,
        failures=[message],
    )
    metadata = {
        "schema_version": "e4_1.v1",
        "run_id": run_id,
        "run_timestamp": now_utc_iso(),
        "manifest_path": str(manifest_path),
        "output_directory": str(output_directory),
        "document_limit": document_limit,
        "question_limit": question_limit,
        "status": summary.status,
        "diagnostic_label": "NON-PRODUCTION TIMEOUT DIAGNOSTIC",
        "message": message,
    }
    _write_artifacts(output_directory, summary=summary, records=[], mappings=[], metadata=metadata)
    return summary


def write_blocked_diagnostic_state(
    output_directory: Path,
    *,
    split: str,
    run_id: str,
    manifest_path: Path,
    baseline_directory: Path,
    document_limit: int,
    question_limit: int,
    scorable_question_limit: int,
    production_timeout_seconds: float,
    diagnostic_timeout_seconds: float,
    evidence: dict[str, Any],
) -> E4_1RunSummary:
    """Write a blocked diagnostic without turning unattempted questions into failures."""

    validation = validate_manifest(manifest_path)
    records = sorted(validation.records, key=lambda item: item.evaluation_id)[:document_limit]
    questions = build_questions(records, question_limit)
    scorable_keys: set[str] = set()
    baseline_answers = baseline_directory / "answers.jsonl"
    if baseline_answers.is_file():
        for line in baseline_answers.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if payload.get("configuration") == "hybrid" and payload.get("ground_truth_status") == "SCORABLE":
                    scorable_keys.add(payload["question_key"])
    selected_keys = {
        question.question_key
        for question in questions
        if question.question_key in scorable_keys
    }
    selected_keys = set(list(dict.fromkeys(
        question.question_key for question in questions if question.question_key in selected_keys
    ))[:scorable_question_limit])
    production_baseline = _load_production_baseline(baseline_directory, selected_keys)
    baseline_summary_path = baseline_directory / "summary.json"
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8")) if baseline_summary_path.is_file() else {}
    summary = E4_1RunSummary(
        status="CONTROLLED_FAILURE",
        split=split,
        documents_requested=document_limit,
        documents_prepared=baseline_summary.get("documents_prepared", len(records)),
        documents_indexed=baseline_summary.get("documents_indexed", 0),
        questions_available=len(questions),
        questions_scorable_available=len(scorable_keys),
        questions_selected=len(selected_keys),
        production_timeout_baseline=production_baseline,
        warmups={"hybrid_reranked": evidence.get("rag_warmup", {})},
        failures=[evidence.get("message", "The real E4.1 diagnostic was blocked before measured questions.")],
    )
    metadata = {
        "schema_version": "e4_1.v1",
        "run_id": run_id,
        "run_timestamp": now_utc_iso(),
        "dataset": "docvqa",
        "split": split,
        "manifest_path": str(manifest_path),
        "output_directory": str(output_directory),
        "document_limit": document_limit,
        "question_limit": question_limit,
        "scorable_question_limit": scorable_question_limit,
        "questions_selected": sorted(selected_keys),
        "diagnostic_label": "NON-PRODUCTION TIMEOUT DIAGNOSTIC (BLOCKED BEFORE MEASURED LOOP)",
        "models": {"ollama_model": "llama3.2:3b", "ollama_base_url": "http://127.0.0.1:11434"},
        "timeout_path": {
            "responsible_layer": "OllamaClient -> httpx.AsyncClient -> /api/generate",
            "production_timeout_seconds": production_timeout_seconds,
            "effective_timeout_seconds": diagnostic_timeout_seconds,
            "override_applied": True,
            "retry_count": 0,
            "separate_runner_timeout": False,
            "benchmark_warmup_safety_timeout_seconds": diagnostic_timeout_seconds + 30,
        },
        "diagnostic_evidence": evidence,
        "production_baseline_directory": str(baseline_directory),
        "completion_rate_scope": "not measurable; no selected question entered the measured loop",
        "answer_metrics_scope": "not measurable; no generated answer was received in the diagnostic run",
        "command": "write_blocked_diagnostic_state",
        "platform": platform.platform(),
        "python_version": sys.version,
    }
    _write_artifacts(output_directory, summary=summary, records=[], mappings=[], metadata=metadata)
    return summary


async def run_e4_1(
    options: E4_1RunOptions,
    *,
    settings: Settings | None = None,
    run_id: str,
) -> E4_1RunSummary:
    """Run E4.1 over a deterministic scorable subset using existing production services."""

    if options.output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing E4.1 run directory: {options.output_directory}")
    if options.top_k <= 0 or options.scorable_question_limit <= 0:
        raise ValueError("top_k and scorable_question_limit must be greater than zero")
    options.output_directory.mkdir(parents=True, exist_ok=False)
    try:
        try:
            validation = validate_manifest(options.manifest_path)
        except ManifestValidationError as exc:
            return write_controlled_state(
                options.output_directory,
                split=options.split,
                run_id=run_id,
                message=str(exc),
                manifest_path=options.manifest_path,
                document_limit=options.document_limit,
                question_limit=options.question_limit,
            )
        if validation.dataset != "docvqa" or validation.split != options.split:
            return write_controlled_state(
                options.output_directory,
                split=options.split,
                run_id=run_id,
                message=f"Manifest is {validation.dataset}/{validation.split}, expected docvqa/{options.split}.",
                manifest_path=options.manifest_path,
                document_limit=options.document_limit,
                question_limit=options.question_limit,
            )
        records = sorted(validation.records, key=lambda item: item.evaluation_id)[: options.document_limit]
        all_questions = build_questions(records, options.question_limit)
        if not all_questions:
            return write_controlled_state(
                options.output_directory,
                split=options.split,
                run_id=run_id,
                message="Prepared DocVQA documents contain no question/answer pairs.",
                manifest_path=options.manifest_path,
                document_limit=options.document_limit,
                question_limit=options.question_limit,
            )
        effective_settings = settings or Settings()
        if options.database_url_override:
            effective_settings = effective_settings.model_copy(update={"database_url": options.database_url_override})
        database = create_database(effective_settings)
        if database is None:
            return write_controlled_state(
                options.output_directory,
                split=options.split,
                run_id=run_id,
                message="E4.1 requires PostgreSQL/pgvector; provide DATABASE_URL or --database-url.",
                manifest_path=options.manifest_path,
                document_limit=options.document_limit,
                question_limit=options.question_limit,
            )
        repository = DocumentRepository(database)
        storage_directory = options.output_directory / "ingest_staging"
        ingestion_service = PDFIngestionService(settings=effective_settings, storage_directory=storage_directory)
        management_service = DocumentManagementService(
            ingestion_service=ingestion_service,
            repository=repository,
            storage_directory=storage_directory,
            persistence_required=True,
        )
        embedding_service = EmbeddingService(settings=effective_settings)
        indexing_service = DocumentIndexingService(
            repository=repository,
            chunker=StructureAwareChunker(settings=effective_settings),
            embedding_service=embedding_service,
        )
        evaluation_scope_limit = max(effective_settings.rag_max_selected_documents, options.document_limit)
        search_settings = effective_settings.model_copy(update={"rag_max_selected_documents": evaluation_scope_limit})
        benchmark_settings = effective_benchmark_settings(effective_settings, options.generation_timeout_seconds).model_copy(
            update={"rag_max_selected_documents": evaluation_scope_limit}
        )
        search_service = SearchService(
            repository=repository,
            embedding_service=embedding_service,
            settings=search_settings,
            reranker=CrossEncoderReranker(settings=effective_settings),
        )
        mappings: list[CorpusMapping] = []
        mapping_by_evaluation_id: dict[str, CorpusMapping] = {}
        chunks_by_evaluation_id: dict[str, list[IndexedChunk]] = {}
        failures: list[str] = []
        try:
            if options.reuse_indexed_corpus:
                mappings, chunks_by_evaluation_id, failures = _reuse_existing_corpus(records, repository)
            else:
                for record in records:
                    mapping = CorpusMapping(
                        evaluation_id=record.evaluation_id,
                        source_record_id=record.source_record_id,
                        source_document_id=record.source_document_id,
                        local_pdf_path=record.local_pdf_path,
                        original_filename=Path(record.local_pdf_path).name,
                    )
                    source_path = resolve_local_path(options.manifest_path, record.local_pdf_path)
                    upload = LocalPDFUpload(source_path)
                    response: DocumentIngestionResponse | None = None
                    try:
                        response = await management_service.ingest(upload)
                        mapping = mapping.model_copy(
                            update={
                                "document_id": response.document_id,
                                "stored_filename": response.stored_filename,
                                "checksum_sha256": response.checksum_sha256,
                            }
                        )
                        if response.status is not DocumentStatus.READY:
                            mapping = mapping.model_copy(update={"processing_status": "failed", "failure_reason": "DOCUMENT_PROCESSING_FAILED"})
                            failures.append(f"{record.evaluation_id}: ingestion status {response.status.value}")
                        else:
                            index_result = indexing_service.index_document(response.document_id)
                            _document, chunks = repository.get_document_with_chunks(response.document_id)
                            projected = _chunk_projection(response.document_id, chunks)
                            mapping = mapping.model_copy(
                                update={
                                    "processing_status": "indexed",
                                    "indexed_chunk_ids": [chunk.chunk_id for chunk in projected],
                                    "indexed_chunk_count": index_result.chunks_created,
                                }
                            )
                            chunks_by_evaluation_id[record.evaluation_id] = projected
                    except Exception as exc:
                        message = exc.public_message if isinstance(exc, DocumentIngestionError) else str(exc)
                        mapping = mapping.model_copy(update={"processing_status": "failed", "failure_reason": _failure_reason(exc)})
                        failures.append(f"{record.evaluation_id}: {message}")
                    finally:
                        upload.close()
                        if response is not None:
                            (storage_directory / response.stored_filename).unlink(missing_ok=True)
                    mappings.append(mapping)
            mapping_by_evaluation_id = {mapping.evaluation_id: mapping for mapping in mappings}

            indexed_document_ids = [
                mapping.document_id
                for mapping in mappings
                if mapping.processing_status == "indexed" and mapping.document_id is not None
            ]
            rag_service = RAGService(
                search_service=search_service,
                ollama_client=OllamaClient(settings=benchmark_settings),
                settings=benchmark_settings,
            )
            ground_truth: dict[str, E3QuestionGroundTruth] = {}
            for question in all_questions:
                mapping = mapping_by_evaluation_id.get(question.evaluation_id)
                ground_truth[question.question_key] = build_question_ground_truth(
                    question,
                    target_document_id=mapping.document_id if mapping else None,
                    chunks=chunks_by_evaluation_id.get(question.evaluation_id),
                    document_indexed=bool(mapping and mapping.processing_status == "indexed"),
                )
            scorable_questions = [
                question
                for question in all_questions
                if ground_truth[question.question_key].status == "SCORABLE"
            ]
            selected_questions = scorable_questions[: options.scorable_question_limit]
            if not selected_questions:
                return write_controlled_state(
                    options.output_directory,
                    split=options.split,
                    run_id=run_id,
                    message="The selected bounded corpus contains no scorable questions.",
                    manifest_path=options.manifest_path,
                    document_limit=options.document_limit,
                    question_limit=options.question_limit,
                )

            configuration_records: dict[str, list[E4_1QuestionRecord]] = {}
            warmups: dict[str, Any] = {}
            for configuration, rerank in CONFIGURATIONS:
                logger.info("E4.1 configuration started configuration=%s", configuration)
                result, warmup = await _run_configuration(
                    configuration=configuration,
                    rerank=rerank,
                    questions=selected_questions,
                    ground_truth=ground_truth,
                    indexed_document_ids=indexed_document_ids,
                    rag_service=rag_service,
                    top_k=options.top_k,
                )
                configuration_records[configuration] = result
                warmups[configuration] = warmup
                if warmup.get("status") != "completed":
                    failures.append(
                        f"{configuration}: measured question loop not started because warm-up failed ({warmup.get('error', 'unknown')})."
                    )
                    break

            summaries: dict[str, E4_1ConfigurationSummary] = {
                configuration: summarize_configuration(
                    records_for_config,
                    configuration=configuration,
                    search_mode=SearchMode.HYBRID.value,
                    rerank=configuration == "hybrid_reranked",
                    top_k=options.top_k,
                )
                for configuration, records_for_config in configuration_records.items()
            }
            selected_keys = {question.question_key for question in selected_questions}
            production_baseline = _load_production_baseline(
                options.production_baseline_directory,
                selected_keys,
            )
            summary = E4_1RunSummary(
                status="completed",
                split=options.split,
                documents_requested=options.document_limit,
                documents_prepared=len(records),
                documents_indexed=sum(mapping.processing_status == "indexed" for mapping in mappings),
                questions_available=len(all_questions),
                questions_scorable_available=len(scorable_questions),
                questions_selected=len(selected_questions),
                configurations=summaries,
                production_timeout_baseline=production_baseline,
                warmups=warmups,
                failures=failures,
            )
            effective_timeout = benchmark_settings.ollama_timeout_seconds
            metadata = {
                "schema_version": "e4_1.v1",
                "run_id": run_id,
                "run_timestamp": now_utc_iso(),
                "dataset": "docvqa",
                "split": options.split,
                "manifest_path": str(options.manifest_path),
                "output_directory": str(options.output_directory),
                "document_limit": options.document_limit,
                "question_limit": options.question_limit,
                "scorable_question_limit": options.scorable_question_limit,
                "questions_selected": [question.question_key for question in selected_questions],
                "top_k": options.top_k,
                "keep_indexed": options.keep_indexed,
                "reuse_indexed_corpus": options.reuse_indexed_corpus,
                "database": _safe_database_target(effective_settings.resolved_database_url),
                "models": {
                    "embedding_model": effective_settings.embedding_model,
                    "embedding_dimension": effective_settings.embedding_dimension,
                    "reranker_model": effective_settings.reranker_model,
                    "ollama_model": effective_settings.ollama_model,
                    "ollama_base_url": effective_settings.ollama_base_url,
                },
                "timeout_path": {
                    "responsible_layer": "OllamaClient -> httpx.AsyncClient -> /api/generate",
                    "production_timeout_seconds": effective_settings.ollama_timeout_seconds,
                    "effective_timeout_seconds": effective_timeout,
                    "override_applied": options.generation_timeout_seconds is not None,
                    "retry_count": 0,
                    "separate_runner_timeout": False,
                    "benchmark_warmup_safety_timeout_seconds": effective_timeout + 30.0,
                    "benchmark_safety_timeout_is_production_behavior": False,
                    "separate_generation_timeout": False,
                },
                "diagnostic_label": "NON-PRODUCTION TIMEOUT DIAGNOSTIC" if options.generation_timeout_seconds is not None else "PRODUCTION-CONFIGURATION CONTROL",
                "prompt_and_generation": {
                    "ollama_temperature": effective_settings.ollama_temperature,
                    "stream": False,
                    "num_predict": None,
                    "rag_max_context_chars": effective_settings.rag_max_context_chars,
                    "answer_path": "RAGService.ask -> SearchService -> RAGContextBuilder -> existing grounded prompt -> OllamaClient",
                },
                "answer_normalization_rules": ANSWER_NORMALIZATION_RULES,
                "anls_definition": ANLS_REFERENCE,
                "exact_match_definition": EM_NORMALIZATION,
                "metric_answer_definition": "Raw response with only provided [S#] labels and narrow Markdown presentation markers removed; whitespace collapsed; no semantic rewriting, gold-guided extraction, fuzzy matching, or LLM judge.",
                "warmups": warmups,
                "warmup_definition": "One explicit cold/model warm-up request per configuration, excluded from measured quality and latency; measured requests follow the warm-up.",
                "production_baseline_directory": str(options.production_baseline_directory) if options.production_baseline_directory else None,
                "production_path": "DocumentManagementService -> PDFIngestionService -> DocumentIndexingService -> existing SearchService -> RAGService -> existing OllamaClient",
                "command": " ".join(sys.argv),
                "platform": platform.platform(),
                "python_version": sys.version,
            }
            if not options.keep_indexed and not options.reuse_indexed_corpus:
                cleaned = cleanup_run_documents(repository, mappings, storage_directory)
                cleaned_by_id = {mapping.evaluation_id: mapping for mapping in cleaned}
                mappings = [cleaned_by_id.get(mapping.evaluation_id, mapping) for mapping in mappings]
            _write_artifacts(
                options.output_directory,
                summary=summary,
                records=[record for config in CONFIGURATIONS for record in configuration_records[config[0]]],
                mappings=mappings,
                metadata=metadata,
            )
            return summary
        finally:
            database.engine.dispose()
    except Exception:
        raise
