"""Controlled real-corpus E4 runner over the production RAG service."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
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
from evaluation.e3.metrics import ANSWER_NORMALIZATION_RULES, build_question_ground_truth, normalize_answer
from evaluation.e3.models import CorpusMapping, E3Question, E3QuestionGroundTruth, IndexedChunk
from evaluation.e3.runner import LocalPDFUpload, build_questions, _chunk_projection
from evaluation.e4.metrics import (
    ANLS_REFERENCE,
    EM_NORMALIZATION,
    anls_score,
    compare_configurations,
    extract_citation_labels,
    normalized_exact_match,
    summarize_configuration,
)
from evaluation.e4.models import E4ConfigurationSummary, E4QuestionRecord, E4RunSummary
from evaluation.manifests import now_utc_iso
from evaluation.schemas import EvaluationDocument
from evaluation.validation import ManifestValidationError, resolve_local_path, validate_manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class E4RunOptions:
    """Bounded real-corpus E4 runtime options."""

    manifest_path: Path
    output_directory: Path
    split: str
    document_limit: int
    question_limit: int
    top_k: int = 5
    keep_indexed: bool = False
    database_url_override: str | None = None


CONFIGURATIONS: tuple[tuple[str, bool], ...] = (
    ("hybrid", False),
    ("hybrid_reranked", True),
)


def _safe_database_target(database_url: str | None) -> dict[str, Any]:
    """Record only non-secret database location metadata."""

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


def _question_payload(question: E3Question, ground_truth: E3QuestionGroundTruth) -> dict[str, Any]:
    """Serialize the deterministic question and E3 ground truth once."""

    return {**question.model_dump(mode="json"), "ground_truth": ground_truth.model_dump(mode="json")}


def _failure_reason(exc: Exception) -> str:
    """Map controlled provider/application failures to stable artifact codes."""

    message = str(exc).lower()
    if isinstance(exc, OllamaServiceError):
        if "timed out" in message or "within" in message:
            return "OLLAMA_TIMEOUT"
        if "model" in message and "unavailable" in message:
            return "OLLAMA_MODEL_MISSING"
        if "unavailable" in message or "start ollama" in message:
            return "OLLAMA_UNAVAILABLE"
        return "OLLAMA_ERROR"
    if isinstance(exc, RAGServiceError):
        if "citation" in message:
            return "INVALID_CITATION"
        return "GROUNDING_REJECTED"
    if "search" in message or "retriev" in message:
        return "RETRIEVAL_ERROR"
    return type(exc).__name__.upper()


def _source_metrics(
    response: AskResponse,
    *,
    target_document_id: UUID | None,
    relevant_chunk_ids: set[UUID],
    accepted_answers: list[str],
) -> dict[str, Any]:
    """Calculate citation/evidence facts from the actual production response."""

    labels = extract_citation_labels(response.answer)
    sources = {source.source_id: source for source in response.sources}
    valid_sources = [sources[label] for label in labels if label in sources]
    valid_labels = len(valid_sources)
    document_hit = bool(
        target_document_id is not None
        and any(source.document_id == target_document_id for source in valid_sources)
    )
    all_true_document = bool(valid_sources) and target_document_id is not None and all(
        source.document_id == target_document_id for source in valid_sources
    )
    cited_gold = sum(source.chunk_id in relevant_chunk_ids for source in valid_sources)
    gold_hit = cited_gold > 0
    answer_supported = any(
        normalize_answer(answer) in normalize_answer(source.excerpt)
        for answer in accepted_answers
        if normalize_answer(answer)
        for source in valid_sources
    )
    return {
        "labels": labels,
        "valid_labels": valid_labels,
        "document_hit": document_hit,
        "all_true_document": all_true_document if labels else None,
        "gold_hit": gold_hit,
        "gold_count": cited_gold,
        "gold_precision": cited_gold / valid_labels if valid_labels else None,
        "answer_supported": answer_supported if valid_sources else None,
    }


def _base_record(
    question: E3Question,
    ground_truth: E3QuestionGroundTruth,
    configuration: str,
    *,
    status: str,
    reason_code: str | None = None,
) -> E4QuestionRecord:
    """Create a bounded zero-answer record for controlled states."""

    return E4QuestionRecord(
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
    )


def _response_record(
    question: E3Question,
    ground_truth: E3QuestionGroundTruth,
    configuration: str,
    response: AskResponse,
) -> E4QuestionRecord:
    """Project a real production AskResponse into deterministic E4 facts."""

    source_metrics = _source_metrics(
        response,
        target_document_id=ground_truth.target_document_id,
        relevant_chunk_ids=set(ground_truth.relevant_chunk_ids),
        accepted_answers=question.accepted_answers,
    )
    is_abstention = not response.sources or response.answer == NO_RESULTS_ANSWER
    status = "ABSTAINED" if is_abstention else "ANSWERED"
    reason = "RETRIEVAL_NO_RELEVANT_EVIDENCE" if is_abstention else None
    return E4QuestionRecord(
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
        reason_code=reason,
        answer=response.answer,
        model=response.model,
        citations=response.citations,
        citations_valid=response.citations_valid,
        sources=[source.model_dump(mode="json") for source in response.sources],
        anls=anls_score(response.answer, question.accepted_answers),
        exact_match=normalized_exact_match(response.answer, question.accepted_answers),
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
        # Current production AskResponse does not expose separate context or grounding timers.
        context_build_time_ms=None,
        generation_time_ms=response.generation_time_ms,
        grounding_verification_time_ms=None,
        total_pipeline_time_ms=response.total_time_ms,
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
) -> tuple[list[E4QuestionRecord], dict[str, Any]]:
    """Warm and then run one configuration over the exact shared question set."""

    measured: list[E4QuestionRecord] = []
    warmup: dict[str, Any] = {"configuration": configuration, "excluded_from_metrics": True}
    warmup_question = next(
        (question for question in questions if ground_truth[question.question_key].status == "SCORABLE"),
        None,
    )
    if warmup_question is not None and indexed_document_ids:
        started = time.perf_counter()
        try:
            response = await rag_service.ask(
                AskRequest(
                    question=warmup_question.question,
                    top_k=top_k,
                    search_mode=SearchMode.HYBRID,
                    rerank=rerank,
                    filters=SearchFilters(document_ids=indexed_document_ids),
                )
            )
            warmup.update(
                {
                    "question_key": warmup_question.question_key,
                    "wall_clock_time_ms": round((time.perf_counter() - started) * 1000, 3),
                    "retrieval_time_ms": response.retrieval_time_ms,
                    "reranking_time_ms": response.rerank_time_ms,
                    "generation_time_ms": response.generation_time_ms,
                    "total_pipeline_time_ms": response.total_time_ms,
                }
            )
        except Exception as exc:  # warm-up failure is retained; measured queries still run
            warmup.update(
                {
                    "question_key": warmup_question.question_key,
                    "wall_clock_time_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error": _failure_reason(exc),
                }
            )

    for question in questions:
        gt = ground_truth[question.question_key]
        if gt.status == "DOCUMENT_PROCESSING_FAILED":
            measured.append(_base_record(question, gt, configuration, status="DOCUMENT_PROCESSING_FAILED", reason_code=gt.reason_code))
            continue
        if gt.status == "INVALID_GROUND_TRUTH":
            measured.append(_base_record(question, gt, configuration, status="INVALID_GROUND_TRUTH", reason_code=gt.reason_code))
            continue
        if gt.status == "ANSWER_NOT_INDEXED":
            # There is no supported evidence to send to the LLM; preserve this as an end-to-end zero.
            measured.append(_base_record(question, gt, configuration, status="ANSWER_NOT_INDEXED", reason_code=gt.reason_code))
            continue
        started = time.perf_counter()
        try:
            response = await rag_service.ask(
                AskRequest(
                    question=question.question,
                    top_k=top_k,
                    search_mode=SearchMode.HYBRID,
                    rerank=rerank,
                    filters=SearchFilters(document_ids=indexed_document_ids),
                )
            )
            measured.append(_response_record(question, gt, configuration, response))
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            status = "GROUNDING_REJECTED" if isinstance(exc, RAGServiceError) else "GENERATION_FAILED"
            record = _base_record(question, gt, configuration, status=status, reason_code=_failure_reason(exc))
            record.total_pipeline_time_ms = elapsed
            record.error = str(exc)
            measured.append(record)
    return measured, warmup


def _metrics_csv(configurations: dict[str, E4ConfigurationSummary]) -> list[dict[str, Any]]:
    """Flatten summary fields for simple spreadsheet inspection."""

    rows: list[dict[str, Any]] = []
    for name, summary in configurations.items():
        for scope, metrics in (("end_to_end", summary.end_to_end), ("scorable", summary.scorable)):
            rows.extend(
                [
                    {"configuration": name, "metric": f"{scope}_anls", "value": metrics.anls},
                    {"configuration": name, "metric": f"{scope}_exact_match", "value": metrics.exact_match},
                ]
            )
        for field, value in summary.citation.model_dump().items():
            rows.append({"configuration": name, "metric": field, "value": value})
        for stage, stats in summary.latency.items():
            for field, value in stats.model_dump().items():
                rows.append({"configuration": name, "metric": f"{stage}_{field}", "value": value})
    return rows


def _report_markdown(summary: E4RunSummary, metadata: dict[str, Any]) -> str:
    """Render an auditable E4 report without hiding failures or unavailable stages."""

    lines = [
        "# DocuIntel Evaluation E4",
        "",
        f"- Status: `{summary.status}`",
        f"- Run ID: `{metadata['run_id']}`",
        f"- Split: `{summary.split}`",
        f"- Ollama model: `{metadata.get('models', {}).get('ollama_model')}`",
        "",
    ]
    if summary.status != "completed":
        lines.extend([f"- Reason: `{summary.reason_code}`", "", metadata.get("message", "No metrics were produced.")])
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "## Corpus",
            "",
            f"- Documents prepared: `{summary.documents_prepared}`; indexed: `{summary.documents_indexed}`",
            f"- Questions attempted: `{summary.questions_attempted}`; scorable: `{summary.questions_scorable}`",
            f"- Answer-indexability rate: `{summary.answer_indexability_rate}`",
            "",
            "## Question Coverage",
            "",
            "| Configuration | Answered | Abstained | Failed | Coverage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary.configurations.items():
        lines.append(
            f"| {name} | {item.questions_answered} | {item.questions_abstained} | {item.questions_failed} | {item.answer_coverage_rate} |"
        )
    lines.extend(
        [
            "",
            "## End-to-End Answer Accuracy",
            "",
            "| Configuration | ANLS | Exact Match | Scorable ANLS | Scorable Exact Match | Gold Evidence Citation | Median Total ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary.configurations.items():
        lines.append(
            f"| {name} | {item.end_to_end.anls} | {item.end_to_end.exact_match} | "
            f"{item.scorable.anls} | {item.scorable.exact_match} | "
            f"{item.citation.gold_evidence_citation_hit_rate} | "
            f"{item.latency['total_pipeline_time_ms'].median_ms} |"
        )
    lines.extend(
        [
            "",
            "## Citation Quality and Evidence Grounding",
            "",
            "Metrics are deterministic source-label/document/chunk checks; no LLM judge is used.",
            "",
            "| Configuration | Presence | Reference validity | Document hit | All citations true document | Gold evidence hit | Gold precision | Answer supported |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary.configurations.items():
        citation = item.citation
        lines.append(
            f"| {name} | {citation.citation_presence_rate} | {citation.citation_reference_validity_rate} | "
            f"{citation.citation_document_hit_rate} | {citation.all_citations_true_document_rate} | "
            f"{citation.gold_evidence_citation_hit_rate} | {citation.gold_evidence_citation_precision} | "
            f"{citation.answer_supported_by_cited_evidence_rate} |"
        )
    lines.extend(["", "## Abstentions and Failures", ""])
    for name, item in summary.configurations.items():
        lines.append(f"- `{name}` abstention rate `{item.abstention_rate}`, correct abstentions `{item.correct_abstention_count}`")
        lines.append(f"- `{name}` failures: `{json.dumps(item.failures, sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Performance",
            "",
            "Stage timers are exposed by the existing AskResponse. Context-build and grounding-verification timers are unavailable in the production API and remain null, not fabricated.",
            "",
            "| Configuration | Stage | Mean ms | Median ms | P95 ms | Samples |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary.configurations.items():
        for stage, stats in item.latency.items():
            lines.append(f"| {name} | {stage} | {stats.mean_ms} | {stats.median_ms} | {stats.p95_ms} | {stats.samples} |")
    lines.extend(
        [
            "",
            "## Hybrid vs Hybrid + Reranker",
            "",
            "```json",
            json.dumps(summary.deltas, indent=2, sort_keys=True),
            "```",
            "",
            "## Limitations",
            "",
            "- ANLS is implemented compatibly with the public DocVQA reference evaluator: lowercase Unicode-NFKC strings, character Levenshtein similarity, and the 0.5 normalized-distance cutoff.",
            "- Exact Match uses E3's conservative literal normalization and is reported separately from ANLS.",
            "- A current RAG response exposes bounded source excerpts, not full chunks; answer-support diagnostics therefore use those returned excerpts.",
            "- Module 11's summary grounding verifier is not part of the ordinary AskResponse path; no separate LLM judge was introduced.",
            "",
            "## Reproduction",
            "",
            f"- Command: `{metadata.get('command')}`",
            f"- Manifest: `{metadata.get('manifest_path')}`",
            f"- Artifacts: `{metadata.get('output_directory')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(
    output_directory: Path,
    *,
    summary: E4RunSummary,
    question_records: list[dict[str, Any]],
    answer_records: list[dict[str, Any]],
    mappings: list[CorpusMapping],
    metadata: dict[str, Any],
) -> None:
    """Write all required bounded E4 artifacts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for filename, values in (("per_question.jsonl", question_records), ("answers.jsonl", answer_records)):
        with (output_directory / filename).open("w", encoding="utf-8", newline="\n") as stream:
            for value in values:
                stream.write(json.dumps(value, sort_keys=True) + "\n")
    with (output_directory / "corpus_mapping.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for mapping in mappings:
            stream.write(json.dumps(mapping.model_dump(mode="json"), sort_keys=True) + "\n")
    with (output_directory / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["configuration", "metric", "value"])
        writer.writeheader()
        writer.writerows(_metrics_csv(summary.configurations))
    (output_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "report.md").write_text(_report_markdown(summary, metadata), encoding="utf-8")


def write_controlled_state(
    output_directory: Path,
    *,
    split: str,
    run_id: str,
    reason_code: str,
    message: str,
    manifest_path: Path,
    document_limit: int,
    question_limit: int,
) -> E4RunSummary:
    """Write a controlled no-metrics state for missing prerequisites."""

    summary = E4RunSummary(
        status="DOCVQA_DATA_REQUIRED" if reason_code == "DOCVQA_DATA_REQUIRED" else "CONTROLLED_FAILURE",
        reason_code=reason_code,
        split=split,
        documents_requested=document_limit,
        documents_prepared=0,
        documents_indexed=0,
        questions_attempted=0,
        questions_scorable=0,
        questions_unscorable=0,
        failures=[message],
    )
    metadata = {
        "schema_version": "e4.v1",
        "run_id": run_id,
        "run_timestamp": now_utc_iso(),
        "manifest_path": str(manifest_path),
        "output_directory": str(output_directory),
        "document_limit": document_limit,
        "question_limit": question_limit,
        "status": summary.status,
        "reason_code": reason_code,
        "message": message,
        "models": {},
    }
    _write_artifacts(
        output_directory,
        summary=summary,
        question_records=[],
        answer_records=[],
        mappings=[],
        metadata=metadata,
    )
    return summary


async def run_e4(options: E4RunOptions, *, settings: Settings | None = None, run_id: str) -> E4RunSummary:
    """Ingest the bounded E3 corpus and evaluate the current production RAG path."""

    if options.top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if options.output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing E4 run directory: {options.output_directory}")
    options.output_directory.mkdir(parents=True, exist_ok=False)
    try:
        try:
            validation = validate_manifest(options.manifest_path)
        except ManifestValidationError as exc:
            return write_controlled_state(
                options.output_directory,
                split=options.split,
                run_id=run_id,
                reason_code="DOCVQA_DATA_REQUIRED" if "does not exist" in str(exc).lower() else "CONTROLLED_FAILURE",
                message=str(exc),
                manifest_path=options.manifest_path,
                document_limit=options.document_limit,
                question_limit=options.question_limit,
            )
        if not validation.records:
            return write_controlled_state(
                options.output_directory,
                split=options.split,
                run_id=run_id,
                reason_code="DOCVQA_DATA_REQUIRED",
                message="No prepared DocVQA records are available.",
                manifest_path=options.manifest_path,
                document_limit=options.document_limit,
                question_limit=options.question_limit,
            )
        if validation.dataset != "docvqa" or validation.split != options.split:
            return write_controlled_state(
                options.output_directory,
                split=options.split,
                run_id=run_id,
                reason_code="CONTROLLED_FAILURE",
                message=f"Manifest is {validation.dataset}/{validation.split}, expected docvqa/{options.split}.",
                manifest_path=options.manifest_path,
                document_limit=options.document_limit,
                question_limit=options.question_limit,
            )
        records = sorted(validation.records, key=lambda item: item.evaluation_id)[: options.document_limit]
        questions = build_questions(records, options.question_limit)
        if not questions:
            return write_controlled_state(
                options.output_directory,
                split=options.split,
                run_id=run_id,
                reason_code="DOCVQA_QUESTIONS_REQUIRED",
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
                reason_code="DATABASE_REQUIRED",
                message="E4 requires PostgreSQL/pgvector; provide DATABASE_URL or --database-url.",
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
        search_settings = effective_settings.model_copy(
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
                mapping_by_evaluation_id[record.evaluation_id] = mapping

            indexed_document_ids = [
                mapping.document_id for mapping in mappings if mapping.processing_status == "indexed" and mapping.document_id is not None
            ]
            # The production RAG safety guard defaults to 20 selected documents, while
            # this bounded E3/E4 corpus can contain 24 indexed documents.  Raise only
            # the evaluation instance's validation ceiling so every run-owned document
            # remains in scope; search, prompts, models, and filters are otherwise unchanged.
            rag_settings = effective_settings.model_copy(
                update={
                    "rag_max_selected_documents": max(
                        search_settings.rag_max_selected_documents,
                        len(indexed_document_ids),
                    )
                }
            )
            rag_service = RAGService(
                search_service=search_service,
                ollama_client=OllamaClient(settings=rag_settings),
                settings=rag_settings,
            )
            ground_truth: dict[str, E3QuestionGroundTruth] = {}
            question_records = []
            for question in questions:
                mapping = mapping_by_evaluation_id.get(question.evaluation_id)
                gt = build_question_ground_truth(
                    question,
                    target_document_id=mapping.document_id if mapping else None,
                    chunks=chunks_by_evaluation_id.get(question.evaluation_id),
                    document_indexed=bool(mapping and mapping.processing_status == "indexed"),
                )
                ground_truth[question.question_key] = gt
                question_records.append(_question_payload(question, gt))

            configuration_records: dict[str, list[E4QuestionRecord]] = {}
            warm_timings: dict[str, Any] = {}
            for configuration, rerank in CONFIGURATIONS:
                logger.info("E4 configuration started configuration=%s", configuration)
                result, warmup = await _run_configuration(
                    configuration=configuration,
                    rerank=rerank,
                    questions=questions,
                    ground_truth=ground_truth,
                    indexed_document_ids=indexed_document_ids,
                    rag_service=rag_service,
                    top_k=options.top_k,
                )
                configuration_records[configuration] = result
                warm_timings[configuration] = warmup

            summaries = {
                configuration: summarize_configuration(
                    records_for_config,
                    configuration=configuration,
                    search_mode=SearchMode.HYBRID.value,
                    rerank=configuration == "hybrid_reranked",
                    top_k=options.top_k,
                )
                for configuration, records_for_config in configuration_records.items()
            }
            summary = E4RunSummary(
                status="completed",
                split=options.split,
                documents_requested=options.document_limit,
                documents_prepared=len(records),
                documents_indexed=sum(mapping.processing_status == "indexed" for mapping in mappings),
                questions_attempted=len(questions),
                questions_scorable=sum(
                    ground_truth[question.question_key].status == "SCORABLE" for question in questions
                ),
                questions_unscorable=sum(
                    ground_truth[question.question_key].status != "SCORABLE" for question in questions
                ),
                answer_indexability_rate=(
                    sum(ground_truth[question.question_key].status == "SCORABLE" for question in questions) / len(questions)
                    if questions
                    else None
                ),
                configurations=summaries,
                deltas=compare_configurations(summaries["hybrid"], summaries["hybrid_reranked"]),
                failures=failures,
            )
            metadata = {
                "schema_version": "e4.v1",
                "run_id": run_id,
                "run_timestamp": now_utc_iso(),
                "dataset": "docvqa",
                "split": options.split,
                "manifest_path": str(options.manifest_path),
                "output_directory": str(options.output_directory),
                "document_limit": options.document_limit,
                "question_limit": options.question_limit,
                "top_k": options.top_k,
                "keep_indexed": options.keep_indexed,
                "database": _safe_database_target(effective_settings.resolved_database_url),
                "models": {
                    "embedding_model": effective_settings.embedding_model,
                    "embedding_dimension": effective_settings.embedding_dimension,
                    "reranker_model": effective_settings.reranker_model,
                    "ollama_model": effective_settings.ollama_model,
                    "ollama_base_url": effective_settings.ollama_base_url,
                },
                "prompt_and_generation": {
                    "ollama_temperature": effective_settings.ollama_temperature,
                    "rag_max_context_chars": effective_settings.rag_max_context_chars,
                    "answer_path": "RAGService.ask -> SearchService -> RAGContextBuilder -> existing grounded prompt -> OllamaClient",
                },
                "evaluation_only_overrides": {
                    "rag_max_selected_documents": rag_settings.rag_max_selected_documents,
                    "reason": "The existing 20-document request guard was raised only for this run-owned 24-document bounded corpus; production code and default settings were not changed.",
                },
                "answer_normalization_rules": ANSWER_NORMALIZATION_RULES,
                "anls_definition": ANLS_REFERENCE,
                "exact_match_definition": EM_NORMALIZATION,
                "warm_timings": warm_timings,
                "warm_timing_definition": "One warm-up RAG request per configuration, excluded from all measured quality and latency aggregates; the measured loop then uses the identical question set.",
                "unavailable_stage_timers": ["context_build_time_ms", "grounding_verification_time_ms"],
                "latency_aggregation_definition": "Each stage aggregate includes every per-question record with a measured value, including controlled failures; unavailable stage values remain excluded.",
                "production_path": "DocumentManagementService -> PDFIngestionService -> DocumentIndexingService -> existing SearchService -> RAGService -> existing OllamaClient",
                "command": " ".join(sys.argv),
                "platform": platform.platform(),
                "python_version": sys.version,
            }
            answer_records = [
                record.model_dump(mode="json")
                for config in CONFIGURATIONS
                for record in configuration_records[config[0]]
            ]
            if not options.keep_indexed:
                cleaned = cleanup_run_documents(repository, mappings, storage_directory)
                cleaned_by_id = {mapping.evaluation_id: mapping for mapping in cleaned}
                mappings = [cleaned_by_id.get(mapping.evaluation_id, mapping) for mapping in mappings]
            _write_artifacts(
                options.output_directory,
                summary=summary,
                question_records=question_records,
                answer_records=answer_records,
                mappings=mappings,
                metadata=metadata,
            )
            return summary
        finally:
            database.engine.dispose()
    except Exception:
        # The run directory is intentionally retained for diagnosis; cleanup is handled after successful setup.
        raise
