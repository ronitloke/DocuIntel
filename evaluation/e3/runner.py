"""Controlled E3 corpus ingestion, indexing, retrieval, and artifact runner."""

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
from typing import Any, Iterable
from urllib.parse import urlsplit
from uuid import UUID

from app.core.config import Settings
from app.core.exceptions import DocumentIngestionError
from app.db.repository import DocumentRepository
from app.db.session import Database, create_database
from app.models.documents import DocumentStatus, DocumentIngestionResponse
from app.models.search import SearchFilters, SearchMode, SearchRequest
from app.services.chunking.structure_aware import StructureAwareChunker
from app.services.documents.document_management import DocumentManagementService
from app.services.documents.indexing import DocumentIndexingService
from app.services.documents.pdf_ingestion import PDFIngestionService
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.reranking.cross_encoder import CrossEncoderReranker
from app.services.retrieval.search import SearchService
from evaluation.e3.cleanup import cleanup_run_documents
from evaluation.e3.metrics import (
    ANSWER_NORMALIZATION_RULES,
    DEFAULT_E3_K_VALUES,
    aggregate_method_results,
    build_question_ground_truth,
    compare_metric_deltas,
    evaluate_ranked_ids,
)
from evaluation.e3.models import (
    CorpusMapping,
    E3MethodSummary,
    E3Question,
    E3QuestionGroundTruth,
    E3QuestionMethodResult,
    E3ResultItem,
    E3RunSummary,
    IndexedChunk,
)
from evaluation.manifests import now_utc_iso
from evaluation.schemas import EvaluationDocument
from evaluation.validation import ManifestValidationError, resolve_local_path, validate_manifest

logger = logging.getLogger(__name__)


class E3ControlledFailure(RuntimeError):
    """A controlled state that must be reported without fabricated metrics."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class LocalPDFUpload:
    """Minimal async upload adapter for the existing ingestion service."""

    content_type = "application/pdf"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.filename = path.name
        self._stream = path.open("rb")

    async def read(self, size: int = -1) -> bytes:
        """Read one upload chunk."""

        return self._stream.read(size)

    def close(self) -> None:
        """Close the source stream."""

        self._stream.close()


@dataclass(frozen=True, slots=True)
class E3RunOptions:
    """Bounded E3 runtime options."""

    manifest_path: Path
    output_directory: Path
    split: str
    document_limit: int
    question_limit: int
    top_k: int = 10
    keep_indexed: bool = False
    database_url_override: str | None = None


METHODS: tuple[tuple[str, SearchMode, bool], ...] = (
    ("keyword", SearchMode.KEYWORD, False),
    ("semantic", SearchMode.SEMANTIC, False),
    ("hybrid", SearchMode.HYBRID, False),
    ("hybrid_reranked", SearchMode.HYBRID, True),
)


def build_questions(records: Iterable[EvaluationDocument], question_limit: int) -> list[E3Question]:
    """Select one deterministic bounded question set shared by every method."""

    if question_limit <= 0:
        raise ValueError("question_limit must be greater than zero.")
    questions: list[E3Question] = []
    for record in sorted(records, key=lambda item: item.evaluation_id):
        for qa_pair in record.qa_pairs:
            question_key = f"{record.evaluation_id}:{qa_pair.question_id}"
            questions.append(
                E3Question(
                    question_key=question_key,
                    evaluation_id=record.evaluation_id,
                    source_record_id=record.source_record_id,
                    source_document_id=record.source_document_id,
                    question_id=qa_pair.question_id,
                    question=qa_pair.question,
                    accepted_answers=list(qa_pair.accepted_answers),
                    page_number=qa_pair.page_number,
                    local_pdf_path=record.local_pdf_path,
                )
            )
            if len(questions) >= question_limit:
                return questions
    return questions


def _safe_database_target(database_url: str | None) -> dict[str, Any]:
    """Record database location without credentials or the full connection URL."""

    if not database_url:
        return {"configured": False}
    parsed = urlsplit(database_url.replace("postgresql+psyc", "postgresql", 1))
    return {
        "configured": True,
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": parsed.port,
        "database": parsed.path.lstrip("/") or None,
    }


def _chunk_projection(document_id: UUID, chunks: Iterable[Any]) -> list[IndexedChunk]:
    """Detach only retrieval-relevant chunk metadata from SQLAlchemy objects."""

    return [
        IndexedChunk(
            chunk_id=chunk.id,
            document_id=document_id,
            sequence_number=chunk.sequence_number,
            text=chunk.text,
            start_page=chunk.start_page,
            end_page=chunk.end_page,
            section_heading=chunk.section_heading,
        )
        for chunk in sorted(chunks, key=lambda item: (item.sequence_number, str(item.id)))
    ]


def _result_item(result: Any) -> E3ResultItem:
    """Project a production SearchResult without exposing unbounded text."""

    return E3ResultItem(
        rank=result.rank,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        filename=result.original_filename,
        sequence_number=result.sequence_number,
        start_page=result.start_page,
        end_page=result.end_page,
        excerpt=result.text[:500],
        semantic_score=result.semantic_score,
        keyword_score=result.keyword_score,
        hybrid_score=result.hybrid_score,
        base_rank=result.base_rank,
        rerank_score=result.rerank_score,
    )


def _method_request(
    question: E3Question,
    mode: SearchMode,
    rerank: bool,
    top_k: int,
    document_ids: list[UUID],
) -> SearchRequest:
    """Create the same benchmark corpus filter for every retrieval method."""

    return SearchRequest(
        query=question.question,
        mode=mode,
        top_k=top_k,
        rerank=rerank,
        filters=SearchFilters(document_ids=document_ids),
    )


def _empty_method_result(
    question: E3Question,
    method: str,
    ground_truth: E3QuestionGroundTruth,
    *,
    error: str,
) -> E3QuestionMethodResult:
    """Represent a failed search as a visible zero-retrieval result."""

    return E3QuestionMethodResult(
        question_key=question.question_key,
        method=method,
        relevant_chunk_ids=ground_truth.relevant_chunk_ids,
        reciprocal_rank=0.0,
        error=error,
    )


def _method_metrics(
    question: E3Question,
    method: str,
    response: Any,
    ground_truth: E3QuestionGroundTruth,
    wall_clock_time_ms: float,
) -> E3QuestionMethodResult:
    """Calculate E3 metrics from the unchanged production SearchResponse."""

    items = [_result_item(result) for result in response.results]
    chunk_ids = [item.chunk_id for item in items]
    document_ids = [item.document_id for item in items]
    metrics = evaluate_ranked_ids(
        chunk_ids,
        ground_truth.relevant_chunk_ids,
        ranked_document_ids=document_ids,
        target_document_id=ground_truth.target_document_id,  # type: ignore[arg-type]
    )
    return E3QuestionMethodResult(
        question_key=question.question_key,
        method=method,
        result_items=items,
        relevant_chunk_ids=ground_truth.relevant_chunk_ids,
        first_relevant_rank=metrics["first_relevant_rank"],
        reciprocal_rank=metrics["reciprocal_rank"],
        recall_at_k=metrics["recall_at_k"],
        hit_at_k=metrics["hit_at_k"],
        document_hit_at_k=metrics["document_hit_at_k"],
        retrieval_time_ms=response.retrieval_time_ms,
        reranking_time_ms=response.rerank_time_ms,
        total_retrieval_pipeline_ms=response.total_search_time_ms,
        wall_clock_time_ms=wall_clock_time_ms,
    )


def _deltas(methods: dict[str, E3MethodSummary]) -> dict[str, Any]:
    """Calculate explicitly labeled absolute and percentage-point comparisons."""

    result: dict[str, Any] = {}
    if "semantic" in methods and "hybrid" in methods:
        result["hybrid_vs_semantic"] = compare_metric_deltas(
            methods["semantic"], methods["hybrid"], label="hybrid minus semantic"
        )
    if "hybrid" in methods and "hybrid_reranked" in methods:
        result["hybrid_reranked_vs_hybrid"] = compare_metric_deltas(
            methods["hybrid"], methods["hybrid_reranked"], label="hybrid+reranker minus hybrid"
        )
    return result


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    """Write deterministic JSONL with one bounded record per line."""

    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")


def _metrics_csv(methods: dict[str, E3MethodSummary]) -> list[dict[str, Any]]:
    """Flatten method summaries into a comparison-friendly CSV."""

    rows: list[dict[str, Any]] = []
    for method, summary in methods.items():
        for metric_name, values in (
            ("recall_at_k", summary.recall_at_k),
            ("hit_at_k", summary.hit_at_k),
            ("document_hit_at_k", summary.document_hit_at_k),
        ):
            for k, value in values.items():
                rows.append({"method": method, "metric": f"{metric_name}@{k}", "value": value})
        rows.extend(
            [
                {"method": method, "metric": "mrr", "value": summary.mrr},
                {"method": method, "metric": "mean_retrieval_time_ms", "value": summary.retrieval_latency.mean_ms},
                {"method": method, "metric": "median_retrieval_time_ms", "value": summary.retrieval_latency.median_ms},
                {"method": method, "metric": "p95_retrieval_time_ms", "value": summary.retrieval_latency.p95_ms},
                {"method": method, "metric": "mean_reranking_time_ms", "value": summary.reranking_latency.mean_ms},
                {"method": method, "metric": "median_reranking_time_ms", "value": summary.reranking_latency.median_ms},
                {"method": method, "metric": "p95_reranking_time_ms", "value": summary.reranking_latency.p95_ms},
                {"method": method, "metric": "mean_total_pipeline_ms", "value": summary.total_pipeline_latency.mean_ms},
                {"method": method, "metric": "median_total_pipeline_ms", "value": summary.total_pipeline_latency.median_ms},
                {"method": method, "metric": "p95_total_pipeline_ms", "value": summary.total_pipeline_latency.p95_ms},
            ]
        )
    return rows


def _report_markdown(summary: E3RunSummary, metadata: dict[str, Any]) -> str:
    """Render the E3 comparison report without inventing unavailable values."""

    lines = [
        "# DocuIntel Evaluation E3 — controlled DocVQA retrieval benchmark",
        "",
        f"- Status: `{summary.status}`",
        f"- Split: `{summary.split}`",
        f"- Run ID: `{metadata['run_id']}`",
        "",
    ]
    if summary.status != "completed":
        lines.extend(
            [
                f"Reason: `{summary.reason_code}`",
                "",
                "No retrieval metrics were produced because the controlled benchmark prerequisites were unavailable.",
                "",
                "Expected official DocVQA input:",
                "- `data/evaluation/raw/docvqa/`",
                "- A validation JSON/JSONL metadata file whose filename contains `validation`.",
                "- Records containing question ID, question text, accepted answers, and image reference.",
                "- Referenced official page images resolvable relative to that directory.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Corpus and question coverage",
            "",
            f"- Documents prepared: `{summary.documents_prepared}`",
            f"- Documents indexed: `{summary.documents_indexed}`",
            f"- Questions attempted: `{summary.questions_attempted}`",
            f"- Questions scorable: `{summary.questions_scorable}`",
            f"- Questions unscorable: `{summary.questions_unscorable}`",
            f"- Answer-indexability rate: `{summary.answer_indexability_rate}`",
            "",
            "## Retrieval comparison",
            "",
            "| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Median retrieval ms | Median reranking ms | Median total ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, item in summary.methods.items():
        lines.append(
            f"| {method} | {item.recall_at_k.get('1')} | {item.recall_at_k.get('3')} | "
            f"{item.recall_at_k.get('5')} | {item.recall_at_k.get('10')} | {item.mrr} | "
            f"{item.retrieval_latency.median_ms} | {item.reranking_latency.median_ms} | "
            f"{item.total_pipeline_latency.median_ms} |"
        )
    lines.extend(["", "## Document Hit@K", "", "| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 |", "|---|---:|---:|---:|---:|"])
    for method, item in summary.methods.items():
        lines.append(
            f"| {method} | {item.document_hit_at_k.get('1')} | {item.document_hit_at_k.get('3')} | "
            f"{item.document_hit_at_k.get('5')} | {item.document_hit_at_k.get('10')} |"
        )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "Deltas are reported as absolute changes; recall deltas additionally show percentage points.",
            "",
            "```json",
            json.dumps(summary.deltas, indent=2, sort_keys=True),
            "```",
            "",
            "## Method and timing notes",
            "",
            f"- Embedding model: `{metadata['models']['embedding_model']}`",
            f"- Reranking model: `{metadata['models']['reranker_model']}`",
            f"- Candidate settings: `{json.dumps(metadata['candidate_settings'], sort_keys=True)}`",
            "- Warm metrics are measured after a bounded warm-up query for each method; cold timings are recorded separately in run_metadata.json.",
            "- Ground truth is binary and literal: no LLM judgement, embeddings, or fuzzy matching define relevance.",
            "- Retrieval metrics use only SCORABLE questions; ANSWER_NOT_INDEXED and processing failures remain in coverage counts.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(
    output_directory: Path,
    *,
    summary: E3RunSummary,
    question_records: list[dict[str, Any]],
    retrieval_records: list[dict[str, Any]],
    mappings: list[CorpusMapping],
    metadata: dict[str, Any],
) -> None:
    """Write all required E3 artifacts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_directory / "per_question.jsonl", question_records)
    _write_jsonl(output_directory / "retrieval_results.jsonl", retrieval_records)
    _write_jsonl(
        output_directory / "corpus_mapping.jsonl",
        [mapping.model_dump(mode="json") for mapping in mappings],
    )
    with (output_directory / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "metric", "value"])
        writer.writeheader()
        writer.writerows(_metrics_csv(summary.methods))
    (output_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "report.md").write_text(
        _report_markdown(summary, metadata), encoding="utf-8"
    )


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
) -> E3RunSummary:
    """Write a no-metrics E3 result for missing official data or prerequisites."""

    summary = E3RunSummary(
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
        "schema_version": "e3.v1",
        "run_id": run_id,
        "run_timestamp": now_utc_iso(),
        "dataset": "docvqa",
        "split": split,
        "manifest_path": str(manifest_path),
        "document_limit": document_limit,
        "question_limit": question_limit,
        "status": summary.status,
        "reason_code": reason_code,
        "message": message,
        "models": {},
        "candidate_settings": {},
    }
    _write_artifacts(
        output_directory,
        summary=summary,
        question_records=[],
        retrieval_records=[],
        mappings=[],
        metadata=metadata,
    )
    return summary


async def run_e3(
    options: E3RunOptions,
    *,
    settings: Settings | None = None,
    run_id: str,
) -> E3RunSummary:
    """Execute E3 with real ingestion/indexing/retrieval or a controlled state."""

    if options.top_k < max(DEFAULT_E3_K_VALUES):
        raise ValueError("top_k must be at least 10 so Recall@10 uses the same result pool.")
    if options.output_directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing E3 run directory: {options.output_directory}"
        )
    options.output_directory.mkdir(parents=True, exist_ok=False)

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
            message=(
                "No prepared DocVQA records are available. Supply official metadata/images under "
                "data/evaluation/raw/docvqa and run the E1 preparation command."
            ),
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
        effective_settings = effective_settings.model_copy(
            update={"database_url": options.database_url_override}
        )
    database = create_database(effective_settings)
    if database is None:
        return write_controlled_state(
            options.output_directory,
            split=options.split,
            run_id=run_id,
            reason_code="DATABASE_REQUIRED",
            message="E3 requires PostgreSQL/pgvector; provide DATABASE_URL or --database-url.",
            manifest_path=options.manifest_path,
            document_limit=options.document_limit,
            question_limit=options.question_limit,
        )

    repository = DocumentRepository(database)
    storage_directory = options.output_directory / "ingest_staging"
    ingestion_service = PDFIngestionService(
        settings=effective_settings,
        storage_directory=storage_directory,
    )
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
    reranker = CrossEncoderReranker(settings=effective_settings)
    search_service = SearchService(
        repository=repository,
        embedding_service=embedding_service,
        settings=effective_settings,
        reranker=reranker,
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
                    mapping = mapping.model_copy(
                        update={"processing_status": "failed", "failure_reason": "DOCUMENT_PROCESSING_FAILED"}
                    )
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
                mapping = mapping.model_copy(
                    update={"processing_status": "failed", "failure_reason": type(exc).__name__}
                )
                failures.append(f"{record.evaluation_id}: {message}")
            finally:
                upload.close()
                if response is not None:
                    (storage_directory / response.stored_filename).unlink(missing_ok=True)
            mappings.append(mapping)
            mapping_by_evaluation_id[record.evaluation_id] = mapping

        indexed_document_ids = [
            mapping.document_id
            for mapping in mappings
            if mapping.processing_status == "indexed" and mapping.document_id is not None
        ]
        indexed_document_ids = [document_id for document_id in indexed_document_ids if document_id is not None]
        ground_truth: dict[str, E3QuestionGroundTruth] = {}
        question_records: list[dict[str, Any]] = []
        for question in questions:
            mapping = mapping_by_evaluation_id.get(question.evaluation_id)
            target_document_id = mapping.document_id if mapping else None
            gt = build_question_ground_truth(
                question,
                target_document_id=target_document_id,
                chunks=chunks_by_evaluation_id.get(question.evaluation_id),
                document_indexed=bool(mapping and mapping.processing_status == "indexed"),
            )
            ground_truth[question.question_key] = gt
            question_records.append(
                {
                    **question.model_dump(mode="json"),
                    "ground_truth": gt.model_dump(mode="json"),
                }
            )
        scorable_questions = [question for question in questions if ground_truth[question.question_key].status == "SCORABLE"]
        method_results: dict[str, list[E3QuestionMethodResult]] = {method: [] for method, _, _ in METHODS}
        retrieval_records: list[dict[str, Any]] = []
        cold_timings: dict[str, dict[str, Any]] = {}

        if scorable_questions and indexed_document_ids:
            warmup_question = scorable_questions[0]
            for method, mode, rerank_enabled in METHODS:
                started = time.perf_counter()
                try:
                    response = search_service.search(
                        _method_request(
                            warmup_question,
                            mode,
                            rerank_enabled,
                            options.top_k,
                            indexed_document_ids,
                        )
                    )
                    cold_timings[method] = {
                        "wall_clock_time_ms": round((time.perf_counter() - started) * 1000, 3),
                        "retrieval_time_ms": response.retrieval_time_ms,
                        "reranking_time_ms": response.rerank_time_ms,
                        "total_retrieval_pipeline_ms": response.total_search_time_ms,
                        "warmup_question_key": warmup_question.question_key,
                    }
                except Exception as exc:
                    cold_timings[method] = {
                        "error": str(exc),
                        "wall_clock_time_ms": round((time.perf_counter() - started) * 1000, 3),
                        "warmup_question_key": warmup_question.question_key,
                    }

        for question in scorable_questions:
            gt = ground_truth[question.question_key]
            for method, mode, rerank_enabled in METHODS:
                started = time.perf_counter()
                try:
                    response = search_service.search(
                        _method_request(
                            question,
                            mode,
                            rerank_enabled,
                            options.top_k,
                            indexed_document_ids,
                        )
                    )
                    result = _method_metrics(
                        question,
                        method,
                        response,
                        gt,
                        round((time.perf_counter() - started) * 1000, 3),
                    )
                except Exception as exc:
                    message = exc.public_message if isinstance(exc, DocumentIngestionError) else str(exc)
                    result = _empty_method_result(question, method, gt, error=message)
                    result.wall_clock_time_ms = round((time.perf_counter() - started) * 1000, 3)
                method_results[method].append(result)
                retrieval_records.append(
                    {
                        "question_key": question.question_key,
                        "method": method,
                        **result.model_dump(mode="json"),
                    }
                )

        method_summaries = {
            method: aggregate_method_results(
                method,
                method_results[method],
                scorable_questions=len(scorable_questions),
                candidate_count=(
                    min(
                        max(
                            effective_settings.rerank_candidate_count,
                            options.top_k * effective_settings.rerank_candidate_multiplier,
                        ),
                        effective_settings.rerank_max_candidates,
                    )
                    if rerank_enabled
                    else (options.top_k * effective_settings.search_candidate_multiplier if mode is SearchMode.HYBRID else options.top_k)
                ),
            )
            for method, mode, rerank_enabled in METHODS
        }
        summary = E3RunSummary(
            status="completed",
            split=options.split,
            documents_requested=options.document_limit,
            documents_prepared=len(records),
            documents_indexed=sum(mapping.processing_status == "indexed" for mapping in mappings),
            questions_attempted=len(questions),
            questions_scorable=len(scorable_questions),
            questions_unscorable=len(questions) - len(scorable_questions),
            answer_indexability_rate=len(scorable_questions) / len(questions) if questions else None,
            methods=method_summaries,
            deltas=_deltas(method_summaries),
            failures=failures,
        )
        metadata = {
            "schema_version": "e3.v1",
            "run_id": run_id,
            "run_timestamp": now_utc_iso(),
            "dataset": "docvqa",
            "split": options.split,
            "manifest_path": str(options.manifest_path),
            "document_limit": options.document_limit,
            "question_limit": options.question_limit,
            "top_k": options.top_k,
            "k_values": list(DEFAULT_E3_K_VALUES),
            "keep_indexed": options.keep_indexed,
            "database": _safe_database_target(effective_settings.resolved_database_url),
            "models": {
                "embedding_model": effective_settings.embedding_model,
                "embedding_dimension": effective_settings.embedding_dimension,
                "reranker_model": effective_settings.reranker_model,
            },
            "candidate_settings": {
                "search_candidate_multiplier": effective_settings.search_candidate_multiplier,
                "rerank_candidate_count": effective_settings.rerank_candidate_count,
                "rerank_candidate_multiplier": effective_settings.rerank_candidate_multiplier,
                "rerank_max_candidates": effective_settings.rerank_max_candidates,
                "hybrid_rrf_k": effective_settings.hybrid_rrf_k,
            },
            "answer_normalization_rules": ANSWER_NORMALIZATION_RULES,
            "cold_timings": cold_timings,
            "warm_timing_definition": "Measured query results after one warm-up query per method; warm-up is excluded from quality and warm latency aggregates.",
            "production_path": "DocumentManagementService -> PDFIngestionService -> DocumentIndexingService -> StructureAwareChunker/EmbeddingService -> SearchService -> repository semantic/keyword/hybrid -> CrossEncoderReranker",
            "command": " ".join(sys.argv),
            "platform": platform.platform(),
            "python_version": sys.version,
        }
        if not options.keep_indexed:
            mappings = cleanup_run_documents(repository, mappings, storage_directory)
        else:
            mappings = [mapping.model_copy(update={"cleaned_up": False}) for mapping in mappings]
        _write_artifacts(
            options.output_directory,
            summary=summary,
            question_records=question_records,
            retrieval_records=retrieval_records,
            mappings=mappings,
            metadata=metadata,
        )
        return summary
    finally:
        database.engine.dispose()
