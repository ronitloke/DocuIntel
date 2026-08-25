"""Direct, persistence-free E2 benchmark execution and artifact generation."""

from __future__ import annotations

import asyncio
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytesseract

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    CorruptedPDFError,
    DocumentIngestionError,
    DocumentStorageError,
    EncryptedPDFError,
    PDFProcessingError,
)
from app.models.documents import DocumentIngestionResponse, DocumentStatus
from app.services.documents.pdf_ingestion import PDFIngestionService, UploadStream
from app.services.ocr.tesseract_ocr import TesseractOCRService
from evaluation.manifests import now_utc_iso
from evaluation.schemas import EvaluationBoundingBox, EvaluationDocument, EvaluationLayoutRegion
from evaluation.validation import resolve_local_path, validate_manifest
from evaluation.e2.metrics import LayoutEvaluation, TextEvaluation, evaluate_layout, evaluate_text


class LocalPDFUpload:
    """Minimal async upload adapter that feeds a local manifest PDF to the app service."""

    content_type = "application/pdf"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.filename = path.name
        self._stream = path.open("rb")

    async def read(self, size: int = -1) -> bytes:
        """Read the next upload chunk using the same interface as FastAPI uploads."""

        return self._stream.read(size)

    def close(self) -> None:
        """Close the local upload stream."""

        self._stream.close()


@dataclass(frozen=True, slots=True)
class E2DocumentResult:
    """Per-document benchmark result written to JSONL."""

    evaluation_id: str
    source_record_id: str
    status: str
    reason_code: str | None
    processing_time_ms: float | None
    page_count: int
    pages_with_native_text: int | None
    pages_processed_by_ocr: int | None
    unresolved_ocr_pages: int | None
    text: dict[str, Any] | None
    layout: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible per-document data."""

        return asdict(self)


def _reference_text(record: EvaluationDocument) -> str:
    """Join normalized FUNSD word annotations in their manifest/source order."""

    return " ".join(entity.text for entity in record.entities)


def _prediction_text(response: DocumentIngestionResponse) -> str:
    """Join page text in page order without changing OCR content."""

    return "\n".join(page.text for page in sorted(response.pages, key=lambda item: item.page_number))


def _layout_regions_in_pdf_space(record: EvaluationDocument) -> list[EvaluationLayoutRegion]:
    """Scale DocLayNet COCO boxes into the PDF page coordinate space when declared."""

    source_metadata = record.metadata.get("source_metadata", {})
    if not isinstance(source_metadata, dict):
        return list(record.layout_regions)
    try:
        source_width = float(source_metadata.get("coco_width"))
        source_height = float(source_metadata.get("coco_height"))
    except (TypeError, ValueError):
        return list(record.layout_regions)
    if source_width <= 0 or source_height <= 0:
        return list(record.layout_regions)
    page_sizes = {page.page_number: (page.width, page.height) for page in record.pages}
    transformed: list[EvaluationLayoutRegion] = []
    for region in record.layout_regions:
        target_width, target_height = page_sizes.get(
            region.page_number, (source_width, source_height)
        )
        box = region.bounding_box
        transformed.append(
            region.model_copy(
                update={
                    "bounding_box": EvaluationBoundingBox(
                        x0=box.x0 * target_width / source_width,
                        y0=box.y0 * target_height / source_height,
                        x1=box.x1 * target_width / source_width,
                        y1=box.y1 * target_height / source_height,
                    )
                }
            )
        )
    return transformed


def _failure_reason(exc: BaseException) -> str:
    """Map application exceptions to stable machine-readable benchmark reasons."""

    mapping = {
        CorruptedPDFError: "CORRUPTED_PDF",
        EncryptedPDFError: "ENCRYPTED_PDF",
        DocumentStorageError: "STORAGE_FAILURE",
        PDFProcessingError: "PDF_PROCESSING_FAILURE",
    }
    for error_type, reason in mapping.items():
        if isinstance(exc, error_type):
            return reason
    if isinstance(exc, DocumentIngestionError):
        return f"INGESTION_{type(exc).__name__.removesuffix('Error').upper()}"
    return "UNEXPECTED_FAILURE"


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return an inclusive linear-interpolation percentile, defined for small n."""

    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _aggregate_text(results: list[E2DocumentResult]) -> dict[str, Any] | None:
    scored = [result.text for result in results if result.text is not None]
    if not scored:
        return None
    output: dict[str, Any] = {"documents": len(scored)}
    for variant in ("strict", "whitespace_normalized"):
        values = [score[variant] for score in scored]
        character_distance = sum(int(value["character_edit_distance"]) for value in values)
        word_distance = sum(int(value["word_edit_distance"]) for value in values)
        reference_characters = sum(int(value["reference_characters"]) for value in values)
        reference_words = sum(int(value["reference_words"]) for value in values)
        output[variant] = {
            "cer": character_distance / reference_characters if reference_characters else None,
            "wer": word_distance / reference_words if reference_words else None,
            "mean_cer": statistics.fmean(float(value["cer"]) for value in values),
            "mean_wer": statistics.fmean(float(value["wer"]) for value in values),
            "character_edit_distance": character_distance,
            "word_edit_distance": word_distance,
            "reference_characters": reference_characters,
            "reference_words": reference_words,
        }
    return output


def _aggregate_layout(results: list[E2DocumentResult]) -> dict[str, Any] | None:
    scored = [result.layout for result in results if result.layout is not None]
    if not scored:
        return None
    true_positives = sum(int(item["true_positives"]) for item in scored)
    false_positives = sum(int(item["false_positives"]) for item in scored)
    false_negatives = sum(int(item["false_negatives"]) for item in scored)
    denominator_precision = true_positives + false_positives
    denominator_recall = true_positives + false_negatives
    precision = true_positives / denominator_precision if denominator_precision else None
    recall = true_positives / denominator_recall if denominator_recall else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (0.0 if precision is not None and recall is not None else None)
    )
    matched_ious = [
        float(match["iou"])
        for item in scored
        for match in item.get("matches", [])
    ]
    return {
        "documents": len(scored),
        "iou_threshold": scored[0]["iou_threshold"],
        "comparable_ground_truth": sum(int(item["comparable_ground_truth"]) for item in scored),
        "comparable_predictions": sum(int(item["comparable_predictions"]) for item in scored),
        "unsupported_ground_truth": sum(int(item["unsupported_ground_truth"]) for item in scored),
        "unsupported_predictions": sum(int(item["unsupported_predictions"]) for item in scored),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": statistics.fmean(matched_ious) if matched_ious else None,
    }


def _tesseract_metadata(service: TesseractOCRService) -> dict[str, Any]:
    """Capture executable/version facts without storing secrets or full environment data."""

    command = service.resolved_command
    available = service.is_available()
    version: str | None = None
    if available:
        try:
            version = str(pytesseract.get_tesseract_version())
        except (OSError, RuntimeError, pytesseract.TesseractError):
            version = None
    return {"executable": command, "available": available, "version": version}


def _summary(
    records: list[E2DocumentResult],
    *,
    dataset: str,
    split: str,
    manifest_path: Path,
    iou_threshold: float,
) -> dict[str, Any]:
    attempted = [record for record in records if record.status in {"success", "failed"}]
    successful = [record for record in records if record.status == "success"]
    skipped = [record for record in records if record.status == "skipped"]
    failed = [record for record in records if record.status == "failed"]
    timings = [record.processing_time_ms for record in attempted if record.processing_time_ms is not None]
    failure_reasons: dict[str, int] = {}
    for record in failed + skipped:
        if record.reason_code:
            failure_reasons[record.reason_code] = failure_reasons.get(record.reason_code, 0) + 1
    return {
        "schema_version": "e2.v1",
        "dataset": dataset,
        "split": split,
        "manifest_path": str(manifest_path),
        "reliability": {
            "processed_documents": len(records),
            "attempted_documents": len(attempted),
            "successful_documents": len(successful),
            "failed_documents": len(failed),
            "skipped_documents": len(skipped),
            "processing_success_rate": len(successful) / len(attempted) if attempted else None,
            "failure_reasons": failure_reasons,
        },
        "text_accuracy": _aggregate_text(records),
        "layout_accuracy": _aggregate_layout(records),
        "performance": {
            "documents_with_timing": len(timings),
            "mean_processing_time_ms": statistics.fmean(timings) if timings else None,
            "median_processing_time_ms": statistics.median(timings) if timings else None,
            "p95_processing_time_ms": _percentile(timings, 0.95),
            "timing_scope": "all attempted documents, including failed attempts",
        },
        "layout_metric_scope": (
            "Only compatible DocLayNet labels are scored: Section-header, Text, "
            "List-item, and Table. Unsupported labels/predictions are counted "
            "and excluded from TP/FP/FN denominators."
        ),
        "layout_evaluation_scope": (
            "All attempted documents with a returned page extraction are included, "
            "including a response marked failed; documents without a returned response "
            "have no layout score."
        ),
        "coordinate_scope": (
            "DocLayNet boxes are scaled from source_metadata.coco_width/coco_height "
            "into the E1 PDF page width/height when those dimensions are present; "
            "otherwise manifest coordinates are used unchanged."
        ),
        "iou_threshold": iou_threshold,
    }


def _metrics_csv(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, values in (
        ("reliability", summary["reliability"]),
        ("performance", summary["performance"]),
    ):
        for metric, value in values.items():
            if isinstance(value, (int, float)) or value is None:
                rows.append({"scope": scope, "metric": metric, "value": value})
    for scope_name, values in (("text_accuracy", summary["text_accuracy"]), ("layout_accuracy", summary["layout_accuracy"])):
        if not values:
            continue
        for metric, value in values.items():
            if isinstance(value, (int, float)) or value is None:
                rows.append({"scope": scope_name, "metric": metric, "value": value})
            elif isinstance(value, dict):
                for nested_metric, nested_value in value.items():
                    if isinstance(nested_value, (int, float)) or nested_value is None:
                        rows.append({"scope": f"{scope_name}.{metric}", "metric": nested_metric, "value": nested_value})
    return rows


def _write_artifacts(
    output_directory: Path,
    *,
    summary: dict[str, Any],
    records: list[E2DocumentResult],
    metadata: dict[str, Any],
) -> None:
    """Write stable, machine-readable and human-readable benchmark artifacts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_directory / "per_document.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    with (output_directory / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scope", "metric", "value"])
        writer.writeheader()
        writer.writerows(_metrics_csv(summary))
    (output_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "report.md").write_text(
        _markdown_report(summary, metadata, records), encoding="utf-8"
    )


def _markdown_report(
    summary: dict[str, Any], metadata: dict[str, Any], records: list[E2DocumentResult]
) -> str:
    """Render a concise report whose absent metrics remain explicitly unavailable."""

    reliability = summary["reliability"]
    performance = summary["performance"]
    lines = [
        f"# DocuIntel Module E2 evaluation — {summary['dataset']} {summary['split']}",
        "",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Run timestamp: `{metadata['run_timestamp']}`",
        f"- Processing success rate: `{reliability['processing_success_rate']}` "
        f"({reliability['successful_documents']}/{reliability['attempted_documents']} attempted)",
        "",
        "## Reliability",
        "",
        "| Status | Documents |",
        "|---|---:|",
        f"| Processed | {reliability['processed_documents']} |",
        f"| Successful | {reliability['successful_documents']} |",
        f"| Failed | {reliability['failed_documents']} |",
        f"| Skipped | {reliability['skipped_documents']} |",
        "",
        "Failure reasons: " + (json.dumps(reliability["failure_reasons"], sort_keys=True) or "none"),
        "",
        "## Text accuracy",
        "",
    ]
    text = summary["text_accuracy"]
    if text is None:
        lines.append("Not evaluated: no successfully processed records produced scorable text.")
    else:
        lines.extend(
            [
                "Metrics are shown for strict normalization and whitespace-normalized normalization.",
                "",
                "| Variant | CER | WER | Mean CER | Mean WER | Documents |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in ("strict", "whitespace_normalized"):
            item = text[variant]
            lines.append(
                f"| {variant} | {item['cer']:.6f} | {item['wer']:.6f} | "
                f"{item['mean_cer']:.6f} | {item['mean_wer']:.6f} | {text['documents']} |"
            )
    lines.extend(["", "## Layout accuracy", ""])
    layout = summary["layout_accuracy"]
    if layout is None:
        lines.append("Not evaluated: no successfully processed records produced layout results.")
    else:
        lines.extend(
            [
                "| IoU threshold | Precision | Recall | F1 | Mean matched IoU | TP | FP | FN |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {layout['iou_threshold']} | {layout['precision']} | {layout['recall']} | "
                f"{layout['f1']} | {layout['mean_matched_iou']} | {layout['true_positives']} | "
                f"{layout['false_positives']} | {layout['false_negatives']} |",
                "",
                summary["layout_metric_scope"],
                summary["layout_evaluation_scope"],
            ]
        )
    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- Mean processing time (ms): `{performance['mean_processing_time_ms']}`",
            f"- Median processing time (ms): `{performance['median_processing_time_ms']}`",
            f"- P95 processing time (ms): `{performance['p95_processing_time_ms']}`",
            f"- Timing scope: {performance['timing_scope']}",
            "",
            "## Reproducibility and limitations",
            "",
            "- The runner calls `PDFIngestionService` directly and does not insert into PostgreSQL.",
            "- OCR metrics use the existing Tesseract word extraction output; no alternate OCR is used.",
            "- CER/WER are edit-distance metrics; no lowercasing, punctuation removal, spelling correction, or semantic matching is applied.",
            "- P95 uses inclusive linear interpolation and is descriptive only for this bounded sample.",
            "- Unsupported layout labels are reported rather than silently mapped to unrelated classes.",
            "",
            f"Evaluated per-document records: {len(records)}.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_e2(
    manifest_path: Path,
    output_directory: Path,
    *,
    expected_dataset: str,
    expected_split: str,
    iou_threshold: float = 0.5,
    settings: Settings | None = None,
) -> Path:
    """Run E2 against one validated manifest and write all required artifacts."""

    validation = validate_manifest(manifest_path)
    if validation.dataset != expected_dataset or validation.split != expected_split:
        raise ValueError(
            f"Manifest dataset/split is {validation.dataset}/{validation.split}, "
            f"expected {expected_dataset}/{expected_split}."
        )
    if expected_dataset not in {"doclaynet", "funsd"}:
        raise ValueError("E2 supports only the DocLayNet and FUNSD manifests.")
    effective_settings = settings or get_settings()
    staging_directory = output_directory / ".ingest_staging"
    service = PDFIngestionService(settings=effective_settings, storage_directory=staging_directory)
    tesseract = service.ocr_service
    results: list[E2DocumentResult] = []
    for record in validation.records:
        has_ground_truth = bool(record.entities) if expected_dataset == "funsd" else bool(record.layout_regions)
        if not has_ground_truth:
            results.append(
                E2DocumentResult(
                    evaluation_id=record.evaluation_id,
                    source_record_id=record.source_record_id,
                    status="skipped",
                    reason_code="NO_GROUND_TRUTH",
                    processing_time_ms=None,
                    page_count=record.page_count,
                    pages_with_native_text=None,
                    pages_processed_by_ocr=None,
                    unresolved_ocr_pages=None,
                    text=None,
                    layout=None,
                )
            )
            continue
        path = resolve_local_path(manifest_path, record.local_pdf_path)
        upload = LocalPDFUpload(path)
        started = time.perf_counter()
        response: DocumentIngestionResponse | None = None
        try:
            response = await service.ingest(upload)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            reason = None
            status = "success"
            if response.status is DocumentStatus.FAILED:
                status = "failed"
                reason = "TESSERACT_UNAVAILABLE" if not tesseract.is_available() else "OCR_FAILED"
            text_metrics: TextEvaluation | None = None
            if expected_dataset == "funsd" and status == "success":
                text_metrics = evaluate_text(_reference_text(record), _prediction_text(response))
            layout_metrics: LayoutEvaluation | None = None
            if expected_dataset == "doclaynet":
                layout_metrics = evaluate_layout(
                    _layout_regions_in_pdf_space(record),
                    response.pages,
                    iou_threshold=iou_threshold,
                )
            results.append(
                E2DocumentResult(
                    evaluation_id=record.evaluation_id,
                    source_record_id=record.source_record_id,
                    status=status,
                    reason_code=reason,
                    processing_time_ms=elapsed_ms,
                    page_count=response.page_count,
                    pages_with_native_text=response.pages_with_native_text,
                    pages_processed_by_ocr=response.pages_processed_by_ocr,
                    unresolved_ocr_pages=response.unresolved_ocr_pages,
                    text=text_metrics.to_dict() if text_metrics else None,
                    layout=layout_metrics.to_dict() if layout_metrics else None,
                )
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            results.append(
                E2DocumentResult(
                    evaluation_id=record.evaluation_id,
                    source_record_id=record.source_record_id,
                    status="failed",
                    reason_code=_failure_reason(exc),
                    processing_time_ms=elapsed_ms,
                    page_count=record.page_count,
                    pages_with_native_text=None,
                    pages_processed_by_ocr=None,
                    unresolved_ocr_pages=None,
                    text=None,
                    layout=None,
                )
            )
        finally:
            upload.close()
            if response is not None:
                (staging_directory / response.stored_filename).unlink(missing_ok=True)

    summary = _summary(
        results,
        dataset=expected_dataset,
        split=expected_split,
        manifest_path=manifest_path,
        iou_threshold=iou_threshold,
    )
    metadata = {
        "schema_version": "e2.v1",
        "run_timestamp": now_utc_iso(),
        "dataset": expected_dataset,
        "split": expected_split,
        "manifest_path": str(manifest_path),
        "output_directory": str(output_directory),
        "python_version": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "settings": {
            "ocr_candidate_char_threshold": effective_settings.ocr_candidate_char_threshold,
            "ocr_language": effective_settings.ocr_language,
            "ocr_render_dpi": effective_settings.ocr_render_dpi,
            "tesseract_cmd": effective_settings.tesseract_cmd,
        },
        "tesseract": _tesseract_metadata(tesseract),
        "iou_threshold": iou_threshold,
        "metric_definitions": {
            "text": TEXT_METRIC_DEFINITION,
            "layout": summary["layout_metric_scope"],
            "performance": summary["performance"]["timing_scope"],
        },
    }
    _write_artifacts(output_directory, summary=summary, records=results, metadata=metadata)
    try:
        staging_directory.rmdir()
    except OSError:
        pass
    return output_directory


TEXT_METRIC_DEFINITION = (
    "CER and WER are Levenshtein edit distance divided by reference character/word "
    "count. Empty reference and empty prediction score 0; empty reference with a "
    "non-empty prediction scores 1.0."
)
