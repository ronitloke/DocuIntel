"""Shared bounded adapter primitives and local PDF materialization helpers."""

from __future__ import annotations

import hashlib
import io
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import pymupdf as fitz
from PIL import Image

from evaluation.manifests import (
    command_options,
    now_utc_iso,
    repository_relative,
    write_manifest,
    write_preparation_metadata,
)
from evaluation.schemas import (
    EvaluationBoundingBox,
    EvaluationDocument,
    EvaluationPage,
    PreparationMetadata,
    PreparationResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DatasetPreparationError(RuntimeError):
    """A source cannot currently be loaded or materialized safely."""

    def __init__(self, message: str, code: str = "DATASET_PREPARATION_ERROR") -> None:
        super().__init__(message)
        self.code = code


class DocVQADataRequired(DatasetPreparationError):
    """Official DocVQA files are required in the user-provided source directory."""

    def __init__(self, source_directory: Path) -> None:
        super().__init__(
            "Place the official DocVQA JSON and referenced "
            f"document images under {source_directory.resolve()}.",
            code="DOCVQA_DATA_REQUIRED",
        )


@dataclass(frozen=True)
class PreparationOptions:
    """Validated options shared by every adapter."""

    dataset: str
    split: str
    limit: int
    output_root: Path
    source_directory: Path | None = None
    revision: str | None = None
    command: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class EvaluationDatasetAdapter(ABC):
    """Common bounded interface consumed by the preparation CLI."""

    dataset_name: str
    source_description: str

    def validate_options(self, options: PreparationOptions) -> None:
        """Validate adapter-specific options before any source access."""

        if options.dataset != self.dataset_name:
            raise ValueError(f"Adapter {self.dataset_name!r} cannot prepare {options.dataset!r}.")
        if not options.split.strip():
            raise ValueError("split must contain non-whitespace text.")
        if options.limit <= 0:
            raise ValueError("limit must be greater than zero.")

    @abstractmethod
    def prepare(self, options: PreparationOptions) -> PreparationResult:
        """Load a bounded source subset and write normalized artifacts."""


def load_huggingface_streaming(dataset_name: str, split: str) -> Any:
    """Load a public dataset lazily, never falling back to an unlimited download."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetPreparationError(
            "The evaluation dependency 'datasets' is not installed. "
            "Install requirements-eval.txt inside the project .venv.",
            code="EVALUATION_DEPENDENCY_REQUIRED",
        ) from exc
    try:
        return load_dataset(dataset_name, split=split, streaming=True)
    except Exception as exc:
        raise DatasetPreparationError(
            f"Could not load {dataset_name!r} split {split!r} in bounded streaming mode: {exc}",
            code="DATASET_SOURCE_UNAVAILABLE",
        ) from exc


def bounded_records(dataset: Iterable[Any], limit: int) -> Iterator[Any]:
    """Yield at most the requested number of source records."""

    for index, record in enumerate(dataset):
        if index >= limit:
            break
        yield record


def source_value(record: Any, *names: str, default: Any = None) -> Any:
    """Read a field from mapping-like or attribute-like dataset records."""

    for name in names:
        if isinstance(record, dict) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def source_record_id(record: Any, fallback: str) -> str:
    """Extract a source ID without inventing semantic ground truth."""

    value = source_value(record, "id", "document_id", "doc_id", "uid", "name", default=None)
    return str(value) if value not in (None, "") else fallback


def safe_metadata(value: Any) -> Any:
    """Keep metadata JSON-serializable without serializing images or dataset objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_metadata(item) for item in value]
    return str(value)


def stable_file_name(evaluation_id: str) -> str:
    """Keep generated filenames short while retaining stable identity."""

    digest = hashlib.sha256(evaluation_id.encode("utf-8")).hexdigest()[:16]
    return f"{evaluation_id[:32]}-{digest}.pdf"


def copy_pdf_source(value: Any, output_path: Path, source_directory: Path | None = None) -> bool:
    """Copy source PDF bytes when a dataset exposes them."""

    candidate = value
    if isinstance(candidate, dict):
        candidate = candidate.get("bytes") or candidate.get("path") or candidate.get("pdf")
    if isinstance(candidate, (bytes, bytearray)) and bytes(candidate).startswith(b"%PDF-"):
        output_path.write_bytes(bytes(candidate))
        return True
    if isinstance(candidate, str):
        path = Path(candidate)
        if not path.is_absolute() and source_directory is not None:
            path = source_directory / path
        if path.is_file() and path.suffix.lower() == ".pdf":
            shutil.copyfile(path, output_path)
            return True
    if isinstance(candidate, Path) and candidate.is_file() and candidate.suffix.lower() == ".pdf":
        shutil.copyfile(candidate, output_path)
        return True
    return False


def materialize_image_pdf(image_value: Any, output_path: Path) -> tuple[int, int]:
    """Place one source image on a same-sized PDF page without altering its pixels."""

    image = open_image(image_value)
    try:
        rgb = image.convert("RGB")
        png = io.BytesIO()
        rgb.save(png, format="PNG")
        width, height = rgb.size
        document = fitz.open()
        try:
            page = document.new_page(width=float(width), height=float(height))
            page.insert_image(page.rect, stream=png.getvalue())
            output_path.parent.mkdir(parents=True, exist_ok=True)
            document.save(output_path)
        finally:
            document.close()
        return width, height
    finally:
        image.close()


def open_image(value: Any) -> Image.Image:
    """Open common Hugging Face/Pillow/local-image representations."""

    candidate = value
    if isinstance(candidate, dict):
        candidate = candidate.get("bytes") or candidate.get("path") or candidate.get("image")
    if isinstance(candidate, Image.Image):
        return candidate.copy()
    if isinstance(candidate, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(candidate))).copy()
    if isinstance(candidate, (str, Path)):
        return Image.open(candidate).copy()
    raise DatasetPreparationError("The source record does not contain a usable image.")


def pdf_page_info(path: Path) -> list[EvaluationPage]:
    """Read page dimensions from a materialized PDF."""

    try:
        with fitz.open(path) as document:
            return [
                EvaluationPage(
                    page_number=index + 1,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                )
                for index, page in enumerate(document)
            ]
    except (RuntimeError, fitz.FileDataError) as exc:
        raise DatasetPreparationError(f"Materialized PDF could not be reopened: {path}") from exc


def normalize_bbox(value: Any, *, format_name: str = "xyxy") -> EvaluationBoundingBox | None:
    """Convert common source bbox encodings without fabricating missing coordinates."""

    values: list[float] | None = None
    if isinstance(value, dict):
        if all(key in value for key in ("x0", "y0", "x1", "y1")):
            values = [float(value[key]) for key in ("x0", "y0", "x1", "y1")]
        elif all(key in value for key in ("x", "y", "width", "height")):
            values = [
                float(value["x"]),
                float(value["y"]),
                float(value["x"]) + float(value["width"]),
                float(value["y"]) + float(value["height"]),
            ]
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        values = [float(item) for item in value[:4]]
        if format_name == "xywh":
            values = [values[0], values[1], values[0] + values[2], values[1] + values[3]]
    if values is None:
        return None
    try:
        return EvaluationBoundingBox(x0=values[0], y0=values[1], x1=values[2], y1=values[3])
    except ValueError:
        return None


def finalize_preparation(
    options: PreparationOptions,
    records: list[EvaluationDocument],
    skipped: int,
    failures: list[str],
    source_description: str | None = None,
    source_revision: str | None = None,
) -> PreparationResult:
    """Write a deterministic manifest and timestamped provenance metadata."""

    output_directory = options.output_root / options.dataset
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.jsonl"
    metadata_path = output_directory / "preparation_metadata.json"
    write_manifest(records, manifest_path)
    metadata = PreparationMetadata(
        dataset=options.dataset,
        source=source_description or options.dataset,
        split=options.split,
        requested_limit=options.limit,
        prepared=len(records),
        skipped=skipped,
        failed=len(failures),
        prepared_at=now_utc_iso(),
        source_revision=source_revision,
        command=options.command,
        options=command_options(
            output_root=repository_relative(options.output_root),
            source_directory=repository_relative(options.source_directory)
            if options.source_directory is not None
            else None,
            revision=options.revision,
            **options.extra,
        ),
        errors=failures,
    )
    write_preparation_metadata(metadata, metadata_path)
    return PreparationResult(
        dataset=options.dataset,
        split=options.split,
        requested=options.limit,
        prepared=len(records),
        skipped=skipped,
        failed=len(failures),
        manifest_path=repository_relative(manifest_path),
        output_directory=repository_relative(output_directory),
        metadata_path=repository_relative(metadata_path),
        errors=failures,
    )
