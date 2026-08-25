"""Bounded adapter for ``docling-project/DocLayNet-v1.2``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from evaluation.datasets.base import (
    DatasetPreparationError,
    EvaluationDatasetAdapter,
    PreparationOptions,
    bounded_records,
    finalize_preparation,
    load_huggingface_streaming,
    materialize_image_pdf,
    normalize_bbox,
    pdf_page_info,
    repository_relative,
    safe_metadata,
    source_record_id,
    source_value,
    stable_file_name,
)
from evaluation.manifests import stable_evaluation_id
from evaluation.schemas import EvaluationDocument, EvaluationLayoutRegion


class DocLayNetAdapter(EvaluationDatasetAdapter):
    """Prepare only a bounded streaming subset and preserve source annotations."""

    dataset_name = "doclaynet"
    source_description = "docling-project/DocLayNet-v1.2"

    CATEGORY_NAMES = {
        1: "Caption",
        2: "Footnote",
        3: "Formula",
        4: "List-item",
        5: "Page-footer",
        6: "Page-header",
        7: "Picture",
        8: "Section-header",
        9: "Table",
        10: "Text",
        11: "Title",
    }

    def prepare(self, options: PreparationOptions):
        self.validate_options(options)
        records: list[EvaluationDocument] = []
        failures: list[str] = []
        seen = 0
        try:
            dataset = load_huggingface_streaming(self.source_description, options.split)
            for index, raw_record in enumerate(bounded_records(dataset, options.limit)):
                seen += 1
                try:
                    records.append(self.normalize_record(raw_record, options, index))
                except (DatasetPreparationError, ValueError, TypeError, OSError) as exc:
                    failures.append(f"record {index}: {exc}")
        except DatasetPreparationError as exc:
            failures.append(f"{exc.code}: {exc}")
        return finalize_preparation(
            options,
            records,
            skipped=max(0, options.limit - seen),
            failures=failures,
            source_description=self.source_description,
            source_revision=options.revision,
        )

    def normalize_record(
        self,
        record: Any,
        options: PreparationOptions,
        index: int,
    ) -> EvaluationDocument:
        """Normalize one defensively inspected DocLayNet record."""

        record_metadata = source_value(record, "metadata", default={})
        metadata_record_id = (
            record_metadata.get("page_hash")
            if isinstance(record_metadata, dict)
            else None
        ) or (
            record_metadata.get("image_id")
            if isinstance(record_metadata, dict)
            else None
        )
        record_id = source_record_id(record, str(metadata_record_id or index))
        evaluation_id = stable_evaluation_id(self.dataset_name, options.split, record_id)
        output_path = options.output_root / self.dataset_name / stable_file_name(evaluation_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_value = source_value(record, "pdf", "pdf_bytes", "source_pdf", "pdf_path")
        image_value = source_value(record, "image", "page_image", "img")
        materialization = "source_pdf"
        if not self._copy_pdf(pdf_value, output_path, options.source_directory):
            if image_value is None:
                raise DatasetPreparationError(
                    "record has neither source PDF bytes/path nor a page image."
                )
            materialize_image_pdf(image_value, output_path)
            materialization = "image_to_pdf"
        pages = pdf_page_info(output_path)
        regions = self._layout_regions(record)
        metadata = {
            "materialization": materialization,
            "source_fields": sorted(str(key) for key in record.keys()) if isinstance(record, dict) else [],
            "source_metadata": safe_metadata(record_metadata),
        }
        for key in ("document_id", "doc_id", "page_no", "page_number", "original_filename"):
            value = source_value(record, key)
            if value is not None:
                metadata[key] = safe_metadata(value)
        if isinstance(record_metadata, dict):
            for key in ("page_no", "original_filename", "image_id", "page_hash"):
                if key in record_metadata:
                    metadata[key] = safe_metadata(record_metadata[key])
        return EvaluationDocument(
            evaluation_id=evaluation_id,
            dataset=self.dataset_name,
            split=options.split,
            source_record_id=record_id,
            source_document_id=self._optional_source_document_id(record),
            local_pdf_path=repository_relative(output_path),
            page_count=len(pages),
            pages=pages,
            layout_regions=regions,
            metadata=metadata,
        )

    @staticmethod
    def _copy_pdf(value: Any, output_path: Path, source_directory: Path | None) -> bool:
        from evaluation.datasets.base import copy_pdf_source

        return copy_pdf_source(value, output_path, source_directory)

    @staticmethod
    def _optional_source_document_id(record: Any) -> str | None:
        record_metadata = source_value(record, "metadata", default={})
        value = source_value(record, "document_id", "doc_id", "document", default=None)
        if value is None and isinstance(record_metadata, dict):
            value = record_metadata.get("original_filename")
        return str(value) if value not in (None, "") else None

    @staticmethod
    def _layout_regions(record: Any) -> list[EvaluationLayoutRegion]:
        regions: list[EvaluationLayoutRegion] = []
        categories = source_value(record, "categories", "category_names", default=DocLayNetAdapter.CATEGORY_NAMES)
        for index, annotation in enumerate(_annotation_items(record)):
            bbox = normalize_bbox(
                source_value(annotation, "bbox", "bounding_box", "box", default=None),
                format_name="xywh",
            )
            label = _annotation_label(annotation, categories)
            if bbox is None or label is None:
                continue
            regions.append(
                EvaluationLayoutRegion(
                    page_number=1,
                    label=label,
                    bounding_box=bbox,
                    text=_optional_text(source_value(annotation, "text", "transcription", default=None)),
                    source_annotation_id=_optional_text(
                        source_value(annotation, "id", "annotation_id", default=None)
                    )
                    or str(index),
                    metadata={"source_bbox_format": "xywh_or_explicit"},
                )
            )
        return regions


def _annotation_items(record: Any) -> Iterable[Any]:
    raw = source_value(record, "annotations", "layout_annotations", "objects", default=[])
    if isinstance(raw, list) and raw:
        return raw
    if isinstance(raw, dict):
        for key in ("annotations", "objects", "items"):
            if isinstance(raw.get(key), list):
                return raw[key]
        boxes = raw.get("bbox") or raw.get("bboxes") or []
        labels = raw.get("category") or raw.get("categories") or raw.get("labels") or []
        ids = raw.get("id") or raw.get("ids") or []
        if isinstance(boxes, list):
            return [
                {
                    "bbox": box,
                    "category": labels[index] if index < len(labels) else None,
                    "id": ids[index] if index < len(ids) else None,
                }
                for index, box in enumerate(boxes)
            ]
    if isinstance(record, dict) and isinstance(record.get("bboxes"), list):
        boxes = record["bboxes"]
        category_ids = record.get("category_id", [])
        pdf_cells = record.get("pdf_cells", [])
        return [
            {
                "bbox": box,
                "category_id": category_ids[index] if index < len(category_ids) else None,
                "text": " ".join(
                    str(cell.get("text", ""))
                    for cell in (pdf_cells[index] if index < len(pdf_cells) and isinstance(pdf_cells[index], list) else [])
                    if isinstance(cell, dict) and cell.get("text")
                ).strip()
                or None,
                "id": index,
            }
            for index, box in enumerate(boxes)
        ]
    return []


def _annotation_label(annotation: Any, categories: Any) -> str | None:
    value = source_value(annotation, "label", "category_name", "name", "category", default=None)
    if value is None:
        value = source_value(annotation, "category_id", "class_id", default=None)
        if isinstance(categories, dict) and value in categories:
            value = categories[value]
        elif isinstance(categories, list) and isinstance(value, int) and value < len(categories):
            value = categories[value]
    return _optional_text(value)


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _as_positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 1
    return number if number >= 1 else 1
