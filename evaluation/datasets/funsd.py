"""Bounded adapter for the scanned-form FUNSD dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from evaluation.schemas import EvaluationDocument, EvaluationEntity


class FUNSDAdapter(EvaluationDatasetAdapter):
    """Materialize source form images as PDFs and preserve word/entity labels."""

    dataset_name = "funsd"
    source_description = "nielsr/funsd"

    def prepare(self, options: PreparationOptions):
        self.validate_options(options)
        records: list[EvaluationDocument] = []
        failures: list[str] = []
        seen = 0
        try:
            dataset = load_huggingface_streaming(self.source_description, options.split)
            tag_names = _feature_label_names(dataset)
            for index, raw_record in enumerate(bounded_records(dataset, options.limit)):
                seen += 1
                try:
                    records.append(
                        self.normalize_record(
                            raw_record,
                            PreparationOptions(
                                dataset=options.dataset,
                                split=options.split,
                                limit=options.limit,
                                output_root=options.output_root,
                                source_directory=options.source_directory,
                                revision=options.revision,
                                command=options.command,
                                extra={**options.extra, "ner_tag_names": tag_names},
                            ),
                            index,
                        )
                    )
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
        """Normalize one FUNSD record without requiring dataset downloads in tests."""

        record_id = source_record_id(record, str(index))
        evaluation_id = stable_evaluation_id(self.dataset_name, options.split, record_id)
        output_path = options.output_root / self.dataset_name / stable_file_name(evaluation_id)
        image_value = source_value(record, "image", "page_image", "img", default=None)
        if image_value is None:
            raise DatasetPreparationError("record does not contain an image.")
        materialize_image_pdf(image_value, output_path)
        pages = pdf_page_info(output_path)
        entities = self._entities(record, options.extra.get("ner_tag_names", {}))
        metadata = {
            "materialization": "image_to_pdf",
            "source_fields": sorted(str(key) for key in record.keys()) if isinstance(record, dict) else [],
            "ground_truth_bbox_format": "source_xyxy",
        }
        for key in ("id", "document_id", "doc_id"):
            value = source_value(record, key)
            if value is not None:
                metadata[key] = safe_metadata(value)
        return EvaluationDocument(
            evaluation_id=evaluation_id,
            dataset=self.dataset_name,
            split=options.split,
            source_record_id=record_id,
            source_document_id=str(record_id),
            local_pdf_path=repository_relative(output_path),
            page_count=len(pages),
            pages=pages,
            entities=entities,
            metadata=metadata,
        )

    @staticmethod
    def _entities(record: Any, tag_names: Any) -> list[EvaluationEntity]:
        direct = source_value(record, "entities", "annotations", default=None)
        if isinstance(direct, list):
            entities = []
            for index, item in enumerate(direct):
                text = _text(source_value(item, "text", "word", "transcription", default=None))
                label = _label(source_value(item, "label", "entity", "ner_tag", default=None), tag_names)
                if text and label:
                    entities.append(
                        EvaluationEntity(
                            page_number=1,
                            text=text,
                            label=label,
                            bounding_box=normalize_bbox(
                                source_value(item, "bbox", "box", "bounding_box", default=None)
                            ),
                            source_annotation_id=_text(source_value(item, "id", default=None)) or str(index),
                        )
                    )
            if entities:
                return entities

        words = source_value(record, "words", "tokens", default=[])
        boxes = source_value(record, "bboxes", "boxes", "bounding_boxes", default=[])
        tags = source_value(record, "ner_tags", "labels", "tags", default=[])
        if not isinstance(words, list):
            return []
        entities = []
        for index, word in enumerate(words):
            if isinstance(word, dict):
                text = _text(source_value(word, "text", "word", default=None))
                bbox_value = source_value(word, "bbox", "box", "bounding_box", default=None)
            else:
                text = _text(word)
                bbox_value = boxes[index] if isinstance(boxes, list) and index < len(boxes) else None
            tag_value = tags[index] if isinstance(tags, list) and index < len(tags) else None
            label = _label(tag_value, tag_names)
            if text and label:
                entities.append(
                    EvaluationEntity(
                        page_number=1,
                        text=text,
                        label=label,
                        bounding_box=normalize_bbox(bbox_value),
                        source_annotation_id=str(index),
                    )
                )
        return entities


def _label(value: Any, names: Any) -> str | None:
    if value is None:
        return None
    if isinstance(names, dict) and value in names:
        value = names[value]
    elif isinstance(names, list) and isinstance(value, int) and value < len(names):
        value = names[value]
    return _text(value)


def _feature_label_names(dataset: Any) -> dict[int, str]:
    """Read ClassLabel names when the live Hugging Face schema exposes them."""

    try:
        feature = dataset.features["ner_tags"]
        class_label = getattr(feature, "feature", feature)
        names = getattr(class_label, "names", None)
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
    except (AttributeError, KeyError, TypeError):
        pass
    return {}


def _text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
