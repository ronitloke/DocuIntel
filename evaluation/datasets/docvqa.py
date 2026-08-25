"""Local/manual DocVQA adapter with bounded question-document materialization."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

from evaluation.datasets.base import (
    DocVQADataRequired,
    DatasetPreparationError,
    EvaluationDatasetAdapter,
    PreparationOptions,
    finalize_preparation,
    materialize_image_pdf,
    pdf_page_info,
    repository_relative,
    safe_metadata,
    source_value,
    stable_file_name,
)
from evaluation.manifests import stable_evaluation_id
from evaluation.schemas import EvaluationDocument, EvaluationQAPair


DOCVQA_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "validation": ("validation", "val"),
    "train": ("train", "training"),
    "test": ("test", "testing"),
}


def canonical_docvqa_split(split: str) -> str:
    """Normalize supported DocVQA split names without changing their meaning."""

    normalized = split.strip().casefold()
    for canonical, aliases in DOCVQA_SPLIT_ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


class DocVQAAdapter(EvaluationDatasetAdapter):
    """Read official local DocVQA release files; never scrape or substitute data."""

    dataset_name = "docvqa"
    source_description = "official DocVQA files supplied under data/evaluation/raw/docvqa"

    def prepare(self, options: PreparationOptions):
        canonical_split = canonical_docvqa_split(options.split)
        effective_options = replace(options, split=canonical_split)
        self.validate_options(effective_options)
        source_directory = effective_options.source_directory or (
            Path(__file__).resolve().parents[2] / "data" / "evaluation" / "raw" / "docvqa"
        )
        records: list[EvaluationDocument] = []
        failures: list[str] = []
        source_missing = False
        try:
            items = self._load_items(source_directory, canonical_split)
            groups = self._group_items(items, source_directory, effective_options.limit)
            for index, (group_key, group) in enumerate(groups.items()):
                try:
                    records.append(
                        self.normalize_group(group_key, group, effective_options, index, source_directory)
                    )
                except (DatasetPreparationError, ValueError, TypeError, OSError) as exc:
                    failures.append(f"document {group_key}: {exc}")
        except DocVQADataRequired as exc:
            failures.append(f"{exc.code}: {exc}")
            source_missing = True
        except DatasetPreparationError as exc:
            failures.append(f"{exc.code}: {exc}")
        prepared_or_failed = len(records) + len(failures)
        return finalize_preparation(
            effective_options,
            records,
            skipped=effective_options.limit
            if source_missing
            else max(0, effective_options.limit - prepared_or_failed),
            failures=failures,
            source_description="official_local_docvqa",
            source_revision=effective_options.revision,
        )

    def normalize_group(
        self,
        group_key: str,
        items: list[dict[str, Any]],
        options: PreparationOptions,
        index: int,
        source_directory: Path,
    ) -> EvaluationDocument:
        """Materialize one image/document and normalize all bounded questions for it."""

        first = items[0]
        image_value = first.get("_resolved_image")
        if image_value is None:
            raise DatasetPreparationError("question record does not reference a readable document image.")
        evaluation_id = stable_evaluation_id(self.dataset_name, options.split, group_key)
        output_path = options.output_root / self.dataset_name / stable_file_name(evaluation_id)
        materialize_image_pdf(image_value, output_path)
        pages = pdf_page_info(output_path)
        if isinstance(image_value, (str, Path)) and pages:
            pages[0] = pages[0].model_copy(
                update={"source_image_path": repository_relative(Path(image_value))}
            )
        qa_pairs = [self._qa_pair(item) for item in items]
        return EvaluationDocument(
            evaluation_id=evaluation_id,
            dataset=self.dataset_name,
            split=options.split,
            source_record_id=group_key,
            source_document_id=_string(first.get("_source_document_id")),
            local_pdf_path=repository_relative(output_path),
            page_count=len(pages),
            pages=pages,
            qa_pairs=qa_pairs,
            metadata={
                "materialization": "image_to_pdf",
                "source_directory": repository_relative(source_directory),
                "source_fields": sorted(key for key in first if not key.startswith("_")),
            },
        )

    @staticmethod
    def _load_items(source_directory: Path, split: str) -> list[dict[str, Any]]:
        if not source_directory.is_dir():
            raise DocVQADataRequired(source_directory)
        aliases = DOCVQA_SPLIT_ALIASES.get(canonical_docvqa_split(split), (split,))
        conventional_names = [
            f"{alias}{suffix}"
            for alias in aliases
            for suffix in (
                "_v1.0_withQT.json",
                "_v1.0.json",
                "_withQT.json",
                ".json",
                ".jsonl",
            )
        ]
        candidates = sorted(
            {
                source_directory / name
                for name in conventional_names
                if (source_directory / name).is_file()
            },
            key=lambda path: path.as_posix().casefold(),
        )
        if not candidates:
            candidates = sorted(
                [*source_directory.rglob("*.json"), *source_directory.rglob("*.jsonl")],
                key=lambda path: path.as_posix().casefold(),
            )
        if not candidates:
            raise DocVQADataRequired(source_directory)
        matching = [DocVQAAdapter._select_metadata_file(candidates, split)]
        items: list[dict[str, Any]] = []
        for path in matching:
            try:
                if path.suffix.lower() == ".jsonl":
                    payloads = [
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                else:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        DocVQAAdapter._validate_split_value(
                            payload.get("dataset_split") or payload.get("split"),
                            split,
                            path,
                        )
                    payloads = payload.get("data", payload) if isinstance(payload, dict) else payload
                if not isinstance(payloads, list):
                    raise ValueError("expected a JSON array or an object containing a data array.")
                for item in payloads:
                    if isinstance(item, dict):
                        DocVQAAdapter._validate_split_value(
                            source_value(item, "data_split", "split", default=None),
                            split,
                            path,
                        )
                        items.append({**item, "_metadata_file": path.as_posix()})
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                if isinstance(exc, DatasetPreparationError):
                    raise
                raise DatasetPreparationError(f"Could not parse DocVQA file {path}: {exc}") from exc
        return items

    @staticmethod
    def _select_metadata_file(candidates: list[Path], split: str) -> Path:
        """Select one metadata file using exact filename tokens and deterministic priority."""

        aliases = DOCVQA_SPLIT_ALIASES.get(split, (split,))
        matches: list[tuple[int, Path]] = []
        for candidate in candidates:
            tokens = set(re.findall(r"[a-z0-9]+", candidate.stem.casefold()))
            priorities = [index for index, alias in enumerate(aliases) if alias in tokens]
            if priorities:
                matches.append((min(priorities), candidate))
        if not matches:
            raise DatasetPreparationError(
                f"No DocVQA metadata file matches split {split!r} under {candidates[0].parent}. "
                f"Accepted filename aliases: {', '.join(aliases)}.",
                code="DOCVQA_SPLIT_DATA_REQUIRED",
            )
        best_priority = min(priority for priority, _path in matches)
        best = [path for priority, path in matches if priority == best_priority]
        if len(best) != 1:
            names = ", ".join(path.name for path in best)
            raise DatasetPreparationError(
                f"Multiple DocVQA metadata files match split {split!r}: {names}.",
                code="DOCVQA_SPLIT_DATA_REQUIRED",
            )
        return best[0]

    @staticmethod
    def _validate_split_value(value: Any, expected_split: str, metadata_path: Path) -> None:
        """Reject official metadata whose declared split conflicts with the request."""

        if value in (None, ""):
            return
        actual_split = canonical_docvqa_split(str(value))
        if actual_split != expected_split:
            raise DatasetPreparationError(
                f"DocVQA metadata {metadata_path.name} declares split {value!r}; "
                f"expected {expected_split!r}.",
                code="DOCVQA_SPLIT_DATA_REQUIRED",
            )

    @classmethod
    def _group_items(
        cls,
        items: list[dict[str, Any]],
        source_directory: Path,
        limit: int,
    ) -> "OrderedDict[str, list[dict[str, Any]]]":
        groups: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
        for item in items:
            image_ref = source_value(item, "image", "image_path", "img", "document_image", default=None)
            group_key = _string(
                source_value(
                    item,
                    "ucsf_document_id",
                    "document_id",
                    "doc_id",
                    "docId",
                    default=None,
                )
            ) or _string(image_ref)
            if group_key and group_key not in groups and len(groups) < limit:
                groups[group_key] = []

        for item in items:
            image_ref = source_value(item, "image", "image_path", "img", "document_image", default=None)
            group_key = _string(
                source_value(
                    item,
                    "ucsf_document_id",
                    "document_id",
                    "doc_id",
                    "docId",
                    default=None,
                )
            ) or _string(image_ref)
            if group_key not in groups:
                continue
            resolved = cls._resolve_image(image_ref, source_directory)
            item_copy = dict(item)
            item_copy["_resolved_image"] = resolved
            item_copy["_source_document_id"] = source_value(
                item,
                "ucsf_document_id",
                "document_id",
                "doc_id",
                "docId",
                default=None,
            )
            groups.setdefault(group_key, []).append(item_copy)
        return groups

    @staticmethod
    def _resolve_image(value: Any, source_directory: Path) -> Any:
        if isinstance(value, dict):
            value = value.get("path") or value.get("image") or value.get("bytes")
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if value in (None, ""):
            return None
        path = Path(str(value))
        candidates = [path if path.is_absolute() else source_directory / path]
        if path.name:
            candidates.extend(
                [
                    source_directory / path.name,
                    source_directory / "documents" / path.name,
                ]
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _qa_pair(item: dict[str, Any]) -> EvaluationQAPair:
        question_id = _string(source_value(item, "questionId", "question_id", "id", default=None))
        question = _string(source_value(item, "question", "question_text", default=None))
        answers_value = source_value(item, "answers", "answer", "ground_truth", default=[])
        answers = _answers(answers_value)
        if not question_id or not question or not answers:
            raise DatasetPreparationError("DocVQA record lacks question ID, question, or accepted answer.")
        page_number = source_value(
            item,
            "page",
            "page_no",
            "page_number",
            "ucsf_document_page_no",
            default=None,
        )
        try:
            page_number = int(page_number) if page_number is not None else None
        except (TypeError, ValueError):
            page_number = None
        metadata = {
            "source_fields": sorted(key for key in item if not key.startswith("_")),
            "metadata_file": safe_metadata(item.get("_metadata_file")),
        }
        for field_name, metadata_name in (
            ("question_types", "question_types"),
            ("question_type", "question_type"),
            ("data_split", "data_split"),
            ("docId", "doc_id"),
            ("ucsf_document_page_no", "source_document_page_no"),
        ):
            if field_name in item:
                metadata[metadata_name] = safe_metadata(item[field_name])
        return EvaluationQAPair(
            question_id=question_id,
            question=question,
            accepted_answers=answers,
            page_number=page_number if page_number and page_number >= 1 else None,
            source_document_id=_string(item.get("_source_document_id")),
            metadata=metadata,
        )


def _answers(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    answers: list[str] = []
    for item in values:
        candidate = item.get("answer") if isinstance(item, dict) else item
        if candidate not in (None, ""):
            text = str(candidate).strip()
            if text and text not in answers:
                answers.append(text)
    return answers


def _string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
