from __future__ import annotations

import json
from pathlib import Path

import pymupdf as fitz
import pytest

from evaluation.manifests import stable_evaluation_id, write_manifest
from evaluation.schemas import EvaluationDocument, EvaluationPage
from evaluation.validation import ManifestValidationError, validate_manifest
from scripts.evaluation_inspect import main as inspect_manifest_main


def _document(path: str, source_id: str) -> EvaluationDocument:
    return EvaluationDocument(
        evaluation_id=stable_evaluation_id("funsd", "test", source_id),
        dataset="funsd",
        split="test",
        source_record_id=source_id,
        local_pdf_path=path,
        page_count=1,
        pages=[EvaluationPage(page_number=1, width=100, height=100)],
    )


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    document.new_page(width=100, height=100)
    document.save(path)
    document.close()


def test_manifest_is_sorted_and_validates_referenced_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "one.pdf"
    _write_pdf(pdf)
    manifest = tmp_path / "manifest.jsonl"
    records = [_document(str(pdf), "b"), _document(str(pdf), "a")]
    write_manifest(records, manifest)
    lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [line["evaluation_id"] for line in lines] == sorted(line["evaluation_id"] for line in lines)
    result = validate_manifest(manifest)
    assert result.statistics.documents == 2
    assert result.statistics.pages == 2


def test_manifest_rejects_duplicate_ids_and_missing_pdf(tmp_path: Path) -> None:
    record = _document(str(tmp_path / "missing.pdf"), "same")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record.model_dump(mode="json")) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError, match="Duplicate"):
        validate_manifest(manifest)

    write_manifest([record], manifest)
    with pytest.raises(ManifestValidationError, match="does not exist"):
        validate_manifest(manifest)


def test_manifest_rejects_malformed_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="line 1"):
        validate_manifest(manifest)


def test_inspector_reports_empty_manifest_as_not_usable(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("", encoding="utf-8")
    assert inspect_manifest_main(["--manifest", str(manifest)]) == 2
    output = capsys.readouterr().out
    assert "Status: empty / not usable" in output
    assert "Status: valid" not in output
