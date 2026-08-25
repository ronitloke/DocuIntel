from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from evaluation.datasets.base import PreparationOptions
from evaluation.datasets.docvqa import DocVQAAdapter


def test_docvqa_reads_local_official_style_json_and_materializes_pdf(tmp_path: Path) -> None:
    source = tmp_path / "docvqa"
    source.mkdir()
    Image.new("RGB", (180, 100), "white").save(source / "page.png")
    (source / "test_v1.0.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "questionId": "q-1",
                        "question": "What is shown?",
                        "answers": [{"answer": "a form"}, {"answer": "a form"}],
                        "image": "page.png",
                        "ucsf_document_id": "doc-1",
                        "page": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    options = PreparationOptions(
        dataset="docvqa",
        split="test",
        limit=1,
        output_root=tmp_path / "processed",
        source_directory=source,
    )
    result = DocVQAAdapter().prepare(options)
    assert result.prepared == 1
    assert result.failed == 0
    manifest = Path(result.manifest_path)
    assert manifest.exists() or Path(result.output_directory, "manifest.jsonl").exists()


def test_docvqa_missing_source_returns_controlled_status(tmp_path: Path) -> None:
    options = PreparationOptions(
        dataset="docvqa",
        split="test",
        limit=5,
        output_root=tmp_path / "processed",
        source_directory=tmp_path / "missing-docvqa",
    )
    result = DocVQAAdapter().prepare(options)
    assert result.prepared == 0
    assert result.failed == 1
    assert "DOCVQA_DATA_REQUIRED" in result.errors[0]


def test_docvqa_limit_bounds_unique_document_groups(tmp_path: Path) -> None:
    source = tmp_path / "docvqa"
    source.mkdir()
    for name in ("one.png", "two.png"):
        Image.new("RGB", (20, 20), "white").save(source / name)
    items = [
        {"image": "one.png", "document_id": "one"},
        {"image": "two.png", "document_id": "two"},
    ]
    groups = DocVQAAdapter._group_items(items, source, limit=1)
    assert list(groups) == ["one"]


def _write_official_docvqa_file(
    source: Path,
    filename: str,
    split: str,
    question_id: int,
    image_name: str = "page.png",
) -> None:
    (source / filename).write_text(
        json.dumps(
            {
                "dataset_name": "SP-DocVQA",
                "dataset_version": "1.0",
                "dataset_split": split,
                "data": [
                    {
                        "questionId": question_id,
                        "question": "What is shown?",
                        "question_types": ["layout", "form"],
                        "image": f"documents/{image_name}",
                        "docId": question_id,
                        "ucsf_document_id": f"doc-{question_id}",
                        "ucsf_document_page_no": "7",
                        "answers": ["a form", "A FORM", "a form"],
                        "data_split": split,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_validation_alias_selects_val_file_not_train(tmp_path: Path) -> None:
    source = tmp_path / "docvqa"
    source.mkdir()
    _write_official_docvqa_file(source, "val_v1.0_withQT.json", "val", 1)
    _write_official_docvqa_file(source, "train_v1.0_withQT.json", "train", 2)
    items = DocVQAAdapter._load_items(source, "validation")
    assert [item["questionId"] for item in items] == [1]
    assert items[0]["_metadata_file"].endswith("val_v1.0_withQT.json")


def test_train_alias_selects_train_file(tmp_path: Path) -> None:
    source = tmp_path / "docvqa"
    source.mkdir()
    _write_official_docvqa_file(source, "val_v1.0_withQT.json", "val", 1)
    _write_official_docvqa_file(source, "train_v1.0_withQT.json", "train", 2)
    items = DocVQAAdapter._load_items(source, "train")
    assert [item["questionId"] for item in items] == [2]


def test_test_alias_selects_test_file(tmp_path: Path) -> None:
    source = tmp_path / "docvqa"
    source.mkdir()
    _write_official_docvqa_file(source, "test_v1.0.json", "test", 3)
    items = DocVQAAdapter._load_items(source, "test")
    assert [item["questionId"] for item in items] == [3]


def test_official_docvqa_fields_answers_page_and_image_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "docvqa"
    source.mkdir()
    image = source / "page.png"
    Image.new("RGB", (180, 100), "white").save(image)
    _write_official_docvqa_file(source, "val_v1.0_withQT.json", "val", 11)

    options = PreparationOptions(
        dataset="docvqa",
        split="validation",
        limit=1,
        output_root=tmp_path / "processed",
        source_directory=source,
    )
    result = DocVQAAdapter().prepare(options)
    assert result.prepared == 1
    manifest = Path(result.manifest_path)
    record = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert record["split"] == "validation"
    assert record["source_document_id"] == "doc-11"
    assert record["qa_pairs"][0]["question_id"] == "11"
    assert record["qa_pairs"][0]["accepted_answers"] == ["a form", "A FORM"]
    assert record["qa_pairs"][0]["page_number"] == 7
    assert record["qa_pairs"][0]["metadata"]["question_types"] == ["layout", "form"]
    assert record["pages"][0]["source_image_path"]
    assert Path(record["pages"][0]["source_image_path"]).resolve() == image.resolve()


def test_missing_validation_metadata_is_controlled_even_when_other_splits_exist(tmp_path: Path) -> None:
    source = tmp_path / "docvqa"
    source.mkdir()
    _write_official_docvqa_file(source, "train_v1.0_withQT.json", "train", 2)
    _write_official_docvqa_file(source, "test_v1.0.json", "test", 3)
    options = PreparationOptions(
        dataset="docvqa",
        split="validation",
        limit=1,
        output_root=tmp_path / "processed",
        source_directory=source,
    )
    result = DocVQAAdapter().prepare(options)
    assert result.prepared == 0
    assert result.failed == 1
    assert "DOCVQA_SPLIT_DATA_REQUIRED" in result.errors[0]
