"""Focused artifact and direct-runner tests for Module E2."""

import json
from pathlib import Path
from uuid import uuid4

import pymupdf

from app.models.documents import DocumentIngestionResponse, DocumentStatus, LayoutElement, PageExtraction, PDFMetadata
from evaluation.e2 import runner
from evaluation.manifests import write_manifest
from evaluation.schemas import EvaluationBoundingBox, EvaluationDocument, EvaluationLayoutRegion, EvaluationPage


def _pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=100, height=100)
    page.insert_text((10, 20), "sample text")
    document.save(path)
    document.close()


def _document(pdf_path: Path) -> EvaluationDocument:
    return EvaluationDocument(
        evaluation_id="doclaynet-test",
        dataset="doclaynet",
        split="validation",
        source_record_id="record-1",
        local_pdf_path=str(pdf_path),
        page_count=1,
        pages=[EvaluationPage(page_number=1, width=100, height=100)],
        layout_regions=[
            EvaluationLayoutRegion(
                page_number=1,
                label="Text",
                bounding_box=EvaluationBoundingBox(x0=0, y0=0, x1=50, y1=50),
            )
        ],
    )


class _UnavailableOCR:
    resolved_command = None

    def is_available(self) -> bool:
        return False


class _FakeIngestionService:
    def __init__(self, *, settings, storage_directory: Path) -> None:
        self.storage_directory = storage_directory
        self.ocr_service = _UnavailableOCR()

    async def ingest(self, upload) -> DocumentIngestionResponse:
        return DocumentIngestionResponse(
            document_id=uuid4(),
            original_filename=upload.filename,
            stored_filename="generated.pdf",
            mime_type="application/pdf",
            file_size_bytes=1,
            checksum_sha256="a" * 64,
            page_count=1,
            pages_with_native_text=1,
            pages_requiring_ocr=0,
            pages_processed_by_ocr=0,
            unresolved_ocr_pages=0,
            status=DocumentStatus.READY,
            metadata=PDFMetadata(
                title=None,
                author=None,
                subject=None,
                keywords=None,
                creator=None,
                producer=None,
                creation_date=None,
                modification_date=None,
            ),
            pages=[
                PageExtraction(
                    page_number=1,
                    text="sample text",
                    character_count=11,
                    has_native_text=True,
                    needs_ocr=False,
                    extraction_method="native",
                    layout_elements=[
                        LayoutElement(element_type="paragraph", text="sample text", bbox=[0, 0, 50, 50])
                    ],
                )
            ],
        )


def test_e2_runner_writes_all_required_artifacts(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _pdf(pdf_path)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest([_document(pdf_path)], manifest_path)
    monkeypatch.setattr(runner, "PDFIngestionService", _FakeIngestionService)

    import asyncio

    output_directory = tmp_path / "results"
    asyncio.run(
        runner.run_e2(
            manifest_path,
            output_directory,
            expected_dataset="doclaynet",
            expected_split="validation",
        )
    )

    expected = {"summary.json", "per_document.jsonl", "metrics.csv", "report.md", "run_metadata.json"}
    assert expected <= {path.name for path in output_directory.iterdir()}
    summary = json.loads((output_directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["reliability"]["processing_success_rate"] == 1
    assert summary["layout_accuracy"]["true_positives"] == 1
    assert "PDFIngestionService" in (output_directory / "report.md").read_text(encoding="utf-8")


def test_doclaynet_boxes_use_preserved_source_dimensions(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _pdf(pdf_path)
    record = _document(pdf_path).model_copy(
        update={"metadata": {"source_metadata": {"coco_width": 1000, "coco_height": 1000}}}
    )
    transformed = runner._layout_regions_in_pdf_space(record)
    box = transformed[0].bounding_box
    assert box.x1 == 5
    assert box.y1 == 5
