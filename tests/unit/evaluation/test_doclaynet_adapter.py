from __future__ import annotations

from pathlib import Path

from PIL import Image

from evaluation.datasets.base import PreparationOptions
from evaluation.datasets.doclaynet import DocLayNetAdapter


def test_doclaynet_normalizes_layout_ground_truth_without_prediction_claims(tmp_path: Path) -> None:
    image = Image.new("RGB", (200, 100), "white")
    options = PreparationOptions(dataset="doclaynet", split="validation", limit=1, output_root=tmp_path)
    record = {
        "id": "doclaynet-1",
        "image": image,
        "document_id": "source-document-1",
        "annotations": [
            {"id": "region-1", "category": "Title", "bbox": [10, 20, 80, 15]},
        ],
    }
    normalized = DocLayNetAdapter().normalize_record(record, options, 0)
    assert normalized.dataset == "doclaynet"
    assert normalized.page_count == 1
    assert Path(normalized.local_pdf_path).is_file() or Path(normalized.local_pdf_path).exists()
    assert normalized.layout_regions[0].label == "Title"
    assert normalized.layout_regions[0].bounding_box.x1 == 90
    assert normalized.metadata["materialization"] == "image_to_pdf"

