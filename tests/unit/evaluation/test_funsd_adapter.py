from __future__ import annotations

from pathlib import Path

from PIL import Image

from evaluation.datasets.base import PreparationOptions
from evaluation.datasets.funsd import FUNSDAdapter


def test_funsd_preserves_words_boxes_and_labels(tmp_path: Path) -> None:
    options = PreparationOptions(
        dataset="funsd",
        split="test",
        limit=1,
        output_root=tmp_path,
        extra={"ner_tag_names": {0: "O", 1: "question"}},
    )
    record = {
        "id": "form-1",
        "image": Image.new("RGB", (240, 120), "white"),
        "words": ["What", "is", "your", "name"],
        "bboxes": [[1, 2, 40, 20], [45, 2, 60, 20], [65, 2, 90, 20], [95, 2, 140, 20]],
        "ner_tags": [1, 0, 0, 1],
    }
    normalized = FUNSDAdapter().normalize_record(record, options, 0)
    assert normalized.page_count == 1
    assert Path(normalized.local_pdf_path).exists()
    assert [entity.text for entity in normalized.entities] == ["What", "is", "your", "name"]
    assert normalized.entities[0].label == "question"
    assert normalized.entities[-1].bounding_box is not None

