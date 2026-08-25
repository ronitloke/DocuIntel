from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluation.manifests import stable_evaluation_id
from evaluation.schemas import (
    EvaluationBoundingBox,
    EvaluationDocument,
    EvaluationPage,
)


def test_evaluation_ids_are_stable_and_source_specific() -> None:
    first = stable_evaluation_id("funsd", "test", "sample-1")
    second = stable_evaluation_id("funsd", "test", "sample-1")
    assert first == second
    assert first != stable_evaluation_id("funsd", "train", "sample-1")


def test_schema_rejects_invalid_geometry_and_page_count() -> None:
    with pytest.raises(ValidationError):
        EvaluationBoundingBox(x0=10, y0=10, x1=5, y1=20)
    with pytest.raises(ValidationError):
        EvaluationDocument(
            evaluation_id="doc-1",
            dataset="funsd",
            split="test",
            source_record_id="1",
            local_pdf_path="data/evaluation/processed/funsd/1.pdf",
            page_count=2,
            pages=[EvaluationPage(page_number=1, width=100, height=100)],
        )

