"""Focused unit tests for the deterministic Module E2 metrics."""

from app.models.documents import LayoutElement, PageExtraction
from evaluation.e2.metrics import evaluate_layout, evaluate_text, levenshtein_distance, normalize_text
from evaluation.schemas import EvaluationBoundingBox, EvaluationLayoutRegion


def _region(label: str, box: list[float], index: int = 0) -> EvaluationLayoutRegion:
    return EvaluationLayoutRegion(
        page_number=1,
        label=label,
        bounding_box=EvaluationBoundingBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
        source_annotation_id=str(index),
    )


def _page(elements: list[LayoutElement]) -> PageExtraction:
    return PageExtraction(
        page_number=1,
        text="",
        character_count=0,
        has_native_text=True,
        needs_ocr=False,
        extraction_method="native",
        layout_elements=elements,
    )


def test_text_normalization_is_conservative_and_explicit() -> None:
    assert normalize_text("Ａ\r\nB\tC") == "A\nB\tC"
    assert normalize_text("Ａ\r\nB\tC", whitespace_normalized=True) == "A B C"


def test_levenshtein_distance_is_exact() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3


def test_text_scores_strict_and_whitespace_variants() -> None:
    score = evaluate_text("hello world", "hello\nworld")
    assert score.strict.cer > 0
    assert score.whitespace_normalized.cer == 0
    assert score.whitespace_normalized.wer == 0


def test_empty_reference_behavior_is_defined() -> None:
    empty = evaluate_text("", "")
    non_empty = evaluate_text("", "text")
    assert empty.strict.cer == 0
    assert empty.strict.wer == 0
    assert non_empty.strict.cer == 1
    assert non_empty.strict.wer == 1


def test_layout_matching_is_class_aware_and_one_to_one() -> None:
    result = evaluate_layout(
        [_region("Text", [0, 0, 50, 50]), _region("Section-header", [60, 0, 90, 20], 1)],
        [_page([
            LayoutElement(element_type="paragraph", text="body", bbox=[0, 0, 50, 50]),
            LayoutElement(element_type="paragraph", text="extra", bbox=[0, 0, 50, 50]),
            LayoutElement(element_type="heading", text="heading", bbox=[60, 0, 90, 20]),
        ])],
    )
    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert [match.ground_truth_index for match in result.matches] == [0, 1]


def test_layout_reports_unsupported_labels_instead_of_mapping_them() -> None:
    result = evaluate_layout(
        [_region("Picture", [0, 0, 50, 50])],
        [_page([LayoutElement(element_type="other", text="image", bbox=[0, 0, 50, 50])])],
    )
    assert result.true_positives == 0
    assert result.comparable_ground_truth == 0
    assert result.unsupported_ground_truth == 1
    assert result.unsupported_predictions == 1
    assert result.unsupported_ground_truth_labels == ("Picture",)
