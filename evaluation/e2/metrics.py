"""Deterministic text and layout metrics used by the E2 benchmark."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from app.models.documents import LayoutElement, PageExtraction
from evaluation.schemas import EvaluationLayoutRegion


TEXT_NORMALIZATION_RULES = (
    "Unicode NFKC normalization, CRLF/CR to LF conversion, and no case, "
    "punctuation, number, spelling, or semantic normalization."
)
WHITESPACE_NORMALIZATION_RULES = (
    "The strict form followed by collapsing every whitespace run to one ASCII "
    "space and stripping leading/trailing spaces."
)


def normalize_text(text: str, *, whitespace_normalized: bool = False) -> str:
    """Apply the benchmark's conservative, deterministic text normalization."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if whitespace_normalized:
        normalized = " ".join(normalized.split())
    return normalized


def levenshtein_distance(source: str, target: str) -> int:
    """Return the exact insertion/deletion/substitution edit distance."""

    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)
    if len(source) > len(target):
        source, target = target, source
    previous = list(range(len(source) + 1))
    for target_index, target_character in enumerate(target, start=1):
        current = [target_index]
        for source_index, source_character in enumerate(source, start=1):
            insertion = current[source_index - 1] + 1
            deletion = previous[source_index] + 1
            substitution = previous[source_index - 1] + (source_character != target_character)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _rate(distance: int, reference_length: int) -> float:
    """Define the empty-reference case instead of silently dividing by zero."""

    if reference_length == 0:
        return 0.0 if distance == 0 else 1.0
    return distance / reference_length


@dataclass(frozen=True, slots=True)
class TextScore:
    """One CER/WER measurement for one normalization policy."""

    character_edit_distance: int
    word_edit_distance: int
    reference_characters: int
    reference_words: int
    predicted_characters: int
    predicted_words: int
    cer: float
    wer: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metric data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class TextEvaluation:
    """Strict and whitespace-normalized text accuracy measurements."""

    strict: TextScore
    whitespace_normalized: TextScore

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metric data."""

        return {
            "strict": self.strict.to_dict(),
            "whitespace_normalized": self.whitespace_normalized.to_dict(),
            "normalization_rules": TEXT_NORMALIZATION_RULES,
            "whitespace_normalization_rules": WHITESPACE_NORMALIZATION_RULES,
        }


def _score_variant(reference: str, prediction: str, *, whitespace_normalized: bool) -> TextScore:
    normalized_reference = normalize_text(reference, whitespace_normalized=whitespace_normalized)
    normalized_prediction = normalize_text(prediction, whitespace_normalized=whitespace_normalized)
    reference_words = normalized_reference.split()
    predicted_words = normalized_prediction.split()
    word_distance = _levenshtein_tokens(reference_words, predicted_words)
    character_distance = levenshtein_distance(normalized_reference, normalized_prediction)
    return TextScore(
        character_edit_distance=character_distance,
        word_edit_distance=word_distance,
        reference_characters=len(normalized_reference),
        reference_words=len(reference_words),
        predicted_characters=len(normalized_prediction),
        predicted_words=len(predicted_words),
        cer=_rate(character_distance, len(normalized_reference)),
        wer=_rate(word_distance, len(reference_words)),
    )


def _levenshtein_tokens(source: list[str], target: list[str]) -> int:
    """Calculate Levenshtein distance for token sequences."""

    if source == target:
        return 0
    if len(source) > len(target):
        source, target = target, source
    previous = list(range(len(source) + 1))
    for target_index, target_token in enumerate(target, start=1):
        current = [target_index]
        for source_index, source_token in enumerate(source, start=1):
            current.append(
                min(
                    current[source_index - 1] + 1,
                    previous[source_index] + 1,
                    previous[source_index - 1] + (source_token != target_token),
                )
            )
        previous = current
    return previous[-1]


def evaluate_text(reference: str, prediction: str) -> TextEvaluation:
    """Evaluate text with both strict and whitespace-normalized views."""

    return TextEvaluation(
        strict=_score_variant(reference, prediction, whitespace_normalized=False),
        whitespace_normalized=_score_variant(reference, prediction, whitespace_normalized=True),
    )


LAYOUT_LABEL_MAPPING: dict[str, str] = {
    "heading": "Section-header",
    "paragraph": "Text",
    "list_item": "List-item",
    "table": "Table",
}


@dataclass(frozen=True, slots=True)
class LayoutMatch:
    """One deterministic ground-truth/prediction match."""

    page_number: int
    ground_truth_index: int
    prediction_index: int
    label: str
    iou: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible match data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LayoutClassEvaluation:
    """Counts and derived metrics for one comparable layout class."""

    label: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible class metrics."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LayoutEvaluation:
    """One-to-one IoU-based layout evaluation."""

    iou_threshold: float
    comparable_ground_truth: int
    comparable_predictions: int
    unsupported_ground_truth: int
    unsupported_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    mean_matched_iou: float | None
    matches: tuple[LayoutMatch, ...] = field(default_factory=tuple)
    per_class: tuple[LayoutClassEvaluation, ...] = field(default_factory=tuple)
    unsupported_ground_truth_labels: tuple[str, ...] = field(default_factory=tuple)
    unsupported_prediction_labels: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible layout metrics."""

        return {
            "iou_threshold": self.iou_threshold,
            "comparable_ground_truth": self.comparable_ground_truth,
            "comparable_predictions": self.comparable_predictions,
            "unsupported_ground_truth": self.unsupported_ground_truth,
            "unsupported_predictions": self.unsupported_predictions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_matched_iou": self.mean_matched_iou,
            "matches": [match.to_dict() for match in self.matches],
            "per_class": [metric.to_dict() for metric in self.per_class],
            "unsupported_ground_truth_labels": list(self.unsupported_ground_truth_labels),
            "unsupported_prediction_labels": list(self.unsupported_prediction_labels),
            "label_mapping": LAYOUT_LABEL_MAPPING,
        }


def _bbox_iou(left: list[float], right: list[float]) -> float:
    intersection_left = max(left[0], right[0])
    intersection_top = max(left[1], right[1])
    intersection_right = min(left[2], right[2])
    intersection_bottom = min(left[3], right[3])
    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _evaluation_bbox_values(region: EvaluationLayoutRegion) -> list[float]:
    """Convert the normalized Pydantic box to the application's list format."""

    box = region.bounding_box
    return [box.x0, box.y0, box.x1, box.y1]


def _derived_metric(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None if precision is None or recall is None else 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_layout(
    ground_truth: Iterable[EvaluationLayoutRegion],
    pages: Iterable[PageExtraction],
    *,
    iou_threshold: float = 0.5,
) -> LayoutEvaluation:
    """Match comparable layout classes with deterministic greedy one-to-one IoU."""

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than zero and at most one.")
    gt_regions = list(ground_truth)
    predictions: list[tuple[int, int, LayoutElement]] = [
        (page.page_number, index, element)
        for page in pages
        for index, element in enumerate(page.layout_elements)
    ]
    comparable_gt = [
        (index, region)
        for index, region in enumerate(gt_regions)
        if region.label in set(LAYOUT_LABEL_MAPPING.values())
    ]
    comparable_predictions = [
        (page_number, index, element)
        for page_number, index, element in predictions
        if element.element_type in LAYOUT_LABEL_MAPPING
    ]
    candidates: list[tuple[float, int, int, int, int, str]] = []
    for gt_index, region in comparable_gt:
        for page_number, prediction_index, element in comparable_predictions:
            if page_number != region.page_number:
                continue
            if LAYOUT_LABEL_MAPPING[element.element_type] != region.label:
                continue
            overlap = _bbox_iou(_evaluation_bbox_values(region), element.bbox)
            if overlap >= iou_threshold:
                candidates.append(
                    (-overlap, region.page_number, gt_index, prediction_index, 0, region.label)
                )
    candidates.sort()
    matched_gt: set[int] = set()
    matched_predictions: set[tuple[int, int]] = set()
    matches: list[LayoutMatch] = []
    for negative_iou, page_number, gt_index, prediction_index, _, label in candidates:
        prediction_key = (page_number, prediction_index)
        if gt_index in matched_gt or prediction_key in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(prediction_key)
        matches.append(
            LayoutMatch(
                page_number=page_number,
                ground_truth_index=gt_index,
                prediction_index=prediction_index,
                label=label,
                iou=-negative_iou,
            )
        )

    matched_prediction_count = len(matched_predictions)
    true_positives = len(matches)
    false_negatives = len(comparable_gt) - true_positives
    false_positives = len(comparable_predictions) - matched_prediction_count
    precision = _derived_metric(true_positives, true_positives + false_positives)
    recall = _derived_metric(true_positives, true_positives + false_negatives)
    per_class: list[LayoutClassEvaluation] = []
    for label in sorted(set(LAYOUT_LABEL_MAPPING.values())):
        gt_count = sum(region.label == label for _, region in comparable_gt)
        prediction_count = sum(
            LAYOUT_LABEL_MAPPING[element.element_type] == label
            for _, _, element in comparable_predictions
        )
        tp = sum(match.label == label for match in matches)
        fp = prediction_count - tp
        fn = gt_count - tp
        class_precision = _derived_metric(tp, tp + fp)
        class_recall = _derived_metric(tp, tp + fn)
        per_class.append(
            LayoutClassEvaluation(
                label=label,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=class_precision,
                recall=class_recall,
                f1=_f1(class_precision, class_recall),
            )
        )
    matched_ious = [match.iou for match in matches]
    unsupported_gt_labels = tuple(
        sorted({region.label for region in gt_regions if region.label not in set(LAYOUT_LABEL_MAPPING.values())})
    )
    unsupported_prediction_labels = tuple(
        sorted({element.element_type for _, _, element in predictions if element.element_type not in LAYOUT_LABEL_MAPPING})
    )
    return LayoutEvaluation(
        iou_threshold=iou_threshold,
        comparable_ground_truth=len(comparable_gt),
        comparable_predictions=len(comparable_predictions),
        unsupported_ground_truth=len(gt_regions) - len(comparable_gt),
        unsupported_predictions=len(predictions) - len(comparable_predictions),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        mean_matched_iou=sum(matched_ious) / len(matched_ious) if matched_ious else None,
        matches=tuple(sorted(matches, key=lambda match: (match.page_number, match.ground_truth_index))),
        per_class=tuple(per_class),
        unsupported_ground_truth_labels=unsupported_gt_labels,
        unsupported_prediction_labels=unsupported_prediction_labels,
    )
