"""Controlled DocVQA ingestion and retrieval evaluation for Module E3."""

from evaluation.e3.metrics import (
    DEFAULT_E3_K_VALUES,
    aggregate_method_results,
    build_question_ground_truth,
    evaluate_ranked_ids,
    normalize_answer,
)
from evaluation.e3.models import (
    CorpusMapping,
    E3MethodSummary,
    E3Question,
    E3QuestionGroundTruth,
)

__all__ = [
    "CorpusMapping",
    "DEFAULT_E3_K_VALUES",
    "E3MethodSummary",
    "E3Question",
    "E3QuestionGroundTruth",
    "aggregate_method_results",
    "build_question_ground_truth",
    "evaluate_ranked_ids",
    "normalize_answer",
]
