"""Load and validate human-editable evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.models import EvaluationCase, EvaluationDataset


def load_dataset(path: Path) -> EvaluationDataset:
    """Load a JSON dataset or JSONL case file and validate it with Pydantic."""

    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return EvaluationDataset(name=path.stem, cases=[EvaluationCase.model_validate(item) for item in cases])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return EvaluationDataset(name=path.stem, cases=payload)
    return EvaluationDataset.model_validate(payload)
