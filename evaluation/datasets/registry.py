"""Registry for E1 dataset adapters."""

from __future__ import annotations

from evaluation.datasets.base import EvaluationDatasetAdapter
from evaluation.datasets.doclaynet import DocLayNetAdapter
from evaluation.datasets.docvqa import DocVQAAdapter
from evaluation.datasets.funsd import FUNSDAdapter


def adapter_registry() -> dict[str, EvaluationDatasetAdapter]:
    """Return fresh adapters keyed by the stable CLI dataset names."""

    return {
        "doclaynet": DocLayNetAdapter(),
        "funsd": FUNSDAdapter(),
        "docvqa": DocVQAAdapter(),
    }


def get_adapter(name: str) -> EvaluationDatasetAdapter:
    """Resolve one supported dataset name with a useful error."""

    adapters = adapter_registry()
    try:
        return adapters[name.casefold()]
    except KeyError as exc:
        supported = ", ".join(sorted(adapters))
        raise ValueError(f"Unsupported evaluation dataset {name!r}; choose one of: {supported}.") from exc

