"""Read-only presentation of the authoritative Module E5 package."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

from streamlit_app.config import PROJECT_ROOT, get_settings


DEFAULT_E5_RESULTS_DIRECTORY = (
    PROJECT_ROOT / "data" / "evaluation" / "results" / "e5" / "final_baseline_20260821_final"
)
REQUIRED_FILES = (
    "summary.json",
    "scorecard.json",
    "retrieval_comparison.csv",
    "limitations.json",
)

LIMITATION_DISPLAY_TITLES = {
    "bounded_e2": "Bounded evaluation sample",
    "doclaynet_failure": "Document processing coverage",
    "ocr_error": "OCR quality",
    "layout_quality": "Layout quality",
    "answer_indexability": "Answer indexability",
    "upload_limit": "Document size limit",
    "conditional_retrieval": "Retrieval evaluation scope",
    "reranking_latency": "Reranking latency",
    "ollama_runtime": "Local LLM runtime",
    "answer_format": "Answer-format evaluation",
    "citation_denominators": "Citation sample size",
    "e41_blocked": "Extended runtime diagnostic",
    "no_global_score": "Metric interpretation",
}


class E5UiArtifactError(RuntimeError):
    """Raised when the explicit E5 package cannot be displayed safely."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E5UiArtifactError(f"Could not read evaluation artifact {path.name}.") from exc
    if not isinstance(value, dict):
        raise E5UiArtifactError(f"Evaluation artifact {path.name} is not a JSON object.")
    return value


def _number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise E5UiArtifactError("The retrieval comparison contains a non-numeric metric.") from exc


def _load_e5_package_cached(directory: str) -> dict[str, Any]:
    root = Path(directory)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise E5UiArtifactError(
            "The authoritative E5 evaluation package is unavailable "
            f"(missing: {', '.join(missing)})."
        )
    summary = _read_json(root / "summary.json")
    scorecard = _read_json(root / "scorecard.json")
    limitations = _read_json(root / "limitations.json")
    records = scorecard.get("records")
    limitation_rows = limitations.get("limitations")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise E5UiArtifactError("The E5 scorecard does not contain valid metric records.")
    if not isinstance(limitation_rows, list) or not all(isinstance(item, dict) for item in limitation_rows):
        raise E5UiArtifactError("The E5 limitations artifact does not contain valid records.")
    try:
        with (root / "retrieval_comparison.csv").open("r", encoding="utf-8", newline="") as handle:
            retrieval = [
                {
                    key: _number(value) if key != "method" else value
                    for key, value in row.items()
                    if value is not None
                }
                for row in csv.DictReader(handle)
            ]
    except (OSError, csv.Error) as exc:
        raise E5UiArtifactError("Could not read the retrieval comparison artifact.") from exc
    if not retrieval:
        raise E5UiArtifactError("The E5 retrieval comparison is empty.")
    return {
        "root": str(root),
        "summary": summary,
        "scorecard": records,
        "retrieval": retrieval,
        "limitations": limitation_rows,
    }


def load_e5_package(directory: str | Path = DEFAULT_E5_RESULTS_DIRECTORY) -> dict[str, Any]:
    """Load the fixed E5 run package; never choose a latest/other run implicitly."""

    return _load_e5_package_cached(str(Path(directory).expanduser().resolve()))


def find_scorecard_metric(
    package: dict[str, Any],
    metric: str,
    *,
    module: str | None = None,
    dataset: str | None = None,
) -> dict[str, Any] | None:
    """Find a scorecard record while retaining its measurement status."""

    for record in package.get("scorecard", []):
        if record.get("metric") != metric:
            continue
        if module is not None and record.get("module") != module:
            continue
        if dataset is not None and record.get("dataset") != dataset:
            continue
        return record
    return None


def format_e5_metric(
    record: dict[str, Any] | None,
    *,
    percent: bool = False,
    decimals: int = 3,
) -> str:
    """Format a measured scorecard record without hiding blocked states."""

    if record is None:
        return "—"
    status = record.get("status", "NOT_MEASURED")
    if status != "MEASURED":
        return str(status)
    value = record.get("value")
    if value is None:
        return "—"
    numeric = float(value)
    if percent:
        return f"{numeric * 100:.{decimals}f}%"
    return f"{numeric:.{decimals}f}"


def retrieval_rows(package: dict[str, Any]) -> list[dict[str, Any]]:
    """Return retrieval rows in the artifact's stable comparable order."""

    labels = {
        "keyword": "Keyword",
        "semantic": "Semantic",
        "hybrid": "Hybrid",
        "hybrid_reranked": "Hybrid + reranker",
    }
    return [
        {**row, "label": labels.get(str(row.get("method")), str(row.get("method")))}
        for row in package.get("retrieval", [])
    ]


def public_provenance(package: dict[str, Any]) -> dict[str, str]:
    """Return repository-relative E5 provenance safe for the public UI."""

    summary = package.get("summary", {})
    sources = summary.get("authoritative_sources", {})
    dataset_labels = {"funsd": "FUNSD", "doclaynet": "DocLayNet", "docvqa": "DocVQA"}
    datasets = {
        dataset_labels.get(str(item.get("dataset", "")).casefold(), str(item.get("dataset", "")))
        for item in sources.values()
        if isinstance(item, dict)
    }
    dataset_order = {"FUNSD": 0, "DocLayNet": 1, "DocVQA": 2}
    splits = {str(item.get("split", "")) for item in sources.values() if isinstance(item, dict)}
    return {
        "Run": str(summary.get("run_id", "—")),
        "Datasets": ", ".join(sorted((dataset for dataset in datasets if dataset), key=lambda value: dataset_order.get(value, 99))) or "—",
        "DocVQA split": "validation" if "validation" in splits else ", ".join(sorted(splits)) or "—",
        "Manifest": "evaluation/e5/baseline_manifest.json",
    }


def limitation_display_title(limitation_id: Any) -> str:
    """Return a human-readable limitation heading without exposing E5 identifiers."""

    key = str(limitation_id or "").strip().casefold()
    return LIMITATION_DISPLAY_TITLES.get(key, "Additional evaluation limitation")


def _metric_value(package: dict[str, Any], metric: str, *, method: str) -> float | None:
    for row in retrieval_rows(package):
        if row.get("method") == method:
            value = row.get(metric)
            return float(value) if value is not None else None
    return None


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _retrieval_table(package: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "Configuration": row["label"],
            "Recall@1": _percent(row.get("recall_at_1")),
            "Recall@3": _percent(row.get("recall_at_3")),
            "Recall@5": _percent(row.get("recall_at_5")),
            "Recall@10": _percent(row.get("recall_at_10")),
            "MRR": f"{row['mrr']:.3f}",
            "Median retrieval (ms)": f"{row['retrieval_median_ms']:.1f}",
            "Median total (ms)": f"{row['total_pipeline_median_ms']:.1f}",
            "Mean rerank (ms)": (
                f"{row['reranking_mean_ms']:.1f}" if row.get("reranking_mean_ms") is not None else "—"
            ),
        }
        for row in retrieval_rows(package)
    ]


def _metric_card(record: dict[str, Any] | None, label: str, *, percent: bool = False) -> None:
    st.metric(label, format_e5_metric(record, percent=percent))
    if record and record.get("denominator"):
        st.caption(str(record["denominator"]))


def render_evaluation() -> None:
    """Render E5 metrics without rerunning or mutating the evaluation."""

    st.title("Evaluation")
    st.caption("Read-only view of the authoritative E5 benchmark package.")
    settings = get_settings()
    try:
        package = load_e5_package(settings.e5_results_directory)
    except E5UiArtifactError as exc:
        st.error(str(exc))
        st.caption("Expected the configured final E5 evaluation package.")
        return

    summary = package["summary"]
    status_counts = summary.get("measurement_status_counts", {})
    with st.container(border=True):
        st.subheader("Evaluation overview")
        st.caption(
            "This page reads stored results only. It does not start benchmark jobs, call the database, "
            "or contact Ollama."
        )
        columns = st.columns(4)
        columns[0].metric("Documents prepared", summary.get("documents_prepared", "—"))
        columns[1].metric("Documents indexed", summary.get("documents_indexed", "—"))
        columns[2].metric("Scorable questions", summary.get("questions_scorable", "—"))
        columns[3].metric("Measured records", status_counts.get("MEASURED", "—"))
        st.caption(
            "Measurement states: "
            f"{status_counts.get('MEASURED', 0)} measured · "
            f"{status_counts.get('BLOCKED', 0)} blocked · "
            f"{status_counts.get('NOT_APPLICABLE', 0)} not applicable."
        )

    understanding_tab, retrieval_tab, rag_tab, limitations_tab = st.tabs(
        ["Document understanding", "Retrieval", "RAG reliability", "Limitations & provenance"]
    )
    with understanding_tab:
        st.subheader("OCR and layout")
        st.caption("Percentages are displayed from the stored E2 scorecard; they are not a single global accuracy.")
        funsd_cols = st.columns(2)
        with funsd_cols[0].container(border=True):
            st.markdown("**FUNSD OCR**")
            _metric_card(find_scorecard_metric(package, "ocr.cer", module="E2", dataset="funsd"), "CER")
        with funsd_cols[1].container(border=True):
            st.markdown("**FUNSD OCR**")
            _metric_card(find_scorecard_metric(package, "ocr.wer", module="E2", dataset="funsd"), "WER")
        layout_metrics = (
            ("layout.precision", "Precision"),
            ("layout.recall", "Recall"),
            ("layout.f1", "F1"),
            ("layout.mean_matched_iou", "Mean matched IoU"),
            ("layout.processing_success_rate", "Processing success"),
        )
        layout_columns = st.columns(len(layout_metrics))
        for column, (metric, label) in zip(layout_columns, layout_metrics, strict=True):
            with column:
                _metric_card(find_scorecard_metric(package, metric, module="E2", dataset="doclaynet"), label, percent=True)

    with retrieval_tab:
        st.subheader("Retrieval quality and latency")
        st.caption("The comparison uses the same controlled DocVQA evaluation cases and denominators for each configuration.")
        st.dataframe(_retrieval_table(package), hide_index=True)
        rows = retrieval_rows(package)
        chart_columns = st.columns(3)
        with chart_columns[0]:
            st.markdown("**Recall@1**")
            st.bar_chart({row["label"]: row["recall_at_1"] for row in rows}, y_label="Recall")
        with chart_columns[1]:
            st.markdown("**MRR**")
            st.bar_chart({row["label"]: row["mrr"] for row in rows}, y_label="MRR")
        with chart_columns[2]:
            st.markdown("**Median total latency**")
            st.bar_chart(
                {row["label"]: row["total_pipeline_median_ms"] for row in rows},
                y_label="Milliseconds",
            )

        hybrid = next((row for row in rows if row.get("method") == "hybrid"), None)
        reranked = next((row for row in rows if row.get("method") == "hybrid_reranked"), None)
        if hybrid and reranked:
            with st.container(border=True):
                st.subheader("Reranking impact")
                impact_columns = st.columns(4)
                impact_columns[0].metric("Recall@1 delta", f"{(reranked['recall_at_1'] - hybrid['recall_at_1']) * 100:.1f} pp")
                impact_columns[1].metric("Recall@5 delta", f"{(reranked['recall_at_5'] - hybrid['recall_at_5']) * 100:.1f} pp")
                impact_columns[2].metric("MRR delta", f"{reranked['mrr'] - hybrid['mrr']:.3f}")
                impact_columns[3].metric(
                    "Median rerank cost",
                    f"{reranked['reranking_median_ms']:.0f} ms",
                )
                st.caption("Reranking improves measured retrieval quality in this bounded benchmark, with a substantial CPU latency cost.")

    with rag_tab:
        st.subheader("End-to-end answer reliability")
        st.warning("CURRENT LOCAL CPU RUNTIME LIMITATION · E4 generation completed very few questions because CPU-mode Ollama timed out frequently.")
        rag_columns = st.columns(2)
        for column, method, title in (
            (rag_columns[0], "hybrid", "Hybrid"),
            (rag_columns[1], "hybrid_reranked", "Hybrid + reranker"),
        ):
            with column.container(border=True):
                st.markdown(f"**{title}**")
                answered = find_scorecard_metric(package, f"{method}.questions_answered", module="E4", dataset="docvqa")
                coverage = find_scorecard_metric(package, f"{method}.answer_coverage_rate", module="E4", dataset="docvqa")
                _metric_card(answered, "Questions answered")
                _metric_card(coverage, "Answer coverage", percent=True)
        st.caption("These are bounded E4 reliability measurements, not a general answer-accuracy score.")

    with limitations_tab:
        st.subheader("Known limitations")
        for limitation in package["limitations"]:
            with st.expander(limitation_display_title(limitation.get("id"))):
                st.write(limitation.get("text") or "No limitation detail was recorded.")
                st.caption(f"Source: {limitation.get('source', 'E5 artifact')}")
        with st.expander("Artifact provenance"):
            st.json(public_provenance(package))


__all__ = [
    "DEFAULT_E5_RESULTS_DIRECTORY",
    "E5UiArtifactError",
    "find_scorecard_metric",
    "format_e5_metric",
    "load_e5_package",
    "limitation_display_title",
    "public_provenance",
    "render_evaluation",
    "retrieval_rows",
]
