"""JSON report persistence, baseline comparison, console summaries, and gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.metrics import apply_quality_gates, compare_baseline
from app.evaluation.models import (
    ComparisonReport,
    QualityGateConfig,
    RAGEvaluationReport,
    RetrievalEvaluationReport,
)


def report_payload(report: Any) -> dict[str, Any]:
    """Convert a Pydantic report to JSON-compatible data."""

    return report.model_dump(mode="json")


def write_report(report: Any, path: Path) -> None:
    """Write one machine-readable report to an ignored runtime location."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report_payload(report), indent=2) + "\n", encoding="utf-8")


def attach_baseline(
    report: RetrievalEvaluationReport | ComparisonReport,
    baseline_path: Path | None,
    *,
    tolerance: float,
) -> None:
    """Attach baseline deltas to a report in place when requested."""

    if baseline_path is None:
        return
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if isinstance(report, RetrievalEvaluationReport):
        report_data = report.model_dump(mode="json")
        report_data["baseline_comparison"] = compare_baseline(
            report_data,
            baseline,
            tolerance=tolerance,
        )
        report.baseline_comparison = report_data["baseline_comparison"]
        return
    baseline_reports = {}
    for item in baseline.get("reports", []):
        configuration = item.get("configuration", {})
        label = configuration.get("mode", "")
        if configuration.get("rerank"):
            label += " + rerank"
        baseline_reports[label] = item
    comparisons: dict[str, Any] = {}
    for item in report.reports:
        key = item.configuration.label
        previous = baseline_reports.get(key)
        if previous is not None:
            comparisons[key] = compare_baseline(
                item.model_dump(mode="json"),
                previous,
                tolerance=tolerance,
            )
    report.baseline_comparison = comparisons


def apply_retrieval_gates(
    report: RetrievalEvaluationReport,
    gates: QualityGateConfig,
) -> None:
    """Attach an explicit quality-gate result to a retrieval report."""

    result = apply_quality_gates(report.summary.model_dump(mode="json"), gates)
    report.quality_gates = result.model_dump(mode="json")


def console_summary(report: Any) -> str:
    """Return a compact, human-readable summary for CLI output."""

    if isinstance(report, ComparisonReport):
        lines = [f"Dataset: {report.dataset}", "Configuration              Success@1  Success@3  MRR  Mean total ms"]
        for item in report.reports:
            summary = item.summary
            lines.append(
                f"{item.configuration.label:<27} "
                f"{summary.success_at_k.get('1', 0):>9.3f}  "
                f"{summary.success_at_k.get('3', 0):>9.3f}  "
                f"{summary.mrr:>4.3f}  "
                f"{summary.mean_total_search_latency_ms if summary.mean_total_search_latency_ms is not None else 0:>14.3f}"
            )
        return "\n".join(lines)
    if isinstance(report, RAGEvaluationReport):
        summary = report.summary
        return (
            f"Dataset: {report.dataset}\n"
            f"Configuration: {report.configuration.label}\n"
            f"Cases: {summary.cases}\n"
            f"Key-fact coverage: {summary.key_fact_coverage:.3f}\n"
            f"Citation presence rate: {summary.citation_presence_rate:.3f}\n"
            f"Citation validity rate: {summary.citation_validity_rate:.3f}\n"
            f"Evidence support rate: {summary.evidence_support_rate:.3f}\n"
            f"Mean generation latency: {summary.mean_generation_latency_ms} ms\n"
            f"Mean total RAG latency: {summary.mean_total_rag_latency_ms} ms"
        )
    summary = report.summary
    return (
        f"Dataset: {report.dataset}\n"
        f"Configuration: {report.configuration.label}\n"
        f"Cases: {summary.cases} (positive={summary.positive_cases}, no-evidence={summary.no_evidence_cases})\n"
        f"Success@1: {summary.success_at_k.get('1', 0):.3f}\n"
        f"Success@3: {summary.success_at_k.get('3', 0):.3f}\n"
        f"MRR: {summary.mrr:.3f}\n"
        f"Mean total search latency: {summary.mean_total_search_latency_ms} ms"
    )
