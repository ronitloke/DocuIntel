"""Read-only presentation of the authoritative Module E5 package."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from streamlit_app.config import PROJECT_ROOT, get_settings


DEFAULT_E5_RESULTS_DIRECTORY = (
    PROJECT_ROOT / "data" / "evaluation" / "results" / "e5" / "final_baseline_20260821_final"
)
PUBLIC_E5_RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation" / "public" / "e5_ui_package"
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
    """Load the fixed E5 run package, with a safe public-checkout fallback.

    Local development uses the authoritative ignored result directory. A clean
    public checkout has only the reviewed four-file package, so the fallback is
    used only when the caller requested the default directory and that local
    package is absent. Explicit custom directories still fail closed.
    """

    requested = Path(directory).expanduser().resolve()
    default_directory = DEFAULT_E5_RESULTS_DIRECTORY.resolve()
    if requested == default_directory and not requested.is_dir():
        requested = PUBLIC_E5_RESULTS_DIRECTORY.resolve()
    return _load_e5_package_cached(str(requested))


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


EVALUATION_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M5 19V9M12 19V5M19 19v-7"/><path d="M3 19h18"/>'
    '<path d="m4 7 5-3 4 2 6-3"/></svg>'
)


def _metric_card(
    record: dict[str, Any] | None,
    label: str,
    *,
    percent: bool = False,
    decimals: int = 3,
) -> None:
    """Render one status-aware metric without converting non-measured states."""

    status = str(record.get("status", "NOT_MEASURED")) if record else "NOT_MEASURED"
    with st.container(border=True):
        st.metric(label, format_e5_metric(record, percent=percent, decimals=decimals))
        if status != "MEASURED":
            st.markdown(
                f'<span class="di-eval-state di-eval-state--{status.casefold()}">{escape(status)}</span>',
                unsafe_allow_html=True,
            )
        if record and record.get("denominator"):
            st.caption(str(record["denominator"]))


def _limitation(package: dict[str, Any], limitation_id: str) -> dict[str, Any] | None:
    return next(
        (item for item in package.get("limitations", []) if str(item.get("id")) == limitation_id),
        None,
    )


def _eval_section_heading(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="di-eval-section-heading"><div class="di-eval-section-kicker">{escape(kicker)}</div>'
        f'<h2>{escape(title)}</h2><p>{escape(copy)}</p></div>',
        unsafe_allow_html=True,
    )


def _render_retrieval_chart(rows: list[dict[str, Any]], metric: str, title: str, y_label: str) -> None:
    values = {
        str(row["label"]): row.get(metric)
        for row in rows
        if row.get(metric) is not None
    }
    with st.container(border=True):
        st.markdown(f'<div class="di-eval-chart-title">{escape(title)}</div>', unsafe_allow_html=True)
        st.bar_chart(values, y_label=y_label, height=220)


def _render_rag_method(package: dict[str, Any], method: str, title: str) -> None:
    with st.container(key=f"evaluation-rag-{method}", border=True):
        st.markdown(f'<div class="di-eval-method-label">{escape(title)}</div>', unsafe_allow_html=True)
        with st.container(
            key=f"evaluation-rag-{method}-metrics",
            horizontal=True,
            gap="small",
            vertical_alignment="top",
        ):
            _metric_card(
                find_scorecard_metric(package, f"{method}.questions_answered", module="E4", dataset="docvqa"),
                "Questions answered",
                decimals=0,
            )
            _metric_card(
                find_scorecard_metric(package, f"{method}.answer_coverage_rate", module="E4", dataset="docvqa"),
                "Answer coverage",
                percent=True,
            )
            _metric_card(
                find_scorecard_metric(package, f"{method}.questions_failed", module="E4", dataset="docvqa"),
                "Questions failed",
                decimals=0,
            )
        with st.container(
            key=f"evaluation-rag-{method}-quality",
            horizontal=True,
            gap="small",
            vertical_alignment="top",
        ):
            _metric_card(
                find_scorecard_metric(package, f"{method}.end_to_end.anls", module="E4", dataset="docvqa"),
                "End-to-end ANLS",
            )
            _metric_card(
                find_scorecard_metric(package, f"{method}.end_to_end.exact_match", module="E4", dataset="docvqa"),
                "End-to-end exact match",
            )


def render_evaluation() -> None:
    """Render the read-only E5 benchmark dashboard without running evaluation work."""

    with st.container(key="evaluation-page"):
        with st.container(
            key="evaluation-page-header",
            horizontal=True,
            gap="medium",
            vertical_alignment="center",
        ):
            with st.container(key="evaluation-page-header-copy", width="stretch"):
                st.markdown('<div class="di-eyebrow">MODEL &amp; SYSTEM EVALUATION</div>', unsafe_allow_html=True)
                st.markdown('<h1 class="di-page-title">Evaluation</h1>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="di-page-subtitle">Inspect reproducible document understanding, retrieval and '
                    'grounded RAG benchmark results.</p>',
                    unsafe_allow_html=True,
                )
            st.markdown(f'<div class="di-evaluation-header-icon">{EVALUATION_ICON}</div>', unsafe_allow_html=True)

        settings = get_settings()
        try:
            package = load_e5_package(settings.e5_results_directory)
        except E5UiArtifactError as exc:
            with st.container(key="evaluation-error-state", border=True):
                st.error(str(exc))
                st.caption("Expected the configured final E5 evaluation package.")
            return

        summary = package["summary"]
        status_counts = summary.get("measurement_status_counts", {})
        with st.container(key="evaluation-overview-card", border=True):
            _eval_section_heading(
                "AUTHORITATIVE BENCHMARK SNAPSHOT",
                "Evaluation overview",
                "A bounded, reproducible package of stored measurements. This page is read-only and never starts benchmark work.",
            )
            with st.container(
                key="evaluation-overview-grid",
                horizontal=True,
                gap="medium",
                vertical_alignment="top",
            ):
                for label, value, accent in (
                    ("Documents prepared", summary.get("documents_prepared", "—"), "indigo"),
                    ("Documents indexed", summary.get("documents_indexed", "—"), "cyan"),
                    ("Scorable questions", summary.get("questions_scorable", "—"), "violet"),
                    ("Measured records", status_counts.get("MEASURED", "—"), "green"),
                ):
                    with st.container(border=True):
                        st.markdown(
                            f'<div class="di-eval-overview-value di-eval-overview-value--{accent}">'
                            f'<span>{escape(str(label))}</span><strong>{escape(str(value))}</strong></div>',
                            unsafe_allow_html=True,
                        )
            with st.container(
                key="evaluation-status-grid",
                horizontal=True,
                gap="small",
                vertical_alignment="top",
            ):
                for label, value, accent in (
                    ("Measured", status_counts.get("MEASURED", 0), "green"),
                    ("Blocked", status_counts.get("BLOCKED", 0), "amber"),
                    ("Not applicable", status_counts.get("NOT_APPLICABLE", 0), "neutral"),
                ):
                    with st.container(border=True):
                        st.markdown(
                            f'<div class="di-eval-status-value di-eval-status-value--{accent}">'
                            f'<span>{escape(str(label))}</span><strong>{escape(str(value))}</strong></div>',
                            unsafe_allow_html=True,
                        )
            st.caption(
                f"Bounded DocVQA configuration: {summary.get('questions_attempted', '—')} attempted questions, "
                f"{summary.get('questions_scorable', '—')} scorable questions."
            )

        understanding_tab, retrieval_tab, rag_tab, limitations_tab = st.tabs(
            ["Document understanding", "Retrieval", "RAG reliability", "Limitations & provenance"],
            key="evaluation-tabs",
        )
        with understanding_tab:
            _eval_section_heading(
                "E2 · DOCUMENT UNDERSTANDING",
                "Document understanding",
                "OCR quality and layout matching remain separate measurements with their original denominators.",
            )
            with st.container(key="evaluation-ocr-card", border=True):
                st.markdown('<div class="di-eval-subsection-title">FUNSD OCR</div>', unsafe_allow_html=True)
                with st.container(
                    key="evaluation-ocr-metrics",
                    horizontal=True,
                    gap="medium",
                    vertical_alignment="top",
                ):
                    _metric_card(
                        find_scorecard_metric(package, "ocr.cer", module="E2", dataset="funsd"),
                        "CER",
                    )
                    _metric_card(
                        find_scorecard_metric(package, "ocr.wer", module="E2", dataset="funsd"),
                        "WER",
                    )
                st.caption("CER = Character Error Rate · WER = Word Error Rate · lower is better.")

            with st.container(key="evaluation-layout-card", border=True):
                st.markdown('<div class="di-eval-subsection-title">DOCLAYNET LAYOUT</div>', unsafe_allow_html=True)
                with st.container(
                    key="evaluation-layout-metrics",
                    horizontal=True,
                    gap="small",
                    vertical_alignment="top",
                ):
                    for metric, label, percent in (
                        ("layout.precision", "Precision", True),
                        ("layout.recall", "Recall", True),
                        ("layout.f1", "F1", True),
                        ("layout.mean_matched_iou", "Mean matched IoU", False),
                        ("layout.processing_success_rate", "Processing success", True),
                    ):
                        _metric_card(
                            find_scorecard_metric(package, metric, module="E2", dataset="doclaynet"),
                            label,
                            percent=percent,
                        )
                st.markdown(
                    '<div class="di-eval-interpretation"><strong>Interpretation</strong> Matched regions achieved '
                    'strong overlap, while overall precision and F1 indicate that layout detection remains a known '
                    'baseline limitation.</div>',
                    unsafe_allow_html=True,
                )

        with retrieval_tab:
            rows = retrieval_rows(package)
            _eval_section_heading(
                "E3 · CONTROLLED DOCVQA RETRIEVAL",
                "Retrieval quality",
                "Four configurations use the same controlled cases and denominators; values come directly from the E5 package.",
            )
            with st.container(key="evaluation-retrieval-table-card", border=True):
                st.dataframe(_retrieval_table(package), hide_index=True, width="stretch")
            st.markdown('<div class="di-eval-subsection-title">Retrieval quality at a glance</div>', unsafe_allow_html=True)
            with st.container(
                key="evaluation-retrieval-quality-charts",
                horizontal=True,
                gap="medium",
                vertical_alignment="top",
            ):
                _render_retrieval_chart(rows, "recall_at_1", "Recall@1", "Recall")
                _render_retrieval_chart(rows, "mrr", "MRR", "Score")

            _eval_section_heading(
                "LATENCY TRADE-OFF",
                "Retrieval latency",
                "The reranker improves early-rank retrieval quality, while the stored CPU measurements show its added cost.",
            )
            with st.container(key="evaluation-retrieval-latency-card", border=True):
                _render_retrieval_chart(rows, "total_pipeline_median_ms", "Median total latency", "Milliseconds")

            hybrid = next((row for row in rows if row.get("method") == "hybrid"), None)
            reranked = next((row for row in rows if row.get("method") == "hybrid_reranked"), None)
            if hybrid and reranked:
                with st.container(key="evaluation-reranking-impact-card", border=True):
                    _eval_section_heading(
                        "CROSSENCODER EFFECT",
                        "Reranking impact",
                        "CrossEncoder reranking improves top-ranked retrieval in this bounded benchmark, with a substantial CPU latency cost.",
                    )
                    with st.container(
                        key="evaluation-reranking-impact-grid",
                        horizontal=True,
                        gap="small",
                        vertical_alignment="top",
                    ):
                        for label, value, accent in (
                            (
                                "Recall@1 improvement",
                                f"{(reranked['recall_at_1'] - hybrid['recall_at_1']) * 100:.1f} pp",
                                "indigo",
                            ),
                            (
                                "Recall@5 improvement",
                                f"{(reranked['recall_at_5'] - hybrid['recall_at_5']) * 100:.1f} pp",
                                "cyan",
                            ),
                            ("MRR improvement", f"{reranked['mrr'] - hybrid['mrr']:.3f}", "green"),
                            ("Median rerank cost", f"{reranked['reranking_median_ms']:.0f} ms", "amber"),
                        ):
                            with st.container(border=True):
                                st.markdown(
                                    f'<div class="di-eval-impact-value di-eval-impact-value--{accent}">'
                                    f'<span>{escape(label)}</span><strong>{escape(value)}</strong></div>',
                                    unsafe_allow_html=True,
                                )
                    st.caption("Recall@10 is unchanged in the stored results; reranking is not presented as universally better.")

        with rag_tab:
            _eval_section_heading(
                "E4 · GROUNDED ANSWER RELIABILITY",
                "RAG reliability",
                "Answer coverage, failures, citations and timing are shown separately so incomplete local generation is visible.",
            )
            runtime_limitation = _limitation(package, "ollama_runtime")
            if runtime_limitation:
                st.warning(str(runtime_limitation.get("text") or "Local LLM runtime limitation recorded."), icon=":material/schedule:")
            with st.container(key="evaluation-rag-measured-card", border=True):
                st.markdown('<div class="di-eval-subsection-title">MEASURED E4 RESULTS</div>', unsafe_allow_html=True)
                with st.container(
                    key="evaluation-rag-methods",
                    horizontal=True,
                    gap="medium",
                    vertical_alignment="top",
                ):
                    _render_rag_method(package, "hybrid", "Hybrid")
                    _render_rag_method(package, "hybrid_reranked", "Hybrid + reranker")
                st.caption("These are bounded E4 reliability measurements, not a generic answer-accuracy score.")

            with st.container(key="evaluation-rag-diagnostics-card", border=True):
                _eval_section_heading(
                    "FAILURE DIAGNOSTICS",
                    "Runtime and response diagnostics",
                    "Failure categories remain separate from measured zero-quality metrics.",
                )
                with st.container(
                    key="evaluation-rag-failure-grid",
                    horizontal=True,
                    gap="small",
                    vertical_alignment="top",
                ):
                    for method, title in (("hybrid", "Hybrid"), ("hybrid_reranked", "Hybrid + reranker")):
                        for metric, label in (
                            ("failure_ollama_timeout", f"{title} · Ollama timeouts"),
                            ("failure_no_literal_answer_match", f"{title} · no literal answer match"),
                        ):
                            _metric_card(
                                find_scorecard_metric(package, f"{method}.{metric}", module="E4", dataset="docvqa"),
                                label,
                                decimals=0,
                            )

            with st.container(key="evaluation-rag-citation-card", border=True):
                _eval_section_heading(
                    "CITATION EVIDENCE",
                    "Citation measurements",
                    "Citation rates use the answered-response denominators from the authoritative answers artifact.",
                )
                with st.container(
                    key="evaluation-rag-citation-grid",
                    horizontal=True,
                    gap="medium",
                    vertical_alignment="top",
                ):
                    for method, title in (("hybrid", "Hybrid"), ("hybrid_reranked", "Hybrid + reranker")):
                        with st.container(border=True):
                            st.markdown(f'<div class="di-eval-method-label">{escape(title)}</div>', unsafe_allow_html=True)
                            _metric_card(
                                find_scorecard_metric(package, f"{method}.citation_presence_rate", module="E4", dataset="docvqa"),
                                "Citation presence",
                                percent=True,
                            )
                            _metric_card(
                                find_scorecard_metric(package, f"{method}.citation_reference_validity_rate", module="E4", dataset="docvqa"),
                                "Reference validity",
                                percent=True,
                            )
                citation_limitation = _limitation(package, "citation_denominators")
                if citation_limitation:
                    st.caption(str(citation_limitation.get("text") or "Citation denominator limitation recorded."))

            with st.container(key="evaluation-rag-response-format-card", border=True):
                response_limitation = _limitation(package, "answer_format")
                _eval_section_heading(
                    "RESPONSE FORMAT",
                    "Answer-format concern",
                    str(response_limitation.get("text") if response_limitation else "Response-format limitation recorded in the E5 package."),
                )

            with st.container(key="evaluation-e41-card", border=True):
                _eval_section_heading(
                    "E4.1 · EXTENDED DIAGNOSTIC",
                    "Blocked diagnostic state",
                    "The extended loop did not enter measured generation; blocked values are shown as BLOCKED, not zero.",
                )
                with st.container(
                    key="evaluation-e41-metrics",
                    horizontal=True,
                    gap="small",
                    vertical_alignment="top",
                ):
                    for metric, label in (
                        ("extended_question_loop_completion_rate", "Question loop completion"),
                        ("extended_answer_anls", "Extended answer ANLS"),
                        ("extended_answer_exact_match", "Extended exact match"),
                        ("direct_minimal_probe_wall_clock_ms", "Minimal direct probe (ms)"),
                    ):
                        _metric_card(
                            find_scorecard_metric(package, metric, module="E4.1", dataset="docvqa"),
                            label,
                            percent=metric.endswith("completion_rate"),
                            decimals=1 if metric.endswith("wall_clock_ms") else 3,
                        )

        with limitations_tab:
            _eval_section_heading(
                "METHODOLOGY & TRACEABILITY",
                "Limitations and provenance",
                "Human-readable limitations and safe repository-relative provenance from the authoritative E5 package.",
            )
            with st.container(key="evaluation-limitations-card", border=True):
                for limitation in package["limitations"]:
                    with st.expander(limitation_display_title(limitation.get("id"))):
                        st.write(limitation.get("text") or "No limitation detail was recorded.")
                        st.caption(f"Source: {limitation.get('source', 'E5 artifact')}")
            with st.container(key="evaluation-provenance-card", border=True):
                _eval_section_heading(
                    "SAFE PROVENANCE",
                    "Benchmark provenance",
                    "The page exposes only the reviewed run, datasets, split and repository-relative manifest.",
                )
                provenance = public_provenance(package)
                for label, value in provenance.items():
                    st.markdown(
                        f'<div class="di-eval-provenance-row"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>',
                        unsafe_allow_html=True,
                    )


__all__ = [
    "DEFAULT_E5_RESULTS_DIRECTORY",
    "E5UiArtifactError",
    "PUBLIC_E5_RESULTS_DIRECTORY",
    "find_scorecard_metric",
    "format_e5_metric",
    "load_e5_package",
    "limitation_display_title",
    "public_provenance",
    "render_evaluation",
    "retrieval_rows",
]
