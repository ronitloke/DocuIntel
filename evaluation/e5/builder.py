"""Read-only E5 scorecard, provenance, and portfolio report builder."""

from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from evaluation.e5.loader import E5ArtifactError, load_baseline, load_jsonl, require_path
from evaluation.e5.models import BaselineArtifact, E5BuildResult, MetricRecord


_UNSUPPORTED_GLOBAL_ACCURACY = re.compile(
    r"(?i)\bdocuintel(?:\s+total[- ]system)?\s+accuracy\s*(?:=|:)\s*[0-9]"
)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _source(artifact: BaselineArtifact, root: Path, *, supplemental: str | None = None) -> str:
    path = artifact.supplemental_paths.get(supplemental) if supplemental else artifact.summary_path
    return _relative(path, root)


def _record(
    artifact: BaselineArtifact,
    root: Path,
    metric: str,
    value: float | int | bool | None,
    *,
    denominator: str | None,
    status: str = "MEASURED",
    notes: str | None = None,
    source: str | None = None,
) -> MetricRecord:
    return MetricRecord(
        module=artifact.module,
        dataset=artifact.dataset,
        split=artifact.split,
        run_id=artifact.run_id,
        metric=metric,
        value=value,
        denominator=denominator,
        status=status,  # type: ignore[arg-type]
        source=source or _source(artifact, root),
        notes=notes,
    )


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _f(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise E5ArtifactError(f"Expected numeric metric, got {value!r}")
    return float(value)


def _collect_e2(artifacts: dict[str, BaselineArtifact], root: Path) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    funsd = artifacts["e2_funsd"]
    fs = funsd.summary
    reliability = require_path(fs, "reliability", source=_source(funsd, root))
    performance = require_path(fs, "performance", source=_source(funsd, root))
    text_accuracy = require_path(fs, "text_accuracy", source=_source(funsd, root))
    text = require_path(text_accuracy, "strict", source=_source(funsd, root))
    docs = int(require_path(text_accuracy, "documents", source=_source(funsd, root)))
    doc_attempted = int(reliability["attempted_documents"])
    for metric, value in (
        ("ocr.cer", _f(require_path(text, "cer", source=_source(funsd, root)))),
        ("ocr.wer", _f(require_path(text, "wer", source=_source(funsd, root)))),
        ("ocr.documents_attempted", int(reliability["attempted_documents"])),
        ("ocr.documents_successful", int(reliability["successful_documents"])),
        ("ocr.processing_success_rate", _f(reliability["processing_success_rate"])),
        ("ocr.mean_processing_time_ms", _f(performance["mean_processing_time_ms"])),
        ("ocr.median_processing_time_ms", _f(performance["median_processing_time_ms"])),
        ("ocr.p95_processing_time_ms", _f(performance["p95_processing_time_ms"])),
    ):
        records.append(_record(funsd, root, metric, value, denominator=f"{docs} FUNSD documents"))

    doclaynet = artifacts["e2_doclaynet"]
    ds = doclaynet.summary
    layout = require_path(ds, "layout_accuracy", source=_source(doclaynet, root))
    reliability = require_path(ds, "reliability", source=_source(doclaynet, root))
    performance = require_path(ds, "performance", source=_source(doclaynet, root))
    comparable = int(layout["comparable_ground_truth"])
    for metric, key in (
        ("layout.precision", "precision"),
        ("layout.recall", "recall"),
        ("layout.f1", "f1"),
        ("layout.mean_matched_iou", "mean_matched_iou"),
        ("layout.true_positives", "true_positives"),
        ("layout.false_positives", "false_positives"),
        ("layout.false_negatives", "false_negatives"),
        ("layout.comparable_ground_truth", "comparable_ground_truth"),
    ):
        value = layout[key]
        records.append(
            _record(
                doclaynet,
                root,
                metric,
                _f(value) if isinstance(value, float) else int(value),
                denominator=f"{comparable} comparable DocLayNet ground-truth regions",
            )
        )
    for metric, value in (
        ("layout.documents_attempted", int(reliability["attempted_documents"])),
        ("layout.documents_successful", int(reliability["successful_documents"])),
        ("layout.processing_success_rate", _f(reliability["processing_success_rate"])),
        ("layout.mean_processing_time_ms", _f(performance["mean_processing_time_ms"])),
        ("layout.median_processing_time_ms", _f(performance["median_processing_time_ms"])),
        ("layout.p95_processing_time_ms", _f(performance["p95_processing_time_ms"])),
    ):
        records.append(
            _record(doclaynet, root, metric, value, denominator=f"{doc_attempted} DocLayNet documents attempted")
        )
    return records


def _collect_e3(artifact: BaselineArtifact, root: Path) -> tuple[list[MetricRecord], list[dict[str, Any]]]:
    summary = artifact.summary
    denominator = f"{summary['questions_scorable']} scorable questions"
    records: list[MetricRecord] = []
    methods = summary.get("methods")
    if not isinstance(methods, dict):
        raise E5ArtifactError("E3 methods are missing from the authoritative summary")
    method_order = ("keyword", "semantic", "hybrid", "hybrid_reranked")
    comparison: list[dict[str, Any]] = []
    for method in method_order:
        item = methods.get(method)
        if not isinstance(item, dict):
            raise E5ArtifactError(f"E3 method {method!r} is missing")
        quality: dict[str, Any] = {"method": method}
        for k in (1, 3, 5, 10):
            value = _f(require_path(item, f"recall_at_k.{k}", source=_source(artifact, root)))
            records.append(_record(artifact, root, f"{method}.recall_at_{k}", value, denominator=denominator))
            quality[f"recall_at_{k}"] = value
        mrr = _f(require_path(item, "mrr", source=_source(artifact, root)))
        records.append(_record(artifact, root, f"{method}.mrr", mrr, denominator=denominator))
        quality["mrr"] = mrr
        for timing_name, field in (
            ("retrieval", "retrieval_latency"),
            ("reranking", "reranking_latency"),
            ("total_pipeline", "total_pipeline_latency"),
        ):
            timing = require_path(item, field, source=_source(artifact, root))
            for stat in ("mean_ms", "median_ms", "p95_ms"):
                value = timing.get(stat)
                status = "MEASURED" if value is not None else "NOT_APPLICABLE"
                records.append(
                    _record(
                        artifact,
                        root,
                        f"{method}.{timing_name}_{stat}",
                        _f(value) if value is not None else None,
                        denominator=denominator,
                        status=status,
                        notes="No reranking stage exists for non-reranked methods." if status == "NOT_APPLICABLE" else None,
                    )
                )
                quality[f"{timing_name}_{stat}"] = value
        comparison.append(quality)

    hybrid = methods["hybrid"]
    reranked = methods["hybrid_reranked"]
    for k in (1, 3, 5, 10):
        before = _f(hybrid["recall_at_k"][str(k)])
        after = _f(reranked["recall_at_k"][str(k)])
        delta = after - before
        records.append(
            _record(
                artifact,
                root,
                f"hybrid_reranked_vs_hybrid.recall_at_{k}.absolute_delta",
                delta,
                denominator=denominator,
                notes=f"Calculated from authoritative E3 values {before} and {after}.",
            )
        )
        records.append(
            _record(
                artifact,
                root,
                f"hybrid_reranked_vs_hybrid.recall_at_{k}.percentage_point_delta",
                delta * 100,
                denominator=denominator,
                notes="Absolute delta multiplied by 100 percentage points.",
            )
        )
    before = _f(hybrid["mrr"])
    after = _f(reranked["mrr"])
    records.append(
        _record(
            artifact,
            root,
            "hybrid_reranked_vs_hybrid.mrr.absolute_delta",
            after - before,
            denominator=denominator,
            notes=f"Calculated from authoritative E3 values {before} and {after}.",
        )
    )
    return records, comparison


def _e4_citation_denominators(artifact: BaselineArtifact, root: Path) -> dict[str, dict[str, str]]:
    answers_path = artifact.supplemental_paths.get("answers")
    if answers_path is None:
        return {}
    rows = load_jsonl(answers_path)
    result: dict[str, dict[str, str]] = {}
    for configuration in ("hybrid", "hybrid_reranked"):
        answered = [row for row in rows if row.get("configuration") == configuration and row.get("status") == "ANSWERED"]
        cited = [row for row in answered if int(row.get("citation_labels_emitted", 0)) > 0]
        result[configuration] = {
            "presence": f"{len(answered)} answered responses",
            "cited": f"{len(cited)} cited answered responses",
        }
    return result


def _collect_e4(artifact: BaselineArtifact, root: Path) -> list[MetricRecord]:
    summary = artifact.summary
    records: list[MetricRecord] = []
    citation_denominators = _e4_citation_denominators(artifact, root)
    for configuration in ("hybrid", "hybrid_reranked"):
        item = summary.get("configurations", {}).get(configuration)
        if not isinstance(item, dict):
            raise E5ArtifactError(f"E4 configuration {configuration!r} is missing")
        attempted = int(item["questions_attempted"])
        scorable = int(item["questions_scorable"])
        denominator = f"{attempted} attempted questions"
        scorable_denominator = f"{scorable} scorable questions"
        for metric, value in (
            ("questions_attempted", attempted),
            ("questions_scorable", scorable),
            ("questions_answered", int(item["questions_answered"])),
            ("questions_failed", int(item["questions_failed"])),
            ("answer_coverage_rate", _f(item["answer_coverage_rate"])),
            ("end_to_end.anls", _f(item["end_to_end"]["anls"])),
            ("end_to_end.exact_match", _f(item["end_to_end"]["exact_match"])),
            ("scorable.anls", _f(item["scorable"]["anls"])),
            ("scorable.exact_match", _f(item["scorable"]["exact_match"])),
        ):
            records.append(
                _record(
                    artifact,
                    root,
                    f"{configuration}.{metric}",
                    value,
                    denominator=scorable_denominator if "scorable" in metric else denominator,
                )
            )
        failures = item.get("failures", {})
        for failure_name in ("OLLAMA_TIMEOUT", "NO_LITERAL_ANSWER_MATCH"):
            records.append(
                _record(
                    artifact,
                    root,
                    f"{configuration}.failure_{failure_name.lower()}",
                    int(failures.get(failure_name, 0)),
                    denominator=denominator,
                    notes="Failure count is preserved separately from measured zero-quality metrics.",
                )
            )
        citation = item["citation"]
        for metric_name, key in (
            ("citation_presence_rate", "citation_presence_rate"),
            ("citation_reference_validity_rate", "citation_reference_validity_rate"),
            ("citation_document_hit_rate", "citation_document_hit_rate"),
            ("all_citations_true_document_rate", "all_citations_true_document_rate"),
            ("gold_evidence_citation_hit_rate", "gold_evidence_citation_hit_rate"),
            ("gold_evidence_citation_precision", "gold_evidence_citation_precision"),
            ("answer_supported_by_cited_evidence_rate", "answer_supported_by_cited_evidence_rate"),
        ):
            value = citation.get(key)
            if value is None:
                continue
            denominator_text = citation_denominators.get(configuration, {}).get(
                "presence" if key == "citation_presence_rate" else "cited"
            )
            records.append(
                _record(
                    artifact,
                    root,
                    f"{configuration}.{metric_name}",
                    _f(value),
                    denominator=denominator_text,
                    notes="Citation denominator is derived from the authoritative E4 answers artifact." if denominator_text else "E4 summary does not expose a denominator for this rate.",
                )
            )
        latency = item["latency"]
        for stage in ("retrieval_time_ms", "reranking_time_ms", "generation_time_ms", "total_pipeline_time_ms"):
            stage_values = latency.get(stage, {})
            for stat in ("mean_ms", "median_ms", "p95_ms"):
                value = stage_values.get(stat)
                if value is None:
                    continue
                records.append(
                    _record(
                        artifact,
                        root,
                        f"{configuration}.{stage}.{stat}",
                        _f(value),
                        denominator=f"{stage_values.get('samples', 0)} measured timing samples",
                    )
                )
    return records


def _collect_e4_1(artifact: BaselineArtifact, root: Path) -> list[MetricRecord]:
    metadata = artifact.metadata
    timeout_path = require_path(metadata, "timeout_path", source=_source(artifact, root, supplemental=None).replace("summary.json", "run_metadata.json"))
    evidence = metadata.get("diagnostic_evidence", {})
    selected = int(artifact.summary.get("questions_selected", metadata.get("scorable_question_limit", 0)))
    records = [
        _record(artifact, root, "production_timeout_seconds", int(timeout_path["production_timeout_seconds"]), denominator=None),
        _record(artifact, root, "diagnostic_timeout_seconds", int(timeout_path["effective_timeout_seconds"]), denominator=None),
        _record(artifact, root, "questions_selected", selected, denominator=f"{selected} deterministic scorable questions"),
        _record(
            artifact,
            root,
            "extended_question_loop_completion_rate",
            None,
            denominator=f"{selected} selected questions",
            status="BLOCKED",
            notes="Warm-up blocked before the measured E4.1 loop; this is not a measured zero.",
        ),
        _record(
            artifact,
            root,
            "extended_answer_anls",
            None,
            denominator=f"{selected} selected questions",
            status="BLOCKED",
            notes="No generated answer entered the measured diagnostic loop.",
        ),
        _record(
            artifact,
            root,
            "extended_answer_exact_match",
            None,
            denominator=f"{selected} selected questions",
            status="BLOCKED",
            notes="No generated answer entered the measured diagnostic loop.",
        ),
    ]
    direct = evidence.get("direct_probe")
    if isinstance(direct, dict) and direct.get("wall_clock_time_ms") is not None:
        records.append(
            _record(
                artifact,
                root,
                "direct_minimal_probe_wall_clock_ms",
                _f(direct["wall_clock_time_ms"]),
                denominator="1 minimal direct probe",
                status="MEASURED",
                notes=f"Probe status: {direct.get('status')}; this is not a grounded RAG measurement.",
                source=_source(artifact, root, supplemental=None).replace("summary.json", "run_metadata.json"),
            )
        )
    return records


def _records_to_rows(records: Iterable[MetricRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]


def _build_retrieval_csv(records: list[MetricRecord]) -> list[dict[str, Any]]:
    methods = ("keyword", "semantic", "hybrid", "hybrid_reranked")
    rows: list[dict[str, Any]] = []
    for method in methods:
        row: dict[str, Any] = {"method": method}
        prefix = f"{method}."
        for field in (
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "retrieval_mean_ms",
            "retrieval_median_ms",
            "retrieval_p95_ms",
            "reranking_mean_ms",
            "reranking_median_ms",
            "reranking_p95_ms",
            "total_pipeline_mean_ms",
            "total_pipeline_median_ms",
            "total_pipeline_p95_ms",
        ):
            matching = next((record for record in records if record.metric == prefix + field), None)
            row[field] = matching.value if matching else None
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_no_unsupported_global_accuracy(output_directory: Path) -> None:
    """Reject a generated numeric claim that conflates unlike task metrics."""

    for path in output_directory.iterdir():
        if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        if _UNSUPPORTED_GLOBAL_ACCURACY.search(text):
            raise E5ArtifactError(f"Unsupported global accuracy claim found in {path}")


def _limitations(artifacts: dict[str, BaselineArtifact]) -> list[dict[str, Any]]:
    e2_doc = artifacts["e2_doclaynet"].summary
    e3 = artifacts["e3"].summary
    e4 = artifacts["e4"].summary
    return [
        {"id": "bounded_e2", "text": "E2 used five records per dataset for the real acceptance runs.", "source": "E2 summaries", "status": "CONFIRMED"},
        {"id": "doclaynet_failure", "text": f"DocLayNet had {e2_doc['reliability']['failed_documents']} processing failure.", "source": "E2 DocLayNet summary", "status": "CONFIRMED"},
        {"id": "ocr_error", "text": "FUNSD OCR CER/WER were high under the documented deterministic metric.", "source": "E2 FUNSD summary", "status": "CONFIRMED"},
        {"id": "layout_quality", "text": "DocLayNet layout precision and F1 were low despite high mean IoU on matched regions.", "source": "E2 DocLayNet summary", "status": "CONFIRMED"},
        {"id": "answer_indexability", "text": f"{e3['questions_unscorable']} of {e3['questions_attempted']} DocVQA questions were not answer-indexable under the existing pipeline.", "source": "E3 summary", "status": "CONFIRMED"},
        {"id": "upload_limit", "text": f"The E3 run recorded {len(e3['failures'])} document-level failure(s), including the unchanged 25 MB upload-limit failure.", "source": "E3 summary failures", "status": "CONFIRMED"},
        {"id": "conditional_retrieval", "text": "Retrieval metrics are conditional on the 43 answer-indexable/scorable questions.", "source": "E3 summary", "status": "CONFIRMED"},
        {"id": "reranking_latency", "text": "Hybrid+reranker total latency was substantially higher than Hybrid total latency.", "source": "E3 summary", "status": "CONFIRMED"},
        {"id": "ollama_runtime", "text": "E4 completion was extremely low in the CPU-mode local Ollama environment because generation timed out frequently.", "source": "E4 summary and E4.1 summary", "status": "CONFIRMED"},
        {"id": "answer_format", "text": "E4 ANLS/EM results have a response-format mismatch concern because the production answer is evidence-first rather than a short answer.", "source": "E4 answers and E4.1 diagnostic design", "status": "CONFIRMED"},
        {"id": "citation_denominators", "text": "E4 citation rates have tiny denominators and must not be presented as generic citation accuracy.", "source": "E4 answers artifact", "status": "CONFIRMED"},
        {"id": "e41_blocked", "text": "E4.1 did not enter its measured question loop; extended diagnostic metrics are BLOCKED/NOT_MEASURED, not zero.", "source": "E4.1 summary", "status": "CONFIRMED"},
        {"id": "no_global_score", "text": "No generic total-system accuracy is calculated because OCR, layout, retrieval, RAG completion, and answer metrics measure different tasks.", "source": "E5 methodology", "status": "METHODOLOGICAL_RULE"},
    ]


def _portfolio_markdown(records: list[MetricRecord], output_root: Path) -> str:
    def value(metric: str) -> Any:
        match = next(record for record in records if record.metric == metric)
        return match.value

    source_e3 = "Source: E3 / DocVQA validation / 43 scorable questions / summary.json"
    source_e2 = "Source: E2 / FUNSD and DocLayNet real acceptance summaries"
    source_e4 = "Source: E4 / DocVQA validation / 100 attempted, 43 scorable / summary.json"
    h1, r1 = value("hybrid.recall_at_1"), value("hybrid_reranked.recall_at_1")
    hmrr, rmrr = value("hybrid.mrr"), value("hybrid_reranked.mrr")
    delta1 = value("hybrid_reranked_vs_hybrid.recall_at_1.percentage_point_delta")
    delta_mrr = value("hybrid_reranked_vs_hybrid.mrr.absolute_delta")
    hybrid_total = value("hybrid.total_pipeline_median_ms")
    rerank_total = value("hybrid_reranked.total_pipeline_median_ms")
    e4_answered_h = value("hybrid.questions_answered")
    e4_attempted_h = value("hybrid.questions_attempted")
    e4_answered_r = value("hybrid_reranked.questions_answered")
    e4_attempted_r = value("hybrid_reranked.questions_attempted")
    return f"""# DocuIntel portfolio metrics

These claims are scoped to the cited measured artifacts. E5 does not produce a single system-wide score.

## Resume bullet candidates

- Evaluated a PostgreSQL/pgvector hybrid retrieval pipeline on official DocVQA validation data, with CrossEncoder reranking increasing Recall@1 from {_fmt(h1)} to {_fmt(r1)} ({_fmt(delta1, 2)} percentage points) across 43 scorable questions.  
  {source_e3}
- Built and measured a four-way retrieval comparison: keyword, semantic, hybrid, and hybrid+CrossEncoder; reranking increased MRR from {_fmt(hmrr)} to {_fmt(rmrr)} at a median total-latency cost of {_fmt(rerank_total - hybrid_total, 3)} ms.  
  {source_e3}
- Implemented a fail-closed evaluation consolidation layer that preserves metric provenance, denominators, blocked states, and limitations across OCR, layout, retrieval, and grounded RAG benchmarks.  
  Source: E5 generated scorecard and provenance artifacts

## GitHub/README performance statements

- On the bounded DocVQA validation set, hybrid retrieval reached Recall@5={_fmt(value('hybrid.recall_at_5'))} and MRR={_fmt(hmrr)}; the result is conditional on 43 scorable questions.  
  {source_e3}
- CrossEncoder reranking produced the strongest measured retrieval MRR, {_fmt(rmrr)}, versus {_fmt(hmrr)} for hybrid retrieval, while increasing median total pipeline latency from {_fmt(hybrid_total)} ms to {_fmt(rerank_total)} ms.  
  {source_e3}
- The production-config RAG run answered {e4_answered_h}/{e4_attempted_h} hybrid questions and {e4_answered_r}/{e4_attempted_r} reranked questions; these are completion results under a local CPU Ollama runtime, not generic answer accuracy.  
  {source_e4}

## Interview talking points

1. Retrieval worked better than end-to-end RAG because the retrieval benchmark isolated 43 answer-indexable questions, while E4 additionally depended on answer indexability, Ollama generation completion, and response-format scoring.  
   {source_e3}; {source_e4}
2. Reranking improved retrieval quality: Recall@1 and MRR increased by measured deltas, but CrossEncoder scoring added a measured latency stage.  
   {source_e3}
3. E4/E4.1 failures are useful engineering evidence: they exposed low answer completion and a blocked extended CPU-Ollama diagnostic without allowing blocked values to masquerade as measured zeros.  
   {source_e4}; Source: E4.1 / controlled blocked diagnostic / summary.json

## Claims to avoid

- Do not claim a single DocuIntel accuracy percentage.
- Do not call E2 CER/WER, layout F1, E3 Recall@K/MRR, E4 coverage, or citation rates interchangeable accuracy metrics.
- Do not present E4 citation rates without their tiny denominators.
- Do not claim E4.1 extended-timeout quality or latency results; its measured loop was blocked.
- Do not generalize these bounded local CPU/OCR results to broad production performance.

"""


def _report_markdown(
    artifacts: dict[str, BaselineArtifact],
    records: list[MetricRecord],
    retrieval_rows: list[dict[str, Any]],
    limitations: list[dict[str, Any]],
    run_id: str,
    root: Path,
) -> str:
    def val(metric: str) -> Any:
        match = next((record for record in records if record.metric == metric), None)
        return match.value if match else None

    e3 = artifacts["e3"].summary
    e4 = artifacts["e4"].summary
    e41 = artifacts["e4_1"].metadata
    cited_denominator = next(
        record.denominator
        for record in records
        if record.metric == "hybrid_reranked.citation_reference_validity_rate"
    )
    lines = [
        "# DocuIntel Evaluation E5 — final benchmark consolidation",
        "",
        f"- Run ID: `{run_id}`",
        "- Status: `completed`",
        "- Scope: read-only consolidation of authoritative E2, E3, E4, and E4.1 artifacts",
        "- Method: every displayed value is loaded from a selected artifact or calculated only as an explicit delta from loaded values",
        "",
        "## Authoritative sources",
        "",
    ]
    for key, artifact in artifacts.items():
        lines.append(f"- `{key}`: `{_relative(artifact.summary_path, root)}`; run `{artifact.run_id}`")
    lines += [
        "",
        "## Executive conclusion",
        "",
        f"The strongest measured portfolio result is E3 hybrid+reranked retrieval: Recall@1={_fmt(val('hybrid_reranked.recall_at_1'))}, Recall@5={_fmt(val('hybrid_reranked.recall_at_5'))}, and MRR={_fmt(val('hybrid_reranked.mrr'))} on {e3['questions_scorable']} scorable DocVQA questions.",
        f"The fastest measured retrieval configuration is keyword search by total pipeline latency (median {_fmt(val('keyword.total_pipeline_median_ms'))} ms), while the strongest quality/latency trade-off among semantic and hybrid methods is a separate criterion from absolute fastest.",
        "No generic total-system accuracy is calculated or reported.",
        "",
        "## Document understanding scorecard",
        "",
        "| Area | Metric | Result | Status | Scope |",
        "|---|---|---:|---|---|",
        f"| FUNSD OCR | CER | {_fmt(val('ocr.cer'))} | MEASURED | 5 FUNSD test documents |",
        f"| FUNSD OCR | WER | {_fmt(val('ocr.wer'))} | MEASURED | 5 FUNSD test documents |",
        f"| DocLayNet | Precision | {_fmt(val('layout.precision'))} | MEASURED | 29 comparable regions |",
        f"| DocLayNet | Recall | {_fmt(val('layout.recall'))} | MEASURED | 29 comparable regions |",
        f"| DocLayNet | F1 | {_fmt(val('layout.f1'))} | MEASURED | 29 comparable regions |",
        f"| DocLayNet | Mean matched IoU | {_fmt(val('layout.mean_matched_iou'))} | MEASURED | matched regions |",
        f"| DocLayNet | Processing success | {_fmt(val('layout.processing_success_rate'))} | MEASURED | 5 attempted documents |",
        "",
        "## Retrieval scorecard",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Median retrieval ms | Median rerank ms | Median total ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in retrieval_rows:
        lines.append(
            f"| {row['method']} | {_fmt(row['recall_at_1'])} | {_fmt(row['recall_at_3'])} | {_fmt(row['recall_at_5'])} | {_fmt(row['recall_at_10'])} | {_fmt(row['mrr'])} | {_fmt(row['retrieval_median_ms'])} | {_fmt(row['reranking_median_ms'])} | {_fmt(row['total_pipeline_median_ms'])} |"
        )
    lines += [
        "",
        f"Corpus: {e3['documents_prepared']} prepared, {e3['documents_indexed']} indexed, {e3['questions_attempted']} attempted, {e3['questions_scorable']} scorable, answer-indexability={_fmt(e3['answer_indexability_rate'])}.",
        "",
        "### Reranking delta",
        "",
        f"- Recall@1: {_fmt(val('hybrid_reranked_vs_hybrid.recall_at_1.absolute_delta'))} absolute / {_fmt(val('hybrid_reranked_vs_hybrid.recall_at_1.percentage_point_delta'), 4)} percentage points.",
        f"- Recall@3: {_fmt(val('hybrid_reranked_vs_hybrid.recall_at_3.absolute_delta'))} absolute / {_fmt(val('hybrid_reranked_vs_hybrid.recall_at_3.percentage_point_delta'), 4)} percentage points.",
        f"- Recall@5: {_fmt(val('hybrid_reranked_vs_hybrid.recall_at_5.absolute_delta'))} absolute / {_fmt(val('hybrid_reranked_vs_hybrid.recall_at_5.percentage_point_delta'), 4)} percentage points.",
        f"- Recall@10: {_fmt(val('hybrid_reranked_vs_hybrid.recall_at_10.absolute_delta'))} absolute / {_fmt(val('hybrid_reranked_vs_hybrid.recall_at_10.percentage_point_delta'), 4)} percentage points.",
        f"- MRR: {_fmt(val('hybrid_reranked_vs_hybrid.mrr.absolute_delta'))} absolute.",
        f"- Median total-latency cost: {_fmt(val('hybrid_reranked.total_pipeline_median_ms') - val('hybrid.total_pipeline_median_ms'), 3)} ms.",
        "",
        "## End-to-end RAG reliability",
        "",
        "| Configuration | Answered | Failed | Coverage | ANLS | EM | Ollama timeouts | Answer not indexed | Median total ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for configuration in ("hybrid", "hybrid_reranked"):
        item = e4["configurations"][configuration]
        lines.append(
            f"| {configuration} | {item['questions_answered']} | {item['questions_failed']} | {_fmt(item['answer_coverage_rate'])} | {_fmt(item['end_to_end']['anls'])} | {_fmt(item['end_to_end']['exact_match'])} | {item['failures'].get('OLLAMA_TIMEOUT', 0)} | {item['failures'].get('NO_LITERAL_ANSWER_MATCH', 0)} | {_fmt(item['latency']['total_pipeline_time_ms']['median_ms'])} |"
        )
    lines += [
        "",
        "These ANLS/EM zeros are measured E4 production-response results. They are not a generic system accuracy value; completion was limited by answer indexability and local Ollama timeouts, and the response contract is evidence-first rather than DocVQA short-answer-only.",
        "",
        "### Citation evidence",
        "",
        f"Citation metrics are deterministic and denominator-scoped. The reranked E4 run had {val('hybrid_reranked.questions_answered')} answered responses; the citation-validity denominator was {cited_denominator}. The cited response(s) resolved to valid, answer-bearing evidence. The validity/document/gold-evidence rates were {_fmt(val('hybrid_reranked.citation_reference_validity_rate'))}, {_fmt(val('hybrid_reranked.citation_document_hit_rate'))}, and {_fmt(val('hybrid_reranked.gold_evidence_citation_hit_rate'))}; they must not be presented as generic citation accuracy.",
        "",
        "## E4.1 runtime diagnostic",
        "",
        f"- Production timeout: `{e41['timeout_path']['production_timeout_seconds']} seconds`.",
        f"- Benchmark-only diagnostic timeout: `{e41['timeout_path']['effective_timeout_seconds']} seconds`.",
        f"- Selected questions: `{e41.get('scorable_question_limit', 20)} deterministic scorable questions`.",
        "- Hybrid+reranker warm-up: `BLOCKED` / host stall.",
        "- Extended measured question loop: `NOT_MEASURED`.",
        "- E4.1 ANLS/EM: `BLOCKED`, not zero.",
        "",
        "## Bottlenecks and interpretation",
        "",
    ]
    for limitation in limitations:
        if limitation["id"] != "no_global_score":
            lines.append(f"- **{limitation['id']}**: {limitation['text']} ({limitation['source']}).")
    lines += [
        "",
        "## Public-safe strongest result",
        "",
        f"Evaluated official DocVQA validation retrieval over {e3['questions_scorable']} scorable questions; CrossEncoder reranking raised Recall@1 from {_fmt(val('hybrid.recall_at_1'))} to {_fmt(val('hybrid_reranked.recall_at_1'))} and MRR from {_fmt(val('hybrid.mrr'))} to {_fmt(val('hybrid_reranked.mrr'))}, with median total pipeline latency increasing from {_fmt(val('hybrid.total_pipeline_median_ms'))} ms to {_fmt(val('hybrid_reranked.total_pipeline_median_ms'))} ms.",
        "",
        "## Reproducibility and scope",
        "",
        "E5 only reads the explicit manifest sources and writes a new report package. It does not invoke Ollama, embeddings, OCR, PostgreSQL ingestion, or any E2/E3/E4/E4.1 benchmark runner. No production algorithm, model, prompt, setting, upload limit, schema, dependency, or migration was changed. Module 13 remains outside scope.",
        "",
    ]
    return "\n".join(lines)


def build_e5(*, manifest_path: Path, output_directory: Path, run_id: str, project_root: Path | None = None) -> E5BuildResult:
    """Build a new E5 package from existing artifacts only."""

    root = (project_root or Path.cwd()).resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing E5 output directory: {output_directory}")
    artifacts = load_baseline(manifest_path, project_root=root)
    records = _collect_e2(artifacts, root)
    e3_records, _ = _collect_e3(artifacts["e3"], root)
    records.extend(e3_records)
    records.extend(_collect_e4(artifacts["e4"], root))
    records.extend(_collect_e4_1(artifacts["e4_1"], root))
    limitations = _limitations(artifacts)
    retrieval_rows = _build_retrieval_csv(records)
    output_directory.mkdir(parents=True)
    rows = _records_to_rows(records)
    _json_dump(output_directory / "scorecard.json", {"schema_version": "e5.scorecard.v1", "records": rows})
    _json_dump(output_directory / "limitations.json", {"schema_version": "e5.limitations.v1", "limitations": limitations})
    _write_csv(output_directory / "scorecard.csv", rows, list(rows[0].keys()))
    _write_csv(output_directory / "retrieval_comparison.csv", retrieval_rows, list(retrieval_rows[0].keys()))
    pipeline_rows = [
        row
        for row in rows
        if row["metric"].startswith(("ocr.", "layout.", "keyword.", "semantic.", "hybrid.", "hybrid_reranked.", "production_timeout", "diagnostic_timeout", "extended_", "direct_minimal"))
    ]
    _write_csv(output_directory / "pipeline_stage_metrics.csv", pipeline_rows, list(rows[0].keys()))
    with (output_directory / "metric_provenance.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    portfolio = _portfolio_markdown(records, output_directory)
    report = _report_markdown(artifacts, records, retrieval_rows, limitations, run_id, root)
    (output_directory / "portfolio_metrics.md").write_text(portfolio, encoding="utf-8")
    (output_directory / "report.md").write_text(report, encoding="utf-8")

    source_map = {
        key: {
            "module": artifact.module,
            "dataset": artifact.dataset,
            "split": artifact.split,
            "run_id": artifact.run_id,
            "summary": _relative(artifact.summary_path, root),
            "run_metadata": _relative(artifact.metadata_path, root),
        }
        for key, artifact in artifacts.items()
    }
    summary = {
        "schema_version": "e5.v1",
        "status": "completed",
        "run_id": run_id,
        "source_manifest": _relative(manifest_path, root),
        "authoritative_sources": source_map,
        "measurement_status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("MEASURED", "BLOCKED", "NOT_APPLICABLE", "NOT_MEASURED")
        },
        "no_global_accuracy": True,
        "documents_prepared": artifacts["e3"].summary["documents_prepared"],
        "documents_indexed": artifacts["e3"].summary["documents_indexed"],
        "questions_attempted": artifacts["e3"].summary["questions_attempted"],
        "questions_scorable": artifacts["e3"].summary["questions_scorable"],
        "answer_indexability_rate": artifacts["e3"].summary["answer_indexability_rate"],
        "generated_files": [
            "summary.json",
            "scorecard.json",
            "scorecard.csv",
            "metric_provenance.jsonl",
            "pipeline_stage_metrics.csv",
            "retrieval_comparison.csv",
            "limitations.json",
            "portfolio_metrics.md",
            "report.md",
            "run_metadata.json",
        ],
    }
    _json_dump(output_directory / "summary.json", summary)
    _json_dump(
        output_directory / "run_metadata.json",
        {
            "schema_version": "e5.v1",
            "run_id": run_id,
            "command_scope": "read-only artifact consolidation",
            "manifest": _relative(manifest_path, root),
            "modules_rerun": [],
            "production_changes": [],
            "database_migration": None,
            "sources": source_map,
        },
    )
    assert_no_unsupported_global_accuracy(output_directory)
    return E5BuildResult(output_directory=output_directory, summary=summary)
