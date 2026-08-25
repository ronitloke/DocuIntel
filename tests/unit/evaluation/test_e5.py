"""Focused synthetic tests for the read-only E5 consolidation layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.e5.builder import _build_retrieval_csv, _collect_e3, _limitations, _portfolio_markdown, assert_no_unsupported_global_accuracy
from evaluation.e5.loader import E5ArtifactError, load_baseline
from evaluation.e5.models import BaselineArtifact, MetricRecord


def _source(root: Path, key: str, module: str, dataset: str, split: str, run_id: str) -> dict[str, object]:
    if module == "E2":
        directory = root / "data" / "evaluation" / "results" / "e2" / run_id / dataset
    else:
        directory = root / "data" / "evaluation" / "results" / module.lower().replace(".", "_") / run_id
    directory.mkdir(parents=True)
    summary = {
        "schema_version": {"E2": "e2.v1", "E3": "e3.v1", "E4": "e4.v1", "E4.1": "e4_1.v1"}[module],
        "dataset": dataset,
        "split": split,
    }
    metadata = {"schema_version": summary["schema_version"], "dataset": dataset, "split": split}
    if module == "E3":
        summary["status"] = "completed"
    if module == "E4":
        summary["status"] = "completed"
    if module == "E4.1":
        summary["status"] = "CONTROLLED_FAILURE"
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return {
        "module": module,
        "dataset": dataset,
        "split": split,
        "run_id": run_id,
        "summary": str(directory / "summary.json"),
        "run_metadata": str(directory / "run_metadata.json"),
    }


def _manifest(tmp_path: Path) -> Path:
    sources = {
        "e2_funsd": _source(tmp_path, "e2_funsd", "E2", "funsd", "test", "e2-run"),
        "e2_doclaynet": _source(tmp_path, "e2_doclaynet", "E2", "doclaynet", "validation", "e2-run"),
        "e3": _source(tmp_path, "e3", "E3", "docvqa", "validation", "e3-run"),
        "e4": _source(tmp_path, "e4", "E4", "docvqa", "validation", "e4-run"),
        "e4_1": _source(tmp_path, "e4_1", "E4.1", "docvqa", "validation", "e4_1-run"),
    }
    path = tmp_path / "evaluation" / "e5" / "baseline_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": "e5.baseline.v1", "sources": sources}), encoding="utf-8")
    return path


def test_explicit_manifest_selection_and_validation(tmp_path: Path) -> None:
    artifacts = load_baseline(_manifest(tmp_path), project_root=tmp_path)
    assert artifacts["e3"].run_id == "e3-run"
    assert artifacts["e4_1"].summary["status"] == "CONTROLLED_FAILURE"


def test_missing_artifact_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    source = json.loads(manifest.read_text(encoding="utf-8"))
    Path(source["sources"]["e3"]["summary"]).unlink()
    with pytest.raises(E5ArtifactError, match="does not exist"):
        load_baseline(manifest, project_root=tmp_path)


def test_malformed_json_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    source = json.loads(manifest.read_text(encoding="utf-8"))
    Path(source["sources"]["e4"]["summary"]).write_text("{not-json", encoding="utf-8")
    with pytest.raises(E5ArtifactError, match="Could not parse JSON"):
        load_baseline(manifest, project_root=tmp_path)


def test_module_dataset_split_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    source = json.loads(manifest.read_text(encoding="utf-8"))
    source["sources"]["e3"]["dataset"] = "funsd"
    manifest.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(E5ArtifactError, match="inconsistent module/dataset/split"):
        load_baseline(manifest, project_root=tmp_path)


def test_reranker_delta_is_calculated_from_loaded_values(tmp_path: Path) -> None:
    artifact = BaselineArtifact(
        key="e3",
        module="E3",
        dataset="docvqa",
        split="validation",
        run_id="e3-run",
        summary_path=tmp_path / "summary.json",
        metadata_path=tmp_path / "run_metadata.json",
        summary={
            "questions_scorable": 10,
            "methods": {
                method: {
                    "recall_at_k": {"1": 0.2 if method == "hybrid" else 0.3, "3": 0.4, "5": 0.5, "10": 0.6},
                    "mrr": 0.25 if method == "hybrid" else 0.35,
                    "retrieval_latency": {"mean_ms": 1, "median_ms": 1, "p95_ms": 1},
                    "reranking_latency": {"mean_ms": None, "median_ms": None, "p95_ms": None},
                    "total_pipeline_latency": {"mean_ms": 1, "median_ms": 1, "p95_ms": 1},
                }
                for method in ("keyword", "semantic", "hybrid", "hybrid_reranked")
            },
        },
        metadata={},
        supplemental_paths={},
    )
    records, _ = _collect_e3(artifact, tmp_path)
    delta = next(record for record in records if record.metric.endswith("recall_at_1.absolute_delta"))
    assert delta.value == pytest.approx(0.1)
    assert delta.denominator == "10 scorable questions"


def test_measured_zero_and_blocked_are_distinct() -> None:
    measured = MetricRecord("E4", "docvqa", "validation", "e4", "hybrid.anls", 0.0, "43 scorable questions", "MEASURED", "summary.json")
    blocked = MetricRecord("E4.1", "docvqa", "validation", "e4_1", "extended.anls", None, "20 questions", "BLOCKED", "summary.json")
    assert measured.value == 0.0 and measured.status == "MEASURED"
    assert blocked.value is None and blocked.status == "BLOCKED"


def test_provenance_and_denominator_are_preserved() -> None:
    record = MetricRecord(
        "E3",
        "docvqa",
        "validation",
        "e3-run",
        "hybrid.mrr",
        0.5,
        "43 scorable questions",
        "MEASURED",
        "data/evaluation/results/e3/run/summary.json",
    )
    assert record.to_dict()["source"].endswith("summary.json")
    assert record.to_dict()["denominator"] == "43 scorable questions"


def test_not_applicable_latency_and_deterministic_retrieval_order(tmp_path: Path) -> None:
    artifact = BaselineArtifact(
        key="e3",
        module="E3",
        dataset="docvqa",
        split="validation",
        run_id="e3-run",
        summary_path=tmp_path / "summary.json",
        metadata_path=tmp_path / "run_metadata.json",
        summary={
            "questions_scorable": 1,
            "methods": {
                method: {
                    "recall_at_k": {"1": 0.1, "3": 0.1, "5": 0.1, "10": 0.1},
                    "mrr": 0.1,
                    "retrieval_latency": {"mean_ms": 1, "median_ms": 1, "p95_ms": 1},
                    "reranking_latency": {"mean_ms": None, "median_ms": None, "p95_ms": None},
                    "total_pipeline_latency": {"mean_ms": 1, "median_ms": 1, "p95_ms": 1},
                }
                for method in ("keyword", "semantic", "hybrid", "hybrid_reranked")
            },
        },
        metadata={},
        supplemental_paths={},
    )
    records, _ = _collect_e3(artifact, tmp_path)
    assert next(record for record in records if record.metric == "hybrid.reranking_mean_ms").status == "NOT_APPLICABLE"
    rows = _build_retrieval_csv(list(reversed(records)))
    assert [row["method"] for row in rows] == ["keyword", "semantic", "hybrid", "hybrid_reranked"]


def test_limitations_and_portfolio_claims_keep_provenance(tmp_path: Path) -> None:
    def artifact(module: str, dataset: str, summary: dict[str, object]) -> BaselineArtifact:
        return BaselineArtifact(module, module, dataset, "validation", module, tmp_path / "summary.json", tmp_path / "metadata.json", summary, {}, {})

    artifacts = {
        "e2_doclaynet": artifact("E2", "doclaynet", {"reliability": {"failed_documents": 1}}),
        "e2_funsd": artifact("E2", "funsd", {}),
        "e3": artifact("E3", "docvqa", {"questions_unscorable": 2, "questions_attempted": 3, "failures": ["upload"]}),
        "e4": artifact("E4", "docvqa", {}),
        "e4_1": artifact("E4.1", "docvqa", {}),
    }
    limitations = _limitations(artifacts)
    assert any(item["id"] == "answer_indexability" for item in limitations)
    records = []
    values = {
        "hybrid.recall_at_1": 0.2,
        "hybrid_reranked.recall_at_1": 0.3,
        "hybrid_reranked_vs_hybrid.recall_at_1.percentage_point_delta": 10.0,
        "hybrid.mrr": 0.2,
        "hybrid_reranked.mrr": 0.3,
        "hybrid_reranked_vs_hybrid.mrr.absolute_delta": 0.1,
        "hybrid.total_pipeline_median_ms": 1.0,
        "hybrid_reranked.total_pipeline_median_ms": 2.0,
        "hybrid.recall_at_5": 0.2,
        "hybrid.questions_answered": 1,
        "hybrid.questions_attempted": 2,
        "hybrid_reranked.questions_answered": 2,
        "hybrid_reranked.questions_attempted": 2,
    }
    records.extend(MetricRecord("E3", "docvqa", "validation", "e3", metric, value, "2 questions", "MEASURED", "summary.json") for metric, value in values.items())
    portfolio = _portfolio_markdown(records, tmp_path)
    assert "Source: E3" in portfolio


def test_global_accuracy_guard_rejects_numeric_claim(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    path.write_text("DocuIntel accuracy = 0.72", encoding="utf-8")
    with pytest.raises(E5ArtifactError, match="global accuracy"):
        assert_no_unsupported_global_accuracy(tmp_path)


def test_global_accuracy_guard_allows_method_specific_scorecard(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    path.write_text("Recall@5=0.744186; no generic total-system accuracy is reported.", encoding="utf-8")
    assert_no_unsupported_global_accuracy(tmp_path)
