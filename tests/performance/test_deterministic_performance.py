"""Repeatable performance checks for pure local retrieval/processing hot paths."""

import pytest

from scripts.benchmark_performance import run_benchmark


@pytest.mark.performance
def test_pure_pipeline_benchmark_is_repeatable_and_returns_valid_measurements(capsys) -> None:
    """The benchmark reports real timings without requiring external services."""

    report = run_benchmark(iterations=3)
    output = capsys.readouterr().out

    assert set(report) == {"iterations", "benchmarks"}
    assert report["iterations"] == 3
    assert set(report["benchmarks"]) == {
        "chunking",
        "embedding_stub",
        "hybrid_fusion",
        "reranking_stub",
    }
    for measurement in report["benchmarks"].values():
        assert measurement["iterations"] == 3
        assert measurement["result_count"] > 0
        assert measurement["mean_ms"] >= 0
        assert measurement["median_ms"] >= 0
        assert measurement["min_ms"] >= 0
        assert measurement["max_ms"] >= measurement["min_ms"]
    assert "PERFORMANCE" in output
