"""Benchmark deterministic application hot paths without external services.

The embedding and reranking measurements intentionally use small local doubles.
They measure pipeline overhead and determinism, not transformer model throughput.
Use the real API smoke commands in the Module 12 report for model-backed timings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings
from app.db.repository import SearchCandidate
from app.services.chunking.structure_aware import StructureAwareChunker
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.reranking.cross_encoder import CrossEncoderReranker
from app.services.retrieval.search import SearchService


class _EmbeddingModel:
    def encode(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        return [[float((index + len(text)) % 8) for index in range(8)] for text in texts]


class _RerankingModel:
    def predict(self, pairs: list[tuple[str, str]], **_kwargs: Any) -> list[float]:
        return [float("priority" in candidate_text) for _, candidate_text in pairs]


def _measure(
    name: str,
    operation: Callable[[], Any],
    iterations: int,
) -> dict[str, float | int | str]:
    durations: list[float] = []
    result_count = 0
    for _ in range(iterations):
        started = perf_counter()
        result = operation()
        durations.append((perf_counter() - started) * 1000)
        result_count = len(result)
    return {
        "name": name,
        "iterations": iterations,
        "result_count": result_count,
        "mean_ms": round(statistics.fmean(durations), 3),
        "median_ms": round(statistics.median(durations), 3),
        "min_ms": round(min(durations), 3),
        "max_ms": round(max(durations), 3),
    }


def _document() -> SimpleNamespace:
    pages = []
    for page_number in range(1, 4):
        pages.append(
            SimpleNamespace(
                id=UUID(int=page_number),
                page_number=page_number,
                extraction_method="native",
                ocr_applied=False,
                extracted_text="",
                layout_elements=[
                    SimpleNamespace(
                        sequence_order=1,
                        element_type="heading",
                        text=f"Section {page_number}",
                    ),
                    *[
                        SimpleNamespace(
                            sequence_order=index + 2,
                            element_type="paragraph",
                            text=(
                                "This deterministic benchmark paragraph contains a stable "
                                "document processing workload for repeatable timing measurements."
                            ),
                        )
                        for index in range(6)
                    ],
                ],
                tables=[],
            )
        )
    return SimpleNamespace(id=UUID(int=999), pages=pages)


def _candidates(count: int, prefix: str) -> list[SearchCandidate]:
    return [
        SearchCandidate(
            chunk_id=UUID(int=index + 1),
            document_id=UUID(int=1000 + index),
            original_filename=f"{prefix}-{index}.pdf",
            sequence_number=index + 1,
            text=("priority " if index % 7 == 0 else "ordinary ") + "candidate text",
            section_heading="Benchmark",
            start_page=1,
            end_page=1,
            content_type="text",
            contains_ocr=False,
            score=1.0 / (index + 1),
        )
        for index in range(count)
    ]


def run_benchmark(*, iterations: int = 10) -> dict[str, Any]:
    """Measure deterministic local processing and retrieval operations."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    chunker = StructureAwareChunker(target_chars=250, max_chars=400, overlap_chars=40)
    embedding_service = EmbeddingService(
        Settings(embedding_dimension=8),
        model=_EmbeddingModel(),
    )
    settings = Settings(search_candidate_multiplier=4, rerank_candidate_count=20)
    search_service = SearchService(
        repository=None,
        embedding_service=embedding_service,
        settings=settings,
        reranker=CrossEncoderReranker(model=_RerankingModel()),
    )
    semantic = _candidates(50, "semantic")
    keyword = _candidates(50, "keyword")
    keyword.reverse()
    search_results = search_service._fuse(semantic, keyword, 25)

    benchmarks = {
        "chunking": _measure(
            "chunking",
            lambda: chunker.build_chunks(_document()),
            iterations,
        ),
        "embedding_stub": _measure(
            "embedding_stub",
            lambda: embedding_service.embed_texts([f"benchmark text {i}" for i in range(32)]),
            iterations,
        ),
        "hybrid_fusion": _measure(
            "hybrid_fusion",
            lambda: search_service._fuse(semantic, keyword, 25),
            iterations,
        ),
        "reranking_stub": _measure(
            "reranking_stub",
            lambda: search_service._rerank("benchmark query", search_results, 10),
            iterations,
        ),
    }
    report = {"iterations": iterations, "benchmarks": benchmarks}
    for measurement in benchmarks.values():
        print(
            "PERFORMANCE "
            f"{measurement['name']} iterations={measurement['iterations']} "
            f"results={measurement['result_count']} mean_ms={measurement['mean_ms']:.3f} "
            f"median_ms={measurement['median_ms']:.3f}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_benchmark(iterations=args.iterations)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"JSON report: {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
