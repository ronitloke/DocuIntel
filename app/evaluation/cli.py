"""Command-line entry point for Module 9 evaluation runs."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.core.config import Settings
from app.evaluation.dataset import load_dataset
from app.evaluation.models import EvaluationConfiguration, QualityGateConfig
from app.evaluation.rag import RAGEvaluator
from app.evaluation.reporting import (
    apply_retrieval_gates,
    attach_baseline,
    console_summary,
    write_report,
)
from app.evaluation.retrieval import RetrievalEvaluator
from app.evaluation.runner import build_runtime


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options shared by retrieval and RAG commands."""

    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/report.json"))


def _add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    """Add retrieval configuration and optional regression gates."""

    _add_common_arguments(parser)
    parser.add_argument("--mode", choices=("semantic", "keyword", "hybrid"), default="hybrid")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-tolerance", type=float, default=0.0)
    parser.add_argument("--minimum-success-at-3", type=float)
    parser.add_argument("--minimum-mrr", type=float)
    parser.add_argument("--maximum-mean-search-latency-ms", type=float)


def build_parser() -> argparse.ArgumentParser:
    """Build the documented retrieval, compare, and RAG CLI."""

    parser = argparse.ArgumentParser(description="Evaluate the existing DocuIntel retrieval/RAG pipeline.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    retrieval = subcommands.add_parser("retrieval", help="Run retrieval-only evaluation; Ollama is not called.")
    _add_retrieval_arguments(retrieval)

    compare = subcommands.add_parser("compare", help="Compare semantic, keyword, and hybrid retrieval.")
    _add_common_arguments(compare)
    compare.add_argument("--include-rerank", action=argparse.BooleanOptionalAction, default=True)
    compare.add_argument("--baseline", type=Path)
    compare.add_argument("--baseline-tolerance", type=float, default=0.0)

    rag = subcommands.add_parser("rag", help="Run deterministic grounded-answer evaluation with Ollama.")
    _add_common_arguments(rag)
    rag.add_argument("--mode", choices=("semantic", "keyword", "hybrid"), default="hybrid")
    rag.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)

    return parser


def _quality_gates(args: argparse.Namespace) -> QualityGateConfig:
    """Build optional retrieval thresholds from CLI arguments."""

    return QualityGateConfig(
        minimum_success_at_3=args.minimum_success_at_3,
        minimum_mrr=args.minimum_mrr,
        maximum_mean_search_latency_ms=args.maximum_mean_search_latency_ms,
    )


def main(argv: list[str] | None = None) -> int:
    """Run one evaluation command and return a process-style exit code."""

    args = build_parser().parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        runtime = build_runtime(Settings())
        try:
            if args.command == "retrieval":
                configuration = EvaluationConfiguration(
                    mode=args.mode,
                    rerank=args.rerank,
                    top_k=args.top_k,
                )
                evaluator = RetrievalEvaluator(runtime.search_service)
                for check in evaluator.validate_dataset(dataset):
                    print(
                        f"Corpus check: case={check.case_id} question={check.question!r} "
                        f"expected_document={check.expected_document} exists={check.exists} "
                        f"indexed_chunks={check.indexed_chunks}"
                    )
                report = evaluator.evaluate(dataset, configuration)
                attach_baseline(report, args.baseline, tolerance=args.baseline_tolerance)
                gates = _quality_gates(args)
                if any(value is not None for value in gates.model_dump().values()):
                    apply_retrieval_gates(report, gates)
                write_report(report, args.output)
            elif args.command == "compare":
                configurations = [
                    EvaluationConfiguration(mode="semantic", top_k=args.top_k),
                    EvaluationConfiguration(mode="keyword", top_k=args.top_k),
                    EvaluationConfiguration(mode="hybrid", top_k=args.top_k),
                ]
                if args.include_rerank:
                    configurations.append(
                        EvaluationConfiguration(mode="hybrid", rerank=True, top_k=args.top_k)
                    )
                evaluator = RetrievalEvaluator(runtime.search_service)
                for check in evaluator.validate_dataset(dataset):
                    print(
                        f"Corpus check: case={check.case_id} question={check.question!r} "
                        f"expected_document={check.expected_document} exists={check.exists} "
                        f"indexed_chunks={check.indexed_chunks}"
                    )
                report = evaluator.compare(dataset, configurations)
                attach_baseline(report, args.baseline, tolerance=args.baseline_tolerance)
                write_report(report, args.output)
            else:
                configuration = EvaluationConfiguration(
                    mode=args.mode,
                    rerank=args.rerank,
                    top_k=args.top_k,
                )
                report = asyncio.run(RAGEvaluator(runtime.rag_service).evaluate(dataset, configuration))
                write_report(report, args.output)
        finally:
            runtime.close()
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2

    print(console_summary(report))
    print(f"JSON report: {args.output}")
    quality_gates = getattr(report, "quality_gates", None)
    if quality_gates and not quality_gates.get("passed", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
