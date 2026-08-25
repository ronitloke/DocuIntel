"""Run the bounded Evaluation E4.1 timeout/answer-format diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.e4_1.runner import E4_1RunOptions, run_e4_1  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded E4.1 CLI."""

    parser = argparse.ArgumentParser(description="Run the controlled DocVQA E4.1 diagnostic.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--document-limit", type=int, default=25)
    parser.add_argument("--question-limit", type=int, default=100)
    parser.add_argument("--scorable-question-limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--generation-timeout-seconds",
        type=float,
        default=None,
        help="Benchmark-only Ollama HTTP timeout override; omitted means the production setting.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/processed/docvqa/manifest.jsonl"),
    )
    parser.add_argument("--database-url", help="Optional isolated PostgreSQL URL override.")
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/results/e4_1"))
    parser.add_argument("--run-id", help="Unique result directory; defaults to UTC timestamp.")
    parser.add_argument(
        "--production-baseline",
        type=Path,
        default=Path("data/evaluation/results/e4/e4_real_offline_20260821_rerun"),
        help="Preserved E4 directory used only to project production-timeout completion on the same subset.",
    )
    parser.add_argument("--keep-indexed", action="store_true")
    parser.add_argument(
        "--reuse-indexed-corpus",
        action="store_true",
        help="Reuse exact ready/indexed manifest-named documents without ingesting or deleting database records.",
    )
    return parser


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Run E4.1 and print its artifact directory."""

    args = build_parser().parse_args(argv)
    if not 1 <= args.document_limit <= 1000:
        print("--document-limit must be between 1 and 1000.", file=sys.stderr)
        return 2
    if not 1 <= args.question_limit <= 5000:
        print("--question-limit must be between 1 and 5000.", file=sys.stderr)
        return 2
    if not 1 <= args.scorable_question_limit <= 5000:
        print("--scorable-question-limit must be between 1 and 5000.", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("--top-k must be greater than zero.", file=sys.stderr)
        return 2
    if args.generation_timeout_seconds is not None and args.generation_timeout_seconds <= 0:
        print("--generation-timeout-seconds must be greater than zero.", file=sys.stderr)
        return 2
    run_id = args.run_id or datetime.now(UTC).strftime("e4_1_%Y%m%d_%H%M%S_%f")
    output_directory = _resolve(args.output_root) / run_id
    options = E4_1RunOptions(
        manifest_path=_resolve(args.manifest),
        output_directory=output_directory,
        split=args.split,
        document_limit=args.document_limit,
        question_limit=args.question_limit,
        scorable_question_limit=args.scorable_question_limit,
        top_k=args.top_k,
        generation_timeout_seconds=args.generation_timeout_seconds,
        keep_indexed=args.keep_indexed,
        database_url_override=args.database_url,
        production_baseline_directory=_resolve(args.production_baseline),
        reuse_indexed_corpus=args.reuse_indexed_corpus,
    )
    try:
        summary = asyncio.run(run_e4_1(options, run_id=run_id))
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"E4.1 evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Status: {summary.status}")
    print(f"Run directory: {output_directory}")
    print(f"Summary: {output_directory / 'summary.json'}")
    print(f"Report: {output_directory / 'report.md'}")
    return 0 if summary.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
