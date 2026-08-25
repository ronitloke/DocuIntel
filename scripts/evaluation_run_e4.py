"""Run the bounded Evaluation E4 DocVQA end-to-end RAG benchmark."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.e4.runner import E4RunOptions, run_e4  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded E4 CLI."""

    parser = argparse.ArgumentParser(
        description="Run the controlled DocVQA end-to-end RAG answer/grounding benchmark."
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--document-limit", type=int, default=25)
    parser.add_argument("--question-limit", type=int, default=100)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/processed/docvqa/manifest.jsonl"),
    )
    parser.add_argument("--database-url", help="Optional isolated PostgreSQL URL override.")
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/results/e4"))
    parser.add_argument("--run-id", help="Unique result directory name; defaults to UTC timestamp.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--keep-indexed",
        action="store_true",
        help="Keep run-owned benchmark documents after evaluation; default is fail-closed cleanup.",
    )
    return parser


def _resolve(path: Path) -> Path:
    """Resolve CLI paths against the repository root."""

    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Run E4 and return a controlled nonzero status when prerequisites are absent."""

    args = build_parser().parse_args(argv)
    if not 1 <= args.document_limit <= 1000:
        print("--document-limit must be between 1 and 1000.", file=sys.stderr)
        return 2
    if not 1 <= args.question_limit <= 5000:
        print("--question-limit must be between 1 and 5000.", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("--top-k must be greater than zero.", file=sys.stderr)
        return 2
    manifest_path = _resolve(args.manifest)
    run_id = args.run_id or datetime.now(UTC).strftime("e4_%Y%m%d_%H%M%S_%f")
    output_directory = _resolve(args.output_root) / run_id
    options = E4RunOptions(
        manifest_path=manifest_path,
        output_directory=output_directory,
        split=args.split,
        document_limit=args.document_limit,
        question_limit=args.question_limit,
        top_k=args.top_k,
        keep_indexed=args.keep_indexed,
        database_url_override=args.database_url,
    )
    try:
        summary = asyncio.run(run_e4(options, run_id=run_id))
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"E4 evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Status: {summary.status}")
    print(f"Run directory: {output_directory}")
    print(f"Summary: {output_directory / 'summary.json'}")
    print(f"Report: {output_directory / 'report.md'}")
    if summary.status == "DOCVQA_DATA_REQUIRED":
        return 3
    return 0 if summary.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

