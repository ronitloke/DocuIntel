"""Run the bounded Evaluation E3 DocVQA retrieval benchmark."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.datasets.base import PreparationOptions  # noqa: E402
from evaluation.datasets.registry import get_adapter  # noqa: E402
from evaluation.e3.runner import E3RunOptions, run_e3  # noqa: E402
from evaluation.manifests import repository_relative  # noqa: E402
from evaluation.validation import ManifestValidationError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded E3 CLI."""

    parser = argparse.ArgumentParser(
        description="Run the controlled DocVQA ingestion and retrieval benchmark."
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--document-limit", type=int, default=25)
    parser.add_argument("--question-limit", type=int, default=100)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/processed/docvqa/manifest.jsonl"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/evaluation/raw/docvqa"),
        help="Official local DocVQA source directory used by --prepare.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run the existing bounded E1 DocVQA adapter before evaluation.",
    )
    parser.add_argument(
        "--database-url",
        help="Optional isolated PostgreSQL URL override; credentials are not written to artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/evaluation/results/e3"),
    )
    parser.add_argument("--run-id", help="Unique result directory name; defaults to UTC timestamp.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--keep-indexed",
        action="store_true",
        help="Keep run-owned benchmark documents after evaluation; default is fail-closed cleanup.",
    )
    return parser


def _resolve(path: Path) -> Path:
    """Resolve a CLI path against the repository root."""

    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _prepare_if_requested(
    *,
    source_dir: Path,
    manifest_path: Path,
    split: str,
    document_limit: int,
) -> None:
    """Use the existing E1 adapter without introducing a second DocVQA parser."""

    command_args = [argument for argument in sys.argv[1:] if argument != "--prepare"]
    options = PreparationOptions(
        dataset="docvqa",
        split=split,
        limit=document_limit,
        output_root=manifest_path.parent.parent,
        source_directory=source_dir,
        command="python scripts/evaluation_run_e3.py --prepare " + " ".join(command_args),
        extra={"e3_document_limit": document_limit},
    )
    result = get_adapter("docvqa").prepare(options)
    print(
        f"E1 preparation: prepared={result.prepared} skipped={result.skipped} "
        f"failed={result.failed} manifest={result.manifest_path}"
    )
    for error in result.errors:
        print(f"E1 preparation error: {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Run E3 and return controlled nonzero status when prerequisites are absent."""

    args = build_parser().parse_args(argv)
    if not 1 <= args.document_limit <= 1000:
        print("--document-limit must be between 1 and 1000.", file=sys.stderr)
        return 2
    if not 1 <= args.question_limit <= 5000:
        print("--question-limit must be between 1 and 5000.", file=sys.stderr)
        return 2
    if args.top_k < 10:
        print("--top-k must be at least 10 for Recall@10.", file=sys.stderr)
        return 2

    manifest_path = _resolve(args.manifest)
    source_dir = _resolve(args.source_dir)
    if args.prepare:
        _prepare_if_requested(
            source_dir=source_dir,
            manifest_path=manifest_path,
            split=args.split,
            document_limit=args.document_limit,
        )
    run_id = args.run_id or datetime.now(UTC).strftime("e3_%Y%m%d_%H%M%S_%f")
    output_root = _resolve(args.output_root)
    output_directory = output_root / run_id
    options = E3RunOptions(
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
        summary = asyncio.run(run_e3(options, run_id=run_id))
    except (FileExistsError, ManifestValidationError, OSError, ValueError) as exc:
        print(f"E3 evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Status: {summary.status}")
    print(f"Run directory: {output_directory}")
    print(f"Summary: {output_directory / 'summary.json'}")
    print(f"Report: {output_directory / 'report.md'}")
    if summary.status == "DOCVQA_DATA_REQUIRED":
        print(
            "DOCVQA_DATA_REQUIRED: place official validation metadata/images under "
            f"{source_dir} and run scripts/evaluation_prepare.py before rerunning E3.",
            file=sys.stderr,
        )
        return 3
    return 0 if summary.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
