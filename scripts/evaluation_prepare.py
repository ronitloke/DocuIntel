"""Prepare bounded, normalized DocLayNet, FUNSD, or local DocVQA data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.datasets.base import PreparationOptions  # noqa: E402
from evaluation.datasets.registry import get_adapter  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded preparation CLI."""

    parser = argparse.ArgumentParser(
        description="Prepare a bounded, normalized DocuIntel evaluation corpus."
    )
    parser.add_argument("--dataset", choices=("doclaynet", "funsd", "docvqa"), required=True)
    parser.add_argument("--split", required=True, help="Source split, for example validation or test.")
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Maximum number of source documents/images to prepare; this option is required.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/evaluation/processed"),
        help="Root for normalized PDFs and manifests.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Local source directory, required for manually provided DocVQA data.",
    )
    parser.add_argument("--revision", help="Optional source revision recorded in metadata.")
    return parser


def _print_summary(result) -> None:
    print(f"Dataset: {result.dataset}")
    print(f"Split: {result.split}")
    print(f"Requested: {result.requested}")
    print(f"Prepared: {result.prepared}")
    print(f"Skipped: {result.skipped}")
    print(f"Failed: {result.failed}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Output directory: {result.output_directory}")
    print(f"Preparation metadata: {result.metadata_path}")
    for error in result.errors:
        print(f"Error: {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Run preparation and return nonzero when source data was not prepared cleanly."""

    args = build_parser().parse_args(argv)
    if args.limit <= 0:
        print("--limit must be greater than zero.", file=sys.stderr)
        return 2
    if args.limit > 1000:
        print("--limit must be 1000 or smaller for a bounded preparation run.", file=sys.stderr)
        return 2
    source_directory = args.source_dir
    if args.dataset == "docvqa" and source_directory is None:
        source_directory = PROJECT_ROOT / "data" / "evaluation" / "raw" / "docvqa"
    options = PreparationOptions(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        output_root=(PROJECT_ROOT / args.output_root).resolve()
        if not args.output_root.is_absolute()
        else args.output_root.resolve(),
        source_directory=(PROJECT_ROOT / source_directory).resolve()
        if source_directory is not None and not source_directory.is_absolute()
        else source_directory,
        revision=args.revision,
        command="python scripts/evaluation_prepare.py " + " ".join(sys.argv[1:]),
        extra={"source_dir_argument": str(args.source_dir) if args.source_dir else None},
    )
    try:
        result = get_adapter(args.dataset).prepare(options)
    except (ValueError, OSError) as exc:
        print(f"Evaluation preparation failed: {exc}", file=sys.stderr)
        return 2
    _print_summary(result)
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

