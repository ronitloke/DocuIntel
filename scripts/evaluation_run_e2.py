"""Run the real Module E2 ingestion/OCR/layout benchmark."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.e2.runner import run_e2  # noqa: E402
from evaluation.validation import ManifestValidationError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded E2 runner CLI."""

    parser = argparse.ArgumentParser(description="Run the DocuIntel Module E2 benchmark.")
    parser.add_argument("--dataset", choices=("doclaynet", "funsd"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/evaluation/results/e2"),
    )
    parser.add_argument("--run-id", help="Stable output folder name; defaults to a UTC timestamp.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run E2 and print the generated artifact directory."""

    args = build_parser().parse_args(argv)
    if not 0 < args.iou_threshold <= 1:
        print("--iou-threshold must be greater than zero and at most one.", file=sys.stderr)
        return 2
    manifest = args.manifest or Path("data/evaluation/processed") / args.dataset / "manifest.jsonl"
    manifest = manifest if manifest.is_absolute() else PROJECT_ROOT / manifest
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    run_id = args.run_id or datetime.now(UTC).strftime("e2_%Y%m%d_%H%M%S")
    output_directory = output_root / run_id / args.dataset
    try:
        result = asyncio.run(
            run_e2(
                manifest.resolve(),
                output_directory.resolve(),
                expected_dataset=args.dataset,
                expected_split=args.split,
                iou_threshold=args.iou_threshold,
            )
        )
    except (ManifestValidationError, OSError, ValueError) as exc:
        print(f"E2 evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Output directory: {result}")
    print(f"Summary: {result / 'summary.json'}")
    print(f"Report: {result / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
