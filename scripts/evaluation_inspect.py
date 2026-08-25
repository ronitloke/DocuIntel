"""Inspect and validate one prepared E1 JSONL manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.validation import ManifestValidationError, validate_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the manifest inspection CLI."""

    parser = argparse.ArgumentParser(description="Validate a DocuIntel evaluation JSONL manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate and print applicable manifest statistics."""

    args = build_parser().parse_args(argv)
    manifest = (PROJECT_ROOT / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest
    try:
        result = validate_manifest(manifest)
    except ManifestValidationError as exc:
        print(f"Manifest invalid: {exc}", file=sys.stderr)
        return 2
    print(f"Manifest: {manifest}")
    print(f"Dataset: {result.dataset or 'unknown'}")
    print(f"Split: {result.split or 'unknown'}")
    print(f"Documents: {result.statistics.documents}")
    print(f"Pages: {result.statistics.pages}")
    if result.statistics.layout_regions:
        print(f"Layout regions: {result.statistics.layout_regions}")
    if result.statistics.entities:
        print(f"Entities: {result.statistics.entities}")
    if result.statistics.qa_pairs:
        print(f"QA pairs: {result.statistics.qa_pairs}")
    if not result.records:
        print("Status: empty / not usable")
        return 2
    print("Status: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
