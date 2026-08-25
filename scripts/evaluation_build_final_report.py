"""Build the read-only Evaluation E5 final benchmark package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.e5.builder import build_e5  # noqa: E402
from evaluation.e5.loader import E5ArtifactError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the E5 CLI parser."""

    parser = argparse.ArgumentParser(description="Consolidate authoritative DocuIntel evaluation artifacts.")
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/e5/baseline_manifest.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/results/e5"))
    return parser


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Run the read-only E5 builder."""

    args = build_parser().parse_args(argv)
    output = _resolve(args.output_root) / args.run_id
    try:
        result = build_e5(
            manifest_path=_resolve(args.manifest),
            output_directory=output,
            run_id=args.run_id,
            project_root=PROJECT_ROOT,
        )
    except (E5ArtifactError, FileExistsError, OSError, ValueError) as exc:
        print(f"E5 report build failed: {exc}", file=sys.stderr)
        return 2
    print(f"Status: {result.summary['status']}")
    print(f"Run directory: {result.output_directory}")
    print(f"Report: {result.output_directory / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

