"""Create safe, deterministic PDFs for local Module 12.3 acceptance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf as fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "sample_pdfs"


def _add_text_page(document: fitz.Document, heading: str, text: str) -> None:
    page = document.new_page(width=600, height=800)
    page.insert_text((60, 70), heading, fontsize=18, fontname="hebo")
    page.insert_textbox((60, 110, 540, 740), text, fontsize=9, lineheight=1.4)


def _add_table_page(document: fitz.Document, rows: list[list[str]]) -> None:
    page = document.new_page(width=600, height=500)
    page.insert_text((60, 60), "Product schedule", fontsize=18, fontname="hebo")
    x_positions = [60, 230, 330, 450]
    y_positions = [120, 165, 210, 255, 300]
    bottom_y = y_positions[len(rows) + 1]
    for x_position in x_positions:
        page.draw_line((x_position, y_positions[0]), (x_position, bottom_y))
    for y_position in y_positions[: len(rows) + 2]:
        page.draw_line((x_positions[0], y_position), (x_positions[-1], y_position))
    values = [["Product", "Qty", "Price"], *rows]
    for row_index, row in enumerate(values):
        y = 150 + row_index * 45
        for column_index, value in enumerate(row):
            page.insert_text((x_positions[column_index] + 10, y), value, fontsize=10)


def _create_pair(output_directory: Path) -> tuple[Path, Path]:
    base = fitz.open()
    _add_text_page(
        base,
        "Employment Policy",
        "Employees must give thirty days written notice before resignation.",
    )
    _add_text_page(base, "Training", "Annual training is mandatory.")
    _add_text_page(base, "Remote work", "Remote work is not permitted.")
    _add_table_page(base, [["Laptop", "3", "2400"], ["Mouse", "5", "20"]])
    base_path = output_directory / "module12_3_base.pdf"
    base.save(base_path)
    base.close()

    target = fitz.open()
    _add_text_page(
        target,
        "Employment Policy",
        "Employees must give forty-five days written notice before resignation.",
    )
    _add_text_page(target, "Remote work", "Remote work is permitted two days per week.")
    _add_text_page(target, "Expenses", "Expense claims must be submitted within fourteen days.")
    _add_table_page(target, [["Laptop", "4", "2600"], ["Mouse", "5", "20"], ["Keyboard", "2", "80"]])
    target_path = output_directory / "module12_3_target.pdf"
    target.save(target_path)
    target.close()
    return base_path, target_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    base_path, target_path = _create_pair(args.output_directory)
    print(base_path)
    print(target_path)


if __name__ == "__main__":
    main()
