"""Create the safe synthetic PDF used by Module 12.4 acceptance checks."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "sample_pdfs" / "module12_4_pii.pdf"


def create_fixture(output_path: Path = OUTPUT_PATH) -> Path:
    """Write one fictional native-text PDF without modifying uploaded documents."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 72), "DocuIntel Privacy Test Fixture", fontsize=18, fontname="hebo")
    lines = [
        "All values on this page are fictional software-test values.",
        "Email: privacy.test@example.com",
        "Phone: +1 (202) 555-0147",
        "IBAN: GB82 WEST 1234 5698 7654 32",
        "Credit card test number: 4111 1111 1111 1111",
        "Invoice reference: INV-2026-0043",
        "Quantity: 3    Price: 2400    Date: 2026-08-19",
        "The invoice, quantity, price, and date are intentionally non-PII controls.",
    ]
    for index, line in enumerate(lines, start=1):
        page.insert_text((72, 112 + index * 30), line, fontsize=12)
    second_page = document.new_page(width=612, height=792)
    second_page.insert_text((72, 100), "Page two contains no structured PII.", fontsize=12)
    second_page.insert_text((72, 130), "This page exists to verify page-count preservation.", fontsize=12)
    document.set_metadata({"title": "DocuIntel Module 12.4 synthetic PII fixture"})
    try:
        document.save(output_path, garbage=4, deflate=True)
    finally:
        document.close()
    return output_path


if __name__ == "__main__":
    print(create_fixture())
