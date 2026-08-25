"""Create safe local PDFs for DocuIntel OCR and structure demonstrations."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pymupdf as fitz
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "sample_pdfs"


def _save_document(document: fitz.Document, output_path: Path) -> None:
    """Save and close one generated document."""

    try:
        document.save(output_path)
    finally:
        document.close()


def _render_page_as_png(page: fitz.Page, dpi: int = 150) -> bytes:
    """Render a native page into an in-memory PNG for scanned-PDF fixtures."""

    pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    try:
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    finally:
        image.close()


def _create_native_text_pdf(output_path: Path) -> None:
    """Create a native-text PDF with heading, paragraph, and list content."""

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 90), "1. Introduction", fontsize=20, fontname="hebo")
    page.insert_text(
        (72, 135),
        "This native PDF demonstrates page text extraction and heuristic structure.",
        fontsize=11,
    )
    page.insert_text((72, 180), "- Native text is extracted without OCR.", fontsize=11)
    page.insert_text((72, 205), "- Headings and list items are classified heuristically.", fontsize=11)
    _save_document(document, output_path)


def _create_scanned_text_pdf(output_path: Path) -> None:
    """Create an image-only PDF by rasterising a native source page."""

    source = fitz.open()
    source_page = source.new_page()
    source_page.insert_text(
        (72, 120),
        "Scanned sample recovered through Tesseract OCR.",
        fontsize=18,
    )
    source_page.insert_text((72, 170), "OCR confidence is reported when available.", fontsize=16)
    page_rect = source_page.rect
    image_bytes = _render_page_as_png(source_page)
    source.close()

    scanned = fitz.open()
    page = scanned.new_page(width=page_rect.width, height=page_rect.height)
    page.insert_image(page.rect, stream=image_bytes)
    _save_document(scanned, output_path)


def _create_mixed_pdf(output_path: Path) -> None:
    """Create a two-page PDF with one native and one image-only page."""

    document = fitz.open()
    native_page = document.new_page()
    native_page.insert_text((72, 100), "Page 1 remains native text.", fontsize=18)
    source = fitz.open()
    scanned_source_page = source.new_page()
    scanned_source_page.insert_text((72, 100), "Page 2 requires selective OCR.", fontsize=18)
    image_bytes = _render_page_as_png(scanned_source_page)
    page_rect = scanned_source_page.rect
    source.close()
    scanned_page = document.new_page(width=page_rect.width, height=page_rect.height)
    scanned_page.insert_image(scanned_page.rect, stream=image_bytes)
    _save_document(document, output_path)


def _create_layout_table_pdf(output_path: Path) -> None:
    """Create a native PDF containing heuristic layout content and a simple table."""

    document = fitz.open()
    page = document.new_page(width=500, height=500)
    page.insert_text((60, 60), "Payment Schedule", fontsize=22, fontname="hebo")
    page.insert_text(
        (60, 100),
        "The following table lists example payment items.",
        fontsize=11,
    )
    page.insert_text((60, 130), "1. Review the product.", fontsize=11)
    page.insert_text((60, 155), "2. Confirm the quantity.", fontsize=11)

    x_positions = [60, 250, 350, 440]
    y_positions = [200, 245, 290]
    for x_position in x_positions:
        page.draw_line((x_position, y_positions[0]), (x_position, y_positions[-1]))
    for y_position in y_positions:
        page.draw_line((x_positions[0], y_position), (x_positions[-1], y_position))
    page.insert_text((70, 230), "Product", fontsize=10)
    page.insert_text((260, 230), "Qty", fontsize=10)
    page.insert_text((360, 230), "Price", fontsize=10)
    page.insert_text((70, 275), "Laptop", fontsize=10)
    page.insert_text((260, 275), "3", fontsize=10)
    page.insert_text((360, 275), "2400", fontsize=10)
    _save_document(document, output_path)


def create_sample_pdfs(output_directory: Path = DEFAULT_OUTPUT_DIRECTORY) -> None:
    """Generate all local, non-confidential sample PDFs."""

    output_directory.mkdir(parents=True, exist_ok=True)
    _create_native_text_pdf(output_directory / "native_text_sample.pdf")
    _create_scanned_text_pdf(output_directory / "scanned_text_sample.pdf")
    _create_mixed_pdf(output_directory / "mixed_sample.pdf")
    _create_layout_table_pdf(output_directory / "layout_table_sample.pdf")


def main() -> None:
    """Generate sample PDFs under the project sample directory."""

    create_sample_pdfs()


if __name__ == "__main__":
    main()
