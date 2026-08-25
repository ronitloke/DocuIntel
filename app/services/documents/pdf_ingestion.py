"""PDF ingestion, selective OCR, and heuristic document structure extraction."""

from __future__ import annotations

import logging
import hashlib
import re
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Protocol
from uuid import UUID, uuid4

import pymupdf as fitz
from PIL import Image

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    CorruptedPDFError,
    DocumentIngestionError,
    DocumentStorageError,
    EmptyUploadError,
    EncryptedPDFError,
    InvalidUploadError,
    PDFProcessingError,
    UploadTooLargeError,
)
from app.models.documents import (
    DocumentIngestionResponse,
    DocumentStatus,
    ExtractedTable,
    LayoutElement,
    PDFMetadata,
    PageExtraction,
)
from app.services.ocr.tesseract_ocr import OCRResult, OCRService, TesseractOCRService

logger = logging.getLogger(__name__)

PDF_SIGNATURE = b"%PDF-"
READ_CHUNK_SIZE = 1024 * 1024
SUPPORTED_PDF_MIME_TYPES = frozenset({"application/pdf", "application/octet-stream"})
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "uploads"
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][)])\s+")


class UploadStream(Protocol):
    """Minimum async upload interface required by the ingestion service."""

    filename: str | None
    content_type: str | None

    def read(self, size: int = -1) -> Awaitable[bytes]:
        """Read bytes from the upload stream."""


@dataclass(frozen=True, slots=True)
class _NativeTextBlock:
    """Internal normalized representation of one native text block."""

    text: str
    bbox: list[float]
    font_size: float | None
    is_bold: bool


@dataclass(frozen=True, slots=True)
class _UploadWriteResult:
    """Size and deterministic checksum produced while streaming an upload."""

    file_size_bytes: int
    checksum_sha256: str


class PDFIngestionService:
    """Validate, store, extract, and optionally OCR one PDF upload."""

    def __init__(
        self,
        settings: Settings | None = None,
        storage_directory: Path | None = None,
        ocr_service: OCRService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage_directory = storage_directory or DEFAULT_UPLOAD_DIRECTORY
        self.ocr_service = ocr_service or TesseractOCRService(self.settings)

    async def ingest(self, upload: UploadStream) -> DocumentIngestionResponse:
        """Ingest one upload and return structured native/OCR extraction results."""

        filename = upload.filename or ""
        content_type = self._normalise_content_type(upload.content_type)
        logger.info(
            "PDF upload received filename=%s content_type=%s",
            filename,
            content_type or "missing",
        )
        self._validate_upload_headers(filename, content_type)

        document_id = uuid4()
        temporary_path = self.storage_directory / f".{document_id}.uploading"
        stored_filename = f"{document_id}.pdf"
        stored_path = self.storage_directory / stored_filename

        try:
            self._ensure_storage_directory()
            upload_result = await self._write_upload(upload, temporary_path)
            result = self._extract_pdf(
                document_id=document_id,
                original_filename=filename,
                stored_filename=stored_filename,
                source_path=temporary_path,
                mime_type=content_type,
                file_size=upload_result.file_size_bytes,
                checksum_sha256=upload_result.checksum_sha256,
            )
            self._finalise_storage(temporary_path, stored_path)
            logger.info(
                "PDF extraction completed document_id=%s native_text_pages=%s "
                "ocr_pages=%s unresolved_ocr_pages=%s layout_elements=%s tables=%s",
                document_id,
                result.pages_with_native_text,
                result.pages_processed_by_ocr,
                result.unresolved_ocr_pages,
                result.layout_element_count,
                result.table_count,
            )
            return result
        except DocumentIngestionError:
            raise
        except OSError as exc:
            logger.exception("PDF storage failure document_id=%s", document_id)
            raise DocumentStorageError("The PDF could not be stored safely.") from exc
        except Exception as exc:
            logger.exception("Unexpected PDF ingestion failure document_id=%s", document_id)
            raise PDFProcessingError("The PDF could not be processed.") from exc
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "Temporary upload cleanup failed document_id=%s error=%s",
                    document_id,
                    exc,
                )

    @staticmethod
    def _normalise_content_type(content_type: str | None) -> str:
        """Return the MIME type without optional parameters."""

        return (content_type or "").split(";", maxsplit=1)[0].strip().lower()

    @staticmethod
    def _validate_upload_headers(filename: str, content_type: str) -> None:
        """Validate metadata before any file is written to disk."""

        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise InvalidUploadError("Only files with a .pdf extension are supported.")
        if content_type not in SUPPORTED_PDF_MIME_TYPES:
            raise InvalidUploadError(
                "The uploaded file must use a PDF content type such as application/pdf."
            )

    def _ensure_storage_directory(self) -> None:
        """Create the project-local upload directory when needed."""

        try:
            self.storage_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DocumentStorageError("The PDF upload directory is unavailable.") from exc

    async def _write_upload(
        self,
        upload: UploadStream,
        temporary_path: Path,
    ) -> _UploadWriteResult:
        """Stream an upload to a temporary path while enforcing its size limit."""

        maximum_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        total_bytes = 0
        first_chunk = True
        checksum = hashlib.sha256()

        try:
            with temporary_path.open("xb") as destination:
                while True:
                    chunk = await upload.read(READ_CHUNK_SIZE)
                    if not chunk:
                        break
                    if first_chunk:
                        first_chunk = False
                        if not chunk.startswith(PDF_SIGNATURE):
                            raise InvalidUploadError(
                                "The uploaded file does not have a valid PDF signature."
                            )
                    next_total = total_bytes + len(chunk)
                    if next_total > maximum_bytes:
                        raise UploadTooLargeError(
                            f"The PDF exceeds the {self.settings.max_upload_size_mb} MB upload limit."
                        )
                    destination.write(chunk)
                    checksum.update(chunk)
                    total_bytes = next_total
        except DocumentIngestionError:
            raise
        except OSError as exc:
            raise DocumentStorageError("The PDF upload could not be written safely.") from exc

        if total_bytes == 0:
            raise EmptyUploadError("The uploaded PDF is empty.")
        return _UploadWriteResult(
            file_size_bytes=total_bytes,
            checksum_sha256=checksum.hexdigest(),
        )

    def _extract_pdf(
        self,
        document_id: UUID,
        original_filename: str,
        stored_filename: str,
        source_path: Path,
        mime_type: str,
        file_size: int,
        checksum_sha256: str,
    ) -> DocumentIngestionResponse:
        """Open a validated PDF and extract metadata, structure, and page text."""

        try:
            pdf_bytes = source_path.read_bytes()
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                if bool(getattr(document, "needs_pass", False)):
                    raise EncryptedPDFError(
                        "Password-protected PDFs are not supported yet."
                    )

                logger.info(
                    "PDF opened document_id=%s page_count=%s",
                    document_id,
                    document.page_count,
                )
                structured_pages = [
                    (page_number, page, page.get_text("dict"))
                    for page_number, page in enumerate(document, start=1)
                ]
                document_font_sizes = self._document_font_sizes(
                    page_dict for _, _, page_dict in structured_pages
                )
                pages: list[PageExtraction] = []
                next_table_index = 1
                for page_number, page, page_dict in structured_pages:
                    page_result = self._extract_page(
                        page_number=page_number,
                        page=page,
                        page_dict=page_dict,
                        document_font_sizes=document_font_sizes,
                        table_index_start=next_table_index,
                    )
                    pages.append(page_result)
                    next_table_index += len(page_result.tables)
                metadata = self._extract_metadata(document.metadata)
        except DocumentIngestionError:
            raise
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            logger.warning("Corrupted PDF document_id=%s error=%s", document_id, exc)
            raise CorruptedPDFError("The uploaded file could not be parsed as a PDF.") from exc
        except Exception as exc:
            logger.exception("PDF extraction failure document_id=%s", document_id)
            raise PDFProcessingError("The PDF could not be processed.") from exc

        layout_elements = [element for page in pages for element in page.layout_elements]
        tables = [table for page in pages for table in page.tables]
        unresolved_ocr_pages = sum(page.needs_ocr for page in pages)
        return DocumentIngestionResponse(
            document_id=document_id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            mime_type=mime_type,
            file_size_bytes=file_size,
            checksum_sha256=checksum_sha256,
            page_count=len(pages),
            pages_with_native_text=sum(page.has_native_text for page in pages),
            pages_requiring_ocr=sum(page.needs_ocr for page in pages),
            pages_processed_by_ocr=sum(page.ocr_applied for page in pages),
            ocr_failed_pages=sum(
                page.ocr_applied and page.ocr_success is False for page in pages
            ),
            unresolved_ocr_pages=unresolved_ocr_pages,
            heading_count=sum(
                element.element_type == "heading" for element in layout_elements
            ),
            table_count=len(tables),
            layout_element_count=len(layout_elements),
            status=DocumentStatus.FAILED
            if unresolved_ocr_pages
            else DocumentStatus.READY,
            metadata=metadata,
            pages=pages,
        )

    def _extract_page(
        self,
        page_number: int,
        page: fitz.Page,
        page_dict: dict[str, Any],
        document_font_sizes: list[float],
        table_index_start: int,
    ) -> PageExtraction:
        """Extract one page and apply OCR only when the page needs it."""

        native_text = page.get_text("text") or ""
        meaningful_native_text = self._normalise_text(native_text)
        has_native_text = bool(meaningful_native_text)
        needs_ocr = (
            not has_native_text
            or len(meaningful_native_text) < self.settings.ocr_candidate_char_threshold
        )
        tables = self._extract_tables(page, page_number, table_index_start)
        layout_elements = self._extract_layout_elements(
            page_dict=page_dict,
            document_font_sizes=document_font_sizes,
            tables=tables,
        )
        result = PageExtraction(
            page_number=page_number,
            text=native_text,
            character_count=len(native_text),
            has_native_text=has_native_text,
            needs_ocr=needs_ocr,
            extraction_method="native",
            layout_elements=layout_elements,
            tables=tables,
        )

        # A truly blank page has no visual content and should remain an unresolved
        # OCR candidate rather than invoking Tesseract on an empty canvas.
        if needs_ocr and (has_native_text or self._has_image_content(page)):
            return self._apply_ocr(page_number, page, result)
        return result

    def _apply_ocr(
        self,
        page_number: int,
        page: fitz.Page,
        page_result: PageExtraction,
    ) -> PageExtraction:
        """Render a candidate page and merge a controlled OCR result."""

        logger.info("OCR fallback triggered page=%s", page_number)
        base_update: dict[str, Any] = {
            "ocr_applied": True,
            "ocr_success": False,
            "extraction_method": "ocr",
            "needs_ocr": True,
        }
        try:
            if not self.ocr_service.is_available():
                logger.warning("OCR unresolved page=%s because Tesseract is unavailable", page_number)
                return page_result.model_copy(update=base_update)

            pixmap = page.get_pixmap(dpi=self.settings.ocr_render_dpi, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            try:
                ocr_result = self.ocr_service.extract(image)
            finally:
                image.close()
        except Exception as exc:
            logger.exception("OCR rendering or processing failed page=%s", page_number)
            ocr_result = OCRResult(
                text="",
                success=False,
                confidence=None,
                error="OCR rendering or processing failed.",
            )

        if ocr_result.success:
            text = self._normalise_text(ocr_result.text)
            logger.info(
                "OCR succeeded page=%s confidence=%s character_count=%s",
                page_number,
                ocr_result.confidence,
                len(text),
            )
            return page_result.model_copy(
                update={
                    "text": text,
                    "character_count": len(text),
                    "needs_ocr": False,
                    "ocr_applied": True,
                    "ocr_success": True,
                    "ocr_confidence": ocr_result.confidence,
                    "extraction_method": "ocr",
                }
            )

        logger.warning(
            "OCR failed page=%s reason=%s confidence=%s",
            page_number,
            ocr_result.error or "insufficient OCR text",
            ocr_result.confidence,
        )
        failure_text = self._normalise_text(ocr_result.text) or self._normalise_text(
            page_result.text
        )
        return page_result.model_copy(
            update={
                **base_update,
                "text": failure_text,
                "character_count": len(failure_text),
                "ocr_confidence": ocr_result.confidence,
            }
        )

    @staticmethod
    def _has_image_content(page: fitz.Page) -> bool:
        """Return whether a page contains an embedded image suitable for OCR."""

        try:
            return bool(page.get_images(full=True))
        except Exception as exc:
            logger.warning("Could not inspect page image content: %s", exc)
            return False

    @staticmethod
    def _document_font_sizes(page_dicts: Any) -> list[float]:
        """Collect document-wide span sizes for relative heading heuristics."""

        sizes: list[float] = []
        for page_dict in page_dicts:
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        try:
                            size = float(span.get("size", 0))
                        except (TypeError, ValueError):
                            continue
                        if size > 0:
                            sizes.append(size)
        return sizes

    def _extract_layout_elements(
        self,
        page_dict: dict[str, Any],
        document_font_sizes: list[float],
        tables: list[ExtractedTable],
    ) -> list[LayoutElement]:
        """Create heuristic heading, paragraph, list, and table elements."""

        blocks: list[_NativeTextBlock] = []
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            spans = [
                span
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ]
            text = self._normalise_text(" ".join(str(span.get("text", "")) for span in spans))
            bbox = self._normalise_bbox(block.get("bbox"))
            if not text or bbox is None or self._overlaps_table(bbox, tables):
                continue
            sizes = []
            for span in spans:
                try:
                    size = float(span.get("size", 0))
                except (TypeError, ValueError):
                    continue
                if size > 0:
                    sizes.append(size)
            font_size = max(sizes) if sizes else None
            is_bold = any(self._span_is_bold(span) for span in spans)
            blocks.append(
                _NativeTextBlock(
                    text=text,
                    bbox=bbox,
                    font_size=font_size,
                    is_bold=is_bold,
                )
            )

        baseline = median(document_font_sizes) if document_font_sizes else 1.0
        elements = [
            LayoutElement(
                element_type=self._classify_text_block(block, baseline),
                text=block.text,
                bbox=block.bbox,
                font_size=round(block.font_size, 2) if block.font_size else None,
                is_bold=block.is_bold,
            )
            for block in blocks
        ]
        for table in tables:
            table_text = " | ".join(
                part
                for part in [*table.headers, *(cell for row in table.rows for cell in row)]
                if part
            )
            elements.append(
                LayoutElement(
                    element_type="table",
                    text=table_text,
                    bbox=table.bbox,
                    table_index=table.table_index,
                )
            )
        return sorted(elements, key=lambda element: (element.bbox[1], element.bbox[0]))

    @staticmethod
    def _classify_text_block(block: _NativeTextBlock, baseline: float) -> str:
        """Classify a native text block using relative size, emphasis, and prefixes."""

        short_text = len(block.text) <= 140 and len(block.text.split()) <= 20
        relative_size = bool(block.font_size and block.font_size >= baseline * 1.2)
        bold_emphasis = bool(block.is_bold and block.font_size and block.font_size >= baseline * 1.05)
        uppercase_label = (
            block.text.upper() == block.text
            and any(character.isalpha() for character in block.text)
            and len(block.text) <= 100
        )
        if short_text and (relative_size or bold_emphasis or uppercase_label):
            return "heading"
        if LIST_PREFIX_RE.match(block.text):
            return "list_item"
        return "paragraph"

    @staticmethod
    def _span_is_bold(span: dict[str, Any]) -> bool:
        """Read PyMuPDF bold flags defensively across PDF font variants."""

        try:
            flags = int(span.get("flags") or 0)
        except (TypeError, ValueError):
            flags = 0
        return "bold" in str(span.get("font", "")).lower() or bool(flags & 16)

    @staticmethod
    def _extract_tables(
        page: fitz.Page,
        page_number: int,
        table_index_start: int,
    ) -> list[ExtractedTable]:
        """Extract simple native tables, continuing safely when detection fails."""

        try:
            finder = page.find_tables()
            raw_tables = getattr(finder, "tables", [])
            tables: list[ExtractedTable] = []
            for offset, table in enumerate(raw_tables):
                rows = [
                    [PDFIngestionService._normalise_text("" if cell is None else str(cell)) for cell in row]
                    for row in (table.extract() or [])
                ]
                header_object = getattr(table, "header", None)
                headers = [
                    PDFIngestionService._normalise_text("" if header is None else str(header))
                    for header in (getattr(header_object, "names", None) or [])
                ]
                if headers and rows and rows[0] == headers:
                    rows = rows[1:]
                bbox = PDFIngestionService._normalise_bbox(getattr(table, "bbox", None))
                if bbox is None:
                    continue
                tables.append(
                    ExtractedTable(
                        table_index=table_index_start + offset,
                        page_number=page_number,
                        bbox=bbox,
                        headers=headers,
                        rows=rows,
                    )
                )
            return tables
        except Exception as exc:
            logger.warning("Table extraction failed page=%s error=%s", page_number, exc)
            return []

    @staticmethod
    def _overlaps_table(bbox: list[float], tables: list[ExtractedTable]) -> bool:
        """Avoid duplicating table cell text as paragraph layout elements."""

        for table in tables:
            left = max(bbox[0], table.bbox[0])
            top = max(bbox[1], table.bbox[1])
            right = min(bbox[2], table.bbox[2])
            bottom = min(bbox[3], table.bbox[3])
            if right > left and bottom > top:
                return True
        return False

    @staticmethod
    def _normalise_bbox(value: Any) -> list[float] | None:
        """Convert a PyMuPDF bounding box into a validated JSON-friendly list."""

        if value is None:
            return None
        try:
            values = [float(part) for part in value]
        except (TypeError, ValueError):
            return None
        return values if len(values) == 4 else None

    @staticmethod
    def _normalise_text(value: str) -> str:
        """Collapse OCR/layout whitespace without logging or inventing content."""

        return " ".join(value.split())

    @staticmethod
    def _extract_metadata(metadata: dict[str, str | None] | None) -> PDFMetadata:
        """Map PyMuPDF metadata keys without inventing missing values."""

        values = metadata or {}
        return PDFMetadata(
            title=PDFIngestionService._optional_metadata_value(values.get("title")),
            author=PDFIngestionService._optional_metadata_value(values.get("author")),
            subject=PDFIngestionService._optional_metadata_value(values.get("subject")),
            keywords=PDFIngestionService._optional_metadata_value(values.get("keywords")),
            creator=PDFIngestionService._optional_metadata_value(values.get("creator")),
            producer=PDFIngestionService._optional_metadata_value(values.get("producer")),
            creation_date=PDFIngestionService._optional_metadata_value(
                values.get("creationDate")
            ),
            modification_date=PDFIngestionService._optional_metadata_value(
                values.get("modDate")
            ),
        )

    @staticmethod
    def _optional_metadata_value(value: str | None) -> str | None:
        """Convert empty metadata strings to JSON null."""

        if value is None:
            return None
        normalised = str(value).strip()
        return normalised or None

    @staticmethod
    def _finalise_storage(temporary_path: Path, stored_path: Path) -> None:
        """Move a successfully processed temporary file without overwriting."""

        if stored_path.exists():
            raise DocumentStorageError("A document with the generated ID already exists.")
        try:
            temporary_path.rename(stored_path)
        except OSError as exc:
            raise DocumentStorageError("The processed PDF could not be finalised.") from exc
