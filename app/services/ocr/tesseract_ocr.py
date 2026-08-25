"""Tesseract OCR integration for scanned PDF pages."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytesseract
from PIL import Image

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OCRResult:
    """Result of one page-level OCR attempt."""

    text: str
    success: bool
    confidence: float | None
    error: str | None = None


class OCRService(Protocol):
    """Interface used by PDF ingestion for OCR and test doubles."""

    def is_available(self) -> bool:
        """Return whether the configured Tesseract executable can run."""

    def extract(self, image: Image.Image) -> OCRResult:
        """Extract text and confidence from one rendered page image."""


class TesseractOCRService:
    """Resolve and invoke Tesseract without embedding platform paths elsewhere."""

    COMMON_WINDOWS_PATHS = (
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._availability: bool | None = None
        self._resolved_command: str | None = None

    @property
    def resolved_command(self) -> str | None:
        """Return the resolved executable path, if one was found."""

        if self._resolved_command is None:
            self._resolved_command = self._resolve_command()
        return self._resolved_command

    def is_available(self) -> bool:
        """Check the configured Tesseract executable and cache the result."""

        if self._availability is not None:
            return self._availability

        command = self.resolved_command
        if command is None:
            logger.warning(
                "Tesseract is unavailable; set TESSERACT_CMD or add tesseract to PATH."
            )
            self._availability = False
            return False

        pytesseract.pytesseract.tesseract_cmd = command
        try:
            pytesseract.get_tesseract_version()
        except (OSError, RuntimeError, pytesseract.TesseractError) as exc:
            logger.warning("Tesseract cannot be executed at %s: %s", command, exc)
            self._availability = False
            return False

        logger.info("Tesseract OCR available at %s", command)
        self._availability = True
        return True

    def extract(self, image: Image.Image) -> OCRResult:
        """Run word-level Tesseract OCR and calculate mean valid confidence."""

        if not self.is_available():
            return OCRResult(
                text="",
                success=False,
                confidence=None,
                error="Tesseract OCR is unavailable.",
            )

        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.settings.ocr_language,
                output_type=pytesseract.Output.DICT,
            )
        except (OSError, RuntimeError, pytesseract.TesseractError) as exc:
            logger.exception("Tesseract OCR execution failed")
            return OCRResult(
                text="",
                success=False,
                confidence=None,
                error="Tesseract OCR execution failed.",
            )
        except Exception as exc:
            logger.exception("Unexpected Tesseract OCR failure")
            return OCRResult(
                text="",
                success=False,
                confidence=None,
                error="Unexpected Tesseract OCR failure.",
            )

        words: list[str] = []
        confidences: list[float] = []
        raw_words = data.get("text", [])
        raw_confidences = data.get("conf", [])
        for index, raw_word in enumerate(raw_words):
            word = " ".join(str(raw_word).split())
            if not word:
                continue
            words.append(word)
            if index >= len(raw_confidences):
                continue
            try:
                confidence = float(raw_confidences[index])
            except (TypeError, ValueError):
                continue
            if confidence >= 0:
                confidences.append(confidence)

        text = " ".join(words)
        confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
        if len(text) < self.settings.ocr_candidate_char_threshold:
            return OCRResult(
                text=text,
                success=False,
                confidence=confidence,
                error="OCR returned insufficient meaningful text.",
            )

        return OCRResult(
            text=text,
            success=True,
            confidence=confidence,
        )

    def _resolve_command(self) -> str | None:
        """Resolve explicit configuration, PATH, then common Windows install paths."""

        explicit_command = (self.settings.tesseract_cmd or "").strip()
        if explicit_command:
            resolved_explicit = shutil.which(explicit_command)
            if resolved_explicit:
                return resolved_explicit
            if Path(explicit_command).is_file():
                return explicit_command
            logger.error("Configured Tesseract executable does not exist: %s", explicit_command)
            return None

        path_command = shutil.which("tesseract")
        if path_command:
            return path_command

        for candidate in self.COMMON_WINDOWS_PATHS:
            if candidate.is_file():
                return str(candidate)
        return None
