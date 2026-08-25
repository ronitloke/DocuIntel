"""Deterministic high-confidence PII detectors with conservative validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.privacy import PIIType


@dataclass(frozen=True, slots=True)
class PIICandidate:
    """A regex candidate before page and PDF-coordinate projection."""

    pii_type: PIIType
    matched_text: str
    start_offset: int
    end_offset: int
    detector: str
    validation_status: str


EMAIL_RE = re.compile(
    r"(?<![\w.+-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
    r"(?![\w.-])"
)

# Separators and a three-digit first/area group are intentional. This avoids
# classifying dates, quantities, invoice references, and arbitrary long IDs.
PHONE_RE = re.compile(
    r"(?<![\w])"
    r"(?P<country>\+\d{1,3}[ .-]?)?"
    r"(?P<area>\(?\d{3}\)?)[ .-]"
    r"\d{3,4}[ .-]\d{3,4}"
    r"(?![\w])"
)

IBAN_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Za-z]{2}[ -]?\d{2}(?:[ -]?[A-Za-z0-9]){10,34}"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

CREDIT_CARD_RE = re.compile(
    r"(?<![\w])(?:\d[ -]?){13,19}(?![\w])"
)

# ISO 13616 lengths for the common local/test formats supported here. Requiring
# a known country length prevents long arbitrary alphanumeric strings from
# becoming high-confidence IBAN detections.
IBAN_LENGTHS = {
    "AT": 20,
    "BE": 16,
    "CH": 21,
    "DE": 22,
    "DK": 18,
    "ES": 24,
    "FI": 18,
    "FR": 27,
    "GB": 22,
    "IE": 22,
    "IT": 27,
    "LU": 20,
    "NL": 18,
    "NO": 15,
    "PL": 28,
    "PT": 25,
    "SE": 24,
}


def detect_pii(text: str, pii_types: list[PIIType]) -> list[PIICandidate]:
    """Detect supported PII in deterministic page order."""

    candidates: list[PIICandidate] = []
    detectors = {
        PIIType.EMAIL: _detect_emails,
        PIIType.PHONE_NUMBER: _detect_phone_numbers,
        PIIType.IBAN: _detect_ibans,
        PIIType.CREDIT_CARD: _detect_credit_cards,
    }
    for pii_type in pii_types:
        candidates.extend(detectors[pii_type](text))

    # A physical text span is represented only once for a detector category.
    # Keeping the type in the key still permits a future detector to classify a
    # genuinely distinct span that happens to share the same character range.
    unique: dict[tuple[int, int, PIIType, str], PIICandidate] = {}
    for candidate in candidates:
        key = (
            candidate.start_offset,
            candidate.end_offset,
            candidate.pii_type,
            " ".join(candidate.matched_text.split()).casefold(),
        )
        unique.setdefault(key, candidate)
    return sorted(
        unique.values(),
        key=lambda item: (item.start_offset, item.end_offset, item.pii_type.value),
    )


def _detect_emails(text: str) -> list[PIICandidate]:
    return [
        PIICandidate(
            pii_type=PIIType.EMAIL,
            matched_text=match.group(0),
            start_offset=match.start(),
            end_offset=match.end(),
            detector="email_regex",
            validation_status="format_validated",
        )
        for match in EMAIL_RE.finditer(text)
    ]


def _detect_phone_numbers(text: str) -> list[PIICandidate]:
    candidates: list[PIICandidate] = []
    for match in PHONE_RE.finditer(text):
        value = match.group(0)
        digits = re.sub(r"\D", "", value)
        if not 10 <= len(digits) <= 15:
            continue
        candidates.append(
            PIICandidate(
                pii_type=PIIType.PHONE_NUMBER,
                matched_text=value,
                start_offset=match.start(),
                end_offset=match.end(),
                detector="conservative_phone_regex",
                validation_status="format_validated",
            )
        )
    return candidates


def _detect_ibans(text: str) -> list[PIICandidate]:
    candidates: list[PIICandidate] = []
    for match in IBAN_RE.finditer(text):
        value = match.group(0)
        normalized = re.sub(r"[ -]", "", value).upper()
        if not _valid_iban(normalized):
            continue
        candidates.append(
            PIICandidate(
                pii_type=PIIType.IBAN,
                matched_text=value,
                start_offset=match.start(),
                end_offset=match.end(),
                detector="iban_mod97_checksum",
                validation_status="format_and_checksum_validated",
            )
        )
    return candidates


def _detect_credit_cards(text: str) -> list[PIICandidate]:
    candidates: list[PIICandidate] = []
    for match in CREDIT_CARD_RE.finditer(text):
        value = match.group(0).strip()
        digits = re.sub(r"[ -]", "", value)
        if not 13 <= len(digits) <= 19 or not _luhn_valid(digits):
            continue
        candidates.append(
            PIICandidate(
                pii_type=PIIType.CREDIT_CARD,
                matched_text=value,
                start_offset=match.start(),
                end_offset=match.start() + len(value),
                detector="credit_card_luhn",
                validation_status="luhn_validated",
            )
        )
    return candidates


def _valid_iban(value: str) -> bool:
    """Validate country length, alphanumeric format, and ISO 13616 checksum."""

    if len(value) < 5 or not value.isalnum() or not value[:2].isalpha():
        return False
    country = value[:2]
    if country not in IBAN_LENGTHS or len(value) != IBAN_LENGTHS[country]:
        return False
    if not value[2:4].isdigit():
        return False
    rearranged = value[4:] + value[:4]
    numeric = ""
    for character in rearranged:
        numeric += str(ord(character) - ord("A") + 10) if character.isalpha() else character
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def _luhn_valid(digits: str) -> bool:
    """Return whether a digit string passes the Luhn checksum."""

    total = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0
