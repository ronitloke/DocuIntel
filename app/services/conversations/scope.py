"""Internal, migration-free persistence helpers for conversation document scope."""

from __future__ import annotations

import re
from collections.abc import Iterable
from uuid import UUID

_SCOPE_PREFIX = "\x1eDOCUINTEL_SCOPE:"
_SCOPE_SUFFIX = "\x1e"
_SCOPE_PATTERN = re.compile(
    re.escape(_SCOPE_PREFIX) + r"([0-9a-fA-F,-]+)" + re.escape(_SCOPE_SUFFIX)
)


def encode_message_scope(content: str, document_ids: Iterable[UUID]) -> str:
    """Append compact internal scope metadata while preserving user-visible content."""

    unique_ids = list(dict.fromkeys(document_ids))
    normalized = content.strip()
    if not unique_ids:
        return normalized
    encoded_ids = ",".join(str(document_id) for document_id in unique_ids)
    return f"{normalized}{_SCOPE_PREFIX}{encoded_ids}{_SCOPE_SUFFIX}"


def extract_message_scope(content: str) -> list[UUID]:
    """Read the latest valid selected-document scope from internal message metadata."""

    matches = list(_SCOPE_PATTERN.finditer(content))
    if not matches:
        return []
    values: list[UUID] = []
    for raw_id in matches[-1].group(1).split(","):
        try:
            document_id = UUID(raw_id)
        except ValueError:
            return []
        if document_id not in values:
            values.append(document_id)
    return values


def strip_message_scope(content: str) -> str:
    """Remove internal scope metadata before history or public API projection."""

    return _SCOPE_PATTERN.sub("", content).strip()
