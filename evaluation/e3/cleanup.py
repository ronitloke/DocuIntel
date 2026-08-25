"""Fail-closed cleanup for documents created by one E3 run."""

from __future__ import annotations

from pathlib import Path

from app.db.repository import DocumentRepository
from evaluation.e3.models import CorpusMapping


class CleanupSafetyError(RuntimeError):
    """Raised when a mapped document cannot be proven to belong to the run."""


def cleanup_run_documents(
    repository: DocumentRepository,
    mappings: list[CorpusMapping],
    storage_directory: Path,
) -> list[CorpusMapping]:
    """Verify every identity first, then delete only exact run-owned documents/files."""

    owned = [mapping for mapping in mappings if mapping.document_id is not None]
    verified: list[tuple[CorpusMapping, str]] = []
    storage_root = storage_directory.resolve()
    for mapping in owned:
        document = repository.get_document(mapping.document_id)  # type: ignore[arg-type]
        if document is None:
            raise CleanupSafetyError(
                f"Mapped E3 document is missing; refusing cleanup: {mapping.document_id}."
            )
        if document.original_filename != mapping.original_filename:
            raise CleanupSafetyError(
                f"E3 cleanup identity mismatch for {mapping.document_id}: original filename changed."
            )
        if mapping.checksum_sha256 and document.checksum_sha256 != mapping.checksum_sha256:
            raise CleanupSafetyError(
                f"E3 cleanup identity mismatch for {mapping.document_id}: checksum changed."
            )
        if mapping.stored_filename and document.stored_filename != mapping.stored_filename:
            raise CleanupSafetyError(
                f"E3 cleanup identity mismatch for {mapping.document_id}: stored filename changed."
            )
        candidate = (storage_root / document.stored_filename).resolve()
        if candidate.parent != storage_root or candidate.name != document.stored_filename:
            raise CleanupSafetyError(
                f"E3 cleanup refused unsafe stored path for {mapping.document_id}."
            )
        verified.append((mapping, document.stored_filename))

    cleaned: list[CorpusMapping] = []
    for mapping, stored_filename in verified:
        deleted_filename = repository.delete_document(mapping.document_id)  # type: ignore[arg-type]
        if deleted_filename != stored_filename:
            raise CleanupSafetyError(
                f"E3 cleanup returned an unexpected stored filename for {mapping.document_id}."
            )
        (storage_root / stored_filename).unlink(missing_ok=True)
        cleaned.append(mapping.model_copy(update={"cleaned_up": True}))
    return cleaned

