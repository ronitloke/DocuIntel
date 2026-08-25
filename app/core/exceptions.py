"""Application exceptions for controlled document-ingestion failures."""


class DocumentIngestionError(Exception):
    """Base error with a safe public message and HTTP status code."""

    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.public_message = message


class InvalidUploadError(DocumentIngestionError):
    """The upload does not satisfy the basic PDF upload contract."""

    status_code = 415


class EmptyUploadError(DocumentIngestionError):
    """The upload contains no bytes."""

    status_code = 400


class UploadTooLargeError(DocumentIngestionError):
    """The upload exceeds the configured size limit."""

    status_code = 413


class CorruptedPDFError(DocumentIngestionError):
    """The file has a PDF signature but cannot be parsed as a PDF."""

    status_code = 422


class EncryptedPDFError(DocumentIngestionError):
    """The PDF requires a password and is unsupported by Module 1."""

    status_code = 422


class DocumentStorageError(DocumentIngestionError):
    """The accepted PDF could not be safely stored."""

    status_code = 500


class PDFProcessingError(DocumentIngestionError):
    """An unexpected, non-public PDF processing failure."""

    status_code = 500


class DatabaseNotConfiguredError(DocumentIngestionError):
    """The application cannot persist because PostgreSQL is not configured."""

    status_code = 503


class DatabaseUnavailableError(DocumentIngestionError):
    """PostgreSQL is configured but cannot currently be reached."""

    status_code = 503


class DocumentPersistenceError(DocumentIngestionError):
    """A document transaction could not be committed safely."""

    status_code = 500


class DuplicateDocumentError(DocumentIngestionError):
    """The exact PDF checksum already exists in the database."""

    status_code = 409


class DocumentNotFoundError(DocumentIngestionError):
    """The requested document or page does not exist."""

    status_code = 404


class EmbeddingServiceError(DocumentIngestionError):
    """The configured local embedding model could not generate vectors."""

    status_code = 503


class SearchValidationError(DocumentIngestionError):
    """The search request exceeds a configured runtime limit."""

    status_code = 422


class RerankerServiceError(DocumentIngestionError):
    """The configured local cross-encoder could not rerank candidates."""

    status_code = 503


class OllamaServiceError(DocumentIngestionError):
    """The local Ollama provider could not generate an answer."""

    status_code = 503


class RAGServiceError(DocumentIngestionError):
    """The RAG provider returned an unsafe or malformed answer."""

    status_code = 502


class AnalysisContentError(DocumentIngestionError):
    """A document cannot be analyzed because indexed content is unavailable."""

    status_code = 422


class AnalysisResponseError(DocumentIngestionError):
    """The analysis provider returned an unsafe or invalid result."""

    status_code = 502


class StructuredExtractionValidationError(DocumentIngestionError):
    """The structured extraction request or provider result is unsafe."""

    status_code = 422


class TableQueryValidationError(DocumentIngestionError):
    """The table plan cannot be executed against the selected table safely."""

    status_code = 422


class PIIValidationError(DocumentIngestionError):
    """The deterministic PII request or selected evidence is invalid."""

    status_code = 422


class PIIRedactionError(DocumentIngestionError):
    """A safe, irreversible PDF redaction could not be completed."""

    status_code = 422


class PIIArtifactNotFoundError(DocumentIngestionError):
    """A generated redacted artifact is not available for download."""

    status_code = 404


class ConversationNotFoundError(DocumentIngestionError):
    """The requested conversation session does not exist."""

    status_code = 404


class ConversationPersistenceError(DocumentIngestionError):
    """A conversation or message could not be persisted safely."""

    status_code = 500
