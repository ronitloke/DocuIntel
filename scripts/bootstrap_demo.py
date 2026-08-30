"""Populate a running DocuIntel deployment with the reviewed demo corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "http://127.0.0.1:8001"
DEFAULT_FRONTEND_URL = "http://localhost:8501"
DEFAULT_TIMEOUT_SECONDS = 180.0
UPLOAD_PATH = "/api/v1/documents/upload"
DOCUMENTS_PATH = "/api/v1/documents"
READY_PATH = "/ready"


@dataclass(frozen=True, slots=True)
class DemoFixture:
    """One reviewed, repository-tracked fixture in the demo corpus."""

    key: str
    display_name: str
    fixture_path: str
    capabilities: tuple[str, ...]


DEMO_MANIFEST: tuple[DemoFixture, ...] = (
    DemoFixture(
        key="employment_policy",
        display_name="Employment policy",
        fixture_path="data/sample_pdfs/module9-evaluation.pdf",
        capabilities=("qa", "summary", "classification", "extraction"),
    ),
    DemoFixture(
        key="table_intelligence",
        display_name="Table intelligence",
        fixture_path="data/sample_pdfs/layout_table_sample.pdf",
        capabilities=("table_query",),
    ),
    DemoFixture(
        key="comparison_base",
        display_name="Comparison base",
        fixture_path="data/sample_pdfs/module12_3_base.pdf",
        capabilities=("comparison_base",),
    ),
    DemoFixture(
        key="comparison_target",
        display_name="Comparison target",
        fixture_path="data/sample_pdfs/module12_3_target.pdf",
        capabilities=("comparison_target",),
    ),
    DemoFixture(
        key="privacy_redaction",
        display_name="Privacy and redaction",
        fixture_path="data/sample_pdfs/module12_4_pii.pdf",
        capabilities=("pii",),
    ),
    DemoFixture(
        key="ocr_demo",
        display_name="OCR demo",
        fixture_path="data/sample_pdfs/scanned_text_sample.pdf",
        capabilities=("ocr",),
    ),
)


class BootstrapFailure(RuntimeError):
    """A controlled failure suitable for concise terminal output."""


class APIClientProtocol(Protocol):
    """Operations needed by the bootstrap orchestration."""

    def check_ready(self) -> None:
        """Confirm that the running API and required database are ready."""

    def list_documents(self) -> list[dict[str, Any]]:
        """Return document metadata through the public API."""

    def upload_document(self, filename: str, content: bytes) -> dict[str, Any]:
        """Upload one fixture through the public ingestion endpoint."""

    def index_document(self, document_id: str) -> dict[str, Any]:
        """Run the existing production indexing endpoint."""


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    """Counts from one bootstrap run."""

    added: int
    already_present: int
    failed: int


class APIRequestError(BootstrapFailure):
    """A safe, high-level public API failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BootstrapAPIClient:
    """Small standard-library HTTP client for the existing FastAPI contract."""

    def __init__(self, base_url: str, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def check_ready(self) -> None:
        """Require API readiness before attempting any uploads."""

        payload = self._request_json("GET", READY_PATH)
        if not isinstance(payload, Mapping) or payload.get("status") != "healthy":
            raise APIRequestError("DocuIntel API is not ready. Check PostgreSQL readiness first.")

    def list_documents(self) -> list[dict[str, Any]]:
        """List all document identities needed for checksum-based idempotency."""

        page = 1
        documents: list[dict[str, Any]] = []
        while True:
            payload = self._request_json(
                "GET",
                DOCUMENTS_PATH,
                query=f"?page={page}&page_size=100",
            )
            if not isinstance(payload, Mapping) or not isinstance(payload.get("items"), list):
                raise APIRequestError("The document-list response was malformed.")
            documents.extend(item for item in payload["items"] if isinstance(item, Mapping))
            total_pages = payload.get("total_pages", page)
            if not isinstance(total_pages, int) or page >= total_pages:
                return [dict(item) for item in documents]
            page += 1

    def upload_document(self, filename: str, content: bytes) -> dict[str, Any]:
        """Upload a PDF through the same endpoint used by the application."""

        boundary = "----DocuIntelDemoBootstrapBoundary"
        body = _multipart_body(boundary, filename, content)
        payload = self._request_json(
            "POST",
            UPLOAD_PATH,
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        if not isinstance(payload, Mapping) or not payload.get("document_id"):
            raise APIRequestError("The upload response was malformed.")
        return dict(payload)

    def index_document(self, document_id: str) -> dict[str, Any]:
        """Generate chunks and embeddings through the existing index endpoint."""

        payload = self._request_json(
            "POST",
            f"{DOCUMENTS_PATH}/{document_id}/index",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        if not isinstance(payload, Mapping) or payload.get("status") != "indexed":
            raise APIRequestError("The indexing response was malformed.")
        return dict(payload)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        query: str = "",
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        request = Request(
            f"{self.base_url}{path}{query}",
            data=body,
            headers=dict(headers or {"Accept": "application/json"}),
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read()
        except HTTPError as exc:
            raise APIRequestError(
                _http_error_message(exc),
                status_code=exc.code,
            ) from exc
        except (TimeoutError, URLError, OSError, ValueError) as exc:
            raise APIRequestError(
                "DocuIntel API is unavailable. Start the deployment and run the readiness check first."
            ) from exc
        if not raw_body:
            return None
        try:
            return json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIRequestError("The DocuIntel API returned malformed JSON.") from exc


def run_bootstrap(
    api: APIClientProtocol,
    *,
    project_root: Path,
    frontend_url: str = DEFAULT_FRONTEND_URL,
    manifest: Sequence[DemoFixture] = DEMO_MANIFEST,
    emit: Callable[[str], None] = print,
) -> BootstrapSummary:
    """Add missing fixtures and index them without deleting existing documents."""

    api.check_ready()
    existing = {
        str(item.get("checksum_sha256")): item
        for item in api.list_documents()
        if item.get("checksum_sha256")
    }
    added = 0
    already_present = 0
    failed = 0

    emit("DocuIntel demo bootstrap")
    emit("")
    for fixture in manifest:
        relative_path = Path(fixture.fixture_path)
        path = project_root / relative_path
        try:
            content = path.read_bytes()
        except OSError:
            failed += 1
            emit(f"[FAIL] {fixture.display_name}")
            emit(f"     Missing fixture: {fixture.fixture_path}")
            emit("")
            continue

        checksum = hashlib.sha256(content).hexdigest()
        existing_document = existing.get(checksum)
        try:
            if existing_document is not None:
                already_present += 1
                document_id = str(existing_document.get("id", ""))
                if not document_id:
                    raise BootstrapFailure("The existing document identity was missing.")
                if not bool(existing_document.get("is_indexed")):
                    api.index_document(document_id)
                    state = "already present; indexed"
                else:
                    state = "already present"
            else:
                uploaded = api.upload_document(path.name, content)
                document_id = str(uploaded.get("document_id", ""))
                if not document_id:
                    raise BootstrapFailure("The upload did not return a document identity.")
                api.index_document(document_id)
                existing[checksum] = {
                    "id": document_id,
                    "checksum_sha256": checksum,
                    "is_indexed": True,
                }
                added += 1
                state = "added and indexed"
        except APIRequestError as exc:
            if exc.status_code == 409:
                try:
                    duplicate = next(
                        (
                            item
                            for item in api.list_documents()
                            if item.get("checksum_sha256") == checksum
                        ),
                        None,
                    )
                except BootstrapFailure:
                    duplicate = None
                if duplicate is not None and duplicate.get("id"):
                    already_present += 1
                    document_id = str(duplicate["id"])
                    if not bool(duplicate.get("is_indexed")):
                        api.index_document(document_id)
                        state = "already present; indexed"
                    else:
                        state = "already present"
                    emit(f"[OK] {fixture.display_name} ({state})")
                    emit(f"     {fixture.fixture_path}")
                    emit("")
                    continue
                failed += 1
                emit(f"[FAIL] {fixture.display_name}")
                emit("     Duplicate response could not be matched to an existing document.")
            else:
                failed += 1
                emit(f"[FAIL] {fixture.display_name}")
                emit(f"     {_safe_failure_text(str(exc))}")
            emit("")
            continue
        except BootstrapFailure as exc:
            failed += 1
            emit(f"[FAIL] {fixture.display_name}")
            emit(f"     {_safe_failure_text(str(exc))}")
            emit("")
            continue

        emit(f"[OK] {fixture.display_name} ({state})")
        emit(f"     {fixture.fixture_path}")
        emit("")

    emit("Demo corpus ready." if failed == 0 else "Demo corpus completed with failures.")
    emit("")
    emit(f"Documents added: {added}")
    emit(f"Already present: {already_present}")
    emit(f"Failed: {failed}")
    emit("")
    emit("Open DocuIntel:")
    emit(frontend_url)
    return BootstrapSummary(added, already_present, failed)


def _multipart_body(boundary: str, filename: str, content: bytes) -> bytes:
    """Create the one-file multipart body required by FastAPI UploadFile."""

    safe_filename = Path(filename).name
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8")
    return prefix + content + f"\r\n--{boundary}--\r\n".encode("ascii")


def _http_error_message(error: HTTPError) -> str:
    """Convert a public API error to a concise, secret-safe message."""

    try:
        payload = json.loads(error.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        payload = None
    if isinstance(payload, Mapping) and isinstance(payload.get("detail"), str):
        detail = _safe_failure_text(payload["detail"])
        if detail:
            return detail
    return f"The API returned HTTP {error.code}."


def _safe_failure_text(message: str) -> str:
    """Remove connection strings, paths, and multiline detail from terminal output."""

    compact = " ".join(message.split())
    compact = re.sub(r"(?:https?|postgres(?:ql)?):[^\s]+", "[redacted connection]", compact, flags=re.IGNORECASE)
    compact = re.sub(r"[A-Za-z]:\\[^\s]+", "[redacted path]", compact)
    return compact[:240] or "The operation failed."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default=os.getenv("DOCUINTEL_API_BASE_URL", DEFAULT_API_URL),
        help="Running FastAPI base URL (or DOCUINTEL_API_BASE_URL).",
    )
    parser.add_argument(
        "--frontend-url",
        default=os.getenv("DOCUINTEL_STREAMLIT_BASE_URL", DEFAULT_FRONTEND_URL),
        help="Streamlit URL printed after bootstrap (or DOCUINTEL_STREAMLIT_BASE_URL).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("DOCUINTEL_BOOTSTRAP_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the safe, repeatable demo bootstrap command."""

    args = _parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        print("Bootstrap timeout must be greater than zero.")
        return 2
    project_root = Path(__file__).resolve().parents[1]
    api = BootstrapAPIClient(args.api_url, timeout_seconds=args.timeout_seconds)
    try:
        summary = run_bootstrap(
            api,
            project_root=project_root,
            frontend_url=args.frontend_url,
        )
    except APIRequestError as exc:
        print(_safe_failure_text(str(exc)))
        if exc.status_code in (None, 503):
            print("Run: python scripts/check_deployment.py")
        return 1
    except BootstrapFailure as exc:
        print(_safe_failure_text(str(exc)))
        return 1
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
