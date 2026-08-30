"""Focused Module 14.3 demo-bootstrap tests."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

from scripts import bootstrap_demo
from scripts.bootstrap_demo import (
    APIRequestError,
    BootstrapFailure,
    BootstrapSummary,
    DEMO_MANIFEST,
    DemoFixture,
    run_bootstrap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeAPI:
    """Small public-API double that records upload/index operations."""

    def __init__(self, documents: list[dict[str, Any]] | None = None) -> None:
        self.documents = documents or []
        self.uploaded: list[str] = []
        self.indexed: list[str] = []
        self.ready = True
        self.fail_upload_for: set[str] = set()
        self.duplicate_upload_for: set[str] = set()

    def check_ready(self) -> None:
        if not self.ready:
            raise BootstrapFailure("DocuIntel API is unavailable.")

    def list_documents(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.documents]

    def upload_document(self, filename: str, content: bytes) -> dict[str, Any]:
        self.uploaded.append(filename)
        if filename in self.fail_upload_for:
            raise BootstrapFailure("The upload failed safely.")
        if filename in self.duplicate_upload_for:
            raise APIRequestError("An identical PDF has already been uploaded.", status_code=409)
        document_id = f"demo-{len(self.documents) + 1}"
        checksum = hashlib.sha256(content).hexdigest()
        self.documents.append(
            {"id": document_id, "checksum_sha256": checksum, "is_indexed": False}
        )
        return {"document_id": document_id}

    def index_document(self, document_id: str) -> dict[str, Any]:
        self.indexed.append(document_id)
        for document in self.documents:
            if document["id"] == document_id:
                document["is_indexed"] = True
        return {"status": "indexed", "document_id": document_id}


def _fixture(*, path: str | None = None) -> DemoFixture:
    source = DEMO_MANIFEST[0]
    return DemoFixture(source.key, source.display_name, path or source.fixture_path, source.capabilities)


def test_manifest_contains_only_reviewed_safe_demo_fixtures() -> None:
    paths = {item.fixture_path for item in DEMO_MANIFEST}
    assert paths == {
        "data/sample_pdfs/module9-evaluation.pdf",
        "data/sample_pdfs/layout_table_sample.pdf",
        "data/sample_pdfs/module12_3_base.pdf",
        "data/sample_pdfs/module12_3_target.pdf",
        "data/sample_pdfs/module12_4_pii.pdf",
        "data/sample_pdfs/scanned_text_sample.pdf",
    }
    assert all("evaluation/" not in path and "external/" not in path for path in paths)
    assert all((PROJECT_ROOT / path).is_file() for path in paths)


def test_successful_bootstrap_uses_upload_then_index_and_safe_output() -> None:
    api = FakeAPI()
    lines: list[str] = []

    summary = run_bootstrap(api, project_root=PROJECT_ROOT, manifest=(_fixture(),), emit=lines.append)

    assert summary == BootstrapSummary(added=1, already_present=0, failed=0)
    assert api.uploaded == ["module9-evaluation.pdf"]
    assert api.indexed == ["demo-1"]
    assert "Documents added: 1" in lines
    assert "Failed: 0" in lines
    assert not any("postgresql://" in line for line in lines)


def test_existing_checksum_is_already_present_and_not_uploaded() -> None:
    fixture = _fixture()
    checksum = hashlib.sha256((PROJECT_ROOT / fixture.fixture_path).read_bytes()).hexdigest()
    api = FakeAPI([{"id": "existing-1", "checksum_sha256": checksum, "is_indexed": True}])

    summary = run_bootstrap(api, project_root=PROJECT_ROOT, manifest=(fixture,), emit=lambda _line: None)

    assert summary == BootstrapSummary(added=0, already_present=1, failed=0)
    assert api.uploaded == []
    assert api.indexed == []


def test_duplicate_upload_response_is_not_fatal_when_checksum_can_be_matched() -> None:
    fixture = _fixture()
    checksum = hashlib.sha256((PROJECT_ROOT / fixture.fixture_path).read_bytes()).hexdigest()
    existing = {"id": "existing-1", "checksum_sha256": checksum, "is_indexed": True}

    class DuplicateRaceAPI(FakeAPI):
        def __init__(self) -> None:
            super().__init__([existing])
            self.first_list = True

        def list_documents(self) -> list[dict[str, Any]]:
            if self.first_list:
                self.first_list = False
                return []
            return super().list_documents()

        def upload_document(self, filename: str, content: bytes) -> dict[str, Any]:
            self.uploaded.append(filename)
            raise APIRequestError("An identical PDF has already been uploaded.", status_code=409)

    api = DuplicateRaceAPI()

    summary = run_bootstrap(api, project_root=PROJECT_ROOT, manifest=(fixture,), emit=lambda _line: None)

    assert summary == BootstrapSummary(added=0, already_present=1, failed=0)


def test_second_run_is_idempotent_without_duplicate_documents() -> None:
    api = FakeAPI()
    fixture = _fixture()

    first = run_bootstrap(api, project_root=PROJECT_ROOT, manifest=(fixture,), emit=lambda _line: None)
    second = run_bootstrap(api, project_root=PROJECT_ROOT, manifest=(fixture,), emit=lambda _line: None)

    assert first.added == 1
    assert second.already_present == 1
    assert len(api.documents) == 1
    assert api.uploaded == ["module9-evaluation.pdf"]


def test_missing_fixture_is_an_honest_failure() -> None:
    api = FakeAPI()
    lines: list[str] = []

    summary = run_bootstrap(
        api,
        project_root=PROJECT_ROOT,
        manifest=(_fixture(path="data/sample_pdfs/not-a-real-demo.pdf"),),
        emit=lines.append,
    )

    assert summary == BootstrapSummary(added=0, already_present=0, failed=1)
    assert any("data/sample_pdfs/not-a-real-demo.pdf" in line for line in lines)
    assert api.uploaded == []


def test_partial_upload_failure_is_reported_without_claiming_success() -> None:
    api = FakeAPI()
    api.fail_upload_for.add("module9-evaluation.pdf")
    lines: list[str] = []

    summary = run_bootstrap(api, project_root=PROJECT_ROOT, manifest=(_fixture(),), emit=lines.append)

    assert summary == BootstrapSummary(added=0, already_present=0, failed=1)
    assert "Demo corpus completed with failures." in lines
    assert "Failed: 1" in lines


def test_api_unavailability_returns_nonzero_and_no_destructive_method_is_used(
    monkeypatch,
    capsys,
) -> None:
    api = FakeAPI()
    api.ready = False

    monkeypatch.setattr(bootstrap_demo, "BootstrapAPIClient", lambda *_args, **_kwargs: api)
    assert bootstrap_demo.main(["--api-url", "http://api"]) == 1
    assert "DocuIntel API is unavailable." in capsys.readouterr().out

    assert api.uploaded == []
    assert api.indexed == []


def test_bootstrap_uses_public_http_paths_and_not_database_access() -> None:
    source = inspect.getsource(bootstrap_demo)

    assert bootstrap_demo.UPLOAD_PATH == "/api/v1/documents/upload"
    assert "/api/v1/documents/{id}/index" not in source
    assert "/api/v1/documents/" in source
    assert "app.db" not in source
    assert "sqlalchemy" not in source
