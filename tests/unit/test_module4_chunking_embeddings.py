"""Fast Module 4 tests that do not download a transformer model."""

from __future__ import annotations

from types import SimpleNamespace as Namespace

import pytest

from app.core.exceptions import EmbeddingServiceError
from app.services.chunking.structure_aware import StructureAwareChunker
from app.services.embeddings.sentence_transformer import EmbeddingService


def page(
    number: int,
    elements: list[Namespace] | None = None,
    *,
    text: str = "",
    ocr: bool = False,
    tables: list[Namespace] | None = None,
) -> Namespace:
    """Build the small ORM-shaped object graph needed by the chunker."""

    return Namespace(
        id=f"page-{number}",
        page_number=number,
        extracted_text=text,
        ocr_applied=ocr,
        extraction_method="ocr" if ocr else "native",
        layout_elements=elements or [],
        tables=tables or [],
    )


def element(sequence: int, element_type: str, text: str) -> Namespace:
    """Build a persisted layout element-shaped object."""

    return Namespace(sequence_order=sequence, element_type=element_type, text=text)


def test_heading_context_and_page_ranges_are_preserved() -> None:
    """Paragraphs inherit headings across pages until a new heading appears."""

    document = Namespace(
        id="document-1",
        pages=[
            page(
                1,
                [
                    element(1, "heading", "Termination Conditions"),
                    element(2, "paragraph", "Either party may terminate with notice."),
                ],
            ),
            page(
                2,
                [element(1, "paragraph", "The notice must be provided in writing.")],
            ),
            page(
                3,
                [
                    element(1, "heading", "Payment"),
                    element(2, "paragraph", "Invoices are payable within thirty days."),
                ],
            ),
        ],
    )

    chunks = StructureAwareChunker(target_chars=500, max_chars=700, overlap_chars=0).build_chunks(
        document
    )

    assert len(chunks) == 2
    assert chunks[0].section_heading == "Termination Conditions"
    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 2
    assert "notice must be provided" in chunks[0].text
    assert chunks[1].section_heading == "Payment"


def test_oversized_section_splits_with_bounded_overlap() -> None:
    """Long text is split below the configured maximum with same-section overlap."""

    text = " ".join(f"Sentence {index} explains the policy." for index in range(30))
    document = Namespace(
        id="document-2",
        pages=[page(1, [element(1, "paragraph", text)])],
    )

    chunks = StructureAwareChunker(target_chars=120, max_chars=180, overlap_chars=24).build_chunks(
        document
    )

    assert len(chunks) > 1
    assert all(0 < len(chunk.text) <= 180 for chunk in chunks)
    assert any(set(first.text.split()) & set(second.text.split()) for first, second in zip(chunks, chunks[1:]))
    assert [chunk.fingerprint_sha256 for chunk in chunks] == [
        chunk.fingerprint_sha256 for chunk in chunks
    ]


def test_tables_are_structured_and_ocr_provenance_is_retained() -> None:
    """Tables become row-oriented chunks and OCR page state flows to metadata."""

    document = Namespace(
        id="document-3",
        pages=[
            page(
                1,
                text="Recovered OCR paragraph",
                ocr=True,
                tables=[
                    Namespace(
                        headers=["Product", "Quantity"],
                        rows=[["Laptop", "3"], ["Monitor", "8"]],
                    )
                ],
            )
        ],
    )

    chunks = StructureAwareChunker(target_chars=500, max_chars=700, overlap_chars=0).build_chunks(
        document
    )

    assert chunks[0].contains_ocr is True
    table = next(chunk for chunk in chunks if chunk.content_type == "table")
    assert table.text == "Product | Quantity Laptop | 3 Monitor | 8"
    assert table.contains_ocr is True


class FakeEmbeddingModel:
    """Small deterministic encoder for service unit tests."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        self.calls += 1
        return [[float(index)] * 384 for index, _ in enumerate(texts, start=1)]


def test_embedding_service_is_lazy_batched_and_dimension_checked() -> None:
    """The model loads once and the service validates the configured dimension."""

    model = FakeEmbeddingModel()
    loads: list[str] = []

    def load(name: str) -> FakeEmbeddingModel:
        loads.append(name)
        return model

    service = EmbeddingService(model_loader=load)
    assert service.model_loaded is False
    vectors = service.embed_texts(["first", "second"])
    assert service.model_loaded is True
    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    service.embed_texts(["third"])
    assert loads == ["sentence-transformers/all-MiniLM-L6-v2"]
    assert model.calls == 2


def test_embedding_service_rejects_empty_text_and_empty_batch() -> None:
    """Invalid inputs fail before the model is loaded."""

    service = EmbeddingService(model=FakeEmbeddingModel())
    with pytest.raises(EmbeddingServiceError):
        service.embed_texts([])
    with pytest.raises(EmbeddingServiceError):
        service.embed_texts(["  "])
