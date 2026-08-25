"""Constrained-label document classification using structured Ollama output."""

from __future__ import annotations

import logging

from app.core.exceptions import AnalysisResponseError
from app.models.analysis import ClassificationLLMResponse
from app.services.analysis.prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    build_classification_prompt,
)
from app.services.analysis.summarizer import AnalysisChunk, DocumentSummarizer
from app.services.llm.ollama import OllamaClient

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Classify content against only the labels supplied by the caller."""

    def __init__(self, ollama_client: OllamaClient) -> None:
        self.ollama_client = ollama_client

    async def classify(
        self,
        chunks: list[AnalysisChunk],
        *,
        labels: list[str],
        context_max_chars: int,
        batch_max_chars: int,
    ) -> ClassificationLLMResponse:
        """Return a validated result, retrying one invalid label exactly once."""

        batches = DocumentSummarizer(self.ollama_client).build_batches(
            chunks,
            max_chars=batch_max_chars,
        )
        context = self._bound_context([batch.text for batch in batches], context_max_chars)
        for retry in (False, True):
            raw = await self.ollama_client.generate_json(
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                user_prompt=build_classification_prompt(
                    labels=labels,
                    context=context,
                    retry=retry,
                ),
            )
            try:
                result = ClassificationLLMResponse.model_validate(raw)
            except ValueError as exc:
                if retry:
                    raise AnalysisResponseError(
                        "The local model returned an invalid classification response."
                    ) from exc
                continue
            canonical = self._canonical_label(result.selected_label, labels)
            if canonical is not None:
                return result.model_copy(update={"selected_label": canonical})
            if retry:
                raise AnalysisResponseError(
                    "The local model selected a label that was not supplied."
                )
        raise AnalysisResponseError("The local model returned an invalid classification response.")

    @staticmethod
    def _canonical_label(value: str, labels: list[str]) -> str | None:
        """Match case-insensitively while returning the caller's exact label."""

        normalized = value.strip().casefold()
        return next((label for label in labels if label.casefold() == normalized), None)

    @staticmethod
    def _bound_context(contexts: list[str], max_chars: int) -> str:
        """Join document evidence up to the configured approximate character budget."""

        if max_chars <= 0:
            raise ValueError("context_max_chars must be greater than zero.")
        selected: list[str] = []
        used = 0
        for context in contexts:
            separator = 2 if selected else 0
            available = max_chars - used - separator
            if available <= 0:
                break
            selected.append(context[:available])
            used += separator + min(len(context), available)
        return "\n\n".join(selected)
