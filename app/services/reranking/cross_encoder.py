"""Lazy local CrossEncoder reranking for second-stage search."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.core.config import Settings
from app.core.exceptions import RerankerServiceError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RerankInput:
    """Minimal query/candidate content sent to the cross-encoder."""

    candidate_id: UUID
    text: str
    section_heading: str | None = None


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    """A candidate's raw cross-encoder relevance score and original rank."""

    candidate_id: UUID
    base_rank: int
    score: float


class CrossEncoderReranker:
    """Score query/chunk pairs with one lazily loaded reusable CrossEncoder."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: Any | None = None,
        model_loader: Callable[[str, int], Any] | None = None,
    ) -> None:
        configured = settings or Settings()
        self.model_name = configured.reranker_model
        self.batch_size = configured.reranker_batch_size
        self.max_length = configured.reranker_max_length
        self._model = model
        self._model_loader = model_loader

    @property
    def model_loaded(self) -> bool:
        """Return whether the cross-encoder has been loaded."""

        return self._model is not None

    @property
    def device(self) -> str:
        """Return the model's selected device without requiring GPU support."""

        value = getattr(self._model, "device", None)
        return str(value) if value is not None else "unknown"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankInput],
    ) -> list[RerankedCandidate]:
        """Score all candidates in batches and return deterministic descending order."""

        if not candidates:
            return []
        normalised_query = query.strip()
        if not normalised_query:
            raise RerankerServiceError("A non-empty query is required for reranking.")
        pairs = [
            (self._bounded_text(normalised_query), self._candidate_text(candidate))
            for candidate in candidates
        ]
        model = self._load_model()
        try:
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
            scores = [self._score_value(value) for value in values]
        except RerankerServiceError:
            raise
        except Exception as exc:
            logger.exception(
                "Cross-encoder reranking failed model=%s candidate_count=%s",
                self.model_name,
                len(candidates),
            )
            raise RerankerServiceError(
                f"The local reranker model '{self.model_name}' could not score candidates."
            ) from exc
        if len(scores) != len(candidates):
            raise RerankerServiceError("The reranker returned an unexpected score count.")

        reranked = [
            RerankedCandidate(
                candidate_id=candidate.candidate_id,
                base_rank=base_rank,
                score=score,
            )
            for base_rank, (candidate, score) in enumerate(zip(candidates, scores, strict=True), start=1)
        ]
        reranked.sort(key=lambda item: (-item.score, item.base_rank, str(item.candidate_id)))
        logger.info(
            "Cross-encoder reranking completed model=%s candidate_count=%s batch_size=%s "
            "max_length=%s device=%s",
            self.model_name,
            len(reranked),
            self.batch_size,
            self.max_length,
            self.device,
        )
        return reranked

    def _load_model(self) -> Any:
        """Load the CrossEncoder exactly once on first use."""

        if self._model is not None:
            return self._model
        try:
            if self._model_loader is not None:
                self._model = self._model_loader(self.model_name, self.max_length)
            else:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=self.max_length)
            logger.info(
                "Cross-encoder model loaded model=%s max_length=%s device=%s",
                self.model_name,
                self.max_length,
                self.device,
            )
            return self._model
        except Exception as exc:
            logger.exception("Cross-encoder model could not be loaded model=%s", self.model_name)
            raise RerankerServiceError(
                f"The local reranker model '{self.model_name}' is unavailable. "
                "Install sentence-transformers or make the model cache available."
            ) from exc

    def _candidate_text(self, candidate: RerankInput) -> str:
        """Compose bounded model input without changing stored chunk text."""

        heading = candidate.section_heading.strip() if candidate.section_heading else ""
        text = candidate.text.strip()
        representation = f"{heading}\n\n{text}" if heading else text
        return self._bounded_text(representation)

    def _bounded_text(self, value: str) -> str:
        """Apply deterministic character protection before model tokenization."""

        # CrossEncoder also enforces max_length at tokenization time. This
        # conservative bound prevents very large persisted text from entering
        # the model call while leaving the original chunk untouched.
        return value[: max(self.max_length * 4, 128)]

    @staticmethod
    def _score_value(value: Any) -> float:
        """Convert scalar or one-label model output to a raw relevance score."""

        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise RerankerServiceError("The reranker returned a non-scalar score.")
            value = value[0]
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise RerankerServiceError("The reranker returned an invalid score.") from exc
