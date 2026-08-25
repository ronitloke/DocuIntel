"""Lazy local Sentence Transformers embedding service."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable

from app.core.config import Settings
from app.core.exceptions import EmbeddingServiceError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate consistently normalized embeddings with one reusable model instance."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: Any | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        configured = settings or Settings()
        self.model_name = configured.embedding_model
        self.dimension = configured.embedding_dimension
        self.batch_size = configured.embedding_batch_size
        self._model = model
        self._model_loader = model_loader

    @property
    def model_loaded(self) -> bool:
        """Return whether the local model has been loaded into this service."""

        return self._model is not None

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed non-empty text in batches using normalized vectors."""

        if not texts:
            raise EmbeddingServiceError("At least one non-empty text is required for embedding.")
        normalised = [" ".join(text.split()) for text in texts]
        if any(not text for text in normalised):
            raise EmbeddingServiceError("Embedding input contains empty text.")
        model = self._load_model()
        try:
            encoded = model.encode(
                normalised,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Embedding generation failed model=%s", self.model_name)
            raise EmbeddingServiceError(
                f"The local embedding model '{self.model_name}' could not generate vectors."
            ) from exc

        try:
            vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
            result = [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as exc:
            raise EmbeddingServiceError("The embedding model returned invalid vector data.") from exc
        if len(result) != len(normalised):
            raise EmbeddingServiceError("The embedding model returned an unexpected vector count.")
        if any(len(vector) != self.dimension for vector in result):
            raise EmbeddingServiceError(
                f"The embedding model returned a vector with a dimension other than {self.dimension}."
            )
        logger.info(
            "Embedding generation completed model=%s count=%s batch_size=%s dimension=%s normalized=true",
            self.model_name,
            len(result),
            self.batch_size,
            self.dimension,
        )
        return result

    def _load_model(self) -> Any:
        """Load Sentence Transformer once, on the first embedding request."""

        if self._model is not None:
            return self._model
        try:
            if self._model_loader is not None:
                loader = self._model_loader
            else:
                from sentence_transformers import SentenceTransformer

                loader = SentenceTransformer
            self._model = loader(self.model_name)
            logger.info("Embedding model loaded model=%s", self.model_name)
            return self._model
        except Exception as exc:
            logger.exception("Embedding model could not be loaded model=%s", self.model_name)
            raise EmbeddingServiceError(
                f"The local embedding model '{self.model_name}' is unavailable. "
                "Install sentence-transformers or make the model cache available."
            ) from exc
