"""Small asynchronous client for the local Ollama HTTP API."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.core.config import Settings
from app.core.exceptions import OllamaServiceError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Generate one non-streaming answer through Ollama's local API."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.timeout_seconds = settings.ollama_timeout_seconds
        self.temperature = settings.ollama_temperature
        self._transport = transport

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Request one non-streaming completion and return only answer text."""

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = await self._request(payload)
        answer = data.get("response")
        if not isinstance(answer, str) or not answer.strip():
            logger.warning("Ollama response omitted answer text model=%s", self.model)
            raise OllamaServiceError("Ollama returned a malformed response.")
        return answer.strip()

    async def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Request one non-streaming JSON completion for constrained analysis output."""

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        data = await self._request(payload)
        response = data.get("response")
        try:
            decoded = response if isinstance(response, dict) else json.loads(response)
        except (TypeError, ValueError) as exc:
            logger.warning("Ollama returned malformed JSON model=%s", self.model)
            raise OllamaServiceError("Ollama returned a malformed structured response.") from exc
        if not isinstance(decoded, dict):
            logger.warning("Ollama JSON response was not an object model=%s", self.model)
            raise OllamaServiceError("Ollama returned a malformed structured response.")
        return decoded

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one provider request and normalize transport/provider failures."""

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post("/api/generate", json=payload)
        except httpx.TimeoutException as exc:
            logger.warning("Ollama request timed out model=%s", self.model)
            raise OllamaServiceError(
                f"Ollama did not respond within {self.timeout_seconds:g} seconds."
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("Ollama request failed model=%s error=%s", self.model, exc)
            raise OllamaServiceError(
                "The local Ollama service is unavailable. Start Ollama and try again."
            ) from exc

        if response.status_code == 404:
            logger.warning("Ollama model is unavailable model=%s", self.model)
            raise OllamaServiceError(
                f"The configured Ollama model '{self.model}' is unavailable."
            )
        if response.is_error:
            logger.warning(
                "Ollama returned an error status=%s model=%s",
                response.status_code,
                self.model,
            )
            raise OllamaServiceError("Ollama could not generate an answer.")
        try:
            data: Any = response.json()
        except ValueError as exc:
            logger.warning("Ollama returned non-JSON output model=%s", self.model)
            raise OllamaServiceError("Ollama returned a malformed response.") from exc
        if not isinstance(data, dict):
            logger.warning("Ollama response was not a JSON object model=%s", self.model)
            raise OllamaServiceError("Ollama returned a malformed response.")
        return data
