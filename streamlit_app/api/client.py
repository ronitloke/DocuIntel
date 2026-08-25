"""Small, testable HTTP client for the existing DocuIntel FastAPI API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx


class ApiError(RuntimeError):
    """Safe error raised for transport and controlled API failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload


class ApiClient:
    """Centralize HTTP methods, JSON handling, uploads, deletes, and errors."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._client = http_client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        files: Any = None,
    ) -> Any:
        try:
            response = self._client.request(
                method,
                path,
                params=params,
                json=json,
                files=files,
            )
        except httpx.TimeoutException as exc:
            raise ApiError("The DocuIntel API request timed out.") from exc
        except httpx.RequestError as exc:
            raise ApiError("The DocuIntel API is unavailable. Check that FastAPI is running.") from exc

        payload: Any = None
        if response.content:
            try:
                payload = response.json()
            except ValueError:
                payload = None

        if response.is_error:
            raise ApiError(
                self._error_message(payload, response.status_code),
                status_code=response.status_code,
                payload=payload,
            )
        return payload

    @staticmethod
    def _error_message(payload: Any, status_code: int) -> str:
        if isinstance(payload, Mapping):
            detail = payload.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail
            if isinstance(detail, list):
                return "The API rejected the request: " + "; ".join(
                    str(item.get("msg", item)) if isinstance(item, Mapping) else str(item)
                    for item in detail
                )
        return f"The DocuIntel API returned HTTP {status_code}."

    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def post_json(self, path: str, payload: Any) -> Any:
        return self._request("POST", path, json=payload)

    def upload_pdf(self, path: str, filename: str, content: bytes) -> Any:
        return self._request("POST", path, files={"file": (filename, content, "application/pdf")})

    def get_bytes(self, path: str) -> bytes:
        """Retrieve a known binary artifact while preserving safe API errors."""

        try:
            response = self._client.get(path)
        except httpx.TimeoutException as exc:
            raise ApiError("The DocuIntel API request timed out.") from exc
        except httpx.RequestError as exc:
            raise ApiError("The DocuIntel API is unavailable. Check that FastAPI is running.") from exc
        if response.is_error:
            payload: Any = None
            try:
                payload = response.json()
            except ValueError:
                payload = None
            raise ApiError(
                self._error_message(payload, response.status_code),
                status_code=response.status_code,
                payload=payload,
            )
        return response.content

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)
