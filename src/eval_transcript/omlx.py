from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_API_KEY_ENV = "OMLX_API_KEY"


@dataclass(frozen=True)
class OmlxModel:
    id: str
    owned_by: str | None = None


class OmlxClient:
    """Small OpenAI-compatible client for the local oMLX server."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = (base_url or os.getenv("OMLX_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv(DEFAULT_API_KEY_ENV)
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def list_models(self) -> list[OmlxModel]:
        data = self._request_json("GET", "/models")
        return [
            OmlxModel(id=item["id"], owned_by=item.get("owned_by"))
            for item in data.get("data", [])
            if "id" in item
        ]

    def transcribe(self, *, model: str, audio_path: Path, language: str | None = None) -> dict[str, Any]:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        form: dict[str, str] = {"model": model}
        if language:
            form["language"] = language

        with audio_path.open("rb") as audio_file:
            files = {"file": (audio_path.name, audio_file)}
            return self._request_json("POST", "/audio/transcriptions", data=form, files=files)

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout, headers=self.headers, trust_env=False) as client:
            response = client.request(method, url, **kwargs)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise OmlxError(f"oMLX API request failed: {exc.response.status_code} {exc.response.reason_phrase} for {url}") from exc
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError(f"Expected JSON object from {url}, got {type(data).__name__}")
            return data


class OmlxError(RuntimeError):
    """Raised when the local oMLX API returns an error."""
