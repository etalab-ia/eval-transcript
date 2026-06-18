from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_API_KEY_ENV = "ALBERT_API_KEY"
DEFAULT_TRANSCRIPTION_MODEL = "openai/whisper-large-v3"
JSON_TRANSCRIPTION_RESPONSE_FORMATS = {"json", "verbose_json", "diarized_json"}


@dataclass(frozen=True)
class AlbertModel:
    id: str
    type: str | None = None
    owned_by: str | None = None


class AlbertClient:
    """Small client for Albert API's OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("ALBERT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv(DEFAULT_API_KEY_ENV)
        if timeout is not None and timeout <= 0:
            raise AlbertError(f"timeout must be positive, got {timeout}")
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        # Libellé du fournisseur, repris dans les messages d'erreur. Les
        # sous-classes OpenAI-compatibles (cf. OpenRouterClient) le surchargent.
        self.provider_name = "Albert API"

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def list_models(self) -> list[AlbertModel]:
        data = self._request_json("GET", "/models")
        models_list = data.get("data")
        if not isinstance(models_list, list):
            return []
        return [
            AlbertModel(id=item["id"], type=item.get("type"), owned_by=item.get("owned_by"))
            for item in models_list
            if isinstance(item, dict) and "id" in item
        ]

    def transcribe(
        self,
        *,
        model: str,
        audio_path: Path,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if response_format and response_format not in JSON_TRANSCRIPTION_RESPONSE_FORMATS:
            raise AlbertError(f"Unsupported JSON transcription response format: {response_format}")

        form: dict[str, str] = {"model": model}
        if language:
            form["language"] = language
        if prompt:
            form["prompt"] = prompt
        if response_format:
            form["response_format"] = response_format
        if temperature is not None:
            form["temperature"] = str(temperature)

        with audio_path.open("rb") as audio_file:
            files = {"file": (audio_path.name, audio_file)}
            return self._request_json("POST", "/audio/transcriptions", data=form, files=files)

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if response_format is not None:
            payload["response_format"] = response_format
        return self._request_json("POST", "/chat/completions", json=payload)

    def chat_completion_text(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        data = self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AlbertError(f"No choices in chat completion response: {data!r}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise AlbertError(f"No message content in chat completion response: {data!r}")
        return content

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
            response = client.request(method, url, **kwargs)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                message = f"{self.provider_name} request failed: {exc.response.status_code} {exc.response.reason_phrase} for {url}"
                detail = response_error_detail(exc.response)
                if detail:
                    message = f"{message} - {detail}"
                raise AlbertError(message) from exc
            try:
                data = response.json()
            except ValueError as exc:
                raise AlbertError(f"Invalid JSON response from {url}") from exc
            if not isinstance(data, dict):
                raise AlbertError(f"Expected JSON object from {url}, got {type(data).__name__}")
            return data


class AlbertError(RuntimeError):
    """Raised when Albert API returns an error."""


def response_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()
    if not isinstance(data, dict):
        return ""

    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)

    detail = data.get("detail")
    if detail:
        return str(detail)

    return ""
