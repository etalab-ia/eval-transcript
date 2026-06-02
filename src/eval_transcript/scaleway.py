from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import httpx


BASE_URL_TEMPLATE = "https://api.scaleway.ai/{project_id}/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MODEL = "voxtral-small-24b-2507"
DEFAULT_PROMPT = (
    "Transcris l'audio mot à mot dans sa langue d'origine, sans le traduire. "
    "Réponds uniquement avec le texte transcrit, sans commentaire."
)
SUPPORTED_AUDIO_FORMATS = {"mp3", "wav"}
LANGUAGE_NAMES = {
    "fr": "français",
    "en": "anglais",
    "de": "allemand",
    "es": "espagnol",
    "it": "italien",
    "nl": "néerlandais",
    "pt": "portugais",
}


def build_prompt(language: str | None) -> str:
    """Build a transcription prompt that prevents Voxtral from translating the audio.

    Voxtral follows the language of the prompt, so an English prompt yields an English
    translation instead of a verbatim transcription. The default prompt is in French and
    explicitly forbids translation; passing a language hint pins the target language.
    """
    language = language.strip() if language else ""
    if not language:
        return DEFAULT_PROMPT
    name = LANGUAGE_NAMES.get(language.lower(), language)
    return (
        f"Transcris l'audio mot à mot en {name}, sans le traduire. "
        "Réponds uniquement avec le texte transcrit, sans commentaire."
    )


class ScalewayClient:
    """Client for Scaleway Generative APIs audio transcription with Voxtral."""

    def __init__(
        self,
        *,
        secret_key: str | None = None,
        project_id: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.secret_key = secret_key if secret_key is not None else os.getenv("SCW_SECRET_KEY")
        self.project_id = project_id if project_id is not None else os.getenv("SCW_DEFAULT_PROJECT_ID")
        self.base_url = generative_api_base_url(self.project_id)
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

    @property
    def headers(self) -> dict[str, str]:
        if not self.secret_key:
            return {}
        return {"Authorization": f"Bearer {self.secret_key}"}

    def list_models(self, *, name: str | None = None) -> list[str]:
        """List Generative APIs model IDs usable by ``transcribe``.

        Queries the same OpenAI-compatible endpoint as transcription, so the returned IDs
        match exactly what ``--model`` expects. Only ``SCW_SECRET_KEY`` and
        ``SCW_DEFAULT_PROJECT_ID`` are required.
        """
        if not self.secret_key:
            raise ScalewayError("SCW_SECRET_KEY is required to list Scaleway models")
        if not self.project_id:
            raise ScalewayError("SCW_DEFAULT_PROJECT_ID is required to list Scaleway models")
        data = self._request_json("GET", "/models")
        models = data.get("data")
        if not isinstance(models, list):
            return []
        ids = [model.get("id") for model in models if isinstance(model, dict)]
        ids = [model_id for model_id in ids if isinstance(model_id, str) and model_id]
        if name:
            needle = name.lower()
            ids = [model_id for model_id in ids if needle in model_id.lower()]
        return sorted(ids)

    def transcribe(
        self,
        *,
        audio_path: Path,
        model: str = DEFAULT_MODEL,
        prompt: str = DEFAULT_PROMPT,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        top_p: float = 0.95,
    ) -> dict[str, Any]:
        if not self.secret_key:
            raise ScalewayError("SCW_SECRET_KEY is required for Scaleway transcription")
        if not self.project_id:
            raise ScalewayError("SCW_DEFAULT_PROJECT_ID is required for Scaleway transcription")
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        audio_format = audio_path.suffix.lower().lstrip(".")
        if audio_format not in SUPPORTED_AUDIO_FORMATS:
            raise ScalewayError(
                f"Unsupported Scaleway audio format: {audio_path.suffix or '<none>'}; expected .mp3 or .wav"
            )

        encoded_audio = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": encoded_audio, "format": audio_format}},
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        return self._request_json("POST", "/chat/completions", json=payload)

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
            response = client.request(method, url, **kwargs)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                message = f"Scaleway API request failed: {exc.response.status_code} {exc.response.reason_phrase} for {url}"
                detail = response_error_detail(exc.response)
                if detail:
                    message = f"{message} - {detail}"
                raise ScalewayError(message) from exc
            try:
                data = response.json()
            except ValueError as exc:
                raise ScalewayError(f"Invalid JSON response from {url}") from exc
            if not isinstance(data, dict):
                raise ScalewayError(f"Expected JSON object from {url}, got {type(data).__name__}")
            return data


class ScalewayError(RuntimeError):
    """Raised when Scaleway API returns an error."""


def transcription_text(result: dict[str, Any]) -> str:
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


def generative_api_base_url(project_id: str | None) -> str:
    if not project_id:
        return ""
    return BASE_URL_TEMPLATE.format(project_id=project_id)


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
    if error:
        return str(error)

    detail = data.get("detail")
    if detail:
        return str(detail)

    message = data.get("message")
    if message:
        return str(message)

    return ""
