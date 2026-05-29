from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from scaleway import Client as ScalewaySdkClient
from scaleway import ScalewayException
from scaleway.inference.v1 import InferenceV1API


DEFAULT_BASE_URL = "https://api.scaleway.ai/v1"
DEFAULT_REGION = "fr-par"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MODEL = "voxtral-small-24b-2507"
DEFAULT_PROMPT = "Transcribe this audio. Return only the transcription text."
SUPPORTED_AUDIO_FORMATS = {"mp3", "wav"}


@dataclass(frozen=True)
class ScalewayModel:
    id: str
    name: str
    status: str
    region: str


class ScalewayClient:
    """Client for Scaleway Generative APIs audio transcription with Voxtral."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = (base_url or os.getenv("SCALEWAY_BASE_URL") or default_base_url()).rstrip("/")
        self.secret_key = secret_key if secret_key is not None else os.getenv("SCW_SECRET_KEY")
        self.region = region or os.getenv("SCW_DEFAULT_REGION") or DEFAULT_REGION
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        if not self.secret_key:
            return {}
        return {"Authorization": f"Bearer {self.secret_key}"}

    def list_models(self, *, name: str | None = None) -> list[ScalewayModel]:
        sdk_client = ScalewaySdkClient.from_env()
        if not sdk_client.default_region:
            sdk_client.default_region = self.region
        api = InferenceV1API(sdk_client)
        try:
            models = api.list_models_all(region=self.region, name=name)
        except ScalewayException as exc:
            raise ScalewayError(f"Scaleway SDK request failed: {exc}") from exc
        return [
            ScalewayModel(id=model.id, name=model.name, status=str(model.status), region=str(model.region))
            for model in models
        ]

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


def default_base_url() -> str:
    project_id = os.getenv("SCW_DEFAULT_PROJECT_ID")
    if project_id:
        return f"https://api.scaleway.ai/{project_id}/v1"
    return DEFAULT_BASE_URL


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
