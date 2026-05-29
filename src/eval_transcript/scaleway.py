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


BASE_URL_TEMPLATE = "https://api.scaleway.ai/{project_id}/v1"
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
        access_key: str | None = None,
        secret_key: str | None = None,
        organization_id: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.access_key = access_key if access_key is not None else os.getenv("SCW_ACCESS_KEY")
        self.secret_key = secret_key if secret_key is not None else os.getenv("SCW_SECRET_KEY")
        self.organization_id = (
            organization_id if organization_id is not None else os.getenv("SCW_DEFAULT_ORGANIZATION_ID")
        )
        self.project_id = project_id if project_id is not None else os.getenv("SCW_DEFAULT_PROJECT_ID")
        self.base_url = generative_api_base_url(self.project_id)
        self.region = region or os.getenv("SCW_DEFAULT_REGION") or DEFAULT_REGION
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        if not self.secret_key:
            return {}
        return {"Authorization": f"Bearer {self.secret_key}"}

    def list_models(self, *, name: str | None = None) -> list[ScalewayModel]:
        sdk_client = ScalewaySdkClient(
            access_key=self.access_key,
            secret_key=self.secret_key,
            default_organization_id=self.organization_id,
            default_project_id=self.project_id,
            default_region=self.region,
        )
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
