from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elevenlabs.client import ElevenLabs
from elevenlabs.core import ApiError


DEFAULT_API_KEY_ENV = "ELEVENLABS_API_KEY"
DEFAULT_BASE_URL_ENV = "ELEVENLABS_BASE_URL"
DEFAULT_MODEL = "scribe_v2"
DEFAULT_TIMEOUT_SECONDS = 240.0
TRANSCRIPTION_MODEL_IDS = ("scribe_v2", "scribe_v1")


@dataclass(frozen=True)
class ElevenLabsModel:
    id: str


class ElevenLabsClient:
    """Small client for ElevenLabs Speech to Text transcription."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        sdk_client: Any | None = None,
        require_api_key: bool = True,
    ) -> None:
        if timeout is not None and (timeout <= 0 or not math.isfinite(timeout)):
            raise ElevenLabsError(f"timeout must be positive and finite, got {timeout}")
        self.api_key = api_key if api_key is not None else os.getenv(DEFAULT_API_KEY_ENV)
        if require_api_key and not self.api_key:
            raise ElevenLabsError(f"{DEFAULT_API_KEY_ENV} is required for ElevenLabs transcription")
        self.base_url = (base_url if base_url is not None else os.getenv(DEFAULT_BASE_URL_ENV)) or None
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        self._sdk_client = sdk_client

    @property
    def sdk_client(self) -> Any:
        if self._sdk_client is None:
            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._sdk_client = ElevenLabs(**kwargs)
        return self._sdk_client

    def list_models(self) -> list[ElevenLabsModel]:
        return [ElevenLabsModel(id=model_id) for model_id in TRANSCRIPTION_MODEL_IDS]

    def transcribe(
        self,
        *,
        audio_path: Path,
        model: str = DEFAULT_MODEL,
        language: str | None = None,
        tag_audio_events: bool | None = None,
        num_speakers: int | None = None,
        timestamps_granularity: str | None = None,
        diarize: bool | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        no_verbatim: bool | None = None,
    ) -> dict[str, Any]:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if not self.api_key:
            raise ElevenLabsError(f"{DEFAULT_API_KEY_ENV} is required for ElevenLabs transcription")

        request: dict[str, Any] = {"model_id": model}
        if language:
            request["language_code"] = language
        if tag_audio_events is not None:
            request["tag_audio_events"] = tag_audio_events
        if num_speakers is not None:
            request["num_speakers"] = num_speakers
        if timestamps_granularity is not None:
            request["timestamps_granularity"] = timestamps_granularity
        if diarize is not None:
            request["diarize"] = diarize
        if temperature is not None:
            request["temperature"] = temperature
        if seed is not None:
            request["seed"] = seed
        if no_verbatim is not None:
            request["no_verbatim"] = no_verbatim

        try:
            with audio_path.open("rb") as audio_file:
                request["file"] = audio_file
                output = self.sdk_client.speech_to_text.convert(**request)
        except Exception as exc:
            raise ElevenLabsError(f"ElevenLabs transcription failed for {model}: {elevenlabs_error_detail(exc)}") from exc

        result = serializable_response(output)
        if not isinstance(result, dict):
            raise ElevenLabsError(f"Expected ElevenLabs transcription response object, got {type(result).__name__}")
        return result


class ElevenLabsError(RuntimeError):
    """Raised when ElevenLabs transcription is unavailable or fails."""


def serializable_response(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return serializable_response(dataclasses.asdict(value))
    model_dump_attr = getattr(value, "model_dump", None)
    if callable(model_dump_attr):
        try:
            return serializable_response(model_dump_attr(mode="json"))
        except TypeError:
            return serializable_response(model_dump_attr())
    dict_attr = getattr(value, "dict", None)
    if callable(dict_attr):
        return serializable_response(dict_attr())
    if isinstance(value, dict):
        return {str(key): serializable_response(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serializable_response(item) for item in value]
    return value


def elevenlabs_transcription_text(result: dict[str, Any]) -> str:
    text = result.get("text")
    if isinstance(text, str):
        return text

    transcripts = result.get("transcripts")
    if isinstance(transcripts, list):
        transcript_texts = [transcript.get("text") for transcript in transcripts if isinstance(transcript, dict)]
        return "\n".join(text for text in transcript_texts if isinstance(text, str))

    return ""


def elevenlabs_error_detail(exc: Exception) -> str:
    if isinstance(exc, ApiError):
        detail = response_error_detail(exc.body)
        status = f"status {exc.status_code}" if exc.status_code is not None else "API error"
        if detail:
            return f"{status} - {detail}"
        return status
    return str(exc)


def response_error_detail(body: Any) -> str:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            message = detail.get("message")
            if message:
                return str(message)
            status = detail.get("status")
            if status:
                return str(status)
        if detail:
            return str(detail)
        error = body.get("error")
        if error:
            return str(error)
    return ""
