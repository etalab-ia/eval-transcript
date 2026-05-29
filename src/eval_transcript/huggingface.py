from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient


DEFAULT_API_KEY_ENV = "HF_TOKEN"
DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_PROVIDER = "auto"
TOGETHER_ASR_UNSUPPORTED_FRAGMENT = "Task 'automatic-speech-recognition' not supported for provider 'together'"
HF_INFERENCE_UNSUPPORTED_FRAGMENT = "Model not supported by provider hf-inference"
PARAKEET_FASTEST_VALIDATION_FRAGMENT = "nvidia/parakeet-tdt-0.6b-v3:fastest"
PARAKEET_HF_PROVIDER_HELP = (
    "Hugging Face currently advertises nvidia/parakeet-tdt-0.6b-v3 for Inference Providers, "
    "but huggingface_hub removed Together automatic-speech-recognition support in v1.16.1 after a v1.16.0 "
    "multipart-upload regression. See huggingface/huggingface_hub#4164 and huggingface/huggingface_hub#4248. "
    "Until HF ships a patch release that restores Together ASR, use `eval-transcript omlx transcribe "
    "--model parakeet-tdt-0.6b-v3` or pin/test the future huggingface_hub release before enabling this provider."
)


class HuggingFaceClient:
    """Client for Hugging Face Inference Providers ASR smoke tests."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        provider: str = DEFAULT_PROVIDER,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv(DEFAULT_API_KEY_ENV)
        if not self.api_key:
            raise HuggingFaceError(f"{DEFAULT_API_KEY_ENV} is required for Hugging Face transcription")
        self.provider = provider

    def transcribe(self, *, audio_path: Path, model: str = DEFAULT_PARAKEET_MODEL) -> dict[str, Any]:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        provider = None if self.provider == "auto" else self.provider
        client = InferenceClient(provider=provider, api_key=self.api_key)
        try:
            output = client.automatic_speech_recognition(audio_path, model=model)
        except Exception as exc:
            raise helpful_huggingface_error(model=model, provider=self.provider, exc=exc) from exc

        text, chunks = output_text_and_chunks(output)
        result: dict[str, Any] = {"text": text}
        if chunks is not None:
            result["chunks"] = [serializable_chunk(chunk) for chunk in chunks]
        return result


def output_text_and_chunks(output: Any) -> tuple[str, Any | None]:
    if isinstance(output, str):
        return output, None
    if isinstance(output, dict):
        text = output.get("text", "")
        return text if isinstance(text, str) else "", output.get("chunks")
    text = getattr(output, "text", "")
    return text if isinstance(text, str) else "", getattr(output, "chunks", None)


def serializable_chunk(chunk: Any) -> Any:
    if dataclasses.is_dataclass(chunk):
        return dataclasses.asdict(chunk)
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    return chunk


class HuggingFaceError(RuntimeError):
    """Raised when Hugging Face transcription is unavailable or fails."""


def helpful_huggingface_error(*, model: str, provider: str, exc: Exception) -> HuggingFaceError:
    message = str(exc)
    if model.startswith("nvidia/parakeet-tdt-0.6b-v3") and (
        TOGETHER_ASR_UNSUPPORTED_FRAGMENT in message
        or HF_INFERENCE_UNSUPPORTED_FRAGMENT in message
        or PARAKEET_FASTEST_VALIDATION_FRAGMENT in message
    ):
        return HuggingFaceError(PARAKEET_HF_PROVIDER_HELP)
    return HuggingFaceError(f"Hugging Face transcription failed for {model} via provider {provider}: {message}")
