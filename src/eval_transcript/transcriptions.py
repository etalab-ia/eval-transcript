from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRANSCRIPTIONS_DIR = Path("data/transcriptions")


@dataclass(frozen=True)
class TranscriptionOutput:
    text: str
    json_text: str | None = None
    saved_path: Path | None = None


def build_transcription_output(
    *,
    result: dict[str, Any],
    audio_path: Path,
    provider: str,
    model: str,
    save: bool = False,
    output_dir: Path | None = None,
    include_json: bool = False,
) -> TranscriptionOutput:
    text = result.get("text")
    if not isinstance(text, str):
        text = ""
    saved_path = None
    if save or output_dir:
        saved_path = transcription_output_path(
            output_dir=output_dir or DEFAULT_TRANSCRIPTIONS_DIR,
            audio_path=audio_path,
            provider=provider,
            model=model,
        )
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path.write_text(text, encoding="utf-8")

    json_text = json.dumps(result, ensure_ascii=False, indent=2) if include_json else None
    return TranscriptionOutput(text=text, json_text=json_text, saved_path=saved_path)


def transcription_output_path(*, output_dir: Path, audio_path: Path, provider: str, model: str) -> Path:
    return output_dir / safe_filename(audio_path.stem) / f"{safe_filename(provider)}__{safe_filename(model)}.txt"


def print_transcription_output(output: TranscriptionOutput) -> None:
    if output.json_text is not None:
        print(output.json_text)
        return
    if output.saved_path is not None:
        print(output.saved_path)
        return
    print(output.text)


def safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value).strip("._")
