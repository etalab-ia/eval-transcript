from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRANSCRIPTIONS_DIR = Path("data/transcriptions")


@dataclass(frozen=True)
class TranscriptionOutput:
    result: dict[str, Any]
    text: str
    json_output: bool
    save: bool
    output_dir: Path | None
    audio_path: Path
    provider: str
    model: str

    @property
    def should_save(self) -> bool:
        return self.save or self.output_dir is not None

    @property
    def path(self) -> Path:
        return transcription_output_path(
            output_dir=self.output_dir or DEFAULT_TRANSCRIPTIONS_DIR,
            audio_path=self.audio_path,
            provider=self.provider,
            model=self.model,
        )


def print_transcription_output(output: TranscriptionOutput) -> None:
    if output.should_save:
        output.path.parent.mkdir(parents=True, exist_ok=True)
        output.path.write_text(output.text, encoding="utf-8")

    if output.json_output:
        print(json.dumps(output.result, ensure_ascii=False, indent=2))
    elif output.should_save:
        print(output.path)
    else:
        print(output.text)


def transcription_output_path(*, output_dir: Path, audio_path: Path, provider: str, model: str) -> Path:
    return output_dir / safe_filename(audio_path.stem) / f"{safe_filename(provider)}__{safe_filename(model)}.txt"


def transcription_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value).strip("._")
