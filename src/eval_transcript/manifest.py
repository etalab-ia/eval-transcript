from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MANIFEST_PATH = Path("data/manifest.md")
DEFAULT_AUDIO_DIR = Path("data/audio")
DEFAULT_SOURCE_TRUTH_DIR = Path("data/source_truth")
DEFAULT_TRANSCRIPTIONS_DIR = Path("data/transcriptions")
AUDIO_SUFFIXES = {".aac", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
SOURCE_TRUTH_SUFFIXES = {".txt", ".md"}


@dataclass(frozen=True)
class ManifestSample:
    id: str
    audio_path: Path
    source_truth_path: Path | None
    outputs: list[Path]


def discover_samples(
    *,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    source_truth_dir: Path = DEFAULT_SOURCE_TRUTH_DIR,
    transcriptions_dir: Path = DEFAULT_TRANSCRIPTIONS_DIR,
) -> list[ManifestSample]:
    sample_ids = discover_sample_ids(audio_dir=audio_dir, source_truth_dir=source_truth_dir, transcriptions_dir=transcriptions_dir)
    return [
        ManifestSample(
            id=sample_id,
            audio_path=find_audio_path(audio_dir, sample_id),
            source_truth_path=find_source_truth_path(source_truth_dir, sample_id),
            outputs=find_output_paths(transcriptions_dir, sample_id),
        )
        for sample_id in sample_ids
    ]


def discover_sample_ids(*, audio_dir: Path, source_truth_dir: Path, transcriptions_dir: Path) -> list[str]:
    sample_ids: set[str] = set()
    if audio_dir.is_dir():
        sample_ids.update(path.stem for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    if source_truth_dir.is_dir():
        sample_ids.update(path.stem for path in source_truth_dir.iterdir() if path.is_file() and path.suffix.lower() in SOURCE_TRUTH_SUFFIXES)
    if transcriptions_dir.is_dir():
        sample_ids.update(path.name for path in transcriptions_dir.iterdir() if path.is_dir())
    return sorted(sample_ids)


def find_audio_path(audio_dir: Path, sample_id: str) -> Path:
    for suffix in sorted(AUDIO_SUFFIXES):
        path = audio_dir / f"{sample_id}{suffix}"
        if path.is_file():
            return path
    return audio_dir / f"{sample_id}.wav"


def find_source_truth_path(source_truth_dir: Path, sample_id: str) -> Path | None:
    for suffix in sorted(SOURCE_TRUTH_SUFFIXES):
        path = source_truth_dir / f"{sample_id}{suffix}"
        if path.is_file():
            return path
    return None


def find_output_paths(transcriptions_dir: Path, sample_id: str) -> list[Path]:
    sample_dir = transcriptions_dir / sample_id
    if not sample_dir.is_dir():
        return []
    return sorted(path for path in sample_dir.glob("*.txt") if path.is_file())


def render_manifest(samples: list[ManifestSample]) -> str:
    if not samples:
        lines = ["---", "version: 1", "samples: []"]
    else:
        lines = ["---", "version: 1", "samples:"]
    for sample in samples:
        lines.extend(render_sample(sample))
    lines.extend(
        [
            "---",
            "",
            "# Transcription benchmark manifest",
            "",
            "This file indexes benchmark audio samples, source-of-truth transcripts, and generated model outputs.",
            "",
            "Run `uv run eval-transcript manifest sync` after adding audio, source truth, or transcription outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def render_sample(sample: ManifestSample) -> list[str]:
    lines = [
        f"  - id: {quote_yaml(sample.id)}",
        f"    audio_path: {quote_yaml(sample.audio_path.as_posix())}",
        f"    source_truth_path: {quote_yaml(sample.source_truth_path.as_posix()) if sample.source_truth_path else ''}",
        "    language:",
        "    duration_seconds:",
        "    domain:",
        "    outputs:",
    ]
    if not sample.outputs:
        lines[-1] = "    outputs: []"
        return lines
    for output in sample.outputs:
        provider, model = parse_output_name(output)
        lines.extend(
            [
                f"      - provider: {quote_yaml(provider)}",
                f"        model: {quote_yaml(model)}",
                f"        path: {quote_yaml(output.as_posix())}",
                "        created_at:",
                "        runtime_seconds:",
                "        realtime_factor:",
                "        language_hint:",
            ]
        )
    return lines


def parse_output_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" not in stem:
        return "", stem
    provider, model = stem.split("__", 1)
    return provider, model


def quote_yaml(value: str) -> str:
    return json.dumps(value)
