from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MANIFEST_PATH = Path("data/manifest.md")
DEFAULT_AUDIO_DIR = Path("data/audio")
DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
# Legacy path kept for one release during the source_truth -> ground_truth cutover.
DEFAULT_SOURCE_TRUTH_DIR = Path("data/source_truth")
DEFAULT_TRANSCRIPTIONS_DIR = Path("data/transcriptions")
AUDIO_SUFFIXES = {".aac", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
# Ordered by priority: a sample with both extensions resolves to the first match.
GROUND_TRUTH_SUFFIXES = (".md", ".txt")
# Legacy name kept for one release.
SOURCE_TRUTH_SUFFIXES = GROUND_TRUTH_SUFFIXES
LEGACY_SOURCE_TRUTH_PATH_KEY_WARNING = (
    "Warning: source_truth_path manifest key is deprecated; use ground_truth_path instead."
)
_legacy_source_truth_path_key_warning_printed = False


@dataclass(frozen=True)
class ManifestSample:
    id: str
    audio_path: Path
    ground_truth_path: Path | None
    outputs: list[Path]

    @property
    def source_truth_path(self) -> Path | None:
        """Deprecated alias for the pre-ground_truth manifest field."""

        return self.ground_truth_path


def discover_samples(
    *,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
    source_truth_dir: Path | None = None,
    transcriptions_dir: Path = DEFAULT_TRANSCRIPTIONS_DIR,
) -> list[ManifestSample]:
    resolved_ground_truth_dir = source_truth_dir or ground_truth_dir
    sample_ids = discover_sample_ids(
        audio_dir=audio_dir,
        ground_truth_dir=resolved_ground_truth_dir,
        transcriptions_dir=transcriptions_dir,
    )
    return [
        ManifestSample(
            id=sample_id,
            audio_path=find_audio_path(audio_dir, sample_id),
            ground_truth_path=find_ground_truth_path(resolved_ground_truth_dir, sample_id),
            outputs=find_output_paths(transcriptions_dir, sample_id),
        )
        for sample_id in sample_ids
    ]


def discover_sample_ids(
    *,
    audio_dir: Path,
    ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
    source_truth_dir: Path | None = None,
    transcriptions_dir: Path,
) -> list[str]:
    resolved_ground_truth_dir = source_truth_dir or ground_truth_dir
    sample_ids: set[str] = set()
    if audio_dir.is_dir():
        sample_ids.update(path.stem for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    if resolved_ground_truth_dir.is_dir():
        sample_ids.update(path.stem for path in resolved_ground_truth_dir.iterdir() if path.is_file() and path.suffix.lower() in GROUND_TRUTH_SUFFIXES)
    if transcriptions_dir.is_dir():
        sample_ids.update(path.name for path in transcriptions_dir.iterdir() if path.is_dir())
    return sorted(sample_ids)


def find_audio_path(audio_dir: Path, sample_id: str) -> Path:
    for suffix in sorted(AUDIO_SUFFIXES):
        path = audio_dir / f"{sample_id}{suffix}"
        if path.is_file():
            return path
    return audio_dir / f"{sample_id}.wav"


def find_ground_truth_path(ground_truth_dir: Path, sample_id: str) -> Path | None:
    for suffix in GROUND_TRUTH_SUFFIXES:
        path = ground_truth_dir / f"{sample_id}{suffix}"
        if path.is_file():
            return path
    return None


def find_source_truth_path(source_truth_dir: Path, sample_id: str) -> Path | None:
    """Deprecated alias for find_ground_truth_path during the cutover."""

    return find_ground_truth_path(source_truth_dir, sample_id)


def ground_truth_path_from_manifest_entry(entry: Mapping[str, object]) -> Path | None:
    """Read the canonical ground-truth path from a manifest entry mapping.

    Existing manifests generated before the rename used source_truth_path. Keep
    reading that key during the deprecation window so older manifest fixtures can
    still be consumed by future commands that parse the manifest.
    """

    value = entry.get("ground_truth_path")
    if value:
        return Path(str(value))

    legacy_value = entry.get("source_truth_path")
    if legacy_value:
        warn_legacy_source_truth_path_key()
        return Path(str(legacy_value))
    return None


def warn_legacy_source_truth_path_key() -> None:
    global _legacy_source_truth_path_key_warning_printed
    if _legacy_source_truth_path_key_warning_printed:
        return
    print(LEGACY_SOURCE_TRUTH_PATH_KEY_WARNING, file=sys.stderr)
    _legacy_source_truth_path_key_warning_printed = True


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
            "This file indexes benchmark audio samples, ground-truth transcripts, and generated model outputs.",
            "",
            "Run `uv run eval-transcript manifest sync` after adding audio, ground truth, or transcription outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def render_sample(sample: ManifestSample) -> list[str]:
    lines = [
        f"  - id: {quote_yaml(sample.id)}",
        f"    audio_path: {quote_yaml(sample.audio_path.as_posix())}",
        f"    ground_truth_path: {quote_yaml(sample.ground_truth_path.as_posix()) if sample.ground_truth_path else ''}",
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
