from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_transcript.manifest import (
    DEFAULT_SOURCE_TRUTH_DIR,
    DEFAULT_TRANSCRIPTIONS_DIR,
    discover_sample_ids,
    find_output_paths,
    find_source_truth_path,
    parse_output_name,
)
from eval_transcript.scoring import AggregateScore, NormalizationMode, TranscriptScore, aggregate_scores, score_transcript_pair


@dataclass(frozen=True)
class ScoredTranscript:
    sample_id: str
    provider: str
    model: str
    source_truth_path: Path
    transcription_path: Path
    score: TranscriptScore


class ScoringError(RuntimeError):
    """Raised when transcript scoring inputs cannot be discovered or read."""


def score_sample_outputs(
    sample_id: str,
    *,
    source_truth_dir: Path = DEFAULT_SOURCE_TRUTH_DIR,
    transcriptions_dir: Path = DEFAULT_TRANSCRIPTIONS_DIR,
    normalization: NormalizationMode | str = NormalizationMode.STANDARD,
) -> list[ScoredTranscript]:
    source_truth_path = find_source_truth_path(source_truth_dir, sample_id)
    if source_truth_path is None:
        raise ScoringError(f"Missing source truth for sample {sample_id}: expected {source_truth_dir / f'{sample_id}.md'}")

    output_paths = find_output_paths(transcriptions_dir, sample_id)
    if not output_paths:
        raise ScoringError(f"No transcription outputs found for sample {sample_id}: expected {transcriptions_dir / sample_id}/*.txt")

    reference = source_truth_path.read_text(encoding="utf-8")
    scored: list[ScoredTranscript] = []
    for transcription_path in output_paths:
        hypothesis = transcription_path.read_text(encoding="utf-8")
        provider, model = parse_output_name(transcription_path)
        scored.append(
            ScoredTranscript(
                sample_id=sample_id,
                provider=provider,
                model=model,
                source_truth_path=source_truth_path,
                transcription_path=transcription_path,
                score=score_transcript_pair(reference, hypothesis, normalization=normalization),
            )
        )
    return scored


def score_all_outputs(
    *,
    source_truth_dir: Path = DEFAULT_SOURCE_TRUTH_DIR,
    transcriptions_dir: Path = DEFAULT_TRANSCRIPTIONS_DIR,
    normalization: NormalizationMode | str = NormalizationMode.STANDARD,
) -> list[ScoredTranscript]:
    sample_ids = discover_sample_ids(audio_dir=Path("__missing_audio_dir__"), source_truth_dir=source_truth_dir, transcriptions_dir=transcriptions_dir)
    scored: list[ScoredTranscript] = []
    for sample_id in sample_ids:
        if find_source_truth_path(source_truth_dir, sample_id) is None:
            continue
        if not find_output_paths(transcriptions_dir, sample_id):
            continue
        scored.extend(
            score_sample_outputs(
                sample_id,
                source_truth_dir=source_truth_dir,
                transcriptions_dir=transcriptions_dir,
                normalization=normalization,
            )
        )
    if not scored:
        raise ScoringError(
            f"No scoreable transcript pairs found under {source_truth_dir} and {transcriptions_dir}"
        )
    return scored


def scored_transcripts_to_dict(scored: list[ScoredTranscript]) -> dict[str, Any]:
    aggregate = aggregate_scores([item.score for item in scored])
    return {
        "normalization": scored[0].score.normalization.value if scored else NormalizationMode.STANDARD.value,
        "aggregate": aggregate_to_dict(aggregate),
        "transcripts": [scored_transcript_to_dict(item) for item in scored],
    }


def scored_transcript_to_dict(item: ScoredTranscript) -> dict[str, Any]:
    score = item.score
    return {
        "sample_id": item.sample_id,
        "provider": item.provider,
        "model": item.model,
        "source_truth_path": item.source_truth_path.as_posix(),
        "transcription_path": item.transcription_path.as_posix(),
        "metrics": {
            "wer": score.wer,
            "cer": score.cer,
            "mer": score.mer,
            "wil": score.wil,
            "wip": score.wip,
        },
        "counts": counts_to_dict(score),
    }


def aggregate_to_dict(aggregate: AggregateScore) -> dict[str, Any]:
    return {
        "wer": aggregate.wer,
        "sample_count": aggregate.sample_count,
        "counts": {
            "hits": aggregate.counts.hits,
            "substitutions": aggregate.counts.substitutions,
            "deletions": aggregate.counts.deletions,
            "insertions": aggregate.counts.insertions,
            "reference_tokens": aggregate.counts.reference_tokens,
            "errors": aggregate.counts.errors,
        },
    }


def counts_to_dict(score: TranscriptScore) -> dict[str, int]:
    return {
        "hits": score.counts.hits,
        "substitutions": score.counts.substitutions,
        "deletions": score.counts.deletions,
        "insertions": score.counts.insertions,
        "reference_tokens": score.counts.reference_tokens,
        "errors": score.counts.errors,
    }


def render_scores_text(scored: list[ScoredTranscript]) -> str:
    aggregate = aggregate_scores([item.score for item in scored])
    lines = [
        "sample\tprovider\tmodel\twer\tcer\tS\tD\tI\tN",
    ]
    for item in scored:
        score = item.score
        lines.append(
            "\t".join(
                [
                    item.sample_id,
                    item.provider,
                    item.model,
                    format_rate(score.wer),
                    format_rate(score.cer),
                    str(score.counts.substitutions),
                    str(score.counts.deletions),
                    str(score.counts.insertions),
                    str(score.counts.reference_tokens),
                ]
            )
        )
    lines.extend(
        [
            "",
            "aggregate",
            (
                f"WER: {format_rate(aggregate.wer)}  "
                f"S={aggregate.counts.substitutions} "
                f"D={aggregate.counts.deletions} "
                f"I={aggregate.counts.insertions} "
                f"N={aggregate.counts.reference_tokens}"
            ),
        ]
    )
    return "\n".join(lines)


def render_scores_json(scored: list[ScoredTranscript]) -> str:
    return json.dumps(scored_transcripts_to_dict(scored), ensure_ascii=False, indent=2)


def format_rate(value: float) -> str:
    return f"{value * 100:.2f}%"
