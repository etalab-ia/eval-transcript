from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from eval_transcript.manifest import (
    DEFAULT_GROUND_TRUTH_DIR,
    DEFAULT_TRANSCRIPTIONS_DIR,
    GROUND_TRUTH_SUFFIXES,
    discover_sample_ids,
    find_ground_truth_path,
    find_output_paths,
    parse_output_name,
)
from eval_transcript.scoring import AlignmentOperation, AggregateScore, NormalizationMode, TranscriptScore, aggregate_scores, score_transcript_pair


@dataclass(frozen=True)
class ScoredTranscript:
    sample_id: str
    provider: str
    model: str
    ground_truth_path: Path
    transcription_path: Path
    score: TranscriptScore

    @property
    def source_truth_path(self) -> Path:
        """Deprecated alias for the pre-ground_truth score field."""

        return self.ground_truth_path


@dataclass(frozen=True)
class GroupedScore:
    provider: str
    model: str
    aggregate: AggregateScore
    sample_count: int
    transcript_count: int


ScoreOutputFormat = Literal["text", "json", "markdown", "csv"]


class ScoringError(RuntimeError):
    """Raised when transcript scoring inputs cannot be discovered or read."""


def score_output_paths(
    *,
    sample_id: str,
    ground_truth_path: Path,
    output_paths: list[Path],
    normalization: NormalizationMode | str = NormalizationMode.STANDARD,
) -> list[ScoredTranscript]:
    reference = ground_truth_path.read_text(encoding="utf-8")
    scored: list[ScoredTranscript] = []
    for transcription_path in output_paths:
        hypothesis = transcription_path.read_text(encoding="utf-8")
        provider, model = parse_output_name(transcription_path)
        scored.append(
            ScoredTranscript(
                sample_id=sample_id,
                provider=provider,
                model=model,
                ground_truth_path=ground_truth_path,
                transcription_path=transcription_path,
                score=score_transcript_pair(reference, hypothesis, normalization=normalization),
            )
        )
    return scored


def score_sample_outputs(
    sample_id: str,
    *,
    ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
    source_truth_dir: Path | None = None,
    transcriptions_dir: Path = DEFAULT_TRANSCRIPTIONS_DIR,
    normalization: NormalizationMode | str = NormalizationMode.STANDARD,
) -> list[ScoredTranscript]:
    resolved_ground_truth_dir = source_truth_dir or ground_truth_dir
    ground_truth_path = find_ground_truth_path(resolved_ground_truth_dir, sample_id)
    if ground_truth_path is None:
        expected = " or ".join((resolved_ground_truth_dir / f"{sample_id}{suffix}").as_posix() for suffix in GROUND_TRUTH_SUFFIXES)
        raise ScoringError(f"Missing ground truth for sample {sample_id}: expected {expected}")

    output_paths = find_output_paths(transcriptions_dir, sample_id)
    if not output_paths:
        raise ScoringError(f"No transcription outputs found for sample {sample_id}: expected {transcriptions_dir / sample_id}/*.txt")

    return score_output_paths(
        sample_id=sample_id,
        ground_truth_path=ground_truth_path,
        output_paths=output_paths,
        normalization=normalization,
    )


def score_all_outputs(
    *,
    ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
    source_truth_dir: Path | None = None,
    transcriptions_dir: Path = DEFAULT_TRANSCRIPTIONS_DIR,
    normalization: NormalizationMode | str = NormalizationMode.STANDARD,
) -> list[ScoredTranscript]:
    resolved_ground_truth_dir = source_truth_dir or ground_truth_dir
    sample_ids = discover_sample_ids(
        audio_dir=Path("__missing_audio_dir__"),
        ground_truth_dir=resolved_ground_truth_dir,
        transcriptions_dir=transcriptions_dir,
    )
    scored: list[ScoredTranscript] = []
    for sample_id in sample_ids:
        ground_truth_path = find_ground_truth_path(resolved_ground_truth_dir, sample_id)
        if ground_truth_path is None:
            continue
        output_paths = find_output_paths(transcriptions_dir, sample_id)
        if not output_paths:
            continue
        scored.extend(
            score_output_paths(
                sample_id=sample_id,
                ground_truth_path=ground_truth_path,
                output_paths=output_paths,
                normalization=normalization,
            )
        )
    if not scored:
        raise ScoringError(
            f"No scoreable transcript pairs found under {resolved_ground_truth_dir} and {transcriptions_dir}"
        )
    return scored


def grouped_scores(scored: list[ScoredTranscript]) -> list[GroupedScore]:
    grouped: dict[tuple[str, str], list[ScoredTranscript]] = {}
    for item in scored:
        grouped.setdefault((item.provider, item.model), []).append(item)

    summaries: list[GroupedScore] = []
    for provider, model in sorted(grouped):
        items = grouped[(provider, model)]
        summaries.append(
            GroupedScore(
                provider=provider,
                model=model,
                aggregate=aggregate_scores([item.score for item in items]),
                sample_count=len({item.sample_id for item in items}),
                transcript_count=len(items),
            )
        )
    return summaries


def scored_transcripts_to_dict(scored: list[ScoredTranscript]) -> dict[str, Any]:
    aggregate = aggregate_scores([item.score for item in scored])
    return {
        "normalization": scored[0].score.normalization.value if scored else NormalizationMode.STANDARD.value,
        "aggregate": aggregate_to_dict(
            aggregate,
            sample_count=len({item.sample_id for item in scored}),
            transcript_count=len(scored),
        ),
        "groups": [grouped_score_to_dict(group) for group in grouped_scores(scored)],
        "transcripts": [scored_transcript_to_dict(item) for item in scored],
    }


def grouped_score_to_dict(group: GroupedScore) -> dict[str, Any]:
    return {
        "provider": group.provider,
        "model": group.model,
        "wer": group.aggregate.wer,
        "sample_count": group.sample_count,
        "transcript_count": group.transcript_count,
        "counts": {
            "hits": group.aggregate.counts.hits,
            "substitutions": group.aggregate.counts.substitutions,
            "deletions": group.aggregate.counts.deletions,
            "insertions": group.aggregate.counts.insertions,
            "reference_tokens": group.aggregate.counts.reference_tokens,
            "errors": group.aggregate.counts.errors,
        },
    }


def scored_transcript_to_dict(item: ScoredTranscript) -> dict[str, Any]:
    score = item.score
    return {
        "sample_id": item.sample_id,
        "provider": item.provider,
        "model": item.model,
        "ground_truth_path": item.ground_truth_path.as_posix(),
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


def aggregate_to_dict(aggregate: AggregateScore, *, sample_count: int, transcript_count: int) -> dict[str, Any]:
    return {
        "wer": aggregate.wer,
        "sample_count": sample_count,
        "transcript_count": transcript_count,
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


def render_scores_output(
    scored: list[ScoredTranscript],
    *,
    output_format: ScoreOutputFormat = "text",
    show_alignment: bool = False,
    top_errors: int = 10,
) -> str:
    if output_format == "json":
        return render_scores_json(scored)
    if output_format == "markdown":
        return render_scores_markdown(scored, show_alignment=show_alignment, top_errors=top_errors)
    if output_format == "csv":
        return render_scores_csv(scored)
    return render_scores_text(scored, show_alignment=show_alignment, top_errors=top_errors)


def write_or_print_score_output(content: str, *, output_path: Path | None = None) -> None:
    if output_path is None:
        print(content)
        return
    if output_path.is_dir():
        raise ScoringError(f"Output path must be a file, not a directory: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")
    print(output_path)


def render_scores_text(scored: list[ScoredTranscript], *, show_alignment: bool = False, top_errors: int = 10) -> str:
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
    lines.extend(["", render_grouped_scores_text(grouped_scores(scored))])
    if top_errors > 0:
        lines.extend(["", render_error_summary(scored, top_errors=top_errors)])
    if show_alignment:
        lines.extend(["", render_alignment_blocks(scored)])
    return "\n".join(lines)


def render_grouped_scores_text(groups: list[GroupedScore]) -> str:
    lines = ["by provider/model", "provider\tmodel\twer\tS\tD\tI\tN\tsamples\ttranscripts"]
    for group in groups:
        lines.append(
            "\t".join(
                [
                    group.provider,
                    group.model,
                    format_rate(group.aggregate.wer),
                    str(group.aggregate.counts.substitutions),
                    str(group.aggregate.counts.deletions),
                    str(group.aggregate.counts.insertions),
                    str(group.aggregate.counts.reference_tokens),
                    str(group.sample_count),
                    str(group.transcript_count),
                ]
            )
        )
    return "\n".join(lines)


def render_error_summary(scored: list[ScoredTranscript], *, top_errors: int = 10) -> str:
    substitutions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()
    deletions: Counter[str] = Counter()
    for item in scored:
        for operation in item.score.alignment:
            if operation.type == "substitute":
                for reference_token, hypothesis_token, marker in operation_columns(operation):
                    if marker == "S":
                        substitutions[operation_label((reference_token,), (hypothesis_token,))] += 1
            elif operation.type == "insert":
                for token in operation.hypothesis:
                    insertions[token] += 1
            elif operation.type == "delete":
                for token in operation.reference:
                    deletions[token] += 1

    lines = ["top errors"]
    if not substitutions and not insertions and not deletions:
        lines.append("(no errors)")
        return "\n".join(lines)

    lines.extend(render_counter_section("substitutions", substitutions, top_errors))
    lines.extend(render_counter_section("insertions", insertions, top_errors))
    lines.extend(render_counter_section("deletions", deletions, top_errors))
    return "\n".join(lines)


def render_counter_section(title: str, counter: Counter[str], limit: int) -> list[str]:
    lines = [title + ":"]
    if not counter:
        lines.append("  (none)")
        return lines
    for label, count in counter.most_common(limit):
        lines.append(f"  {label}  {count}")
    return lines


def render_alignment_blocks(scored: list[ScoredTranscript]) -> str:
    lines = ["alignments"]
    for item in scored:
        lines.extend(
            [
                "",
                f"=== {item.sample_id} / {item.provider}__{item.model} ===",
                render_alignment(item.score.alignment),
            ]
        )
    return "\n".join(lines)


def render_alignment(alignment: tuple[AlignmentOperation, ...], *, chunk_size: int = 12) -> str:
    reference_tokens: list[str] = []
    hypothesis_tokens: list[str] = []
    error_tokens: list[str] = []
    for operation in alignment:
        for reference_token, hypothesis_token, marker in operation_columns(operation):
            width = max(len(reference_token), len(hypothesis_token), len(marker), 1)
            reference_tokens.append(reference_token.ljust(width))
            hypothesis_tokens.append(hypothesis_token.ljust(width))
            error_tokens.append(marker.center(width))

    chunks: list[str] = []
    for start in range(0, len(reference_tokens), chunk_size):
        end = start + chunk_size
        chunks.extend(
            [
                "REF: " + " ".join(reference_tokens[start:end]).rstrip(),
                "HYP: " + " ".join(hypothesis_tokens[start:end]).rstrip(),
                "ERR: " + " ".join(error_tokens[start:end]).rstrip(),
                "",
            ]
        )
    return "\n".join(chunks).rstrip()


def operation_columns(operation: AlignmentOperation) -> list[tuple[str, str, str]]:
    if operation.type == "equal":
        return [(reference, hypothesis, "") for reference, hypothesis in zip(operation.reference, operation.hypothesis)]
    if operation.type == "insert":
        return [("*", hypothesis, "I") for hypothesis in operation.hypothesis]
    if operation.type == "delete":
        return [(reference, "*", "D") for reference in operation.reference]

    length = max(len(operation.reference), len(operation.hypothesis), 1)
    return [
        (
            operation.reference[index] if index < len(operation.reference) else "*",
            operation.hypothesis[index] if index < len(operation.hypothesis) else "*",
            "S",
        )
        for index in range(length)
    ]


def operation_label(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> str:
    return f"{' '.join(reference) or '*'} → {' '.join(hypothesis) or '*'}"


def render_scores_markdown(scored: list[ScoredTranscript], *, show_alignment: bool = False, top_errors: int = 10) -> str:
    aggregate = aggregate_scores([item.score for item in scored])
    lines = [
        "# Transcript scoring report",
        "",
        "## Aggregate",
        "",
        "| WER | S | D | I | N |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| {format_rate(aggregate.wer)} | {aggregate.counts.substitutions} | "
            f"{aggregate.counts.deletions} | {aggregate.counts.insertions} | {aggregate.counts.reference_tokens} |"
        ),
        "",
        "## By provider/model",
        "",
        "| Provider | Model | WER | S | D | I | N | Samples | Transcripts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in grouped_scores(scored):
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(group.provider),
                    markdown_cell(group.model),
                    format_rate(group.aggregate.wer),
                    str(group.aggregate.counts.substitutions),
                    str(group.aggregate.counts.deletions),
                    str(group.aggregate.counts.insertions),
                    str(group.aggregate.counts.reference_tokens),
                    str(group.sample_count),
                    str(group.transcript_count),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Transcripts",
            "",
            "| Sample | Provider | Model | WER | CER | S | D | I | N |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in scored:
        score = item.score
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(item.sample_id),
                    markdown_cell(item.provider),
                    markdown_cell(item.model),
                    format_rate(score.wer),
                    format_rate(score.cer),
                    str(score.counts.substitutions),
                    str(score.counts.deletions),
                    str(score.counts.insertions),
                    str(score.counts.reference_tokens),
                ]
            )
            + " |"
        )

    if top_errors > 0:
        lines.extend(["", "## Top errors", "", "```text", render_error_summary(scored, top_errors=top_errors), "```"])
    if show_alignment:
        lines.extend(["", "## Alignments", "", "```text", render_alignment_blocks(scored), "```"])
    return "\n".join(lines)


def render_scores_csv(scored: list[ScoredTranscript]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "sample_id",
            "provider",
            "model",
            "ground_truth_path",
            "transcription_path",
            "wer",
            "cer",
            "mer",
            "wil",
            "wip",
            "hits",
            "substitutions",
            "deletions",
            "insertions",
            "reference_tokens",
            "errors",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for item in scored:
        score = item.score
        writer.writerow(
            {
                "sample_id": item.sample_id,
                "provider": item.provider,
                "model": item.model,
                "ground_truth_path": item.ground_truth_path.as_posix(),
                "transcription_path": item.transcription_path.as_posix(),
                "wer": score.wer,
                "cer": score.cer,
                "mer": score.mer,
                "wil": score.wil,
                "wip": score.wip,
                "hits": score.counts.hits,
                "substitutions": score.counts.substitutions,
                "deletions": score.counts.deletions,
                "insertions": score.counts.insertions,
                "reference_tokens": score.counts.reference_tokens,
                "errors": score.counts.errors,
            }
        )
    return output.getvalue().rstrip("\r\n")


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def render_scores_json(scored: list[ScoredTranscript]) -> str:
    return json.dumps(scored_transcripts_to_dict(scored), ensure_ascii=False, indent=2)


def format_rate(value: float) -> str:
    return f"{value * 100:.2f}%"
