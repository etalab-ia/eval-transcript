from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import jiwer


class NormalizationMode(StrEnum):
    """Supported text normalization levels for transcript scoring."""

    RAW = "raw"
    STANDARD = "standard"


OperationType = Literal["equal", "substitute", "delete", "insert"]


@dataclass(frozen=True)
class ScoreCounts:
    """Edit operation counts for a reference/hypothesis transcript pair."""

    hits: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def reference_tokens(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class AlignmentOperation:
    """A word-level alignment span between a normalized reference and hypothesis."""

    type: OperationType
    reference: tuple[str, ...]
    hypothesis: tuple[str, ...]


@dataclass(frozen=True)
class TranscriptScore:
    """ASR error metrics and alignment for one reference/hypothesis pair."""

    wer: float
    cer: float
    mer: float
    wil: float
    wip: float
    counts: ScoreCounts
    alignment: tuple[AlignmentOperation, ...]
    normalized_reference: str
    normalized_hypothesis: str
    normalization: NormalizationMode


@dataclass(frozen=True)
class AggregateScore:
    """Corpus-level metrics computed from total edit counts across transcript scores."""

    wer: float
    counts: ScoreCounts
    sample_count: int


def score_transcript_pair(
    reference: str,
    hypothesis: str,
    *,
    normalization: NormalizationMode | str = NormalizationMode.STANDARD,
) -> TranscriptScore:
    """Score one reference/hypothesis transcript pair.

    WER, MER, WIL, WIP, and word-level alignment are computed on normalized word tokens.
    CER is computed on normalized characters with spaces removed by jiwer's character transform.
    """

    mode = normalization_mode(normalization)
    normalized_reference = normalize_transcript(reference, mode)
    normalized_hypothesis = normalize_transcript(hypothesis, mode)

    word_output = jiwer.process_words(
        normalized_reference,
        normalized_hypothesis,
        reference_transform=jiwer.wer_default,
        hypothesis_transform=jiwer.wer_default,
    )
    char_output = jiwer.process_characters(
        normalized_reference,
        normalized_hypothesis,
        reference_transform=jiwer.cer_default,
        hypothesis_transform=jiwer.cer_default,
    )

    counts = ScoreCounts(
        hits=word_output.hits,
        substitutions=word_output.substitutions,
        deletions=word_output.deletions,
        insertions=word_output.insertions,
    )
    return TranscriptScore(
        wer=word_output.wer,
        cer=char_output.cer,
        mer=word_output.mer,
        wil=word_output.wil,
        wip=word_output.wip,
        counts=counts,
        alignment=alignment_operations(word_output),
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        normalization=mode,
    )


def aggregate_scores(scores: list[TranscriptScore]) -> AggregateScore:
    """Compute corpus-level WER from total counts, not mean per-sample WER."""

    counts = ScoreCounts(
        hits=sum(score.counts.hits for score in scores),
        substitutions=sum(score.counts.substitutions for score in scores),
        deletions=sum(score.counts.deletions for score in scores),
        insertions=sum(score.counts.insertions for score in scores),
    )
    reference_tokens = counts.reference_tokens
    wer = counts.errors / reference_tokens if reference_tokens else float(counts.insertions > 0)
    return AggregateScore(wer=wer, counts=counts, sample_count=len(scores))


def normalize_transcript(text: str, mode: NormalizationMode | str = NormalizationMode.STANDARD) -> str:
    """Normalize transcript text before scoring.

    The standard preset is intentionally conservative for French: it normalizes Unicode,
    casing, apostrophe variants, punctuation, and whitespace while preserving accents.
    """

    resolved_mode = normalization_mode(mode)
    text = unicodedata.normalize("NFC", text)
    if resolved_mode is NormalizationMode.RAW:
        return text

    text = text.casefold()
    text = text.translate(APOSTROPHE_TRANSLATION)
    text = "".join(" " if is_punctuation(character) else character for character in text)
    return " ".join(text.split())


def alignment_operations(word_output: Any) -> tuple[AlignmentOperation, ...]:
    operations: list[AlignmentOperation] = []
    for sentence_index, chunks in enumerate(word_output.alignments):
        reference_words = word_output.references[sentence_index]
        hypothesis_words = word_output.hypotheses[sentence_index]
        for chunk in chunks:
            operations.append(
                AlignmentOperation(
                    type=chunk.type,
                    reference=tuple(reference_words[chunk.ref_start_idx : chunk.ref_end_idx]),
                    hypothesis=tuple(hypothesis_words[chunk.hyp_start_idx : chunk.hyp_end_idx]),
                )
            )
    return tuple(operations)


def normalization_mode(value: NormalizationMode | str) -> NormalizationMode:
    if isinstance(value, NormalizationMode):
        return value
    try:
        return NormalizationMode(value)
    except ValueError as exc:
        valid_modes = ", ".join(mode.value for mode in NormalizationMode)
        raise ValueError(f"Unsupported normalization mode: {value}; expected one of {valid_modes}") from exc


def is_punctuation(character: str) -> bool:
    return unicodedata.category(character).startswith("P")


APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "ʼ": "'",
        "`": "'",
        "´": "'",
    }
)
