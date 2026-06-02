from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import jiwer


class NormalizationMode(StrEnum):
    """Supported text normalization levels for transcript scoring."""

    RAW = "raw"
    STANDARD = "standard"
    STANDARD_NUMBERS = "standard_numbers"


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
    wer = counts.errors / reference_tokens if reference_tokens else float(counts.insertions)
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

    if resolved_mode is NormalizationMode.STANDARD_NUMBERS:
        # Convert spelled-out cardinals to digits so that e.g. "cinq" and "5"
        # match. Done before casefolding/punctuation stripping because the
        # converter relies on word boundaries.
        text = _spell_out_to_digits(text)

    text = text.casefold()
    text = text.translate(APOSTROPHE_TRANSLATION)
    text = "".join(" " if is_scoring_separator(character) else character for character in text)
    text = " ".join(text.split())

    if resolved_mode is NormalizationMode.STANDARD_NUMBERS:
        text = _canonicalize_numbers(text)
    return text


# French spelled-out ordinals that text2num does not map to a bare digit.
_ORDINAL_WORDS = {
    "premier": "1", "première": "1", "second": "2", "seconde": "2",
    "deuxième": "2", "troisième": "3", "quatrième": "4", "cinquième": "5",
    "sixième": "6", "septième": "7", "huitième": "8", "neuvième": "9", "dixième": "10",
    "premiers": "1", "premières": "1", "seconds": "2", "secondes": "2",
    "deuxièmes": "2", "troisièmes": "3", "quatrièmes": "4", "cinquièmes": "5",
    "sixièmes": "6", "septièmes": "7", "huitièmes": "8", "neuvièmes": "9", "dixièmes": "10",
}
_NUMBER_GAP = re.compile(r"(?<=\d)\s+(?=\d{3}(?!\d))")
_ORDINAL_SUFFIX = re.compile(r"(\d+)(?:er|ère|ere|re|ème|eme|e|nd|nde|de)s?\b")


def _spell_out_to_digits(text: str) -> str:
    """Convert spelled-out French cardinals to digits (e.g. "vingt-cinq" -> "25")."""
    from text_to_num import alpha2digit

    return alpha2digit(text, "fr")


def _canonicalize_numbers(text: str) -> str:
    """Collapse digit formatting so numeric values compare equal regardless of spelling.

    Runs on already standardized (lowercased, de-punctuated) text: strips the digit-group
    spaces left by "2 500", reduces ordinal markers ("5eme" -> "5"), and maps the few
    spelled-out ordinals text2num leaves untouched ("deuxième" -> "2").
    """
    text = _NUMBER_GAP.sub("", text)
    text = _ORDINAL_SUFFIX.sub(r"\1", text)
    return " ".join(_ORDINAL_WORDS.get(word, word) for word in text.split())


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


def is_scoring_separator(character: str) -> bool:
    if character not in _SCORING_SEPARATOR_CACHE:
        _SCORING_SEPARATOR_CACHE[character] = unicodedata.category(character).startswith(("P", "S"))
    return _SCORING_SEPARATOR_CACHE[character]


_SCORING_SEPARATOR_CACHE: dict[str, bool] = {}


APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "ʼ": "'",
        "`": "'",
        "´": "'",
    }
)
