from __future__ import annotations

import unittest

from eval_transcript.scoring import (
    AlignmentOperation,
    NormalizationMode,
    aggregate_scores,
    normalize_transcript,
    score_transcript_pair,
)


class NormalizeTranscriptTests(unittest.TestCase):
    def test_standard_normalization_preserves_accents_and_removes_punctuation(self) -> None:
        self.assertEqual(
            normalize_transcript("L’État — c’est l’été !"),
            "l état c est l été",
        )

    def test_raw_normalization_only_normalizes_unicode(self) -> None:
        self.assertEqual(
            normalize_transcript("État, été", NormalizationMode.RAW),
            "État, été",
        )

    def test_unsupported_normalization_mode_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported normalization mode"):
            normalize_transcript("bonjour", "unknown")


class ScoreTranscriptPairTests(unittest.TestCase):
    def test_scores_word_and_character_errors(self) -> None:
        score = score_transcript_pair("bonjour le monde", "bonjour monde")

        self.assertAlmostEqual(score.wer, 1 / 3)
        self.assertGreater(score.cer, 0)
        self.assertEqual(score.counts.hits, 2)
        self.assertEqual(score.counts.substitutions, 0)
        self.assertEqual(score.counts.deletions, 1)
        self.assertEqual(score.counts.insertions, 0)
        self.assertEqual(score.counts.reference_tokens, 3)
        self.assertEqual(score.counts.errors, 1)

    def test_alignment_operations_preserve_word_spans(self) -> None:
        score = score_transcript_pair("bonjour le monde", "bonjour beau monde")

        self.assertEqual(
            score.alignment,
            (
                AlignmentOperation(type="equal", reference=("bonjour",), hypothesis=("bonjour",)),
                AlignmentOperation(type="substitute", reference=("le",), hypothesis=("beau",)),
                AlignmentOperation(type="equal", reference=("monde",), hypothesis=("monde",)),
            ),
        )

    def test_normalized_text_is_returned_for_auditability(self) -> None:
        score = score_transcript_pair("L’État", "l etat")

        self.assertEqual(score.normalization, NormalizationMode.STANDARD)
        self.assertEqual(score.normalized_reference, "l état")
        self.assertEqual(score.normalized_hypothesis, "l etat")


class AggregateScoreTests(unittest.TestCase):
    def test_aggregate_wer_uses_total_counts_not_mean_sample_wer(self) -> None:
        short_bad = score_transcript_pair("oui", "non")
        long_good = score_transcript_pair("un deux trois quatre cinq", "un deux trois quatre cinq")

        aggregate = aggregate_scores([short_bad, long_good])

        self.assertAlmostEqual(short_bad.wer, 1.0)
        self.assertAlmostEqual(long_good.wer, 0.0)
        self.assertAlmostEqual(aggregate.wer, 1 / 6)
        self.assertEqual(aggregate.sample_count, 2)
        self.assertEqual(aggregate.counts.reference_tokens, 6)
        self.assertEqual(aggregate.counts.errors, 1)


if __name__ == "__main__":
    unittest.main()
