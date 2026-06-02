from __future__ import annotations

import unittest
from pathlib import Path

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

    def test_standard_normalization_replaces_symbols_with_spaces(self) -> None:
        self.assertEqual(
            normalize_transcript("Coût: 5€ + TVA = 6€"),
            "coût 5 tva 6",
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
    def test_aggregate_wer_preserves_empty_reference_insertion_count(self) -> None:
        score = score_transcript_pair("", "hallucinated words here")

        aggregate = aggregate_scores([score])

        self.assertEqual(score.wer, 3.0)
        self.assertEqual(aggregate.wer, score.wer)
        self.assertEqual(aggregate.counts.insertions, 3)

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


class ScoreCliTests(unittest.TestCase):
    def test_score_sample_outputs_scores_each_transcript(self) -> None:
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import score_sample_outputs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            transcriptions_dir = root / "transcriptions"
            sample_dir = transcriptions_dir / "sample-a"
            source_truth_dir.mkdir()
            sample_dir.mkdir(parents=True)
            (source_truth_dir / "sample-a.md").write_text("bonjour le monde", encoding="utf-8")
            (sample_dir / "omlx__parakeet.txt").write_text("bonjour monde", encoding="utf-8")
            (sample_dir / "albert__whisper.txt").write_text("bonjour le monde", encoding="utf-8")

            scored = score_sample_outputs(
                "sample-a",
                source_truth_dir=source_truth_dir,
                transcriptions_dir=transcriptions_dir,
            )

        self.assertEqual([item.provider for item in scored], ["albert", "omlx"])
        self.assertAlmostEqual(scored[0].score.wer, 0.0)
        self.assertAlmostEqual(scored[1].score.wer, 1 / 3)

    def test_score_all_outputs_skips_incomplete_samples(self) -> None:
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import score_all_outputs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            transcriptions_dir = root / "transcriptions"
            source_truth_dir.mkdir()
            (source_truth_dir / "complete.md").write_text("un deux", encoding="utf-8")
            (source_truth_dir / "missing-output.md").write_text("ignored", encoding="utf-8")
            complete_dir = transcriptions_dir / "complete"
            orphan_dir = transcriptions_dir / "missing-truth"
            complete_dir.mkdir(parents=True)
            orphan_dir.mkdir(parents=True)
            (complete_dir / "omlx__model.txt").write_text("un deux", encoding="utf-8")
            (orphan_dir / "omlx__model.txt").write_text("ignored", encoding="utf-8")

            scored = score_all_outputs(
                source_truth_dir=source_truth_dir,
                transcriptions_dir=transcriptions_dir,
            )

        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0].sample_id, "complete")


    def test_render_scores_text_includes_top_errors(self) -> None:
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import render_scores_text, score_sample_outputs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            sample_dir = root / "transcriptions" / "sample-a"
            source_truth_dir.mkdir()
            sample_dir.mkdir(parents=True)
            (source_truth_dir / "sample-a.md").write_text("bonjour le monde", encoding="utf-8")
            (sample_dir / "omlx__model.txt").write_text("bonjour beau monde encore", encoding="utf-8")

            rendered = render_scores_text(
                score_sample_outputs(
                    "sample-a",
                    source_truth_dir=source_truth_dir,
                    transcriptions_dir=root / "transcriptions",
                ),
                top_errors=5,
            )

        self.assertIn("top errors", rendered)
        self.assertIn("substitutions:", rendered)
        self.assertIn("le → beau  1", rendered)
        self.assertIn("insertions:", rendered)
        self.assertIn("encore  1", rendered)
        self.assertIn("deletions:", rendered)
        self.assertIn("(none)", rendered)

    def test_render_scores_text_splits_adjacent_substitutions_for_top_errors(self) -> None:
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import render_scores_text, score_sample_outputs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            sample_dir = root / "transcriptions" / "sample-a"
            source_truth_dir.mkdir()
            sample_dir.mkdir(parents=True)
            (source_truth_dir / "sample-a.md").write_text("alpha beta", encoding="utf-8")
            (sample_dir / "omlx__model.txt").write_text("xray yankee", encoding="utf-8")

            rendered = render_scores_text(
                score_sample_outputs(
                    "sample-a",
                    source_truth_dir=source_truth_dir,
                    transcriptions_dir=root / "transcriptions",
                ),
                top_errors=5,
            )

        self.assertIn("alpha → xray  1", rendered)
        self.assertIn("beta → yankee  1", rendered)
        self.assertNotIn("alpha beta → xray yankee", rendered)

    def test_render_alignment_chunks_long_alignments(self) -> None:
        from eval_transcript.scoring_cli import render_alignment

        score = score_transcript_pair(
            "un deux trois quatre cinq six sept huit neuf dix onze douze treize",
            "un deux trois quatre cinq six sept huit neuf dix onze douze treize",
        )

        rendered = render_alignment(score.alignment, chunk_size=5)

        self.assertEqual(rendered.count("REF:"), 3)
        self.assertEqual(rendered.count("HYP:"), 3)
        self.assertEqual(rendered.count("ERR:"), 3)

    def test_render_scores_text_can_show_alignment_blocks(self) -> None:
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import render_scores_text, score_sample_outputs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            sample_dir = root / "transcriptions" / "sample-a"
            source_truth_dir.mkdir()
            sample_dir.mkdir(parents=True)
            (source_truth_dir / "sample-a.md").write_text("bonjour le monde", encoding="utf-8")
            (sample_dir / "omlx__model.txt").write_text("bonjour beau monde", encoding="utf-8")

            rendered = render_scores_text(
                score_sample_outputs(
                    "sample-a",
                    source_truth_dir=source_truth_dir,
                    transcriptions_dir=root / "transcriptions",
                ),
                show_alignment=True,
                top_errors=0,
            )

        self.assertIn("alignments", rendered)
        self.assertIn("=== sample-a / omlx__model ===", rendered)
        self.assertIn("REF:", rendered)
        self.assertIn("HYP:", rendered)
        self.assertIn("ERR:", rendered)
        self.assertIn("S", rendered)

    def test_render_scores_json_includes_aggregate_and_transcripts(self) -> None:
        import json
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import render_scores_json, score_sample_outputs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            sample_dir = root / "transcriptions" / "sample-a"
            source_truth_dir.mkdir()
            sample_dir.mkdir(parents=True)
            (source_truth_dir / "sample-a.md").write_text("oui", encoding="utf-8")
            (sample_dir / "omlx__model.txt").write_text("non", encoding="utf-8")
            (sample_dir / "albert__model.txt").write_text("oui", encoding="utf-8")

            rendered = render_scores_json(
                score_sample_outputs(
                    "sample-a",
                    source_truth_dir=source_truth_dir,
                    transcriptions_dir=root / "transcriptions",
                )
            )

        data = json.loads(rendered)
        self.assertEqual(data["aggregate"]["sample_count"], 1)
        self.assertEqual(data["aggregate"]["transcript_count"], 2)
        self.assertEqual([item["sample_id"] for item in data["transcripts"]], ["sample-a", "sample-a"])
        self.assertEqual(data["transcripts"][1]["counts"]["substitutions"], 1)


if __name__ == "__main__":
    unittest.main()
