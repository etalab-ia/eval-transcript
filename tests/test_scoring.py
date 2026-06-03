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

    def test_standard_numbers_aligns_spelled_and_digit_cardinals(self) -> None:
        mode = NormalizationMode.STANDARD_NUMBERS
        self.assertEqual(
            normalize_transcript("cinq axes et vingt-cinq mesures", mode),
            normalize_transcript("5 axes et 25 mesures", mode),
        )

    def test_standard_numbers_aligns_ordinals_and_thousands(self) -> None:
        mode = NormalizationMode.STANDARD_NUMBERS
        self.assertEqual(
            normalize_transcript("le deuxième pilier, deux mille cinq cents emplois", mode),
            normalize_transcript("le 2e pilier, 2 500 emplois", mode),
        )
        # ordinaux pluriels et abréviations standard
        self.assertEqual(
            normalize_transcript("les deuxièmes places, les premières lignes", mode),
            normalize_transcript("les 2ndes places, les 1res lignes", mode),
        )
        # les espaces hors séparateur de milliers ne doivent pas être compactés
        self.assertEqual(
            normalize_transcript("page 5 10, entre 15 20 personnes", mode),
            "page 5 10 entre 15 20 personnes",
        )

    def test_standard_numbers_leaves_words_untouched(self) -> None:
        self.assertEqual(
            normalize_transcript("bonjour le monde", NormalizationMode.STANDARD_NUMBERS),
            "bonjour le monde",
        )


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

    def test_score_sample_outputs_accepts_txt_source_truth(self) -> None:
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import score_sample_outputs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            transcriptions_dir = root / "transcriptions"
            sample_dir = transcriptions_dir / "sample-a"
            source_truth_dir.mkdir()
            sample_dir.mkdir(parents=True)
            (source_truth_dir / "sample-a.txt").write_text("bonjour le monde", encoding="utf-8")
            (sample_dir / "omlx__parakeet.txt").write_text("bonjour monde", encoding="utf-8")

            scored = score_sample_outputs(
                "sample-a",
                source_truth_dir=source_truth_dir,
                transcriptions_dir=transcriptions_dir,
            )

        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0].source_truth_path.suffix, ".txt")
        self.assertAlmostEqual(scored[0].score.wer, 1 / 3)

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

    def test_grouped_scores_aggregates_by_provider_and_model(self) -> None:
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import grouped_scores, score_all_outputs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            transcriptions_dir = root / "transcriptions"
            source_truth_dir.mkdir()
            for sample_id, reference in {"sample-a": "un deux", "sample-b": "un deux trois"}.items():
                (source_truth_dir / f"{sample_id}.md").write_text(reference, encoding="utf-8")
                sample_dir = transcriptions_dir / sample_id
                sample_dir.mkdir(parents=True)
                (sample_dir / "omlx__model-a.txt").write_text(reference, encoding="utf-8")
            (transcriptions_dir / "sample-a" / "albert__model-b.txt").write_text("un", encoding="utf-8")

            groups = grouped_scores(
                score_all_outputs(
                    source_truth_dir=source_truth_dir,
                    transcriptions_dir=transcriptions_dir,
                )
            )

        self.assertEqual([(group.provider, group.model) for group in groups], [("albert", "model-b"), ("omlx", "model-a")])
        self.assertEqual(groups[0].sample_count, 1)
        self.assertEqual(groups[0].transcript_count, 1)
        self.assertEqual(groups[0].aggregate.counts.reference_tokens, 2)
        self.assertEqual(groups[0].aggregate.counts.errors, 1)
        self.assertEqual(groups[1].sample_count, 2)
        self.assertEqual(groups[1].transcript_count, 2)
        self.assertEqual(groups[1].aggregate.counts.errors, 0)

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

        self.assertIn("by provider/model", rendered)
        self.assertIn("provider\tmodel\twer\tS\tD\tI\tN\tsamples\ttranscripts", rendered)
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

    def test_render_scores_markdown_includes_tables_and_diagnostics(self) -> None:
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import render_scores_markdown, score_sample_outputs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            sample_dir = root / "transcriptions" / "sample-a"
            source_truth_dir.mkdir()
            sample_dir.mkdir(parents=True)
            (source_truth_dir / "sample-a.md").write_text("bonjour le monde", encoding="utf-8")
            (sample_dir / "omlx__model.txt").write_text("bonjour beau monde", encoding="utf-8")

            rendered = render_scores_markdown(
                score_sample_outputs(
                    "sample-a",
                    source_truth_dir=source_truth_dir,
                    transcriptions_dir=root / "transcriptions",
                ),
                show_alignment=True,
                top_errors=5,
            )

        self.assertIn("# Transcript scoring report", rendered)
        self.assertIn("## By provider/model", rendered)
        self.assertIn("| Provider | Model | WER | S | D | I | N | Samples | Transcripts |", rendered)
        self.assertIn("| Sample | Provider | Model | WER | CER | S | D | I | N |", rendered)
        self.assertIn("## Top errors", rendered)
        self.assertIn("le → beau  1", rendered)
        self.assertIn("## Alignments", rendered)
        self.assertIn("REF:", rendered)

    def test_render_scores_csv_outputs_one_row_per_transcript(self) -> None:
        import csv
        import io
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import render_scores_csv, score_sample_outputs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            sample_dir = root / "transcriptions" / "sample-a"
            source_truth_dir.mkdir()
            sample_dir.mkdir(parents=True)
            (source_truth_dir / "sample-a.md").write_text("oui", encoding="utf-8")
            (sample_dir / "omlx__model.txt").write_text("non", encoding="utf-8")

            rendered = render_scores_csv(
                score_sample_outputs(
                    "sample-a",
                    source_truth_dir=source_truth_dir,
                    transcriptions_dir=root / "transcriptions",
                )
            )

        self.assertNotIn("\r", rendered)
        rows = list(csv.DictReader(io.StringIO(rendered)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "sample-a")
        self.assertEqual(rows[0]["provider"], "omlx")
        self.assertEqual(rows[0]["substitutions"], "1")

    def test_write_or_print_score_output_rejects_directory_paths(self) -> None:
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import ScoringError, write_or_print_score_output

        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ScoringError, "must be a file"):
                write_or_print_score_output("content", output_path=Path(tmp))

    def test_write_or_print_score_output_writes_file_and_prints_path(self) -> None:
        import contextlib
        import io
        from tempfile import TemporaryDirectory

        from eval_transcript.scoring_cli import write_or_print_score_output

        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "reports" / "score.md"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                write_or_print_score_output("content", output_path=output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "content\n")
            self.assertEqual(stdout.getvalue().strip(), str(output_path))

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
        self.assertEqual([(group["provider"], group["model"]) for group in data["groups"]], [("albert", "model"), ("omlx", "model")])
        self.assertEqual(data["groups"][0]["sample_count"], 1)
        self.assertEqual(data["groups"][0]["transcript_count"], 1)
        self.assertEqual([item["sample_id"] for item in data["transcripts"]], ["sample-a", "sample-a"])
        self.assertEqual(data["transcripts"][1]["counts"]["substitutions"], 1)


if __name__ == "__main__":
    unittest.main()
