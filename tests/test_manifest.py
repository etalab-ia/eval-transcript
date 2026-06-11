from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import eval_transcript.manifest as manifest
from eval_transcript.manifest import ManifestSample, discover_samples, ground_truth_path_from_manifest_entry, render_manifest


class ManifestGroundTruthTests(unittest.TestCase):
    def test_render_manifest_uses_ground_truth_path_key(self) -> None:
        rendered = render_manifest(
            [
                ManifestSample(
                    id="sample-a",
                    audio_path=Path("data/audio/sample-a.wav"),
                    ground_truth_path=Path("data/ground_truth/sample-a.md"),
                    outputs=[],
                )
            ]
        )

        self.assertIn("ground_truth_path", rendered)
        self.assertNotIn("source_truth_path", rendered)

    def test_discover_samples_keeps_legacy_source_truth_dir_alias(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            source_truth_dir = root / "source_truth"
            transcriptions_dir = root / "transcriptions"
            audio_dir.mkdir()
            source_truth_dir.mkdir()
            (audio_dir / "sample-a.wav").write_bytes(b"RIFF")
            (source_truth_dir / "sample-a.md").write_text("bonjour", encoding="utf-8")

            samples = discover_samples(
                audio_dir=audio_dir,
                source_truth_dir=source_truth_dir,
                transcriptions_dir=transcriptions_dir,
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].ground_truth_path, source_truth_dir / "sample-a.md")
        self.assertEqual(samples[0].source_truth_path, source_truth_dir / "sample-a.md")

    def test_legacy_source_truth_path_manifest_key_warns_once(self) -> None:
        manifest._legacy_source_truth_path_key_warning_printed = False
        stderr = StringIO()

        with redirect_stderr(stderr):
            first = ground_truth_path_from_manifest_entry({"source_truth_path": "data/source_truth/sample.md"})
            second = ground_truth_path_from_manifest_entry({"source_truth_path": "data/source_truth/other.md"})

        self.assertEqual(first, Path("data/source_truth/sample.md"))
        self.assertEqual(second, Path("data/source_truth/other.md"))
        self.assertEqual(stderr.getvalue().count("source_truth_path manifest key is deprecated"), 1)


if __name__ == "__main__":
    unittest.main()
