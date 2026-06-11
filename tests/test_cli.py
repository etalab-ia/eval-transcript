from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import eval_transcript


@contextlib.contextmanager
def chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ScoringCliPathFlagTests(unittest.TestCase):
    def test_score_sample_accepts_ground_truth_dir_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ground_truth_dir = root / "truth"
            transcriptions_dir = root / "transcriptions"
            sample_dir = transcriptions_dir / "sample-a"
            ground_truth_dir.mkdir()
            sample_dir.mkdir(parents=True)
            (ground_truth_dir / "sample-a.md").write_text("bonjour", encoding="utf-8")
            (sample_dir / "omlx__model.txt").write_text("bonjour", encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "eval-transcript",
                "score",
                "sample",
                "sample-a",
                "--ground-truth-dir",
                str(ground_truth_dir),
                "--transcriptions-dir",
                str(transcriptions_dir),
                "--json",
            ]
            with chdir(root), patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                eval_transcript.main()

        data = json.loads(stdout.getvalue())
        self.assertEqual(data["transcripts"][0]["ground_truth_path"], str(ground_truth_dir / "sample-a.md"))
        self.assertEqual(stderr.getvalue(), "")

    def test_score_sample_accepts_deprecated_source_truth_dir_alias(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_truth_dir = root / "source_truth"
            transcriptions_dir = root / "transcriptions"
            sample_dir = transcriptions_dir / "sample-a"
            source_truth_dir.mkdir()
            sample_dir.mkdir(parents=True)
            (source_truth_dir / "sample-a.md").write_text("bonjour", encoding="utf-8")
            (sample_dir / "omlx__model.txt").write_text("bonjour", encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "eval-transcript",
                "score",
                "sample",
                "sample-a",
                "--source-truth-dir",
                str(source_truth_dir),
                "--transcriptions-dir",
                str(transcriptions_dir),
                "--json",
            ]
            with chdir(root), patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                eval_transcript.main()

        data = json.loads(stdout.getvalue())
        self.assertEqual(data["transcripts"][0]["ground_truth_path"], str(source_truth_dir / "sample-a.md"))
        self.assertIn("--source-truth-dir is deprecated", stderr.getvalue())


class HuggingFaceCliTests(unittest.TestCase):
    def test_huggingface_dataset_ls_prints_sample_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            with patch("eval_transcript.HuggingFaceClient") as client_class:
                client = client_class.return_value
                client.repo_id = "team/corpus"
                client.list_samples.return_value = ["sample-a", "sample-b"]

                stdout = io.StringIO()
                stderr = io.StringIO()
                argv = [
                    "eval-transcript",
                    "huggingface",
                    "dataset",
                    "ls",
                ]
                with chdir(root), patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    eval_transcript.main()

            client_class.assert_called_once()
            self.assertEqual(stdout.getvalue().splitlines(), ["sample-a", "sample-b"])
            self.assertEqual(stderr.getvalue(), "")

    def test_huggingface_dataset_push_requires_token(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = [
                "eval-transcript",
                "huggingface",
                "dataset",
                "push",
                "--audio-dir",
                str(root / "audio"),
                "--ground-truth-dir",
                str(root / "ground_truth"),
            ]
            with chdir(root), patch.dict(os.environ, {}, clear=True), patch("eval_transcript.load_dotenv"), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    eval_transcript.main()
            self.assertIn("HF_TOKEN", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
