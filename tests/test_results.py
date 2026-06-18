from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import eval_transcript
from eval_transcript.results import (
    PlannedUpload,
    PushPlan,
    ResultsClient,
    ResultsError,
    build_push_plan,
    public_sample_ids,
)


@contextlib.contextmanager
def chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class FakeHfApi:
    def __init__(self, *, ground_truth_ids: list[str]) -> None:
        self.files = [f"ground_truth/{sid}.txt" for sid in ground_truth_ids] + [
            f"audio/{sid}.mp3" for sid in ground_truth_ids
        ]
        self.commit_calls: list[dict[str, object]] = []

    def list_repo_files(self, repo_id, *, repo_type, revision=None):
        return list(self.files)

    def create_commit(self, *, repo_id, operations, commit_message, repo_type, token=None):
        self.commit_calls.append(
            {
                "repo_id": repo_id,
                "ops": [(op.path_in_repo, str(op.path_or_fileobj)) for op in operations],
                "message": commit_message,
                "repo_type": repo_type,
                "token": token,
            }
        )


def _make_transcriptions(root: Path, layout: dict[str, list[str]]) -> Path:
    tx = root / "transcriptions"
    for sid, files in layout.items():
        d = tx / sid
        d.mkdir(parents=True)
        for name in files:
            (d / name).write_text("bonjour", encoding="utf-8")
    return tx


class PublicSampleIdsTests(unittest.TestCase):
    def test_extracts_ids_from_ground_truth(self) -> None:
        api = FakeHfApi(ground_truth_ids=["officiel-a", "officiel-b"])
        self.assertEqual(public_sample_ids(api, "corpus"), {"officiel-a", "officiel-b"})


class BuildPushPlanTests(unittest.TestCase):
    def test_keeps_officiels_and_skips_private(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx = _make_transcriptions(
                root,
                {
                    "officiel-a": ["whisperx__large-v2.txt", "omlx__cohere.txt"],
                    "reunion-interne": ["whisperx__large-v2.txt"],  # privé -> ignoré
                },
            )
            api = FakeHfApi(ground_truth_ids=["officiel-a"])
            plan = build_push_plan(
                api=api, corpus_repo="c", results_repo="r", transcriptions_dir=tx
            )
            paths = sorted(u.path_in_repo for u in plan.uploads)
            self.assertEqual(
                paths,
                [
                    "transcriptions/officiel-a/omlx__cohere.txt",
                    "transcriptions/officiel-a/whisperx__large-v2.txt",
                ],
            )
            self.assertEqual(plan.skipped_samples, ["reunion-interne"])

    def test_include_filter(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx = _make_transcriptions(
                root, {"officiel-a": ["whisperx__large-v2.txt", "omlx__cohere.txt"]}
            )
            api = FakeHfApi(ground_truth_ids=["officiel-a"])
            plan = build_push_plan(
                api=api, corpus_repo="c", results_repo="r", transcriptions_dir=tx, include="whisperx"
            )
            self.assertEqual(
                [u.path_in_repo for u in plan.uploads],
                ["transcriptions/officiel-a/whisperx__large-v2.txt"],
            )

    def test_empty_corpus_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            tx = _make_transcriptions(Path(tmp), {"officiel-a": ["whisperx__large-v2.txt"]})
            api = FakeHfApi(ground_truth_ids=[])
            with self.assertRaisesRegex(ResultsError, "No public samples"):
                build_push_plan(api=api, corpus_repo="c", results_repo="r", transcriptions_dir=tx)

    def test_missing_dir_raises(self) -> None:
        api = FakeHfApi(ground_truth_ids=["officiel-a"])
        with self.assertRaisesRegex(ResultsError, "not found"):
            build_push_plan(
                api=api, corpus_repo="c", results_repo="r", transcriptions_dir=Path("/no/such/dir")
            )


class PushTests(unittest.TestCase):
    def test_push_requires_token(self) -> None:
        plan = PushPlan(
            results_repo="r",
            officiels=["officiel-a"],
            uploads=[
                PlannedUpload(
                    "officiel-a", "transcriptions/officiel-a/whisperx__large-v2.txt", Path("x.txt")
                )
            ],
            skipped_samples=[],
        )
        with patch.dict("os.environ", {}, clear=True):
            client = ResultsClient(api=FakeHfApi(ground_truth_ids=["officiel-a"]))
            with self.assertRaisesRegex(ResultsError, "HF_TOKEN"):
                client.push(plan)

    def test_push_builds_single_commit(self) -> None:
        # huggingface_hub.CommitOperationAdd réel expose .path_in_repo / .path_or_fileobj,
        # compatibles avec FakeHfApi -> pas besoin de le patcher.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx = _make_transcriptions(root, {"officiel-a": ["whisperx__large-v2.txt"]})
            api = FakeHfApi(ground_truth_ids=["officiel-a"])
            client = ResultsClient(results_repo="team/results", token="tok", api=api)
            plan = client.plan(transcriptions_dir=tx)
            client.push(plan, message="msg")
            self.assertEqual(len(api.commit_calls), 1)
            call = api.commit_calls[0]
            self.assertEqual(call["repo_id"], "team/results")
            self.assertEqual(call["token"], "tok")
            self.assertEqual(
                call["ops"],
                [
                    (
                        "transcriptions/officiel-a/whisperx__large-v2.txt",
                        str(tx / "officiel-a" / "whisperx__large-v2.txt"),
                    )
                ],
            )


class ResultsCliTests(unittest.TestCase):
    def test_dry_run_prints_plan_without_pushing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx = _make_transcriptions(
                root, {"officiel-a": ["whisperx__large-v2.txt"], "reunion-x": ["whisperx__large-v2.txt"]}
            )
            api = FakeHfApi(ground_truth_ids=["officiel-a"])
            argv = [
                "eval-transcript", "results", "push",
                "--transcriptions-dir", str(tx),
                "--dry-run",
            ]
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("eval_transcript.ResultsClient") as client_cls:
                client = client_cls.return_value
                client.plan.return_value = build_push_plan(
                    api=api, corpus_repo="c", results_repo="team/results", transcriptions_dir=tx
                )
                with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", argv), \
                        contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    eval_transcript.main()
            out = stdout.getvalue()
            self.assertIn("transcriptions/officiel-a/whisperx__large-v2.txt", out)
            self.assertIn("Ignorés (hors corpus public): reunion-x", out)
            self.assertIn("[dry-run]", out)
            client.push.assert_not_called()


if __name__ == "__main__":
    unittest.main()
