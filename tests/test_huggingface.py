from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from eval_transcript.huggingface import (
    DEFAULT_DATASET_REPO,
    HuggingFaceClient,
    HuggingFaceError,
    PullResult,
    _download_module_level,
    _repository_not_found_error,
)


class FakeInfo:
    def __init__(self, private: bool) -> None:
        self.private = private


class FakeOperation:
    def __init__(self, path_in_repo: str, path_or_fileobj: Path) -> None:
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class FakeHfApi:
    def __init__(
        self,
        *,
        files: list[str] | None = None,
        info: FakeInfo | None = None,
        create_repo_calls: list[dict[str, object]] | None = None,
        commit_calls: list[dict[str, object]] | None = None,
        dataset_info_error: BaseException | None = None,
    ) -> None:
        self.files = files or []
        self.info = info
        self.dataset_info_error = dataset_info_error
        self.create_repo_calls = create_repo_calls or []
        self.commit_calls = commit_calls or []
        self.infos: list[dict[str, object]] = []

    def list_repo_files(self, repo_id: str, *, repo_type: str, revision: str | None = None) -> list[str]:
        return list(self.files)

    def dataset_info(self, repo_id: str, *, repo_type: str, revision: str | None = None) -> FakeInfo:
        if self.dataset_info_error is not None:
            raise self.dataset_info_error
        if self.info is None:
            raise _repository_not_found_error()()
        self.infos.append({"repo_id": repo_id, "repo_type": repo_type, "revision": revision})
        return self.info

    def create_repo(
        self, repo_id: str, *, repo_type: str, private: bool, token: str | None, exist_ok: bool
    ) -> None:
        self.create_repo_calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "token": token,
                "exist_ok": exist_ok,
            }
        )

    def create_commit(
        self,
        *,
        repo_id: str,
        operations: list[FakeOperation],
        commit_message: str,
        repo_type: str,
        token: str | None = None,
    ) -> None:
        self.commit_calls.append(
            {
                "repo_id": repo_id,
                "operations": [(op.path_in_repo, str(op.path_or_fileobj)) for op in operations],
                "commit_message": commit_message,
                "repo_type": repo_type,
                "token": token,
            }
        )


def _patch_download(payloads: dict[str, str]) -> patch:
    def fake_download(repo_id: str, filename: str, revision: str | None, token: str | None) -> str:
        if filename not in payloads:
            raise HuggingFaceError(f"Missing fixture download: {filename}")
        return payloads[filename]

    return patch.object(
        __import__("eval_transcript.huggingface", fromlist=["huggingface"]),
        "_download_module_level",
        side_effect=fake_download,
    )


class ConstructorTests(unittest.TestCase):
    def test_default_repo_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {"HF_TOKEN": "tok", "HF_DATASET_REPO": "team/corpus", "HF_ORG": "team"},
            clear=True,
        ):
            client = HuggingFaceClient()
        self.assertEqual(client.repo_id, "team/corpus")

    def test_repo_id_uses_org(self) -> None:
        with patch.dict("os.environ", {"HF_TOKEN": "tok"}, clear=True):
            client = HuggingFaceClient(dataset_repo="my-corpus", org="team")
        self.assertEqual(client.repo_id, "team/my-corpus")

    def test_missing_token_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(HuggingFaceError, "HF_TOKEN"):
                HuggingFaceClient()

    def test_token_explicit_is_accepted(self) -> None:
        client = HuggingFaceClient(token="tok", require_token=False)
        self.assertEqual(client.token, "tok")

    def test_empty_repo_name_raises(self) -> None:
        with patch.dict("os.environ", {"HF_TOKEN": "tok", "HF_DATASET_REPO": ""}, clear=True):
            with self.assertRaisesRegex(HuggingFaceError, "non-empty repository"):
                HuggingFaceClient()


class ListSamplesTests(unittest.TestCase):
    def test_lists_sample_ids(self) -> None:
        api = FakeHfApi(
            files=[
                "audio/sample-a.wav",
                "ground_truth/sample-a.md",
                "audio/sample-b.wav",
            ]
        )
        with patch.dict("os.environ", {"HF_TOKEN": "tok"}, clear=True):
            client = HuggingFaceClient(api=api)
        self.assertEqual(client.list_samples(), ["sample-a", "sample-b"])

    def test_ignores_files_outside_corpus_folders(self) -> None:
        api = FakeHfApi(
            files=[
                "README.md",
                ".gitignore",
                "audio/sample-a.wav",
                "ground_truth/sample-a.md",
                "audio/.gitkeep",
                "training_log.txt",
            ]
        )
        with patch.dict("os.environ", {"HF_TOKEN": "tok"}, clear=True):
            client = HuggingFaceClient(api=api)
        self.assertEqual(client.list_samples(), ["sample-a"])


class PullTests(unittest.TestCase):
    def test_pull_known_samples_writes_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            ground_truth_dir = root / "ground_truth"
            audio_dir.mkdir()
            ground_truth_dir.mkdir()
            downloads = {
                "audio/sample-a.wav": str(root / "src-audio"),
                "ground_truth/sample-a.md": str(root / "src-truth"),
            }
            Path(downloads["audio/sample-a.wav"]).write_bytes(b"RIFF")
            Path(downloads["ground_truth/sample-a.md"]).write_text("bonjour", encoding="utf-8")
            api = FakeHfApi(files=["audio/sample-a.wav", "ground_truth/sample-a.md"])
            with _patch_download(downloads), patch.dict("os.environ", {}, clear=True):
                client = HuggingFaceClient(
                    token="tok",
                    audio_dir=audio_dir,
                    ground_truth_dir=ground_truth_dir,
                    api=api,
                )
                result = client.pull(sample_ids=["sample-a"])

            self.assertIsInstance(result, PullResult)
            self.assertEqual(result.repo_id, DEFAULT_DATASET_REPO)
            self.assertEqual(len(result.samples), 1)
            sample = result.samples[0]
            self.assertEqual(sample.sample_id, "sample-a")
            self.assertEqual(sample.audio_path, audio_dir / "sample-a.wav")
            self.assertEqual(sample.ground_truth_path, ground_truth_dir / "sample-a.md")
            self.assertEqual((audio_dir / "sample-a.wav").read_bytes(), b"RIFF")
            self.assertEqual((ground_truth_dir / "sample-a.md").read_text(encoding="utf-8"), "bonjour")

    def test_pull_streams_with_shutil_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            ground_truth_dir = root / "ground_truth"
            audio_dir.mkdir()
            ground_truth_dir.mkdir()
            src = root / "src-audio"
            src.write_bytes(b"RIFF")
            api = FakeHfApi(files=["audio/sample-a.wav"])

            def fake_copy(source: str, destination: str) -> str:
                # Mirror shutil.copy so post-conditions hold.
                Path(destination).write_bytes(Path(source).read_bytes())
                return destination

            with _patch_download({"audio/sample-a.wav": str(src)}), patch.dict(
                "os.environ", {}, clear=True
            ), patch("eval_transcript.huggingface.shutil.copy", side_effect=fake_copy) as mock_copy:
                client = HuggingFaceClient(
                    token="tok",
                    audio_dir=audio_dir,
                    ground_truth_dir=ground_truth_dir,
                    api=api,
                )
                client.pull(sample_ids=["sample-a"])

            self.assertEqual(mock_copy.call_count, 1)
            (source, destination) = mock_copy.call_args.args
            self.assertEqual(source, str(src))
            self.assertEqual(Path(destination), audio_dir / "sample-a.wav")
            self.assertEqual((audio_dir / "sample-a.wav").read_bytes(), b"RIFF")

    def test_pull_skips_existing_files_without_force(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            existing = audio_dir / "sample-a.wav"
            existing.write_bytes(b"OLD")
            api = FakeHfApi(files=["audio/sample-a.wav"])
            with _patch_download({}), patch.dict("os.environ", {}, clear=True):
                client = HuggingFaceClient(
                    token="tok",
                    audio_dir=audio_dir,
                    ground_truth_dir=root / "ground_truth",
                    api=api,
                )
                result = client.pull(sample_ids=["sample-a"], force=False)

            self.assertEqual(result.samples[0].audio_path, existing)
            self.assertEqual(existing.read_bytes(), b"OLD")

    def test_pull_does_not_match_files_outside_corpus_folders(self) -> None:
        api = FakeHfApi(
            files=["audio/sample-a.wav", "ground_truth/sample-a.md", "README.md"]
        )
        with patch.dict("os.environ", {}, clear=True):
            client = HuggingFaceClient(token="tok", api=api)
        with self.assertRaisesRegex(HuggingFaceError, "not found"):
            client.pull(sample_ids=["README"])

    def test_pull_all_missing_samples_raises(self) -> None:
        api = FakeHfApi(files=["audio/sample-a.wav"])
        with patch.dict("os.environ", {}, clear=True):
            client = HuggingFaceClient(token="tok", api=api)
        with self.assertRaisesRegex(HuggingFaceError, "not found"):
            client.pull(sample_ids=["missing"])

    def test_pull_requires_samples_or_all(self) -> None:
        api = FakeHfApi(files=["audio/sample-a.wav"])
        with patch.dict("os.environ", {}, clear=True):
            client = HuggingFaceClient(token="tok", api=api)
        with self.assertRaisesRegex(HuggingFaceError, "--samples or --all"):
            client.pull()


class PushTests(unittest.TestCase):
    def test_push_refuses_public_repo(self) -> None:
        api = FakeHfApi(info=FakeInfo(private=False))
        with patch.dict("os.environ", {}, clear=True):
            client = HuggingFaceClient(token="tok", api=api)
        with self.assertRaisesRegex(HuggingFaceError, "Refusing to push to public dataset repo"):
            client.push()
        self.assertEqual(api.commit_calls, [])

    def test_push_creates_missing_private_repo_then_commits(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            ground_truth_dir = root / "ground_truth"
            audio_dir.mkdir()
            ground_truth_dir.mkdir()
            (audio_dir / "sample-a.wav").write_bytes(b"RIFF")
            (ground_truth_dir / "sample-a.md").write_text("bonjour", encoding="utf-8")
            api = FakeHfApi(
                dataset_info_error=_repository_not_found_error()("repo not found")
            )
            with patch.dict("os.environ", {}, clear=True):
                client = HuggingFaceClient(
                    token="tok",
                    audio_dir=audio_dir,
                    ground_truth_dir=ground_truth_dir,
                    api=api,
                )
                client.push(message="initial upload")

            self.assertEqual(len(api.create_repo_calls), 1)
            self.assertTrue(api.create_repo_calls[0]["private"])
            self.assertEqual(api.create_repo_calls[0]["repo_id"], client.repo_id)
            self.assertTrue(api.create_repo_calls[0]["exist_ok"])

            self.assertEqual(len(api.commit_calls), 1)
            commit = api.commit_calls[0]
            self.assertEqual(commit["commit_message"], "initial upload")
            self.assertEqual(commit["repo_type"], "dataset")
            self.assertEqual(commit["token"], "tok")
            self.assertEqual(
                sorted(op[0] for op in commit["operations"]),
                ["audio/sample-a.wav", "ground_truth/sample-a.md"],
            )

    def test_push_uses_single_atomic_commit_for_existing_private_repo(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            ground_truth_dir = root / "ground_truth"
            audio_dir.mkdir()
            ground_truth_dir.mkdir()
            (audio_dir / "sample-a.wav").write_bytes(b"RIFF")
            (ground_truth_dir / "sample-a.md").write_text("bonjour", encoding="utf-8")
            api = FakeHfApi(info=FakeInfo(private=True))
            with patch.dict("os.environ", {}, clear=True):
                client = HuggingFaceClient(
                    token="tok",
                    audio_dir=audio_dir,
                    ground_truth_dir=ground_truth_dir,
                    api=api,
                )
                client.push()

        self.assertEqual(api.create_repo_calls, [])
        self.assertEqual(len(api.commit_calls), 1)
        self.assertEqual(
            [op[0] for op in api.commit_calls[0]["operations"]],
            ["audio/sample-a.wav", "ground_truth/sample-a.md"],
        )

    def test_push_propagates_repository_not_found_as_create(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            ground_truth_dir = root / "ground_truth"
            audio_dir.mkdir()
            ground_truth_dir.mkdir()
            (audio_dir / "sample-a.wav").write_bytes(b"RIFF")
            api = FakeHfApi(
                dataset_info_error=_repository_not_found_error()("not found")
            )
            with patch.dict("os.environ", {}, clear=True):
                client = HuggingFaceClient(
                    token="tok",
                    audio_dir=audio_dir,
                    ground_truth_dir=ground_truth_dir,
                    api=api,
                )
                client.push()

        self.assertEqual(len(api.create_repo_calls), 1)
        self.assertEqual(len(api.commit_calls), 1)

    def test_push_does_not_swallow_other_dataset_info_errors(self) -> None:
        api = FakeHfApi(dataset_info_error=RuntimeError("auth failure"))
        with patch.dict("os.environ", {}, clear=True):
            client = HuggingFaceClient(token="tok", api=api)
        with self.assertRaisesRegex(RuntimeError, "auth failure"):
            client.push()


class DownloadHelperTests(unittest.TestCase):
    def test_module_level_hf_hub_download_is_used(self) -> None:
        with patch("huggingface_hub.hf_hub_download") as module_level:
            module_level.return_value = "/tmp/whatever"
            result = _download_module_level("team/corpus", "audio/a.wav", None, "tok")
        module_level.assert_called_once_with(
            repo_id="team/corpus",
            filename="audio/a.wav",
            repo_type="dataset",
            revision=None,
            token="tok",
        )
        self.assertEqual(result, "/tmp/whatever")


if __name__ == "__main__":
    unittest.main()
