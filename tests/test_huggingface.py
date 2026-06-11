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
    file_sha256,
)


class FakeInfo:
    def __init__(self, private: bool) -> None:
        self.private = private


class FakeHfApi:
    def __init__(
        self,
        *,
        files: list[str] | None = None,
        downloads: dict[str, str] | None = None,
        info: FakeInfo | None = None,
        create_repo_calls: list[dict[str, object]] | None = None,
        upload_calls: list[dict[str, object]] | None = None,
    ) -> None:
        self.files = files or []
        self.downloads = downloads or {}
        self.info = info
        self.create_repo_calls = create_repo_calls or []
        self.upload_calls = upload_calls or []
        self.infos: list[dict[str, object]] = []

    def list_repo_files(self, repo_id: str, *, repo_type: str, revision: str | None = None) -> list[str]:
        return list(self.files)

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str,
        revision: str | None = None,
        token: str | None = None,
    ) -> str:
        if filename not in self.downloads:
            raise HuggingFaceError(f"Missing fixture download: {filename}")
        return self.downloads[filename]

    def dataset_info(self, repo_id: str, *, repo_type: str, revision: str | None = None) -> FakeInfo:
        if self.info is None:
            raise HuggingFaceError("repo not found")
        self.infos.append({"repo_id": repo_id, "repo_type": repo_type, "revision": revision})
        return self.info

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool, token: str | None, exist_ok: bool) -> None:
        self.create_repo_calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "token": token,
                "exist_ok": exist_ok,
            }
        )

    def upload_file(
        self,
        *,
        path_or_fileobj: Path,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        token: str | None = None,
        commit_message: str | None = None,
    ) -> None:
        self.upload_calls.append(
            {
                "path": str(path_or_fileobj),
                "path_in_repo": path_in_repo,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "token": token,
                "commit_message": commit_message,
            }
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
        api = FakeHfApi(files=["audio/sample-a.wav", "ground_truth/sample-a.md", "audio/sample-b.wav"])
        with patch.dict("os.environ", {"HF_TOKEN": "tok"}, clear=True):
            client = HuggingFaceClient(api=api)
        self.assertEqual(client.list_samples(), ["sample-a", "sample-b"])


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
            api = FakeHfApi(
                files=["audio/sample-a.wav", "ground_truth/sample-a.md"],
                downloads=downloads,
            )
            with patch.dict("os.environ", {}, clear=True):
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

    def test_pull_skips_existing_files_without_force(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            existing = audio_dir / "sample-a.wav"
            existing.write_bytes(b"OLD")
            api = FakeHfApi(files=["audio/sample-a.wav"], downloads={})
            with patch.dict("os.environ", {}, clear=True):
                client = HuggingFaceClient(
                    token="tok",
                    audio_dir=audio_dir,
                    ground_truth_dir=root / "ground_truth",
                    api=api,
                )
            result = client.pull(sample_ids=["sample-a"], force=False)

            self.assertEqual(result.samples[0].audio_path, existing)
            self.assertEqual(existing.read_bytes(), b"OLD")

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

    def test_push_uploads_local_files_to_existing_private_repo(self) -> None:
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
            client.push(message="initial upload")

        upload_paths = sorted(call["path_in_repo"] for call in api.upload_calls)
        self.assertEqual(
            upload_paths,
            ["audio/sample-a.wav", "ground_truth/sample-a.md"],
        )
        self.assertEqual(api.create_repo_calls, [])
        for call in api.upload_calls:
            self.assertEqual(call["token"], "tok")
            self.assertEqual(call["repo_type"], "dataset")

    def test_push_creates_missing_private_repo(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            ground_truth_dir = root / "ground_truth"
            audio_dir.mkdir()
            ground_truth_dir.mkdir()
            api = FakeHfApi()
            with patch.dict("os.environ", {}, clear=True):
                client = HuggingFaceClient(
                    token="tok",
                    audio_dir=audio_dir,
                    ground_truth_dir=ground_truth_dir,
                    api=api,
                )
            client.push()

        self.assertEqual(len(api.create_repo_calls), 1)
        self.assertTrue(api.create_repo_calls[0]["private"])
        self.assertEqual(api.create_repo_calls[0]["repo_id"], client.repo_id)
        self.assertTrue(api.create_repo_calls[0]["exist_ok"])


class HashHelperTests(unittest.TestCase):
    def test_file_sha256_matches_stdlib(self) -> None:
        import hashlib

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            path.write_bytes(b"eval-transcript")
            self.assertEqual(file_sha256(path), hashlib.sha256(b"eval-transcript").hexdigest())


if __name__ == "__main__":
    unittest.main()
