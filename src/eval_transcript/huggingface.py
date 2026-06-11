from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


DEFAULT_API_KEY_ENV = "HF_TOKEN"
DEFAULT_ORG_ENV = "HF_ORG"
DEFAULT_DATASET_REPO = "eval-transcript-corpus"
DEFAULT_AUDIO_DIR = Path("data/audio")
DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_HASH_ALGORITHM = "sha256"
AUDIO_FILE_EXTENSIONS = {".aac", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
GROUND_TRUTH_FILE_EXTENSIONS = {".md", ".txt"}
DATASET_REPO_TYPE = "dataset"


class HuggingFaceError(RuntimeError):
    """Raised when a Hugging Face Hub dataset operation cannot complete safely."""


class HfApiLike(Protocol):
    """Subset of the huggingface_hub.HfApi surface used by this client.

    The protocol keeps the unit tests free of a hard dependency on the real
    huggingface_hub package; the production wiring passes through HfApi
    directly.
    """

    def dataset_info(self, repo_id: str, *, repo_type: str = ..., revision: str | None = None) -> Any: ...

    def list_repo_files(self, repo_id: str, *, repo_type: str = ..., revision: str | None = None) -> list[str]: ...

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str = ...,
        revision: str | None = None,
        token: str | None = None,
    ) -> str: ...

    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str = ...,
        private: bool = ...,
        token: str | None = None,
        exist_ok: bool = ...,
    ) -> Any: ...

    def upload_file(
        self,
        *,
        path_or_fileobj: str | Path | bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = ...,
        token: str | None = None,
        commit_message: str | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class PulledSample:
    sample_id: str
    audio_path: Path | None
    ground_truth_path: Path | None


@dataclass(frozen=True)
class PullResult:
    repo_id: str
    revision: str | None
    samples: list[PulledSample]


class HuggingFaceClient:
    """Client for the eval-transcript benchmark corpus on Hugging Face Hub.

    The corpus is a single HF Dataset with one parquet file per sample. Each row
    carries the audio bytes, the ground-truth text, and a handful of metadata
    fields. This client is read/write only: ASR inference stays with the
    oMLX, Albert, Scaleway, and ElevenLabs providers.
    """

    def __init__(
        self,
        *,
        dataset_repo: str | None = None,
        org: str | None = None,
        token: str | None = None,
        audio_dir: Path = DEFAULT_AUDIO_DIR,
        ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
        require_token: bool = True,
        api: HfApiLike | None = None,
    ) -> None:
        self.dataset_repo = (
            dataset_repo
            if dataset_repo is not None
            else os.getenv("HF_DATASET_REPO", DEFAULT_DATASET_REPO)
        )
        if not self.dataset_repo:
            raise HuggingFaceError("dataset_repo must be a non-empty repository name")
        self.org = org if org is not None else os.getenv(DEFAULT_ORG_ENV)
        self.token = token if token is not None else os.getenv(DEFAULT_API_KEY_ENV)
        if require_token and not self.token:
            raise HuggingFaceError(
                "HF_TOKEN is required; set the HF_TOKEN environment variable or pass token=..."
            )
        self.audio_dir = audio_dir
        self.ground_truth_dir = ground_truth_dir
        self._api: HfApiLike | None = api

    @property
    def repo_id(self) -> str:
        # When HF_DATASET_REPO already contains a namespace (org/dataset), the
        # user has pinned both, so do not double-prepend the org. Only prepend
        # the org when the configured dataset_repo is a bare name.
        if "/" in self.dataset_repo or not self.org:
            return self.dataset_repo
        return f"{self.org}/{self.dataset_repo}"

    def _get_api(self) -> HfApiLike:
        if self._api is not None:
            return self._api
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only in production
            raise HuggingFaceError(
                "huggingface_hub is required for live HF Hub operations"
            ) from exc
        return HfApi(token=self.token)

    def list_samples(self, *, revision: str | None = None) -> list[str]:
        api = self._get_api()
        files = api.list_repo_files(self.repo_id, repo_type=DATASET_REPO_TYPE, revision=revision)
        return _sample_ids_from_repo_files(files)

    def pull(
        self,
        *,
        sample_ids: Iterable[str] | None = None,
        all_samples: bool = False,
        force: bool = False,
        revision: str | None = None,
    ) -> PullResult:
        if not all_samples and not sample_ids:
            raise HuggingFaceError("pull requires --samples or --all")

        api = self._get_api()
        files = list(api.list_repo_files(self.repo_id, repo_type=DATASET_REPO_TYPE, revision=revision))
        available = _sample_ids_from_repo_files(files)

        if all_samples:
            wanted = sorted(available)
        else:
            wanted = sorted(set(sample_ids or ()))
            unknown = sorted(set(wanted) - set(available))
            if unknown:
                raise HuggingFaceError(
                    f"Samples not found in {self.repo_id}: {', '.join(unknown)}"
                )

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        pulled: list[PulledSample] = []
        for sample_id in wanted:
            pulled.append(
                self._pull_one(
                    api,
                    sample_id=sample_id,
                    files=files,
                    force=force,
                    revision=revision,
                )
            )
        return PullResult(repo_id=self.repo_id, revision=revision, samples=pulled)

    def _pull_one(
        self,
        api: HfApiLike,
        *,
        sample_id: str,
        files: Sequence[str],
        force: bool,
        revision: str | None,
    ) -> PulledSample:
        audio_path = self._write_parquet_field(
            api,
            sample_id=sample_id,
            files=files,
            kind="audio",
            target_dir=self.audio_dir,
            extensions=AUDIO_FILE_EXTENSIONS,
            force=force,
            revision=revision,
        )
        ground_truth_path = self._write_parquet_field(
            api,
            sample_id=sample_id,
            files=files,
            kind="ground_truth",
            target_dir=self.ground_truth_dir,
            extensions=GROUND_TRUTH_FILE_EXTENSIONS,
            force=force,
            revision=revision,
        )
        return PulledSample(
            sample_id=sample_id,
            audio_path=audio_path,
            ground_truth_path=ground_truth_path,
        )

    def _write_parquet_field(
        self,
        api: HfApiLike,
        *,
        sample_id: str,
        files: Sequence[str],
        kind: str,
        target_dir: Path,
        extensions: set[str],
        force: bool,
        revision: str | None,
    ) -> Path | None:
        candidates = _candidate_paths_for_field(files, sample_id, extensions)
        if not candidates:
            return None
        # Pick the first matching file; multiple variants for the same sample
        # are an upstream data error and surface as an error during push.
        path_in_repo = candidates[0]
        local_path = target_dir / Path(path_in_repo).name
        if local_path.exists() and not force:
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = api.hf_hub_download(
            self.repo_id,
            path_in_repo,
            repo_type=DATASET_REPO_TYPE,
            revision=revision,
            token=self.token,
        )
        local_path.write_bytes(Path(downloaded).read_bytes())
        return local_path

    def push(
        self,
        *,
        message: str = "Update eval-transcript corpus",
    ) -> None:
        """Upload the local audio + ground_truth directories to the corpus repo.

        Refuses to push to a public dataset repo. The target repo is created
        with ``private=True`` if it does not exist.
        """

        api = self._get_api()
        self._ensure_private_repo(api)
        self._upload_local_files(api, message=message)

    def _ensure_private_repo(self, api: HfApiLike) -> None:
        try:
            info = api.dataset_info(self.repo_id, repo_type=DATASET_REPO_TYPE)
        except Exception:
            api.create_repo(self.repo_id, repo_type=DATASET_REPO_TYPE, private=True, token=self.token, exist_ok=True)
            return
        is_private = bool(getattr(info, "private", True))
        if not is_private:
            raise HuggingFaceError(
                f"Refusing to push to public dataset repo: {self.repo_id}. "
                "Benchmark corpora must stay private."
            )

    def _upload_local_files(self, api: HfApiLike, *, message: str) -> None:
        for path in sorted(self.audio_dir.glob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_FILE_EXTENSIONS:
                continue
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"audio/{path.name}",
                repo_id=self.repo_id,
                repo_type=DATASET_REPO_TYPE,
                token=self.token,
                commit_message=message,
            )
        for path in sorted(self.ground_truth_dir.glob("*")):
            if not path.is_file() or path.suffix.lower() not in GROUND_TRUTH_FILE_EXTENSIONS:
                continue
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"ground_truth/{path.name}",
                repo_id=self.repo_id,
                repo_type=DATASET_REPO_TYPE,
                token=self.token,
                commit_message=message,
            )


def _sample_ids_from_repo_files(files: Iterable[str]) -> list[str]:
    ids: set[str] = set()
    for path in files:
        stem, suffix = _split_stem_suffix(path)
        if stem and suffix in AUDIO_FILE_EXTENSIONS or suffix in GROUND_TRUTH_FILE_EXTENSIONS:
            ids.add(stem)
    return sorted(ids)


def _candidate_paths_for_field(files: Iterable[str], sample_id: str, extensions: set[str]) -> list[str]:
    matches: list[str] = []
    for path in files:
        stem, suffix = _split_stem_suffix(path)
        if stem != sample_id or suffix not in extensions:
            continue
        matches.append(path)
    return matches


def _split_stem_suffix(path: str) -> tuple[str, str]:
    name = path.rsplit("/", 1)[-1]
    dot_index = name.rfind(".")
    if dot_index <= 0:
        return name, ""
    return name[:dot_index], name[dot_index:]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
