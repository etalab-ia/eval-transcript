from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


DEFAULT_API_KEY_ENV = "HF_TOKEN"
DEFAULT_ORG_ENV = "HF_ORG"
DEFAULT_DATASET_REPO = "eval-transcript-corpus"
DEFAULT_AUDIO_DIR = Path("data/audio")
DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
AUDIO_FILE_EXTENSIONS = {".aac", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
GROUND_TRUTH_FILE_EXTENSIONS = {".md", ".txt"}
DATASET_REPO_TYPE = "dataset"
AUDIO_REPO_SUBDIR = "audio"
GROUND_TRUTH_REPO_SUBDIR = "ground_truth"
CORPUS_REPO_SUBDIRS = frozenset({AUDIO_REPO_SUBDIR, GROUND_TRUTH_REPO_SUBDIR})


class HuggingFaceError(RuntimeError):
    """Raised when a Hugging Face Hub dataset operation cannot complete safely."""


class HfApiLike(Protocol):
    """Subset of the huggingface_hub.HfApi surface used by this client.

    The protocol keeps the unit tests free of a hard dependency on the real
    huggingface_hub package; the production wiring passes through HfApi
    directly. ``hf_hub_download`` is documented as a module-level function in
    huggingface_hub, so it does not belong on this Protocol.
    """

    def dataset_info(self, repo_id: str, *, repo_type: str = ..., revision: str | None = None) -> Any: ...

    def list_repo_files(self, repo_id: str, *, repo_type: str = ..., revision: str | None = None) -> list[str]: ...

    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str = ...,
        private: bool = ...,
        token: str | None = None,
        exist_ok: bool = ...,
    ) -> Any: ...

    def create_commit(
        self,
        *,
        repo_id: str,
        operations: Sequence[Any],
        commit_message: str,
        repo_type: str = ...,
        token: str | None = None,
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

    The corpus is a single HF Dataset with files laid out under ``audio/<id>.<ext>``
    and ``ground_truth/<id>.<ext>``. This client is read/write only: ASR
    inference stays with the oMLX, Albert, Scaleway, and ElevenLabs providers.
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
                    sample_id=sample_id,
                    files=files,
                    force=force,
                    revision=revision,
                )
            )
        return PullResult(repo_id=self.repo_id, revision=revision, samples=pulled)

    def _pull_one(
        self,
        *,
        sample_id: str,
        files: Sequence[str],
        force: bool,
        revision: str | None,
    ) -> PulledSample:
        audio_path = self._write_one(
            sample_id=sample_id,
            files=files,
            target_dir=self.audio_dir,
            extensions=AUDIO_FILE_EXTENSIONS,
            repo_subdir=AUDIO_REPO_SUBDIR,
            force=force,
            revision=revision,
        )
        ground_truth_path = self._write_one(
            sample_id=sample_id,
            files=files,
            target_dir=self.ground_truth_dir,
            extensions=GROUND_TRUTH_FILE_EXTENSIONS,
            repo_subdir=GROUND_TRUTH_REPO_SUBDIR,
            force=force,
            revision=revision,
        )
        return PulledSample(
            sample_id=sample_id,
            audio_path=audio_path,
            ground_truth_path=ground_truth_path,
        )

    def _write_one(
        self,
        *,
        sample_id: str,
        files: Sequence[str],
        target_dir: Path,
        extensions: set[str],
        repo_subdir: str,
        force: bool,
        revision: str | None,
    ) -> Path | None:
        candidates = _candidate_paths_for_field(files, sample_id, repo_subdir, extensions)
        if not candidates:
            return None
        path_in_repo = candidates[0]
        local_path = target_dir / Path(path_in_repo).name
        if local_path.exists() and not force:
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = _download_module_level(self.repo_id, path_in_repo, revision, self.token)
        # shutil.copy streams the file to keep memory low for large audio assets.
        shutil.copy(downloaded, local_path)
        return local_path

    def push(
        self,
        *,
        message: str = "Update eval-transcript corpus",
    ) -> None:
        """Upload the local audio + ground_truth directories to the corpus repo.

        Refuses to push to a public dataset repo. The target repo is created
        with ``private=True`` if it does not exist. All files are uploaded in a
        single atomic commit.
        """

        api = self._get_api()
        self._ensure_private_repo(api)
        operations = self._collect_commit_operations()
        if operations:
            api.create_commit(
                repo_id=self.repo_id,
                operations=operations,
                commit_message=message,
                repo_type=DATASET_REPO_TYPE,
                token=self.token,
            )

    def _ensure_private_repo(self, api: HfApiLike) -> None:
        try:
            info = api.dataset_info(self.repo_id, repo_type=DATASET_REPO_TYPE)
        except _repository_not_found_error():
            api.create_repo(
                self.repo_id,
                repo_type=DATASET_REPO_TYPE,
                private=True,
                token=self.token,
                exist_ok=True,
            )
            return
        is_private = bool(getattr(info, "private", True))
        if not is_private:
            raise HuggingFaceError(
                f"Refusing to push to public dataset repo: {self.repo_id}. "
                "Benchmark corpora must stay private."
            )

    def _collect_commit_operations(self) -> list[Any]:
        try:
            from huggingface_hub import CommitOperationAdd
        except ImportError as exc:  # pragma: no cover - exercised only in production
            raise HuggingFaceError(
                "huggingface_hub is required for live HF Hub operations"
            ) from exc

        operations: list[Any] = []
        for directory, repo_subdir, extensions in (
            (self.audio_dir, AUDIO_REPO_SUBDIR, AUDIO_FILE_EXTENSIONS),
            (self.ground_truth_dir, GROUND_TRUTH_REPO_SUBDIR, GROUND_TRUTH_FILE_EXTENSIONS),
        ):
            for path in sorted(directory.glob("*")):
                if not path.is_file() or path.suffix.lower() not in extensions:
                    continue
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"{repo_subdir}/{path.name}",
                        path_or_fileobj=path,
                    )
                )
        return operations


def _sample_ids_from_repo_files(files: Iterable[str]) -> list[str]:
    ids: set[str] = set()
    for path in files:
        stem, suffix = _split_stem_suffix(path)
        repo_subdir = _repo_subdir(path)
        if not stem or repo_subdir is None:
            continue
        extensions = (
            AUDIO_FILE_EXTENSIONS if repo_subdir == AUDIO_REPO_SUBDIR else GROUND_TRUTH_FILE_EXTENSIONS
        )
        if suffix in extensions:
            ids.add(stem)
    return sorted(ids)


def _candidate_paths_for_field(
    files: Iterable[str], sample_id: str, repo_subdir: str, extensions: set[str]
) -> list[str]:
    matches: list[str] = []
    for path in files:
        stem, suffix = _split_stem_suffix(path)
        if stem != sample_id or suffix not in extensions or _repo_subdir(path) != repo_subdir:
            continue
        matches.append(path)
    return matches


def _split_stem_suffix(path: str) -> tuple[str, str]:
    name = Path(path).name
    p = Path(name)
    return p.stem, p.suffix


def _repo_subdir(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) < 2:
        return None
    subdir = parts[0]
    return subdir if subdir in CORPUS_REPO_SUBDIRS else None


def _download_module_level(
    repo_id: str, filename: str, revision: str | None, token: str | None
) -> str:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only in production
        raise HuggingFaceError(
            "huggingface_hub is required for live HF Hub operations"
        ) from exc
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=DATASET_REPO_TYPE,
        revision=revision,
        token=token,
    )


def _repository_not_found_error() -> type[BaseException]:
    """Return huggingface_hub.errors.RepositoryNotFoundError, or Exception as a safe fallback.

    Falling back keeps the unit tests independent of the real package while
    still scoping the production catch to a specific HF error class.
    """
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:  # pragma: no cover - exercised only in production
        return Exception
    return RepositoryNotFoundError
