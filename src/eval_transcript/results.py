"""Push des transcripts locaux vers le dataset de resultats Hugging Face.

Sert a verser sur le dataset de resultats les sorties des modeles tournant en
local (WhisperX, Kyutai, Cohere via MLX), qui ne passent pas par la CI remote.

Garde-fou de souverainete / RGPD : seuls les transcripts des samples presents
dans le corpus PUBLIC (cote ``ground_truth/``) sont pousses. Un sample absent du
corpus public (reunion interne, sample retire) est ignore -> aucune donnee privee
ne part sur le Hub. L'allowlist est derivee du corpus live, pas codee en dur.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


DEFAULT_CORPUS_REPO = "AgentPublic/eval-stt-officiels"
DEFAULT_RESULTS_REPO = "AgentPublic/eval-stt-results"
DEFAULT_TRANSCRIPTIONS_DIR = Path("data/transcriptions")

CORPUS_REPO_ENV = "EVAL_CORPUS_REPO"
RESULTS_REPO_ENV = "EVAL_RESULTS_REPO"
TOKEN_ENV = "HF_TOKEN"

DATASET_REPO_TYPE = "dataset"
GROUND_TRUTH_REPO_SUBDIR = "ground_truth"
TRANSCRIPTIONS_REPO_SUBDIR = "transcriptions"
TRANSCRIPT_SUFFIX = ".txt"


class ResultsError(RuntimeError):
    """Raised when local results cannot be pushed safely."""


def _hf_http_error_types() -> tuple[type[BaseException], ...]:
    """huggingface_hub HTTP error class, or an empty tuple (catches nothing) if absent.

    Lets the live calls surface a clean ``ResultsError`` (e.g. invalid token, repo
    not found) instead of a raw traceback, while keeping the unit tests free of a
    hard huggingface_hub dependency.
    """
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:  # pragma: no cover - exercised only in production
        return ()
    return (HfHubHTTPError,)


class HfApiLike(Protocol):
    """Subset of huggingface_hub.HfApi used here (keeps unit tests dependency-free)."""

    def list_repo_files(
        self, repo_id: str, *, repo_type: str = ..., revision: str | None = None
    ) -> list[str]: ...

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
class PlannedUpload:
    sample_id: str
    path_in_repo: str
    local_path: Path


@dataclass(frozen=True)
class PushPlan:
    results_repo: str
    officiels: list[str]
    uploads: list[PlannedUpload]
    skipped_samples: list[str]


def public_sample_ids(api: HfApiLike, corpus_repo: str, *, revision: str | None = None) -> set[str]:
    """Sample IDs declared public by the corpus dataset (one per ``ground_truth/<id>`` file)."""
    try:
        files = api.list_repo_files(corpus_repo, repo_type=DATASET_REPO_TYPE, revision=revision)
    except _hf_http_error_types() as exc:
        raise ResultsError(f"Failed to read corpus dataset {corpus_repo}: {exc}") from exc
    prefix = f"{GROUND_TRUTH_REPO_SUBDIR}/"
    return {
        Path(f[len(prefix):]).stem
        for f in files
        if f.startswith(prefix) and f.endswith(TRANSCRIPT_SUFFIX)
    }


def build_push_plan(
    *,
    api: HfApiLike,
    corpus_repo: str,
    results_repo: str,
    transcriptions_dir: Path,
    include: str | None = None,
) -> PushPlan:
    """Plan the upload, keeping only transcripts of samples in the public corpus."""
    officiels = public_sample_ids(api, corpus_repo)
    if not officiels:
        raise ResultsError(f"No public samples found in corpus repo {corpus_repo}")
    if not transcriptions_dir.exists():
        raise ResultsError(f"Transcriptions directory not found: {transcriptions_dir}")

    uploads: list[PlannedUpload] = []
    skipped: set[str] = set()
    for sample_dir in sorted(p for p in transcriptions_dir.iterdir() if p.is_dir()):
        sid = sample_dir.name
        if sid not in officiels:
            skipped.add(sid)
            continue
        for txt in sorted(sample_dir.glob(f"*{TRANSCRIPT_SUFFIX}")):
            if include and include not in txt.name:
                continue
            uploads.append(
                PlannedUpload(
                    sample_id=sid,
                    path_in_repo=f"{TRANSCRIPTIONS_REPO_SUBDIR}/{sid}/{txt.name}",
                    local_path=txt,
                )
            )
    return PushPlan(
        results_repo=results_repo,
        officiels=sorted(officiels),
        uploads=uploads,
        skipped_samples=sorted(skipped),
    )


class ResultsClient:
    """Reads the public corpus and pushes local transcripts to the results dataset."""

    def __init__(
        self,
        *,
        corpus_repo: str | None = None,
        results_repo: str | None = None,
        token: str | None = None,
        api: HfApiLike | None = None,
    ) -> None:
        self.corpus_repo = corpus_repo or os.getenv(CORPUS_REPO_ENV, DEFAULT_CORPUS_REPO)
        self.results_repo = results_repo or os.getenv(RESULTS_REPO_ENV, DEFAULT_RESULTS_REPO)
        self.token = token if token is not None else os.getenv(TOKEN_ENV)
        self._api = api

    def _get_api(self) -> HfApiLike:
        if self._api is not None:
            return self._api
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only in production
            raise ResultsError("huggingface_hub is required for live HF Hub operations") from exc
        return HfApi(token=self.token)

    def plan(
        self, *, transcriptions_dir: Path = DEFAULT_TRANSCRIPTIONS_DIR, include: str | None = None
    ) -> PushPlan:
        return build_push_plan(
            api=self._get_api(),
            corpus_repo=self.corpus_repo,
            results_repo=self.results_repo,
            transcriptions_dir=transcriptions_dir,
            include=include,
        )

    def push(self, plan: PushPlan, *, message: str = "Push local transcripts (officiels only)") -> None:
        if not plan.uploads:
            return
        if not self.token:
            raise ResultsError(
                "HF_TOKEN is required to push results; set the HF_TOKEN environment variable or pass token=..."
            )
        try:
            from huggingface_hub import CommitOperationAdd  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only in production
            raise ResultsError("huggingface_hub is required for live HF Hub operations") from exc
        operations = [
            CommitOperationAdd(path_in_repo=u.path_in_repo, path_or_fileobj=str(u.local_path))
            for u in plan.uploads
        ]
        try:
            self._get_api().create_commit(
                repo_id=plan.results_repo,
                repo_type=DATASET_REPO_TYPE,
                operations=operations,
                commit_message=message,
                token=self.token,
            )
        except _hf_http_error_types() as exc:
            raise ResultsError(f"Failed to push to results dataset {plan.results_repo}: {exc}") from exc
