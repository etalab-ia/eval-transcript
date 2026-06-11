from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataMigrationResult:
    source_dir: Path
    target_dir: Path
    moved: bool
    message: str


class DataMigrationError(RuntimeError):
    """Raised when a local data layout migration cannot be completed safely."""


def migrate_source_truth_to_ground_truth(
    *,
    source_dir: Path = Path("data/source_truth"),
    target_dir: Path = Path("data/ground_truth"),
    force: bool = False,
) -> DataMigrationResult:
    """Move the legacy source_truth directory to ground_truth.

    The migration is intentionally conservative: if the legacy source directory is
    absent, the command is a no-op. If both source and target exist, callers must
    pass force=True before the directories are merged.
    """

    if not source_dir.exists():
        return DataMigrationResult(
            source_dir=source_dir,
            target_dir=target_dir,
            moved=False,
            message=f"No legacy ground-truth directory to migrate: {source_dir}",
        )

    if not source_dir.is_dir():
        raise DataMigrationError(f"Legacy source path is not a directory: {source_dir}")

    if target_dir.exists():
        if not target_dir.is_dir():
            raise DataMigrationError(f"Ground-truth target path is not a directory: {target_dir}")
        if not force:
            raise DataMigrationError(
                f"Ground-truth target already exists: {target_dir}. Use --force to merge {source_dir} into it."
            )
        moved_count = merge_directory(source_dir=source_dir, target_dir=target_dir)
        remove_empty_directory(source_dir)
        return DataMigrationResult(
            source_dir=source_dir,
            target_dir=target_dir,
            moved=True,
            message=f"Merged {moved_count} path(s) from {source_dir} into {target_dir}",
        )

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))
    return DataMigrationResult(
        source_dir=source_dir,
        target_dir=target_dir,
        moved=True,
        message=f"Moved {source_dir} to {target_dir}",
    )


def merge_directory(*, source_dir: Path, target_dir: Path) -> int:
    moved_count = 0
    for child in source_dir.iterdir():
        target = target_dir / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
        moved_count += 1
    return moved_count


def remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except OSError as exc:
        raise DataMigrationError(f"Could not remove migrated source directory: {path}") from exc
