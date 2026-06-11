from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eval_transcript.data_migrate import DataMigrationError, migrate_source_truth_to_ground_truth


class DataMigrationTests(unittest.TestCase):
    def test_migrates_source_truth_directory_to_ground_truth(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "data" / "source_truth"
            target_dir = root / "data" / "ground_truth"
            source_dir.mkdir(parents=True)
            (source_dir / ".gitkeep").write_text("", encoding="utf-8")
            (source_dir / "sample.md").write_text("bonjour", encoding="utf-8")

            result = migrate_source_truth_to_ground_truth(source_dir=source_dir, target_dir=target_dir)

            self.assertTrue(result.moved)
            self.assertFalse(source_dir.exists())
            self.assertEqual((target_dir / "sample.md").read_text(encoding="utf-8"), "bonjour")
            self.assertTrue((target_dir / ".gitkeep").exists())

    def test_missing_source_truth_directory_is_noop(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = migrate_source_truth_to_ground_truth(
                source_dir=root / "data" / "source_truth",
                target_dir=root / "data" / "ground_truth",
            )

            self.assertFalse(result.moved)
            self.assertIn("No legacy", result.message)

    def test_existing_ground_truth_requires_force(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "data" / "source_truth"
            target_dir = root / "data" / "ground_truth"
            source_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            (source_dir / "sample.md").write_text("bonjour", encoding="utf-8")

            with self.assertRaisesRegex(DataMigrationError, "already exists"):
                migrate_source_truth_to_ground_truth(source_dir=source_dir, target_dir=target_dir)

    def test_force_merges_into_existing_ground_truth(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "data" / "source_truth"
            target_dir = root / "data" / "ground_truth"
            source_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            (source_dir / "sample.md").write_text("bonjour", encoding="utf-8")
            (target_dir / "existing.md").write_text("salut", encoding="utf-8")

            result = migrate_source_truth_to_ground_truth(source_dir=source_dir, target_dir=target_dir, force=True)

            self.assertTrue(result.moved)
            self.assertFalse(source_dir.exists())
            self.assertEqual((target_dir / "sample.md").read_text(encoding="utf-8"), "bonjour")
            self.assertEqual((target_dir / "existing.md").read_text(encoding="utf-8"), "salut")


if __name__ == "__main__":
    unittest.main()
