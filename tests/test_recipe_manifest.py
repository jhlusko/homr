import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from training.transformer.recipe_manifest import write_recipe_manifest


class TestRecipeManifest(unittest.TestCase):
    def test_writes_resolved_index_paths_and_digests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "some_index.txt"
            index_path.write_text("a.tokens\n")
            out_dir = Path(tmp) / "checkpoint_folder"

            written = write_recipe_manifest(
                checkpoint_dir=str(out_dir),
                git_root=tmp,
                indexes={"ossq": str(index_path)},
            )

            manifest = json.loads(Path(written).read_text())
            self.assertEqual(manifest["indexes"]["ossq"]["path"], str(index_path))
            self.assertIsNotNone(manifest["indexes"]["ossq"]["sha256"])

    def test_missing_index_file_records_no_digest_rather_than_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "checkpoint_folder"

            written = write_recipe_manifest(
                checkpoint_dir=str(out_dir),
                git_root=tmp,
                indexes={"missing": str(Path(tmp) / "does_not_exist.txt")},
            )

            manifest = json.loads(Path(written).read_text())
            self.assertIsNone(manifest["indexes"]["missing"]["sha256"])

    def test_records_the_commit_of_a_real_git_checkout(self) -> None:
        # Exercised against this repo checkout rather than a fabricated tempdir repo,
        # since building a throwaway git repo just to assert on its HEAD adds
        # indirection without checking anything a real checkout wouldn't.
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "checkpoint_folder"
            git_root = Path(__file__).resolve().parents[1]
            expected = subprocess.run(
                ["git", "-C", str(git_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

            written = write_recipe_manifest(
                checkpoint_dir=str(out_dir), git_root=str(git_root), indexes={}
            )

            manifest = json.loads(Path(written).read_text())
            self.assertEqual(manifest["commit"], expected)


if __name__ == "__main__":
    unittest.main()
