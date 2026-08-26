import json
import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.package_dataset import (
    materialise,
    relative_to_root,
    rewrite_ground_truth,
    rewrite_index,
    verify,
    write_manifest,
)


class Fixture(unittest.TestCase):
    """A dataset root on disk. Every test here needs real symlinks, so no mocking."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name) / "dataset"
        (self.root / "images").mkdir(parents=True)
        self.addCleanup(self.directory.cleanup)

    def source(self, name: str, content: bytes = b"pixels") -> Path:
        """A file *outside* the dataset - what the build machine's links point at."""
        path = Path(self.directory.name) / "elsewhere" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        return path

    def link(self, name: str) -> Path:
        target = self.source(name)
        link = self.root / "images" / name
        link.symlink_to(target)
        return link


class TestRelativeToRoot(Fixture):
    def test_it_strips_the_root_prefix(self) -> None:
        absolute = str(self.root / "images" / "a.png")

        self.assertEqual(relative_to_root(absolute, self.root), "images/a.png")

    def test_a_path_outside_the_root_is_left_alone(self) -> None:
        # Rewriting it would produce ../../.. escapes that break on extraction.
        outside = "/opt/somewhere/else.png"

        self.assertEqual(relative_to_root(outside, self.root), outside)

    def test_it_emits_forward_slashes(self) -> None:
        # An index written on Linux has to read on Windows.
        result = relative_to_root(str(self.root / "images" / "a.png"), self.root)

        self.assertNotIn("\\", result)


class TestMaterialise(Fixture):
    def test_a_symlink_becomes_a_real_file_with_the_same_bytes(self) -> None:
        link = self.link("a.png")

        self.assertTrue(materialise(link))
        self.assertFalse(link.is_symlink())
        self.assertEqual(link.read_bytes(), b"pixels")

    def test_a_real_file_is_left_untouched(self) -> None:
        real = self.root / "images" / "real.png"
        real.write_bytes(b"already here")

        self.assertFalse(materialise(real))
        self.assertEqual(real.read_bytes(), b"already here")

    def test_a_dangling_link_raises_rather_than_shipping_an_empty_file(self) -> None:
        # This is the failure the whole module exists for: a corpus copied without
        # -L arrives complete by file count and empty by content. Refusing loudly
        # here is the only way it does not reach a downloader.
        link = self.root / "images" / "gone.png"
        link.symlink_to(Path(self.directory.name) / "elsewhere" / "never-existed.png")

        with self.assertRaises(FileNotFoundError):
            materialise(link)

    def test_it_leaves_no_temporary_file_behind(self) -> None:
        materialise(self.link("a.png"))

        names = sorted(p.name for p in (self.root / "images").iterdir())
        self.assertEqual(names, ["a.png"])


class TestRewriteIndex(Fixture):
    def index(self, *lines: str) -> Path:
        path = self.root / "train_index.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def beside(self, *names: str) -> None:
        """Files laid out next to the index - the shipped arrangement."""
        for name in names:
            (self.root / name).write_bytes(b"pixels")

    def test_absolute_paths_become_relative(self) -> None:
        self.beside("a.png", "a.txt")
        path = self.index(f"{self.root}/images/a.png,{self.root}/images/a.txt")

        self.assertEqual(rewrite_index(path, self.root), 1)
        self.assertEqual(path.read_text(encoding="utf-8"), "a.png,a.txt\n")

    def test_a_path_from_the_build_machine_is_relocated(self) -> None:
        # The real case. These indexes describe /workspace/b0/phase7bar/train/... on
        # the box that built them; the files ship beside the index. A rewrite that
        # only handled paths already under root would leave every row absolute.
        self.beside("sq7383977_0003_0001_1.png", "sq7383977_0003_0001_1.txt")
        path = self.index(
            "/workspace/b0/phase7bar/train/sq7383977_0003_0001_1.png,"
            "/workspace/b0/phase7bar/train/sq7383977_0003_0001_1.txt"
        )

        rewrite_index(path, self.root)

        self.assertEqual(
            path.read_text(encoding="utf-8"),
            "sq7383977_0003_0001_1.png,sq7383977_0003_0001_1.txt\n",
        )

    def test_a_path_it_cannot_resolve_is_left_for_verify_to_catch(self) -> None:
        # Guessing here would be the dangerous move: a plausible-looking path that
        # resolves to the wrong file is exactly the silent failure this module is
        # for. Leaving it absolute means verify() reports it.
        path = self.index("/workspace/b0/phase7bar/train/absent.png,/workspace/b0/a.txt")

        rewrite_index(path, self.root)

        self.assertTrue(path.read_text(encoding="utf-8").startswith("/workspace"))

    def test_a_comma_in_the_image_path_survives(self) -> None:
        # OSSQ files pages by composer, so paths contain commas: "Haydn,_Joseph/...".
        # Splitting from the left truncates the name and the row silently points at
        # a file that does not exist. Three separate sites in this codebase had it.
        self.beside("Haydn,_Joseph.png", "a.txt")
        path = self.index(f"{self.root}/images/Haydn,_Joseph.png,{self.root}/a.txt")

        rewrite_index(path, self.root)

        self.assertEqual(path.read_text(encoding="utf-8"), "Haydn,_Joseph.png,a.txt\n")

    def test_blank_lines_are_dropped(self) -> None:
        path = self.index("a.png,a.txt", "", "b.png,b.txt")

        rewrite_index(path, self.root)

        self.assertEqual(len(path.read_text(encoding="utf-8").strip().splitlines()), 2)

    def test_already_relative_rows_are_not_counted_as_changed(self) -> None:
        path = self.index("images/a.png,images/a.txt")

        self.assertEqual(rewrite_index(path, self.root), 0)


class TestRewriteGroundTruth(Fixture):
    def document(self, *pages: str) -> Path:
        path = self.root / "score.boxes.json"
        path.write_text(
            json.dumps({"matches": [{"page_image": p, "boxes": []} for p in pages]}),
            encoding="utf-8",
        )
        return path

    def test_absolute_page_references_are_rewritten(self) -> None:
        path = self.document(f"{self.root}/images/p1.png")

        self.assertEqual(rewrite_ground_truth(path, self.root), 1)

        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["matches"][0]["page_image"], "images/p1.png")

    def test_other_fields_are_preserved(self) -> None:
        path = self.document(f"{self.root}/images/p1.png")

        rewrite_ground_truth(path, self.root)

        self.assertIn("boxes", json.loads(path.read_text(encoding="utf-8"))["matches"][0])

    def test_a_document_with_no_matches_is_handled(self) -> None:
        path = self.root / "empty.json"
        path.write_text(json.dumps({}), encoding="utf-8")

        self.assertEqual(rewrite_ground_truth(path, self.root), 0)


class TestVerify(Fixture):
    def test_a_clean_dataset_reports_nothing(self) -> None:
        (self.root / "images" / "a.png").write_bytes(b"pixels")
        (self.root / "index.txt").write_text("images/a.png,images/a.png\n", encoding="utf-8")

        self.assertEqual(verify(self.root), [])

    def test_a_remaining_symlink_is_reported(self) -> None:
        self.link("a.png")

        self.assertTrue(any("symlink" in p for p in verify(self.root)))

    def test_an_absolute_path_in_an_index_is_reported(self) -> None:
        (self.root / "index.txt").write_text("/opt/a.png,/opt/a.txt\n", encoding="utf-8")

        self.assertTrue(any("absolute path" in p for p in verify(self.root)))

    def test_a_relative_path_pointing_nowhere_is_reported(self) -> None:
        # Relative and wrong is the subtle case: it passes a grep for "/" and still
        # fails on first read.
        (self.root / "index.txt").write_text("images/absent.png,a.txt\n", encoding="utf-8")

        self.assertTrue(any("missing file" in p for p in verify(self.root)))


class TestWriteManifest(Fixture):
    def test_it_counts_files_and_bytes(self) -> None:
        (self.root / "images" / "a.png").write_bytes(b"12345")

        manifest = write_manifest(self.root, self.root / "MANIFEST.json")

        self.assertEqual(manifest["files"], 1)
        self.assertEqual(manifest["bytes"], 5)

    def test_the_digest_changes_when_a_file_changes_size(self) -> None:
        image = self.root / "images" / "a.png"
        image.write_bytes(b"12345")
        before = write_manifest(self.root, self.root / "m1.json")["tree_digest"]

        image.write_bytes(b"123456789")
        after = write_manifest(self.root, self.root / "m2.json")["tree_digest"]

        self.assertNotEqual(before, after)

    def test_the_digest_is_stable_when_a_manifest_is_already_present(self) -> None:
        # The downloader checks the digest against a tree that *contains* the
        # manifest; we compute it against a tree that does not yet. If the manifest
        # counted itself the two would never agree, and the checksum we ship would
        # be one nobody can reproduce.
        (self.root / "images" / "a.png").write_bytes(b"12345")
        out = self.root / "MANIFEST.json"

        first = write_manifest(self.root, out)["tree_digest"]
        second = write_manifest(self.root, out)["tree_digest"]

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
