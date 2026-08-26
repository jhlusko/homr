import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.convert_ossq import MissingSegments, segments_dir


def _work(root: Path, *, unaligned: bool = True, scanned: bool = True) -> Path:
    work = root / "Composer" / "Work"
    if unaligned:
        (work / "musicxml" / "unaligned").mkdir(parents=True)
    if scanned:
        (work / "musicxml" / "scanned" / "systemwise").mkdir(parents=True)
    work.mkdir(parents=True, exist_ok=True)
    return work


class TestSegmentsDir(unittest.TestCase):
    def test_the_synthetic_track_reads_unaligned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = _work(Path(tmp))

            self.assertEqual(segments_dir(work, "synthetic").name, "unaligned")

    def test_the_scanned_track_reads_its_own_systemwise(self) -> None:
        # The bug: both tracks read `unaligned`, which is keyed to the synthetic
        # pagination. A score rendering to 24 pages can scan to 22, so the same
        # (page, system) names different music - and because both directories hold the
        # same number of segments, every guard downstream still passes.
        with tempfile.TemporaryDirectory() as tmp:
            work = _work(Path(tmp))

            resolved = segments_dir(work, "scanned")

            self.assertEqual(resolved.name, "systemwise")
            self.assertEqual(resolved.parent.name, "scanned")

    def test_the_two_tracks_never_resolve_to_the_same_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = _work(Path(tmp))

            self.assertNotEqual(segments_dir(work, "synthetic"), segments_dir(work, "scanned"))

    def test_a_missing_scanned_directory_raises_rather_than_falling_back(self) -> None:
        # Falling back to `unaligned` is exactly the failure this prevents, and it
        # would look like a successful conversion.
        with tempfile.TemporaryDirectory() as tmp:
            work = _work(Path(tmp), scanned=False)

            with self.assertRaises(MissingSegments):
                segments_dir(work, "scanned")

    def test_a_missing_synthetic_directory_also_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = _work(Path(tmp), unaligned=False)

            with self.assertRaises(MissingSegments):
                segments_dir(work, "synthetic")

    def test_the_error_names_the_directory_it_wanted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = _work(Path(tmp), scanned=False)

            with self.assertRaises(MissingSegments) as caught:
                segments_dir(work, "scanned")

        self.assertIn("systemwise", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
