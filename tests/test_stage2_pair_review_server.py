import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training.omr_datasets.stage2_pair_review_server import (
    ReviewState,
    load_manifest,
    parse_stem,
    pitch_summary,
)


class TestParseStem(unittest.TestCase):
    def test_parses_score_id_system_and_voice(self) -> None:
        self.assertEqual(parse_stem("IMSLP89026-sys5-v0"), ("IMSLP89026", 5, 0))

    def test_score_id_may_itself_contain_hyphens(self) -> None:
        self.assertEqual(parse_stem("IMSLP-weird-89026-sys12-v1"), ("IMSLP-weird-89026", 12, 1))

    def test_unrecognized_stem_returns_none(self) -> None:
        self.assertIsNone(parse_stem("not-a-pair-stem"))


class TestLoadManifest(unittest.TestCase):
    def test_parses_csv_lines_into_annotated_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.txt"
            manifest.write_text(
                "/a/IMSLP1-sys0-v0.png,/a/IMSLP1-sys0-v0.tokens\n"
                "/a/IMSLP1-sys1-v0.png,/a/IMSLP1-sys1-v0.tokens\n"
            )

            entries = load_manifest(manifest)

            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["score_id"], "IMSLP1")
            self.assertEqual(entries[0]["system"], 0)
            self.assertEqual(entries[1]["system"], 1)

    def test_sorted_by_score_then_system_then_voice(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.txt"
            manifest.write_text(
                "/a/IMSLP2-sys0-v1.png,/a/IMSLP2-sys0-v1.tokens\n"
                "/a/IMSLP1-sys1-v0.png,/a/IMSLP1-sys1-v0.tokens\n"
                "/a/IMSLP1-sys0-v0.png,/a/IMSLP1-sys0-v0.tokens\n"
            )

            entries = load_manifest(manifest)

            self.assertEqual(
                [e["stem"] for e in entries],
                ["IMSLP1-sys0-v0", "IMSLP1-sys1-v0", "IMSLP2-sys0-v1"],
            )

    def test_lines_with_unparseable_stems_are_skipped(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.txt"
            manifest.write_text(
                "/a/garbage.png,/a/garbage.tokens\n"
                "/a/IMSLP1-sys0-v0.png,/a/IMSLP1-sys0-v0.tokens\n"
            )

            entries = load_manifest(manifest)

            self.assertEqual(len(entries), 1)

    def test_blank_lines_are_ignored(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.txt"
            manifest.write_text("\n/a/IMSLP1-sys0-v0.png,/a/IMSLP1-sys0-v0.tokens\n\n")

            entries = load_manifest(manifest)

            self.assertEqual(len(entries), 1)


class TestReviewState(unittest.TestCase):
    def _make_state(self, tmp: str) -> ReviewState:
        manifest = Path(tmp) / "manifest.txt"
        manifest.write_text(
            "/a/IMSLP1-sys0-v0.png,/a/IMSLP1-sys0-v0.tokens\n"
            "/a/IMSLP1-sys1-v0.png,/a/IMSLP1-sys1-v0.tokens\n"
            "/a/IMSLP2-sys0-v0.png,/a/IMSLP2-sys0-v0.tokens\n"
        )
        return ReviewState(manifest, Path(tmp) / "judgments.json")

    def test_score_ids_lists_each_score_once_sorted(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            self.assertEqual(state.score_ids(), ["IMSLP1", "IMSLP2"])

    def test_progress_before_any_judgment_is_zero_of_total(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            self.assertEqual(state.progress("IMSLP1"), (0, 2))

    def test_save_judgment_persists_and_is_reflected_in_progress(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            state.save_judgment("IMSLP1-sys0-v0", "good", "")

            self.assertEqual(state.progress("IMSLP1"), (1, 2))
            saved = json.loads((Path(tmp) / "judgments.json").read_text())
            self.assertEqual(saved["IMSLP1-sys0-v0"]["judgment"], "good")

    def test_save_judgment_rejects_an_unknown_judgment_value(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            with self.assertRaises(ValueError):
                state.save_judgment("IMSLP1-sys0-v0", "maybe", "")

    def test_a_second_judgment_overwrites_the_first_for_the_same_pair(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            state.save_judgment("IMSLP1-sys0-v0", "bad", "")
            state.save_judgment("IMSLP1-sys0-v0", "good", "")

            saved = json.loads((Path(tmp) / "judgments.json").read_text())
            self.assertEqual(saved["IMSLP1-sys0-v0"]["judgment"], "good")


class TestRenderedPath(unittest.TestCase):
    def test_none_when_no_rendered_dir_configured(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.txt"
            manifest.write_text("/a/IMSLP1-sys0-v0.png,/a/IMSLP1-sys0-v0.tokens\n")
            state = ReviewState(manifest, Path(tmp) / "judgments.json")

            self.assertIsNone(state.rendered_path("IMSLP1-sys0-v0"))

    def test_none_when_the_file_does_not_exist_yet(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.txt"
            manifest.write_text("/a/IMSLP1-sys0-v0.png,/a/IMSLP1-sys0-v0.tokens\n")
            rendered_dir = Path(tmp) / "rendered"
            rendered_dir.mkdir()
            state = ReviewState(manifest, Path(tmp) / "judgments.json", rendered_dir)

            self.assertIsNone(state.rendered_path("IMSLP1-sys0-v0"))

    def test_returns_the_path_once_it_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.txt"
            manifest.write_text("/a/IMSLP1-sys0-v0.png,/a/IMSLP1-sys0-v0.tokens\n")
            rendered_dir = Path(tmp) / "rendered"
            rendered_dir.mkdir()
            (rendered_dir / "IMSLP1-sys0-v0.png").write_bytes(b"fake")
            state = ReviewState(manifest, Path(tmp) / "judgments.json", rendered_dir)

            self.assertEqual(state.rendered_path("IMSLP1-sys0-v0"), rendered_dir / "IMSLP1-sys0-v0.png")


class TestPitchSummary(unittest.TestCase):
    def _write_tokens(self, tmp: str, content: str) -> Path:
        path = Path(tmp) / "sample.tokens"
        path.write_text(content)
        return path

    def test_notes_and_barlines_appear_in_order(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._write_tokens(
                tmp,
                "clef_G2 _ _ _ _ upper\n"
                "note_4 E5 b _ _ upper\n"
                "barline . . . . .\n"
                "note_4 C5 _ _ _ upper\n",
            )

            self.assertEqual(pitch_summary(str(path)), "E5b | C5")

    def test_a_rest_shows_as_the_word_rest_not_its_placeholder_pitch(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._write_tokens(
                tmp,
                "note_4 A4 b _ _ upper\n"
                "rest_8 _ _ _ _ upper\n"
                "note_8 A4 b _ _ upper\n",
            )

            self.assertEqual(pitch_summary(str(path)), "A4b rest A4b")

    def test_sharp_lift_is_rendered(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._write_tokens(tmp, "note_4 F5 # _ _ upper\n")

            self.assertEqual(pitch_summary(str(path)), "F5#")


if __name__ == "__main__":
    unittest.main()
