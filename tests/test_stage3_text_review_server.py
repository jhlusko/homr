import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training.omr_datasets.stage3_text_review_server import (
    ReviewState,
    load_matches,
    render_index,
    render_score_page,
)


def _match(text: str, kind: str, page_index: int, page_image: str = "p1.png") -> dict:
    return {
        "box": {"left": 10, "top": 20, "width": 30, "height": 15},
        "text": text,
        "score": 0.9,
        "kind": kind,
        "matched_fraction": 0.95,
        "page_index": page_index,
        "page_image": page_image,
    }


class TestLoadMatches(unittest.TestCase):
    def test_annotates_each_match_with_its_own_score_id(self) -> None:
        with TemporaryDirectory() as tmp:
            matches_dir = Path(tmp)
            (matches_dir / "IMSLP1.json").write_text(
                json.dumps({"score_id": "IMSLP1", "matches": [_match("Wort", "lyric", 0)]})
            )

            entries = load_matches(matches_dir)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["score_id"], "IMSLP1")
            self.assertEqual(entries[0]["text"], "Wort")

    def test_keys_are_unique_within_a_score_by_page_and_order(self) -> None:
        with TemporaryDirectory() as tmp:
            matches_dir = Path(tmp)
            (matches_dir / "IMSLP1.json").write_text(
                json.dumps(
                    {
                        "score_id": "IMSLP1",
                        "matches": [
                            _match("a", "lyric", 0),
                            _match("b", "lyric", 0),
                            _match("c", "lyric", 1),
                        ],
                    }
                )
            )

            entries = load_matches(matches_dir)

            keys = [e["key"] for e in entries]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(keys, ["0-0", "0-1", "1-0"])

    def test_reads_every_score_file_in_the_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            matches_dir = Path(tmp)
            (matches_dir / "IMSLP1.json").write_text(
                json.dumps({"score_id": "IMSLP1", "matches": [_match("a", "lyric", 0)]})
            )
            (matches_dir / "IMSLP2.json").write_text(
                json.dumps({"score_id": "IMSLP2", "matches": [_match("b", "dynamic", 0)]})
            )

            entries = load_matches(matches_dir)

            self.assertEqual({e["score_id"] for e in entries}, {"IMSLP1", "IMSLP2"})


class TestReviewState(unittest.TestCase):
    def _make_state(self, tmp: str) -> ReviewState:
        matches_dir = Path(tmp) / "matches"
        matches_dir.mkdir()
        (matches_dir / "IMSLP1.json").write_text(
            json.dumps(
                {
                    "score_id": "IMSLP1",
                    "matches": [_match("a", "lyric", 0), _match("b", "dynamic", 0)],
                }
            )
        )
        return ReviewState(matches_dir, Path(tmp) / "judgments.json", [Path(tmp) / "pngs"])

    def test_score_ids_lists_scores_with_at_least_one_match(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            self.assertEqual(state.score_ids(), ["IMSLP1"])

    def test_progress_before_any_judgment(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            self.assertEqual(state.progress("IMSLP1"), (0, 2))

    def test_save_judgment_persists_and_is_reflected_in_progress(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            state.save_judgment("IMSLP1/0-0", "good", "")

            self.assertEqual(state.progress("IMSLP1"), (1, 2))
            saved = json.loads((Path(tmp) / "judgments.json").read_text())
            self.assertEqual(saved["IMSLP1/0-0"]["judgment"], "good")

    def test_save_judgment_rejects_unknown_judgment_value(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            with self.assertRaises(ValueError):
                state.save_judgment("IMSLP1/0-0", "maybe", "")

    def test_page_path_returns_none_when_no_pngs_dir_has_the_score(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            self.assertIsNone(state.page_path("IMSLP1", "p1.png"))

    def test_page_path_finds_the_score_in_whichever_pngs_dir_has_it(self) -> None:
        with TemporaryDirectory() as tmp:
            pngs_a = Path(tmp) / "pngs_a"
            pngs_b = Path(tmp) / "pngs_b"
            (pngs_b / "IMSLP1").mkdir(parents=True)
            (pngs_b / "IMSLP1" / "p1.png").write_bytes(b"fake")
            matches_dir = Path(tmp) / "matches"
            matches_dir.mkdir()
            (matches_dir / "IMSLP1.json").write_text(
                json.dumps({"score_id": "IMSLP1", "matches": [_match("a", "lyric", 0)]})
            )
            state = ReviewState(matches_dir, Path(tmp) / "judgments.json", [pngs_a, pngs_b])

            found = state.page_path("IMSLP1", "p1.png")

            self.assertEqual(found, pngs_b / "IMSLP1" / "p1.png")


class TestRenderFunctions(unittest.TestCase):
    def _make_state(self, tmp: str) -> ReviewState:
        matches_dir = Path(tmp) / "matches"
        matches_dir.mkdir()
        (matches_dir / "IMSLP1.json").write_text(
            json.dumps({"score_id": "IMSLP1", "matches": [_match("Wort", "lyric", 0)]})
        )
        return ReviewState(matches_dir, Path(tmp) / "judgments.json", [Path(tmp) / "pngs"])

    def test_render_index_default_base_path_links_are_root_relative(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            html = render_index(state)

            self.assertIn('href="/score/IMSLP1"', html)

    def test_render_index_with_base_path_prefixes_links(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            html = render_index(state, base_path="/text")

            self.assertIn('href="/text/score/IMSLP1"', html)

    def test_render_score_page_returns_none_for_unknown_score(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            self.assertIsNone(render_score_page(state, "IMSLP999"))

    def test_render_score_page_with_base_path_prefixes_crop_and_api_urls(self) -> None:
        with TemporaryDirectory() as tmp:
            state = self._make_state(tmp)

            html = render_score_page(state, "IMSLP1", base_path="/text")

            self.assertIn('src="/text/crop/IMSLP1/0-0"', html)
            self.assertIn("fetch('/text/api/judge'", html)


if __name__ == "__main__":
    unittest.main()
