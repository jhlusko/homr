import json
import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.ossq_pair_review_server import (
    ReviewState,
    read_index,
    read_predictions,
    render_index,
    render_staff,
    safe_staff,
)


def _index(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.write_text("\n".join(f"{i},{t}" for i, t in rows) + "\n", encoding="utf-8")
    return path


def _predictions(path: Path, rows: list[tuple[str, list[str], list[str]]]) -> Path:
    path.write_text(
        "\n".join(
            json.dumps({"tokens": t, "pitch_reference": r, "pitch_predicted": p})
            for t, r, p in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class TestReadIndex(unittest.TestCase):
    def test_it_keys_by_the_token_filename(self) -> None:
        # The two tracks share token filenames and nothing else, so that is the key.
        with tempfile.TemporaryDirectory() as tmp:
            path = _index(Path(tmp) / "i.txt", [("/a/x.png", "/a/sq1_0001_0001_1.txt")])

            self.assertEqual(list(read_index(path)), ["sq1_0001_0001_1.txt"])


class TestReadPredictions(unittest.TestCase):
    def test_accuracy_is_matches_over_reference_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _predictions(
                Path(tmp) / "p.jsonl", [("/a/s.txt", ["C4", "D4", "E4", "F4"], ["C4", "D4", "X", "Y"])]
            )

            self.assertAlmostEqual(read_predictions(path)["s.txt"]["accuracy"], 0.5)

    def test_a_missing_file_is_empty_rather_than_an_error(self) -> None:
        self.assertEqual(read_predictions(Path("/nonexistent.jsonl")), {})


class TestStaves(unittest.TestCase):
    def _state(self, tmp: Path, scanned_acc: dict[str, float] | None = None) -> ReviewState:
        names = ["a.txt", "b.txt", "c.txt"]
        syn = _index(tmp / "syn.txt", [(f"/s/{n}.png", f"/s/{n}") for n in names])
        scn = _index(tmp / "scn.txt", [(f"/c/{n}.png", f"/c/{n}") for n in names])
        preds = None
        if scanned_acc is not None:
            rows = []
            for n, acc in scanned_acc.items():
                total = 10
                right = int(round(acc * total))
                rows.append((f"/c/{n}", ["C4"] * total, ["C4"] * right + ["X"] * (total - right)))
            preds = _predictions(tmp / "scn.jsonl", rows)
        return ReviewState(syn, scn, tmp / "j.json", None, preds)

    def test_only_shared_staves_are_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            syn = _index(tmp / "syn.txt", [("/s/a.png", "/s/a.txt"), ("/s/z.png", "/s/z.txt")])
            scn = _index(tmp / "scn.txt", [("/c/a.png", "/c/a.txt")])

            state = ReviewState(syn, scn, tmp / "j.json")

            self.assertEqual(state.staves(), ["a.txt"])

    def test_worst_scanned_accuracy_comes_first(self) -> None:
        # The claim under review is about collapsed staves, so they must be the ones a
        # reviewer sees without paging.
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), {"a.txt": 0.9, "b.txt": 0.1, "c.txt": 0.5})

            self.assertEqual(state.staves()[0], "b.txt")

    def test_unmeasured_staves_sort_last_not_first(self) -> None:
        # An absent measurement is not evidence of a collapse; sorting it first would
        # fill the page with rows that say nothing.
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), {"a.txt": 0.2})

            self.assertEqual(state.staves()[0], "a.txt")


class TestJudgments(unittest.TestCase):
    def test_a_judgment_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            syn = _index(tmp / "s.txt", [("/s/a.png", "/s/a.txt")])
            scn = _index(tmp / "c.txt", [("/c/a.png", "/c/a.txt")])
            state = ReviewState(syn, scn, tmp / "j.json")

            state.save_judgment("a.txt", "crops-differ", "shifted one part")

            self.assertEqual(state.judgments()["a.txt"]["judgment"], "crops-differ")

    def test_judgments_survive_a_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            syn = _index(tmp / "s.txt", [("/s/a.png", "/s/a.txt")])
            scn = _index(tmp / "c.txt", [("/c/a.png", "/c/a.txt")])
            ReviewState(syn, scn, tmp / "j.json").save_judgment("a.txt", "crops-match", "")

            self.assertIn("a.txt", ReviewState(syn, scn, tmp / "j.json").judgments())


class TestRendering(unittest.TestCase):
    def _state(self, tmp: Path) -> ReviewState:
        syn = _index(tmp / "s.txt", [("/s/a.png", "/s/a.txt")])
        scn = _index(tmp / "c.txt", [("/c/a.png", "/c/a.txt")])
        preds = _predictions(tmp / "p.jsonl", [("/c/a.txt", ["C4", "D4"], ["C4", "X"])])
        return ReviewState(syn, scn, tmp / "j.json", None, preds)

    def test_the_index_lists_the_staff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            body = render_index(self._state(Path(tmp)))

        self.assertIn("a.txt", body)

    def test_the_staff_page_shows_both_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            body = render_staff(self._state(Path(tmp)), "a.txt")

        self.assertIn("track=scanned", body)
        self.assertIn("track=synthetic", body)

    def test_an_unknown_staff_renders_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(render_staff(self._state(Path(tmp)), "absent.txt"))

    def test_staff_names_with_markup_cannot_inject_html(self) -> None:
        self.assertIsNone(safe_staff("<script>alert(1)</script>"))
        self.assertIsNone(safe_staff("../../etc/passwd"))
        self.assertEqual(safe_staff("sq1_0001_0001_1.txt"), "sq1_0001_0001_1.txt")


if __name__ == "__main__":
    unittest.main()


class TestNextUnjudged(unittest.TestCase):
    """Recording a verdict should hand back the next thing to look at."""

    def _state(self, tmp: Path, names: list[str]) -> ReviewState:
        syn = _index(tmp / "syn.txt", [(f"/s/{n}.png", f"/s/{n}") for n in names])
        scn = _index(tmp / "scn.txt", [(f"/c/{n}.png", f"/c/{n}") for n in names])
        return ReviewState(syn, scn, tmp / "j.json")

    def test_it_advances_to_the_following_staff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), ["a.txt", "b.txt", "c.txt"])

            self.assertEqual(state.next_unjudged("a.txt"), "b.txt")

    def test_already_judged_staves_are_skipped(self) -> None:
        # Returning to a reviewed run must not walk back through decisions made.
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), ["a.txt", "b.txt", "c.txt"])
            state.save_judgment("b.txt", "crops-match", "")

            self.assertEqual(state.next_unjudged("a.txt"), "c.txt")

    def test_it_wraps_to_reach_staves_before_the_entry_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), ["a.txt", "b.txt", "c.txt"])
            state.save_judgment("c.txt", "crops-match", "")

            self.assertEqual(state.next_unjudged("c.txt"), "a.txt")

    def test_nothing_left_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), ["a.txt", "b.txt"])
            for n in ("a.txt", "b.txt"):
                state.save_judgment(n, "crops-match", "")

            self.assertIsNone(state.next_unjudged("a.txt"))

    def test_an_unknown_current_staff_still_yields_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), ["a.txt", "b.txt"])

            self.assertEqual(state.next_unjudged("absent.txt"), "a.txt")
