import unittest

from training.omr_datasets.make_review_sets import sample_spread_across_scores


class TestSampleSpreadAcrossScores(unittest.TestCase):
    def test_one_prolific_score_cannot_dominate(self) -> None:
        """The previous review drew 22 of 50 judged items from two scores, which made
        the many-to-many signal impossible to separate from a per-score failure."""
        stems = [f"BIG-sys{i}-v0" for i in range(90)] + [
            f"S{j}-sys0-v0" for j in range(9)
        ]
        picked = sample_spread_across_scores(stems, 10, "eval")
        from_big = sum(1 for s in picked if s.startswith("BIG-"))
        self.assertEqual(len(picked), 10)
        self.assertLessEqual(from_big, 2, picked)

    def test_it_is_deterministic(self) -> None:
        stems = [f"S{j}-sys{i}-v0" for j in range(5) for i in range(5)]
        self.assertEqual(
            sample_spread_across_scores(stems, 7, "eval"),
            sample_spread_across_scores(stems, 7, "eval"),
        )

    def test_different_sets_draw_different_samples(self) -> None:
        stems = [f"S{j}-sys{i}-v0" for j in range(6) for i in range(6)]
        self.assertNotEqual(
            sample_spread_across_scores(stems, 8, "eval"),
            sample_spread_across_scores(stems, 8, "pseudo"),
        )

    def test_asking_for_more_than_exists_returns_everything(self) -> None:
        stems = ["A-sys0-v0", "B-sys1-v0"]
        self.assertEqual(sorted(sample_spread_across_scores(stems, 50, "eval")), stems)

    def test_unparseable_stems_are_dropped(self) -> None:
        self.assertEqual(sample_spread_across_scores(["not-a-stem"], 5, "eval"), [])


if __name__ == "__main__":
    unittest.main()


class TestIdenticalPanesAreDropped(unittest.TestCase):
    """Two methods can choose different measure ranges whose slices render the same
    tokens - an all-rest passage being the common case. Asking which of two identical
    panes is right is a question with no answer."""

    def test_a_pair_with_identical_labels_is_not_offered_for_review(self) -> None:
        import tempfile
        from pathlib import Path

        from training.omr_datasets.make_review_sets import build_set

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.png").write_bytes(b"png")
            same = root / "same.tokens"
            same.write_text("rest_4 _ _ _ _ upper\n", encoding="utf-8")
            other = root / "other.tokens"
            other.write_text("note_4 C4 _ _ _ upper\n", encoding="utf-8")
            left = {"S-sys0-v0": f"{root/'a.png'},{same}",
                    "S-sys1-v0": f"{root/'a.png'},{same}"}
            right = {"S-sys0-v0": f"{root/'a.png'},{same}",
                     "S-sys1-v0": f"{root/'a.png'},{other}"}
            out = root / "out"
            build_set("t", ["S-sys0-v0", "S-sys1-v0"], left, right, out, 10)
            import json
            ids = [x["id"] for x in json.load(open(out / "t" / "manifest.json"))]
            self.assertEqual(ids, ["S-sys1-v0"])
