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
