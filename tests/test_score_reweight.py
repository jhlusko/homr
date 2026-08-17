import tempfile
import unittest
from pathlib import Path

from training.transformer.domain_gap import Pair
from training.omr_datasets.score_reweight import (
    ScoreWeight,
    collapse_rates,
    reweight_index,
    repeat_counts,
)


def _pair(score: str, drop: float, index: int = 1) -> Pair:
    return Pair(f"{score}_0001_0001_{index}.txt", 0.9, 0.9 - drop, 10)


class TestCollapseRates(unittest.TestCase):
    def test_the_rate_is_the_share_past_the_threshold(self) -> None:
        pairs = [_pair("a", 0.6, 1), _pair("a", 0.6, 2), _pair("a", 0.1, 3), _pair("a", 0.1, 4)]

        self.assertAlmostEqual(collapse_rates(pairs)["a"], 0.5)

    def test_scores_are_kept_separate(self) -> None:
        pairs = [_pair("a", 0.9, 1), _pair("b", 0.0, 1)]

        rates = collapse_rates(pairs)

        self.assertGreater(rates["a"], rates["b"])


class TestRepeatCounts(unittest.TestCase):
    def test_a_score_below_the_floor_keeps_its_natural_weight(self) -> None:
        # Sampling noise in a low collapse rate should not produce a meaningless x1.02.
        weights = repeat_counts({"a": 0.05}, floor=0.1)

        self.assertEqual(weights["a"].repeats, 1)

    def test_a_higher_collapse_rate_gets_more_repeats(self) -> None:
        weights = repeat_counts({"a": 0.9, "b": 0.2}, floor=0.1, max_repeats=6)

        self.assertGreater(weights["a"].repeats, weights["b"].repeats)

    def test_the_repeat_count_is_capped(self) -> None:
        # An uncapped multiplier turns a handful of documents into most of an epoch -
        # the same risk 27.50 named for loss weighting, applied to sampling. The worst
        # score in the batch always reaches the cap by construction, since the scale is
        # calibrated against it.
        weights = repeat_counts({"a": 1.0, "b": 0.1}, floor=0.1, max_repeats=6)

        self.assertEqual(weights["a"].repeats, 6)

    def test_scaling_is_against_the_observed_worst_not_a_hypothetical_100_percent(self) -> None:
        # The first version scaled toward rate=1.0. On real data whose worst score is
        # 21.9%, that put every real score within rounding distance of x1 - a scale
        # calibrated to data nothing produces does not fire on the data that exists.
        weights = repeat_counts({"a": 0.219, "b": 0.05}, floor=0.1, max_repeats=6)

        self.assertEqual(weights["a"].repeats, 6)

    def test_a_batch_where_nothing_clears_the_floor_repeats_nothing(self) -> None:
        weights = repeat_counts({"a": 0.05, "b": 0.02}, floor=0.1, max_repeats=6)

        self.assertEqual(weights["a"].repeats, 1)
        self.assertEqual(weights["b"].repeats, 1)

    def test_an_empty_batch_produces_no_weights(self) -> None:
        self.assertEqual(repeat_counts({}), {})

    def test_a_fully_clean_score_at_exactly_the_floor_is_untouched(self) -> None:
        weights = repeat_counts({"a": 0.1}, floor=0.1)

        self.assertEqual(weights["a"].repeats, 1)


class TestReweightIndex(unittest.TestCase):
    def test_lines_are_repeated_by_their_scores_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(
                "img_a.png,/data/a_0001_0001_1.txt\nimg_b.png,/data/b_0001_0001_1.txt\n",
                encoding="utf-8",
            )
            weights = {"a": ScoreWeight("a", 0.9, 3), "b": ScoreWeight("b", 0.0, 1)}

            before, after = reweight_index(index, weights, directory / "out.txt")
            lines = (directory / "out.txt").read_text(encoding="utf-8").splitlines()

        self.assertEqual(before, 2)
        self.assertEqual(after, 4)
        self.assertEqual(sum(1 for line in lines if "a_0001" in line), 3)
        self.assertEqual(sum(1 for line in lines if "b_0001" in line), 1)

    def test_an_unmeasured_score_is_kept_once_not_guessed_at(self) -> None:
        # Repeating a score with no measurement would be acting on a weight that was never
        # computed, not on evidence.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text("img.png,/data/unmeasured_0001_0001_1.txt\n", encoding="utf-8")

            _, after = reweight_index(index, {}, directory / "out.txt")

        self.assertEqual(after, 1)

    def test_blank_lines_are_dropped_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(
                "img.png,/data/a_0001_0001_1.txt\n\n\n", encoding="utf-8"
            )
            weights = {"a": ScoreWeight("a", 0.9, 4)}

            _, after = reweight_index(index, weights, directory / "out.txt")

        self.assertEqual(after, 4)


if __name__ == "__main__":
    unittest.main()
