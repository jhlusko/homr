import unittest

from training.transformer.domain_gap import Pair, describe, pair_up


class TestPairUp(unittest.TestCase):
    """The two tracks write the same token filenames into different directories, which is
    what makes a staff comparable with itself."""

    def test_only_staves_present_in_both_are_compared(self) -> None:
        pairs = pair_up({"a.txt": (1.0, 10), "b.txt": (1.0, 10)}, {"a.txt": (0.5, 10)})

        self.assertEqual([p.name for p in pairs], ["a.txt"])

    def test_the_drop_is_synthetic_minus_scanned(self) -> None:
        pairs = pair_up({"a.txt": (0.9, 10)}, {"a.txt": (0.4, 10)})

        self.assertAlmostEqual(pairs[0].drop, 0.5)

    def test_nothing_in_common_is_reported_not_crashed(self) -> None:
        self.assertIn("no staves in common", describe(pair_up({"a.txt": (1.0, 1)}, {})))


class TestDescribe(unittest.TestCase):
    def test_a_uniform_gap_puts_about_a_tenth_in_the_worst_tenth(self) -> None:
        # The number that separates "scans are harder" from "some crops are broken".
        pairs = [Pair(f"s{i}_0_0_1.txt", 0.9, 0.7, 100) for i in range(100)]

        self.assertIn("10.0%", describe(pairs))

    def test_a_concentrated_gap_puts_most_of_it_there(self) -> None:
        pairs = [Pair(f"s{i}_0_0_1.txt", 0.9, 0.9, 100) for i in range(90)]
        pairs += [Pair(f"b{i}_0_0_1.txt", 0.9, 0.0, 100) for i in range(10)]

        self.assertIn("100.0%", describe(pairs))

    def test_collapse_is_reported_as_a_rate_per_score(self) -> None:
        # A count of scores is not evidence of clustering unless the total is known: the
        # first version said "326 staves from 9 scores" where the split had 9 scores.
        pairs = [Pair("alpha_0_0_1.txt", 1.0, 0.0, 10), Pair("beta_0_0_1.txt", 1.0, 1.0, 10)]

        report = describe(pairs)

        self.assertIn("2 score(s) in this split", report)
        self.assertIn("alpha", report)

    def test_staves_that_hold_up_are_counted(self) -> None:
        pairs = [Pair("a_0_0_1.txt", 0.9, 0.88, 10), Pair("b_0_0_1.txt", 0.9, 0.1, 10)]

        self.assertIn("1 (50.0%)", describe(pairs))


if __name__ == "__main__":
    unittest.main()
