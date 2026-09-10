import unittest

from training.ocr.recognizer_errors import by_bucket, confusions, report


class TestByBucket(unittest.TestCase):
    """A rate says how often; a rate per bucket says whether the cause is structural."""

    def test_each_bucket_gets_its_own_rate(self) -> None:
        pairs = [("va", "va"), ("gues", "gnes"), ("des", "des")]

        buckets = by_bucket(pairs, len)

        self.assertEqual(buckets[2][0], 1.0)
        self.assertEqual(buckets[4][0], 0.0)

    def test_the_bucket_size_is_reported(self) -> None:
        # A 0% bucket of one crop is not the same finding as a 0% bucket of a thousand.
        pairs = [("a", "a"), ("bb", "bb"), ("cc", "xx")]

        self.assertEqual(by_bucket(pairs, len)[2][2], 2)

    def test_cer_is_per_character_not_per_crop(self) -> None:
        pairs = [("abcd", "abcx")]

        self.assertAlmostEqual(by_bucket(pairs, len)[4][1], 0.25)

    def test_an_empty_bucket_does_not_divide_by_zero(self) -> None:
        self.assertEqual(by_bucket([], len), {})


class TestConfusions(unittest.TestCase):
    def test_a_dropped_character_is_named(self) -> None:
        # 'senkt' -> 'senk' was the pattern that suggested a truncation bug; naming the
        # character is what turns a suspicion into something to look up.
        self.assertIn(("dropped 't'", 1), confusions([("senkt", "senk")]))

    def test_an_added_character_is_named(self) -> None:
        self.assertIn(("added '\\xa0'", 1), confusions([("Zü", "Zü\xa0")]))

    def test_correct_predictions_contribute_nothing(self) -> None:
        self.assertEqual(confusions([("va", "va")]), [])

    def test_counts_accumulate_across_crops(self) -> None:
        found = dict(confusions([("senkt", "senk"), ("deckt", "deck")]))

        self.assertEqual(found["dropped 't'"], 2)


class TestReport(unittest.TestCase):
    def test_every_cut_appears(self) -> None:
        text = report([("va", "va"), ("lung,", "lun,")], seen={"va"})

        self.assertIn("by syllable length", text)
        self.assertIn("punctuation", text)
        self.assertIn("unseen", text)

    def test_the_headline_matches_the_pairs(self) -> None:
        text = report([("va", "va"), ("des", "des")], seen=set())

        self.assertIn("exact 100.0%", text)


if __name__ == "__main__":
    unittest.main()
