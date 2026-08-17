import collections
import unittest

from training.omr_datasets.resolve_baseline import Agreement, nearest, single_lyric_line


class TestNearest(unittest.TestCase):
    """The rule under test: a syllable belongs to the note whose centre is closest."""

    def test_it_picks_the_closest_centre(self) -> None:
        self.assertEqual(nearest(52.0, [10.0, 50.0, 90.0]), 1)

    def test_a_tie_resolves_left(self) -> None:
        # Deterministic rather than arbitrary, so a rerun gives the same number.
        self.assertEqual(nearest(30.0, [10.0, 50.0]), 0)

    def test_a_syllable_past_the_last_note_takes_the_last(self) -> None:
        self.assertEqual(nearest(500.0, [10.0, 50.0]), 1)


class TestAgreement(unittest.TestCase):
    def test_a_correct_pick_is_offset_zero(self) -> None:
        agreement = Agreement()
        agreement.observe(chosen=3, true_index=3, held=False)

        self.assertEqual(agreement.correct, 1)
        self.assertEqual(agreement.offsets[0], 1)

    def test_the_offset_says_which_way_it_missed(self) -> None:
        # "One note early" is a different problem from "one note late", and a single
        # accuracy would not distinguish them.
        agreement = Agreement()
        agreement.observe(chosen=2, true_index=3, held=False)

        self.assertEqual(agreement.offsets[-1], 1)

    def test_melismas_are_scored_separately(self) -> None:
        # 27.42 predicts melismas are where nearest-x breaks; a single number could not
        # confirm or refute that.
        agreement = Agreement()
        agreement.observe(chosen=0, true_index=0, held=True)
        agreement.observe(chosen=5, true_index=1, held=True)
        agreement.observe(chosen=2, true_index=2, held=False)

        self.assertEqual(agreement.melismatic, 2)
        self.assertEqual(agreement.melismatic_correct, 1)

    def test_the_report_separates_held_from_single_notes(self) -> None:
        agreement = Agreement()
        agreement.observe(chosen=0, true_index=0, held=False)
        agreement.observe(chosen=9, true_index=1, held=True)

        report = agreement.describe()

        self.assertIn("one note only: 1/1", report)
        self.assertIn("held (melisma): 0/1", report)

    def test_nothing_scored_reports_rather_than_dividing_by_zero(self) -> None:
        self.assertEqual(Agreement().describe(), "nothing to score")

    def test_each_agreement_gets_its_own_counter(self) -> None:
        # A mutable default shared between instances would pool every run's results.
        first, second = Agreement(), Agreement()
        first.observe(chosen=1, true_index=0, held=False)

        self.assertEqual(second.offsets, collections.Counter())


if __name__ == "__main__":
    unittest.main()


class TestSingleLyricLine(unittest.TestCase):
    """A Lied for two voices puts verse-1 syllables under both staves, and the vertical
    separation this file relies on stops meaning anything."""

    def _box(self, top: int, height: int = 16) -> dict:
        return {"top": top, "bottom": top + height, "left": 0, "right": 30}

    def test_one_line_despite_capitals_and_accents_raising_some_tops(self) -> None:
        boxes = [self._box(100), self._box(94), self._box(101)]

        self.assertTrue(single_lyric_line(boxes))

    def test_two_staves_are_recognised_as_two_lines(self) -> None:
        # Measured on a real score: 777 and 2395, which swallowed the whole system.
        boxes = [self._box(777), self._box(2395)]

        self.assertFalse(single_lyric_line(boxes))

    def test_a_single_syllable_is_trivially_one_line(self) -> None:
        self.assertTrue(single_lyric_line([self._box(100)]))
