import unittest
from fractions import Fraction

from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.make_metre_review_set import (
    TOLERANCE,
    disagreement,
    implied_numerator,
    measure_durations,
)

BRANCHES = ("rhythm", "pitch", "lift", "articulation", "slur", "position")


def note(rhythm):
    return EncodedSymbol(rhythm, "C4", "_", "_", "_", "upper")


def record(ref_rhythms, pred_rhythms):
    r = {"tokens": "t"}
    for b in BRANCHES:
        r[f"{b}_reference"] = [
            x if b == "rhythm" else ("C4" if b == "pitch" else "upper" if b == "position" else "_")
            for x in ref_rhythms
        ]
        r[f"{b}_predicted"] = [
            x if b == "rhythm" else ("C4" if b == "pitch" else "upper" if b == "position" else "_")
            for x in pred_rhythms
        ]
    return r


def bar(rhythms):
    return rhythms + ["barline"]


class TestImpliedNumerator(unittest.TestCase):
    def test_four_quarters_is_a_whole_note_of_duration(self) -> None:
        syms = [note("note_4")] * 4 + [EncodedSymbol("barline")]
        self.assertEqual(implied_numerator(syms), Fraction(1))

    def test_three_quarters_is_three_quarters(self) -> None:
        syms = [note("note_4")] * 3 + [EncodedSymbol("barline")]
        self.assertEqual(implied_numerator(syms), Fraction(3, 4))

    def test_an_empty_stream_implies_nothing(self) -> None:
        self.assertEqual(implied_numerator([]), Fraction(0))


class TestDisagreement(unittest.TestCase):
    def test_identical_streams_agree(self) -> None:
        r = record(bar(["note_4"] * 4), bar(["note_4"] * 4))
        _, _, differs = disagreement(r)
        self.assertFalse(differs)

    def test_six_eight_against_four_four_disagrees(self) -> None:
        """The defect this exists for: a label in compound metre against a reading of
        the printed simple metre. Six eighths is 3/4 of a whole; four quarters is a
        whole."""
        r = record(bar(["note_8"] * 6), bar(["note_4"] * 4))
        label, predicted, differs = disagreement(r)
        self.assertTrue(differs)
        self.assertEqual((label, predicted), (Fraction(3, 4), Fraction(1)))

    def test_a_missing_side_is_not_a_disagreement(self) -> None:
        """Nothing to compare is not evidence of conflict - the mistake that made
        'reverse says empty' mean 'no music here'."""
        r = record([], bar(["note_4"] * 4))
        self.assertFalse(disagreement(r)[2])

    def test_a_rounding_wobble_is_within_tolerance(self) -> None:
        self.assertGreater(TOLERANCE, Fraction(0))
        r = record(bar(["note_4"] * 4), bar(["note_4"] * 4 + ["note_64"]))
        label, predicted, differs = disagreement(r)
        self.assertLessEqual(abs(label - predicted), TOLERANCE)
        self.assertFalse(differs)


class TestPerMeasureComparison(unittest.TestCase):
    """A median hides the reported case - "the last two bars should be 3/4" - because
    two changed measures in a system of six do not move it."""

    def test_measure_durations_are_returned_in_order(self) -> None:
        syms = [note("note_4")] * 4 + [EncodedSymbol("barline")] \
             + [note("note_4")] * 3 + [EncodedSymbol("barline")]
        self.assertEqual(measure_durations(syms), [Fraction(1), Fraction(3, 4)])

    def test_a_change_in_the_last_bars_is_caught(self) -> None:
        ref = bar(["note_4"] * 4) + bar(["note_4"] * 4) + bar(["note_4"] * 4) \
            + bar(["note_4"] * 3) + bar(["note_4"] * 3)
        pred = bar(["note_4"] * 4) * 5
        _, _, differs = disagreement(record(ref, pred))
        self.assertTrue(differs, "a metre change in the tail must be caught")

    def test_that_same_case_is_invisible_to_the_median(self) -> None:
        ref = bar(["note_4"] * 4) + bar(["note_4"] * 4) + bar(["note_4"] * 4) \
            + bar(["note_4"] * 3) + bar(["note_4"] * 3)
        pred = bar(["note_4"] * 4) * 5
        from training.omr_datasets.make_metre_review_set import symbols_from
        a = implied_numerator(symbols_from(record(ref, pred), "reference"))
        b = implied_numerator(symbols_from(record(ref, pred), "predicted"))
        self.assertEqual(a, b, "the median is blind here, which is why it is not the test")


if __name__ == "__main__":
    unittest.main()
