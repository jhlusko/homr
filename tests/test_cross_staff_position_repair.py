"""Verified position repair: a fix is applied only when it lands the barlines exactly.

`propose_majority_position_corrections` deliberately has no applier, because it knows
where a divergence starts but not what caused it. This turns it into a search hint and
verifies the result, so nothing is ever guessed.
"""

import unittest
from fractions import Fraction

from homr.cross_staff_position_repair import (
    agrees_exactly,
    barline_count,
    majority_barline_sequence,
    measure_span,
    repair_position_divergence,
)
from homr.transformer.vocabulary import EncodedSymbol


def _n(rhythm: str = "note_4") -> EncodedSymbol:
    return EncodedSymbol(rhythm=rhythm, pitch="C4")


def _bar() -> EncodedSymbol:
    return EncodedSymbol(rhythm="barline")


def _staff(*durations: str) -> list[EncodedSymbol]:
    """One staff: each argument is a measure, given as space-separated rhythms."""
    out: list[EncodedSymbol] = []
    for measure in durations:
        out.extend(_n(r) for r in measure.split())
        out.append(_bar())
    return out


GOOD = "note_4 note_4"  # two quarters = 1/2 per measure


class TestMeasureSpan(unittest.TestCase):
    def test_spans_are_barline_delimited_and_zero_based(self) -> None:
        staff = _staff(GOOD, GOOD, GOOD)
        self.assertEqual(measure_span(staff, 0), (0, 2))
        self.assertEqual(measure_span(staff, 1), (3, 5))
        self.assertEqual(measure_span(staff, 2), (6, 8))

    def test_a_measure_that_does_not_exist_is_not_invented(self) -> None:
        self.assertIsNone(measure_span(_staff(GOOD), 5))
        self.assertIsNone(measure_span(_staff(GOOD), -1))

    def test_counts_repeats_as_barlines_like_the_generator_does(self) -> None:
        self.assertEqual(barline_count([_n(), EncodedSymbol(rhythm="repeat_forward"), _n()]), 1)


class TestMajority(unittest.TestCase):
    def test_needs_four_staves_and_a_clear_majority(self) -> None:
        agree = _staff(GOOD, GOOD, GOOD)
        self.assertIsNone(majority_barline_sequence([agree, agree, agree]))
        self.assertEqual(
            majority_barline_sequence([agree, agree, agree, agree]),
            (Fraction(1, 2), Fraction(1), Fraction(3, 2)),
        )

    def test_a_tie_yields_no_majority(self) -> None:
        a = _staff(GOOD, GOOD)
        b = _staff("note_4 note_4 note_4", GOOD)
        self.assertIsNone(majority_barline_sequence([a, a, b, b]))


class TestAgreesExactly(unittest.TestCase):
    def setUp(self) -> None:
        self.majority = (Fraction(1, 2), Fraction(1), Fraction(3, 2))

    def test_exact_agreement_is_accepted(self) -> None:
        self.assertTrue(agrees_exactly(_staff(GOOD, GOOD, GOOD), self.majority))

    def test_merely_better_is_rejected(self) -> None:
        # First barline right, the rest still wrong: this is the candidate Phase 1's
        # reranker would happily keep, and exactly what must not be applied here.
        better = _staff(GOOD, "note_4 note_4 note_4", GOOD)
        self.assertFalse(agrees_exactly(better, self.majority))

    def test_a_truncated_candidate_is_rejected(self) -> None:
        self.assertFalse(agrees_exactly(_staff(GOOD, GOOD), self.majority))


class TestRepairPositionDivergence(unittest.TestCase):
    def setUp(self) -> None:
        self.good = _staff(GOOD, GOOD, GOOD)
        # Staff 1 has an extra eighth in measure 1, shifting every later barline.
        self.bad = _staff(GOOD, "note_4 note_4 note_8", GOOD)
        self.staves = [self.good, self.bad, self.good, self.good]
        self.margins = [[(7, 0.5)] * len(s) for s in self.staves]

    def test_an_exact_fix_inside_the_divergent_measure_is_applied(self) -> None:
        calls: list[tuple[int, int, int]] = []

        def fork(staff_index: int, step: int, alt: int):
            calls.append((staff_index, step, alt))
            return self.good

        out = repair_position_divergence(self.staves, self.staves, self.margins, fork)
        self.assertEqual(list(out), [1])
        self.assertEqual(out[1], self.good)
        # Only the offending staff is re-decoded, and only inside its bad measure.
        self.assertTrue(calls)
        self.assertEqual({c[0] for c in calls}, {1})
        span = measure_span(self.bad, 1)
        self.assertTrue(all(span[0] <= c[1] < span[1] for c in calls), calls)

    def test_a_candidate_that_only_improves_is_refused(self) -> None:
        def fork(staff_index: int, step: int, alt: int):
            return _staff(GOOD, "note_4 note_4 note_4", GOOD)

        self.assertEqual(repair_position_divergence(self.staves, self.staves, self.margins, fork), {})

    def test_a_failed_fork_is_survivable(self) -> None:
        def fork(staff_index: int, step: int, alt: int):
            return None

        self.assertEqual(repair_position_divergence(self.staves, self.staves, self.margins, fork), {})

    def test_agreeing_staves_are_never_forked(self) -> None:
        calls = []

        def fork(staff_index: int, step: int, alt: int):
            calls.append(step)
            return self.good

        out = repair_position_divergence(
            [self.good] * 4, [self.good] * 4, [[(7, 0.5)] * len(self.good)] * 4, fork
        )
        self.assertEqual(out, {})
        self.assertEqual(calls, [])

    def test_misaligned_margins_decline_rather_than_fork_the_wrong_step(self) -> None:
        calls = []

        def fork(staff_index: int, step: int, alt: int):
            calls.append(step)
            return self.good

        short = [m[:-2] for m in self.margins]
        self.assertEqual(repair_position_divergence(self.staves, self.staves, short, fork), {})
        self.assertEqual(calls, [])

    def test_a_raw_sequence_with_different_measures_declines(self) -> None:
        calls = []

        def fork(staff_index: int, step: int, alt: int):
            calls.append(step)
            return self.good

        raw = list(self.staves)
        raw[1] = _staff(GOOD, GOOD)  # fewer measures than the filtered staff
        out = repair_position_divergence(
            self.staves, raw, [[(7, 0.5)] * len(s) for s in raw], fork
        )
        self.assertEqual(out, {})
        self.assertEqual(calls, [])

    def test_forking_stops_at_max_forks(self) -> None:
        calls = []

        def fork(staff_index: int, step: int, alt: int):
            calls.append(step)
            return None

        repair_position_divergence(self.staves, self.staves, self.margins, fork, max_forks=2)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
