import unittest
from fractions import Fraction

from homr.cross_staff_consistency import analyze_system
from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.audit_label_consistency import (
    OVERFULL_RATIO,
    measure_durations,
    overfull_bars,
)


def n(rhythm, position="upper"):
    return EncodedSymbol(rhythm, "C4", "_", "_", "_", position)


def bar(rhythms, position="upper"):
    return [n(r, position) for r in rhythms] + [EncodedSymbol("barline")]


class TestOverfullBars(unittest.TestCase):
    """An implied tuplet - engraved with no bracket and no numeral - is recorded by
    neither the transcription nor the model, so both write plain note values and the
    bar comes out OVERFULL. Cross-staff checks cannot see it: both staves are equally
    overfull."""

    def test_a_bar_longer_than_the_prevailing_one_is_flagged(self) -> None:
        staff = bar(["note_4"] * 4) + bar(["note_4"] * 4) + bar(["note_4"] * 6) \
              + bar(["note_4"] * 4)
        self.assertEqual(overfull_bars(staff), [2])

    def test_a_short_bar_is_never_flagged(self) -> None:
        """Pickups and final bars are short, not long, and are ordinary."""
        staff = bar(["note_4"] * 2) + bar(["note_4"] * 4) + bar(["note_4"] * 4) \
              + bar(["note_4"] * 4)
        self.assertEqual(overfull_bars(staff), [])

    def test_a_uniform_staff_is_silent(self) -> None:
        self.assertEqual(overfull_bars(bar(["note_4"] * 4) * 4), [])

    def test_too_few_bars_to_know_what_prevailing_means(self) -> None:
        """With two bars an overfull one would define the norm and hide itself."""
        self.assertEqual(overfull_bars(bar(["note_4"] * 4) + bar(["note_4"] * 8)), [])

    def test_the_ratio_leaves_headroom(self) -> None:
        self.assertGreater(OVERFULL_RATIO, 1)

    def test_measure_durations_are_in_whole_notes(self) -> None:
        self.assertEqual(measure_durations(bar(["note_4"] * 4)), [Fraction(1)])


class TestCrossStaffOnLabels(unittest.TestCase):
    """The reviewer's case: one bar where the treble carries a dotted half and the bass
    a plain half. check_measure_durations compares MEDIANS for robustness and cannot
    see a single divergent bar; check_barline_positions can, and says so in its own
    docstring."""

    def _system(self):
        upper = bar(["note_4"] * 4, "upper") + bar(["note_2."], "upper") \
              + bar(["note_4"] * 4, "upper")
        lower = bar(["note_4"] * 4, "lower") + bar(["note_2"], "lower") \
              + bar(["note_4"] * 4, "lower")
        return [upper, lower]

    def test_a_single_divergent_bar_is_caught(self) -> None:
        kinds = {f.kind for f in analyze_system(self._system())}
        self.assertIn("barline_position_mismatch", kinds, kinds)

    def test_agreeing_staves_produce_nothing(self) -> None:
        staff = bar(["note_4"] * 4, "upper") + bar(["note_2."], "upper")
        other = bar(["note_4"] * 4, "lower") + bar(["note_2."], "lower")
        duration_findings = [
            f for f in analyze_system([staff, other])
            if "duration" in f.kind or "barline_position" in f.kind
        ]
        self.assertEqual(duration_findings, [])


if __name__ == "__main__":
    unittest.main()


class TestSingleStaffOnly(unittest.TestCase):
    """Neither shaping of a grand staff gives usable durations: left whole, chord
    grouping takes a minimum and the median bar comes out 13/16 against 3/4; split by
    position, symbols carrying no position belong to neither half and the mismatch rate
    reads 96%. Both are artefacts, so grand staves are excluded and counted."""

    def test_a_single_staff_voice_qualifies(self) -> None:
        from training.omr_datasets.audit_label_consistency import is_single_staff
        self.assertTrue(is_single_staff(bar(["note_4"] * 4)))

    def test_a_grand_staff_voice_does_not(self) -> None:
        from training.omr_datasets.audit_label_consistency import is_single_staff
        grand = [n("note_4", "upper"), n("note_2", "lower"), EncodedSymbol("barline")]
        self.assertFalse(is_single_staff(grand))


if __name__ == "__main__":
    unittest.main()
