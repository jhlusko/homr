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


class TestGrandStaffReconstruction(unittest.TestCase):
    """`chord` is a SEPARATOR joining simultaneous symbols, so a grand staff cannot be
    split symbol by symbol: it carries no position and belongs to whichever staves its
    neighbours are on. Validated on the corpus - both reconstructed staves give a
    median bar of 3/4, matching genuine single staves, against 5/6 unsplit."""

    def _grand(self):
        return [
            n("clef_G2", "upper"), EncodedSymbol("chord"), n("clef_F4", "lower"),
            EncodedSymbol("keySignature_0"),
            n("note_4", "upper"), EncodedSymbol("chord"), n("note_4", "lower"),
            EncodedSymbol("barline"),
        ]

    def test_a_chord_spanning_both_staves_is_partitioned(self) -> None:
        from training.omr_datasets.audit_label_consistency import split_grand_staff
        upper, lower = split_grand_staff(self._grand())
        self.assertEqual([s.rhythm for s in upper if s.rhythm.startswith("note")], ["note_4"])
        self.assertEqual([s.rhythm for s in lower if s.rhythm.startswith("note")], ["note_4"])

    def test_system_wide_symbols_reach_both_staves(self) -> None:
        from training.omr_datasets.audit_label_consistency import split_grand_staff
        for staff in split_grand_staff(self._grand()):
            self.assertIn("keySignature_0", [s.rhythm for s in staff])
            self.assertIn("barline", [s.rhythm for s in staff])

    def test_each_staff_keeps_its_own_clef(self) -> None:
        from training.omr_datasets.audit_label_consistency import split_grand_staff
        upper, lower = split_grand_staff(self._grand())
        self.assertIn("clef_G2", [s.rhythm for s in upper])
        self.assertIn("clef_F4", [s.rhythm for s in lower])
        self.assertNotIn("clef_F4", [s.rhythm for s in upper])

    def test_a_split_staff_has_the_same_bar_length_as_the_music(self) -> None:
        """The acceptance test: an unsplit grand staff gives a nonsense duration
        because group_into_chords takes the minimum across a chord."""
        from training.omr_datasets.audit_label_consistency import (
            measure_durations, split_grand_staff)
        grand = [n("note_4", "upper"), EncodedSymbol("chord"), n("note_2", "lower"),
                 n("note_4", "upper"), EncodedSymbol("chord"), n("note_2", "lower"),
                 EncodedSymbol("barline")]
        upper, lower = split_grand_staff(grand)
        self.assertEqual(measure_durations(upper), [Fraction(1, 2)])
        self.assertEqual(measure_durations(lower), [Fraction(1)])

    def test_a_single_staff_passes_through_untouched(self) -> None:
        from training.omr_datasets.audit_label_consistency import split_grand_staff
        staff = bar(["note_4"] * 4)
        self.assertEqual(split_grand_staff(staff), [staff])


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
