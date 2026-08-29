import unittest

from homr.transformer.structured_notation import AdvanceClass

from homr.transformer.vocabulary import EncodedSymbol, empty
from training.omr_datasets.staff_merging import create_chord_over_two_staffs


class TestStaffMerging(unittest.TestCase):
    def test_chord_merging(self) -> None:

        result = create_chord_over_two_staffs(
            [
                EncodedSymbol("keySignature_0", empty, empty, empty, "upper"),
                EncodedSymbol("timeSignature/4"),
                EncodedSymbol("clef_G2", empty, empty, empty, "upper"),
                EncodedSymbol("note_16", "F4", empty, empty, "upper"),
                EncodedSymbol("clef_F4", empty, empty, empty, "lower"),
                EncodedSymbol("rest_4", empty, empty, empty, "upper"),
                EncodedSymbol("repeatStart"),
            ]
        )

        self.assertEqual(
            [r.rhythm for r in result],
            [
                "repeatStart",
                "clef_G2",
                "chord",
                "clef_F4",
                "keySignature_0",
                "timeSignature/4",
                "note_16",
                "chord",
                "rest_4",
            ],
        )


class TestAdvanceComputation(unittest.TestCase):
    """The onset-delta target: see docs/private/ONSET_REPRESENTATION_RESEARCH.md.

    Uses `merge_upper_and_lower_staff` end to end rather than the private helpers
    directly, so these tests fail if the wiring between them breaks even when each
    piece still passes in isolation.
    """

    def _note(self, rhythm: str) -> EncodedSymbol:
        from homr.transformer.structured_notation import NoteNotation, empty_beam_levels, empty_slur_slots

        notation = NoteNotation(
            beam_levels=empty_beam_levels(), stem="not_applicable", slurs=empty_slur_slots()
        )
        return EncodedSymbol(rhythm, "C4", notation=notation)

    def _advances(self, voices, divisions=24):
        from training.omr_datasets.staff_merging import merge_upper_and_lower_staff

        result = merge_upper_and_lower_staff(voices, divisions=divisions)
        notes = [s for s in result if s.rhythm.startswith(("note", "rest"))]
        return [(s.rhythm, str(s.notation.advance)) for s in notes]

    def test_min_rule_failure_case_from_the_research_report(self) -> None:
        """upper q,q / lower q.,8th: the min-duration rule gets the SECOND group wrong
        (says 1/4, true gap is 1/8) - this is the concrete example the research report
        used to argue the representation is not merely approximate but structurally
        wrong, and it is the case this head exists to fix."""
        from training.omr_datasets.staff_merging import EncodedSymbolWithPos

        upper = [
            EncodedSymbolWithPos(0, self._note("note_4")),
            EncodedSymbolWithPos(24, self._note("note_4")),
        ]
        lower = [
            EncodedSymbolWithPos(0, self._note("note_4.")),
            EncodedSymbolWithPos(36, self._note("note_8")),
        ]
        self.assertEqual(
            self._advances([upper, lower]),
            [
                ("note_4", "not_applicable"),  # upper q @0: not the group's LAST member
                ("note_4.", "4"),              # group @0's last member: true gap = 1/4
                ("note_4", "8"),               # group @24: true gap = 1/8, NOT 1/4
                ("note_8", "not_applicable"),  # last onset of the measure: no next group
            ],
        )

    def test_same_true_onset_advances_by_zero(self) -> None:
        from training.omr_datasets.staff_merging import EncodedSymbolWithPos

        voice = [
            EncodedSymbolWithPos(0, self._note("note_8G"), insert_before=True),
            EncodedSymbolWithPos(0, self._note("note_4")),
            EncodedSymbolWithPos(24, self._note("note_4")),
        ]
        self.assertEqual(
            self._advances([voice]),
            [("note_8G", "zero"), ("note_4", "4"), ("note_4", "not_applicable")],
        )

    def test_divisions_none_is_a_no_op(self) -> None:
        """Every existing caller that does not pass `divisions` must see no change at
        all - this is what makes adding the parameter safe to land without touching the
        kern converters that never learned it."""
        from training.omr_datasets.staff_merging import EncodedSymbolWithPos, merge_upper_and_lower_staff

        voice = [
            EncodedSymbolWithPos(0, self._note("note_4")),
            EncodedSymbolWithPos(24, self._note("note_4")),
        ]
        result = merge_upper_and_lower_staff([voice])
        notes = [s for s in result if s.rhythm.startswith(("note", "rest"))]
        self.assertTrue(all(str(s.notation.advance) == "not_applicable" for s in notes))

    def test_unquantizable_gap_falls_back_to_other(self) -> None:
        from training.omr_datasets.staff_merging import _quantize_advance

        self.assertEqual(str(_quantize_advance(5, 24)), "other")

    def test_quantize_advance_exact_matches(self) -> None:
        from training.omr_datasets.staff_merging import _quantize_advance

        self.assertEqual(str(_quantize_advance(0, 24)), "zero")
        self.assertEqual(str(_quantize_advance(24, 24)), "4")   # quarter
        self.assertEqual(str(_quantize_advance(12, 24)), "8")   # eighth
        self.assertEqual(str(_quantize_advance(96, 24)), "1")   # whole
        self.assertEqual(str(_quantize_advance(36, 24)), "4.")  # dotted quarter


class TestAdvanceFromOwnDuration(unittest.TestCase):
    """The kern path: no position tracking needed, a group's own stated duration IS the
    true advance, by kern's format guarantee (see `_advance_from_own_duration`'s
    docstring for the real GrandStaff example this is built from)."""

    def _note(self, rhythm: str) -> EncodedSymbol:
        from homr.transformer.structured_notation import NoteNotation, empty_beam_levels, empty_slur_slots

        notation = NoteNotation(
            beam_levels=empty_beam_levels(), stem="not_applicable", slurs=empty_slur_slots()
        )
        return EncodedSymbol(rhythm, "C4", notation=notation)

    def test_bass_sixteenths_under_a_treble_quarter(self) -> None:
        """The real GrandStaff shape: bass plays four 16ths in the time of one treble
        quarter. Every group's advance should read 1/16 - a new onset every 16th."""
        from training.omr_datasets.staff_merging import EncodedSymbolWithPos, merge_upper_and_lower_staff

        treble = [EncodedSymbolWithPos(0, self._note("note_4"))]
        bass = [EncodedSymbolWithPos(i, self._note("note_16")) for i in range(4)]
        result = merge_upper_and_lower_staff([treble, bass], advance_from_own_duration=True)
        notes = [s for s in result if s.rhythm.startswith(("note", "rest"))]
        # key=0 is a combined group (treble + bass): only its LAST member (bass, appended
        # second) carries the real target - the treble note is not the canonical member.
        self.assertEqual(
            [(s.rhythm, s.position, str(s.notation.advance)) for s in notes],
            [
                ("note_4", "upper", "not_applicable"),
                ("note_16", "lower", "16"),
                ("note_16", "lower", "16"),
                ("note_16", "lower", "16"),
                ("note_16", "lower", "16"),
            ],
        )

    def test_a_solo_line_gets_its_own_duration_back(self) -> None:
        from training.omr_datasets.staff_merging import EncodedSymbolWithPos, merge_upper_and_lower_staff

        voice = [
            EncodedSymbolWithPos(0, self._note("note_8")),
            EncodedSymbolWithPos(1, self._note("note_4.")),
        ]
        result = merge_upper_and_lower_staff([voice], advance_from_own_duration=True)
        notes = [s for s in result if s.rhythm.startswith(("note", "rest"))]
        self.assertEqual([str(s.notation.advance) for s in notes], ["8", "4."])

    def test_a_group_with_no_notes_is_not_applicable(self) -> None:
        from training.omr_datasets.staff_merging import _advance_from_own_duration

        self.assertEqual(
            _advance_from_own_duration([EncodedSymbol("clef_G2")]),
            AdvanceClass.NOT_APPLICABLE,
        )

    def test_divisions_and_advance_from_own_duration_are_mutually_exclusive(self) -> None:
        from training.omr_datasets.staff_merging import EncodedSymbolWithPos, merge_upper_and_lower_staff

        voice = [EncodedSymbolWithPos(0, self._note("note_4"))]
        with self.assertRaises(ValueError):
            merge_upper_and_lower_staff([voice], divisions=24, advance_from_own_duration=True)
