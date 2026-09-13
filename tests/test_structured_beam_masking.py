"""The head's beam output only means something where the head was trained.

Training masks every beam level above a note's flag count, so at those levels the head
emits an unlearned projection. Before this was masked at decode, `build_beams` wrote a
`<beam>` for 99.2% of flagless notes in a 60-staff sample - half notes carrying beams.
"""

import unittest
import xml.etree.ElementTree as ET

from homr.music_xml_generator import build_beams
from homr.transformer.structured_decode import mask_untrained_beams
from homr.transformer.structured_notation import (
    AdvanceClass,
    BeamLevelState,
    DynamicMark,
    NoteNotation,
    StemDirection,
    TieState,
    written_flags,
)
from homr.transformer.vocabulary import EncodedSymbol

NOISE = (
    BeamLevelState.BEGIN,
    BeamLevelState.BEGIN,
    BeamLevelState.END,
    BeamLevelState.CONTINUE,
)


def notation(levels: tuple[BeamLevelState, ...] = NOISE) -> NoteNotation:
    return NoteNotation(
        beam_levels=levels,
        stem=StemDirection.UP,
        slurs=(),
        tie=TieState.NONE,
        dynamic=DynamicMark.NONE,
        advance=AdvanceClass.NOT_APPLICABLE,
    )


class TestWrittenFlags(unittest.TestCase):
    def test_counts_flags_from_the_written_value(self) -> None:
        self.assertEqual(written_flags("note_2"), 0)
        self.assertEqual(written_flags("note_4"), 0)
        self.assertEqual(written_flags("note_8"), 1)
        self.assertEqual(written_flags("note_16"), 2)
        self.assertEqual(written_flags("note_32"), 3)
        self.assertEqual(written_flags("rest_8"), 1)

    def test_a_triplet_eighth_carries_one_flag(self) -> None:
        """`note_12` lasts a third of a quarter and is still written with one flag."""
        self.assertEqual(written_flags("note_12"), 1)
        self.assertEqual(written_flags("note_24"), 2)

    def test_a_dot_does_not_change_the_flag_count(self) -> None:
        self.assertEqual(written_flags("note_16."), 2)

    def test_a_non_note_has_no_flags_at_all(self) -> None:
        for rhythm in ("barline", "clef_G2", "keySignature_-1", "timeSignature/4"):
            self.assertIsNone(written_flags(rhythm), rhythm)


class TestMaskUntrainedBeams(unittest.TestCase):
    def test_a_flagless_note_keeps_no_beam_level(self) -> None:
        masked = mask_untrained_beams(notation(), "note_2")
        self.assertEqual(set(masked.beam_levels), {BeamLevelState.NOT_APPLICABLE})

    def test_an_eighth_keeps_only_level_one(self) -> None:
        masked = mask_untrained_beams(notation(), "note_8")
        self.assertEqual(masked.beam_levels[0], BeamLevelState.BEGIN)
        self.assertEqual(set(masked.beam_levels[1:]), {BeamLevelState.NOT_APPLICABLE})

    def test_a_sixteenth_keeps_two_levels(self) -> None:
        masked = mask_untrained_beams(notation(), "note_16")
        self.assertEqual(masked.beam_levels[:2], NOISE[:2])
        self.assertEqual(set(masked.beam_levels[2:]), {BeamLevelState.NOT_APPLICABLE})

    def test_a_non_note_keeps_nothing(self) -> None:
        masked = mask_untrained_beams(notation(), "barline")
        self.assertEqual(set(masked.beam_levels), {BeamLevelState.NOT_APPLICABLE})

    def test_everything_applicable_is_returned_unchanged(self) -> None:
        original = notation()
        self.assertIs(mask_untrained_beams(original, "note_64"), original)


class TestNothingUndrawableReachesTheXml(unittest.TestCase):
    def _beams(self, rhythm: str) -> list[str]:
        symbol = EncodedSymbol(rhythm, "G4")
        symbol.notation = mask_untrained_beams(notation(), rhythm)
        element = ET.Element("note")
        build_beams(element, symbol)
        return [child.get("number") or "" for child in element.findall("beam")]

    def test_a_half_note_gets_no_beam_element(self) -> None:
        self.assertEqual(self._beams("note_2"), [])

    def test_an_eighth_gets_exactly_one(self) -> None:
        self.assertEqual(self._beams("note_8"), ["1"])

    def test_no_note_ever_gets_a_beam_above_its_flag_count(self) -> None:
        for rhythm in ("note_2", "note_4", "note_8", "note_12", "note_16", "note_32"):
            flags = written_flags(rhythm) or 0
            for number in self._beams(rhythm):
                self.assertLessEqual(int(number), flags, rhythm)


if __name__ == "__main__":
    unittest.main()
