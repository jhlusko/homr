import unittest
import xml.etree.ElementTree as ET

from homr.music_xml_generator import (
    ConversionState,
    XmlGeneratorArguments,
    generate_xml,
    slur_slot_number,
)
from homr.transformer.structured_notation import (
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    empty_slur_slots,
)
from homr.transformer.vocabulary import EncodedSymbol
from fractions import Fraction


def note(pitch, position, slur="_", notation=None):
    return EncodedSymbol("note_4", pitch, "_", "_", slur, position, notation=notation)


def slur_pairs(symbols):
    xml = generate_xml(XmlGeneratorArguments(None, None, None), [symbols], "")
    out = []
    for n in xml.iter("note"):
        for s in n.iter("slur"):
            out.append((n.findtext("staff"), s.get("type"), s.get("number")))
    return out


CLEFS = [
    EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper"),
    EncodedSymbol("clef_F4", "_", "_", "_", "_", "lower"),
]


class TestCrossStaffSlur(unittest.TestCase):
    """MusicXML pairs slurs by their `number`. Using the STAFF number as that id meant
    a slur starting on the upper staff and ending on the lower emitted number=1 against
    number=2 - an unpaired start and an unpaired stop, silently dropped."""

    def test_a_slur_across_two_staves_shares_one_number(self) -> None:
        syms = CLEFS + [
            note("C5", "upper", "slurStart"),
            note("G4", "upper"),
            note("E3", "lower"),
            note("C3", "lower", "slurStop"),
            EncodedSymbol("barline"),
        ]
        pairs = slur_pairs(syms)
        self.assertEqual([p[1] for p in pairs], ["start", "stop"])
        self.assertNotEqual(pairs[0][0], pairs[1][0], "the endpoints must be on different staves")
        self.assertEqual(pairs[0][2], pairs[1][2], "start and stop must share a number")

    def test_two_overlapping_slurs_on_one_staff_get_different_numbers(self) -> None:
        """The same collision broke same-staff overlaps: both took that staff's number."""
        syms = CLEFS + [
            note("C5", "upper", "slurStart"),
            note("D5", "upper", "slurStart"),
            note("E5", "upper", "slurStop"),
            note("F5", "upper", "slurStop"),
            EncodedSymbol("barline"),
        ]
        starts = [p[2] for p in slur_pairs(syms) if p[1] == "start"]
        self.assertEqual(len(set(starts)), 2, starts)


class TestSidecarSlots(unittest.TestCase):
    """The sidecar keeps each concurrent span in its own canonical slot, so the slot
    index is the pairing information - strictly better than inferring it from
    open/close order, which cannot tell two concurrent slurs apart."""

    def _notation(self, slot, event):
        slots = list(empty_slur_slots())
        slots[slot] = (event, SlurSide.UNSPECIFIED)
        return NoteNotation(beam_levels=(), stem=StemDirection.NOT_APPLICABLE, slurs=tuple(slots))

    def test_the_slot_index_becomes_the_slur_number(self) -> None:
        n = note("C5", "upper", "slurStart", self._notation(1, SlurEvent.START))
        self.assertEqual(slur_slot_number(n, "start"), 2)

    def test_slot_zero_is_number_one(self) -> None:
        n = note("C5", "upper", "slurStart", self._notation(0, SlurEvent.START))
        self.assertEqual(slur_slot_number(n, "start"), 1)

    def test_no_sidecar_means_fall_back(self) -> None:
        self.assertIsNone(slur_slot_number(note("C5", "upper", "slurStart"), "start"))

    def test_a_stop_does_not_match_a_start_slot(self) -> None:
        n = note("C5", "upper", "slurStop", self._notation(0, SlurEvent.START))
        self.assertIsNone(slur_slot_number(n, "stop"))


class TestOpenSpanStack(unittest.TestCase):
    def test_numbers_are_reused_once_freed(self) -> None:
        state = ConversionState(4, Fraction(1))
        first = state.open_slur()
        state.close_slur()
        self.assertEqual(state.open_slur(), first)

    def test_concurrent_spans_take_distinct_numbers(self) -> None:
        state = ConversionState(4, Fraction(1))
        self.assertNotEqual(state.open_slur(), state.open_slur())

    def test_a_stop_with_nothing_open_still_emits(self) -> None:
        """Dropping the element would make the defect vanish from the output instead
        of staying visible in it."""
        self.assertEqual(ConversionState(4, Fraction(1)).close_slur(), 1)


if __name__ == "__main__":
    unittest.main()
