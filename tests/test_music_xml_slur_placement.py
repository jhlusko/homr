"""`build_slurs` writing a slur's placement, not just whether one starts or stops.

Before this, the structured heads' slur-side prediction (0.9094 macro F1 - a real,
validated capability) never reached the generated MusicXML at all: `build_slurs` only
ever read the base six-branch `slur` string ("slurStart"/"slurStop") and wrote a bare
`<slur type="start"/>`, with no `placement` attribute in any code path. Exactly the same
shape as the earlier beam bug this project already found and fixed - a head trained,
evaluated, and silently discarded on the way to MusicXML.
"""

import unittest
import xml.etree.ElementTree as ET

from homr.music_xml_generator import build_slurs
from homr.transformer.structured_notation import NoteNotation, SlurEvent, SlurSide
from homr.transformer.vocabulary import EncodedSymbol


def _note_with(slur: str, notation: NoteNotation | None) -> EncodedSymbol:
    symbol = EncodedSymbol("note_8", "C4", "_", "_", slur, "upper")
    symbol.notation = notation
    return symbol


def _slurs(symbol: EncodedSymbol, slur_number: int = 1) -> list[ET.Element]:
    note = ET.Element("note")
    build_slurs(note, symbol, slur_number)
    return note.find("notations").findall("slur")


def _notation(*slots: tuple[SlurEvent, SlurSide]) -> NoteNotation:
    return NoteNotation(beam_levels=(), stem="up", slurs=tuple(slots))


class TestSlurPlacement(unittest.TestCase):
    def test_a_start_gets_its_predicted_placement(self) -> None:
        symbol = _note_with(
            "slurStart", _notation((SlurEvent.START, SlurSide.BELOW))
        )

        slurs = _slurs(symbol)

        self.assertEqual(len(slurs), 1)
        self.assertEqual(slurs[0].get("type"), "start")
        self.assertEqual(slurs[0].get("placement"), "below")

    def test_a_stop_gets_its_predicted_placement(self) -> None:
        symbol = _note_with(
            "slurStop", _notation((SlurEvent.STOP, SlurSide.ABOVE))
        )

        slurs = _slurs(symbol)

        self.assertEqual(slurs[0].get("type"), "stop")
        self.assertEqual(slurs[0].get("placement"), "above")

    def test_unspecified_side_writes_no_placement_attribute(self) -> None:
        # Unspecified is the common case (most slurs in the corpus carry no explicit
        # placement) - writing a fabricated one would assert something the source
        # never said.
        symbol = _note_with(
            "slurStart", _notation((SlurEvent.START, SlurSide.UNSPECIFIED))
        )

        slurs = _slurs(symbol)

        self.assertIsNone(slurs[0].get("placement"))

    def test_no_heads_means_no_placement_attribute(self) -> None:
        # A checkpoint without the heads must keep producing exactly the old output -
        # a bare <slur> with no placement, not an error.
        symbol = _note_with("slurStart", None)

        slurs = _slurs(symbol)

        self.assertEqual(len(slurs), 1)
        self.assertIsNone(slurs[0].get("placement"))

    def test_start_and_stop_matches_each_element_by_its_own_event(self) -> None:
        # One note closing one span and opening another writes two <slur> elements from
        # a single base-branch token. Slot order is not guaranteed to match write order,
        # so each element must be matched by its own event value, not by position.
        symbol = _note_with(
            "slurStart_slurStop",
            _notation(
                (SlurEvent.STOP, SlurSide.BELOW),
                (SlurEvent.START, SlurSide.ABOVE),
            ),
        )

        slurs = _slurs(symbol)

        self.assertEqual(len(slurs), 2)
        by_type = {s.get("type"): s.get("placement") for s in slurs}
        self.assertEqual(by_type, {"stop": "below", "start": "above"})

    def test_start_and_stop_event_answers_to_both_written_elements(self) -> None:
        # A single slot can legitimately carry START_AND_STOP - one canonical slot
        # closing and reopening at the same note - and should supply placement to
        # whichever element is being written, not just one of them.
        symbol = _note_with(
            "slurStart_slurStop",
            _notation((SlurEvent.START_AND_STOP, SlurSide.BELOW)),
        )

        slurs = _slurs(symbol)

        by_type = {s.get("type"): s.get("placement") for s in slurs}
        self.assertEqual(by_type, {"stop": "below", "start": "below"})

    def test_no_slur_writes_nothing(self) -> None:
        symbol = _note_with("_", _notation((SlurEvent.NONE, SlurSide.BELOW)))

        self.assertEqual(_slurs(symbol), [])

    def test_the_slur_number_is_still_written(self) -> None:
        symbol = _note_with("slurStart", None)

        slurs = _slurs(symbol, slur_number=2)

        self.assertEqual(slurs[0].get("number"), "2")


if __name__ == "__main__":
    unittest.main()
