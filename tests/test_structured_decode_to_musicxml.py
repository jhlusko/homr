"""Logits in, `<beam>` out - the seam that was missing in production.

The heads scored 0.9508 exact beam vector and the generator could write `<beam>`, but
nothing joined them: `generate()` never ran the heads and `Note.notation` stayed None, so
`build_beams` returned early on every note. These tests pin the join, so a regression
shows up as a failure here rather than as scores that quietly carry MuseScore's
auto-beaming again.
"""

import math
import unittest
import xml.etree.ElementTree as ET

from homr.music_xml_generator import build_beams
from homr.transformer.structured_decode import decode_note
from homr.transformer.structured_notation import (
    BEAM_LEVEL_CLASSES,
    SLUR_EVENT_CLASSES,
    STEM_CLASSES,
    BeamLevelState,
    SlurEvent,
    StemDirection,
)
from homr.transformer.vocabulary import EncodedSymbol


def peaked(classes, winner, mass: float = 0.99) -> list[float]:
    index = list(classes).index(winner)
    rest = (1.0 - mass) / (len(classes) - 1)
    return [math.log(mass if i == index else rest) for i in range(len(classes))]


def beams_from(logits: dict[str, list[float]]) -> list[tuple[str, str]]:
    """The whole chain: head logits -> notation -> symbol -> MusicXML `<beam>`."""
    symbol = EncodedSymbol("note_8", "C4", "_", "_", "_", "upper")
    symbol.notation = decode_note(logits).notation
    note = ET.Element("note")
    build_beams(note, symbol)
    return [(element.get("number"), element.text) for element in note.findall("beam")]


class TestLogitsReachMusicXml(unittest.TestCase):
    def test_a_predicted_begin_becomes_a_beam_element(self) -> None:
        beams = beams_from({"beam.level.1": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.BEGIN)})

        self.assertEqual(beams, [("1", "begin")])

    def test_two_levels_produce_two_numbered_beams(self) -> None:
        beams = beams_from(
            {
                "beam.level.1": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.CONTINUE),
                "beam.level.2": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.BEGIN),
            }
        )

        self.assertEqual(beams, [("1", "continue"), ("2", "begin")])

    def test_a_hook_survives_the_whole_chain(self) -> None:
        # Hooks are BeamLevelState values rather than a separate head, and MusicXML
        # spells them with a space ("forward hook"). Both translations happen in
        # different modules, so this is the test that they agree.
        beams = beams_from(
            {"beam.level.1": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.FORWARD_HOOK)}
        )

        self.assertEqual(beams, [("1", "forward hook")])

    def test_not_applicable_writes_no_beam(self) -> None:
        beams = beams_from(
            {"beam.level.1": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.NOT_APPLICABLE)}
        )

        self.assertEqual(beams, [])

    def test_a_flag_writes_no_beam(self) -> None:
        # A flag is a visual level with no neighbour to join; MusicXML has no element
        # for it. Emitting one would beam a note to nothing.
        beams = beams_from({"beam.level.1": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.FLAG)})

        self.assertEqual(beams, [])

    def test_an_uncertain_prediction_still_writes_its_best_guess(self) -> None:
        # Uncertainty changes what the UI offers, never what the score contains. A note
        # with no beam at all would be worse output than a beam the user can revise.
        split = [math.log(1e-6)] * len(BEAM_LEVEL_CLASSES)
        split[list(BEAM_LEVEL_CLASSES).index(BeamLevelState.END)] = math.log(0.52)
        split[list(BEAM_LEVEL_CLASSES).index(BeamLevelState.CONTINUE)] = math.log(0.48)

        prediction = decode_note({"beam.level.1": split})

        self.assertTrue(prediction.uncertain_choices())
        self.assertEqual(beams_from({"beam.level.1": split}), [("1", "end")])

    def test_a_note_without_heads_writes_no_beams(self) -> None:
        # The pre-wiring behaviour, kept deliberately: a checkpoint without the heads
        # must still produce a valid score.
        symbol = EncodedSymbol("note_8", "C4", "_", "_", "_", "upper")
        note = ET.Element("note")

        build_beams(note, symbol)

        self.assertEqual(note.findall("beam"), [])


class TestNonBeamHeadsAlsoArrive(unittest.TestCase):
    def test_the_stem_is_carried_even_though_it_is_not_offered(self) -> None:
        # Not offered as a user choice, but still written - the distinction the whole
        # OFFERED_HEADS policy rests on.
        prediction = decode_note({"stem.direction": peaked(STEM_CLASSES, StemDirection.DOWN)})

        self.assertEqual(prediction.notation.stem, StemDirection.DOWN)
        self.assertEqual(prediction.uncertain_choices(), ())

    def test_slur_events_are_carried(self) -> None:
        prediction = decode_note(
            {"slur.slot.1.event": peaked(SLUR_EVENT_CLASSES, SlurEvent.STOP)}
        )

        self.assertEqual(prediction.notation.slurs[0][0], SlurEvent.STOP)


if __name__ == "__main__":
    unittest.main()
