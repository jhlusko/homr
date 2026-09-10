import unittest
import xml.etree.ElementTree as ET

from homr.music_xml_generator import BEAM_VALUES, build_beams
from homr.transformer.structured_notation import BeamLevelState, NoteNotation
from homr.transformer.vocabulary import EncodedSymbol


def _note_with(*levels: BeamLevelState) -> EncodedSymbol:
    symbol = EncodedSymbol("note_8", "C4", "_", "_", "_", "upper")
    symbol.notation = NoteNotation(
        beam_levels=tuple(levels),
        stem="up",
        slurs=(),
        tie="none",
        dynamic="none",
    )
    return symbol


def _beams(symbol: EncodedSymbol) -> list[tuple[str, str]]:
    note = ET.Element("note")
    build_beams(note, symbol)
    return [(b.get("number"), b.text) for b in note.findall("beam")]


class TestBuildBeams(unittest.TestCase):
    def test_a_begin_beam_is_written_at_level_one(self) -> None:
        self.assertEqual(_beams(_note_with(BeamLevelState.BEGIN)), [("1", "begin")])

    def test_levels_are_numbered_from_one(self) -> None:
        # MusicXML's own convention: level 1 is the eighth-note beam.
        beams = _beams(_note_with(BeamLevelState.BEGIN, BeamLevelState.BEGIN))

        self.assertEqual(beams, [("1", "begin"), ("2", "begin")])

    def test_hooks_use_musicxml_spelling_not_ours(self) -> None:
        # Our labels are snake_case; MusicXML wants a space. Writing "forward_hook"
        # produces a file MuseScore silently ignores.
        self.assertEqual(
            _beams(_note_with(BeamLevelState.FORWARD_HOOK)), [("1", "forward hook")]
        )
        self.assertEqual(
            _beams(_note_with(BeamLevelState.BACKWARD_HOOK)), [("1", "backward hook")]
        )

    def test_a_flag_writes_no_beam(self) -> None:
        # MusicXML has no element meaning "flagged"; absence is how it is expressed, and
        # writing anything here would assert a beam connection that is not there.
        self.assertEqual(_beams(_note_with(BeamLevelState.FLAG)), [])

    def test_a_not_applicable_level_writes_no_beam(self) -> None:
        self.assertEqual(_beams(_note_with(BeamLevelState.NOT_APPLICABLE)), [])

    def test_inapplicable_levels_do_not_shift_the_numbering_of_later_ones(self) -> None:
        # Level 2 must stay level 2 even when level 1 contributes no element, or a
        # 16th-note beam is written as an eighth-note beam.
        beams = _beams(_note_with(BeamLevelState.NOT_APPLICABLE, BeamLevelState.END))

        self.assertEqual(beams, [("2", "end")])

    def test_a_note_without_structured_labels_is_unchanged(self) -> None:
        # Checkpoints trained without the heads, and corpora with no sidecar, must emit
        # exactly what they did before.
        plain = EncodedSymbol("note_8", "C4", "_", "_", "_", "upper")

        self.assertEqual(_beams(plain), [])

    def test_every_connective_state_has_a_mapping(self) -> None:
        # A state with no entry is silently dropped, so a new one must not be able to
        # disappear unnoticed.
        connective = {
            BeamLevelState.BEGIN,
            BeamLevelState.CONTINUE,
            BeamLevelState.END,
            BeamLevelState.FORWARD_HOOK,
            BeamLevelState.BACKWARD_HOOK,
        }

        self.assertEqual({str(s) for s in connective}, set(BEAM_VALUES))

    def test_the_non_connective_states_are_deliberately_absent(self) -> None:
        self.assertNotIn(str(BeamLevelState.FLAG), BEAM_VALUES)
        self.assertNotIn(str(BeamLevelState.NOT_APPLICABLE), BEAM_VALUES)


if __name__ == "__main__":
    unittest.main()
