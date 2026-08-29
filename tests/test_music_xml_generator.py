# ruff: noqa: E501, S101

import unittest
import xml.etree.ElementTree as ET
from fractions import Fraction

from homr.music_xml_generator import (
    ConversionState,
    SymbolChord,
    XmlGeneratorArguments,
    build_note_chord,
    generate_xml,
    rebalance_measure_voices,
)
from homr.transformer.vocabulary import EncodedSymbol, nonote
from training.transformer.training_vocabulary import (
    read_token_lines,
)


def _notes(measure: ET.Element) -> list[ET.Element]:
    return [c for c in measure if c.tag == "note"]


def _pitch(note: ET.Element) -> str:
    p = note.find("pitch")
    if p is None:
        return "rest"
    step = p.findtext("step", "")
    return step


def _duration(note: ET.Element) -> int:
    d = note.findtext("duration")
    return int(d) if d is not None else 0


def _voice(note: ET.Element) -> str:
    return note.findtext("voice", "")


def _staff(note: ET.Element) -> str:
    return note.findtext("staff", "")


def _backups(measure: ET.Element) -> list[int]:
    return [int(c.findtext("duration", "0")) for c in measure if c.tag == "backup"]


def _first_measure(xml: ET.Element) -> ET.Element:
    part = xml.find("part")
    assert part is not None
    m = part.find("measure")
    assert m is not None
    return m


class TestMusicXmlGenerator(unittest.TestCase):
    """
    MusicXML testing is mostly covered by training/validate_music_xml_conversion.py
    This script requires that the data sets are downloaded and converted and uses
    the data sets to check that back and forth conversion works.
    """

    def test_chord_with_different_duratons(self) -> None:
        tabi_measure_18_upper = """clef_G2 . . . . upper
keySignature_4 . . . . .
timeSignature/8 . . . . .
note_4. G3 # _ _ upper &note_4. C4 # _ _ upper&note_16 E4 # _ _ upper
note_16 F4 # _ _ upper
note_4 E4 # _ _ upper
note_8 E4 # _ _ upper
note_8 C4 # _ _ upper
note_8 D4 # _ _ upper
barline . . . . ."""
        tokens = read_token_lines(tabi_measure_18_upper.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        notes = _notes(measure)
        backups = _backups(measure)

        # Pitches in order after rebalancing
        pitches = [_pitch(n) for n in notes]
        self.assertIn("E", pitches)
        self.assertIn("G", pitches)
        self.assertIn("F", pitches)
        self.assertIn("D", pitches)

        # There must be backups due to chord with different durations
        self.assertGreater(len(backups), 0)

        # All notes have a voice and staff assigned
        for note in notes:
            self.assertNotEqual(_voice(note), "")
            self.assertEqual(_staff(note), "1")

    def test_grand_staff_generation(self) -> None:
        grandstaff = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_1 . . . . .
timeSignature/4 . . . . .
note_1 G4 _ _ _ upper&note_1 A3 # _ _ upper&rest_2 _ _ _ _ upper&note_4 G3 _ _ _ lower
rest_4 _ _ _ _ lower
note_2 E4 _ _ _ upper&note_2 C2 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(grandstaff.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        notes = _notes(measure)

        # Both staves must be present
        staves = {_staff(n) for n in notes}
        self.assertIn("1", staves)
        self.assertIn("2", staves)

        # Upper staff notes: G4, A3, rest, E4; lower: G3, rest, C2
        pitches_upper = [_pitch(n) for n in notes if _staff(n) == "1"]
        pitches_lower = [_pitch(n) for n in notes if _staff(n) == "2"]
        self.assertIn("G", pitches_upper)
        self.assertIn("E", pitches_upper)
        self.assertIn("G", pitches_lower)
        self.assertIn("C", pitches_lower)

        # Upper voices are 1-4, lower voices are 5-8
        for note in notes:
            v = int(_voice(note))
            s = int(_staff(note))
            if s == 1:
                self.assertLessEqual(v, 4)
            else:
                self.assertGreaterEqual(v, 5)

    def test_begin_chord_with_standalone_rests(self) -> None:
        """
        If the lower position consists of a standalone rest then start the
        chord with this. That fixes an issue where the upper position
        consists of tuplets because in that case backups must not be used.

        See tabi.jpg measure 9 for an example.
        """
        chord = SymbolChord(
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("rest_8", position="lower"),
            ]
        )
        first, second = chord.into_positions()

        self.assertEqual(first.symbols, [EncodedSymbol("rest_8", position="lower")])
        self.assertEqual(
            second.symbols,
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
            ],
        )

    def test_a_zero_duration_rest_is_dropped_not_a_crash(self) -> None:
        """A grace note with no real pitch (rhythm and pitch heads disagreed: the
        rhythm says "note", the pitch field is still the "doesn't apply" sentinel)
        groups with rests, but every grace note is unconditionally keyed to
        Fraction(0) duration - a rest that takes zero time cannot be written as
        `<note><rest/></note>` plus a zero-duration backup. This used to be an
        unconditional `assert group_duration > Fraction(0)` that crashed the whole
        page (27.9a: `music_xml_generator.py:655`) for one malformed symbol.
        """
        chord = SymbolChord([EncodedSymbol("note_16G", pitch=nonote)])

        result = build_note_chord(chord, ConversionState(4, Fraction(1)), Fraction(0))

        self.assertEqual(result, [])

    def test_a_zero_duration_rest_does_not_crash_the_whole_pipeline(self) -> None:
        measure_with_a_malformed_grace_note = """clef_G2 . . . . upper
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_16G . . . . upper
note_4 C4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(measure_with_a_malformed_grace_note.splitlines())

        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        measure = _first_measure(xml)
        pitches = [_pitch(n) for n in _notes(measure)]
        self.assertIn("C", pitches)

    def test_rebalance_measure_voices_assigns_stable_voices_per_staff(self) -> None:
        measure = ET.Element("measure")

        note1 = self._build_test_note(duration=4, staff=1, voice=1)
        measure.append(note1)
        measure.append(self._build_test_backup(duration=4))

        note2 = self._build_test_note(duration=2, staff=1, voice=1)
        measure.append(note2)

        note3 = self._build_test_note(duration=2, staff=1, voice=1)
        measure.append(note3)

        note4 = self._build_test_note(duration=2, staff=1, voice=1, is_chord=True)
        measure.append(note4)

        measure.append(self._build_test_backup(duration=4))
        note5 = self._build_test_note(duration=4, staff=2, voice=1)
        measure.append(note5)
        measure.append(self._build_test_backup(duration=4))

        note6 = self._build_test_note(duration=2, staff=2, voice=1)
        measure.append(note6)

        rebalance_measure_voices(measure)

        self.assertEqual(self._read_note_voice(note1), "2")
        self.assertEqual(self._read_note_voice(note2), "1")
        self.assertEqual(self._read_note_voice(note3), "1")
        self.assertEqual(self._read_note_voice(note4), "1")
        self.assertEqual(self._read_note_voice(note5), "6")
        self.assertEqual(self._read_note_voice(note6), "5")

    def _build_test_note(
        self, duration: int, staff: int, voice: int, is_chord: bool = False
    ) -> ET.Element:
        note = ET.Element("note")
        if is_chord:
            ET.SubElement(note, "chord")
        ET.SubElement(note, "duration").text = str(duration)
        ET.SubElement(note, "staff").text = str(staff)
        ET.SubElement(note, "voice").text = str(voice)
        return note

    def _build_test_backup(self, duration: int) -> ET.Element:
        backup = ET.Element("backup")
        ET.SubElement(backup, "duration").text = str(duration)
        return backup

    def _read_note_voice(self, note: ET.Element) -> str:
        v = note.findtext("voice")
        self.assertIsNotNone(v)
        return str(v)


class TestTupletParserAcrossInterleavedVoices(unittest.TestCase):
    """A grand-staff measure interleaves both hands' groups by onset
    (group_into_chords), so a hand NOT in a tuplet can land its own extra onset in the
    middle of the tuplet's span. That group must not break the bracket - see
    TupletParser.add_tuplets' docstring, and roundtrip_fidelity.py's real Lieder finding
    (note_12 reading back as note_8) that motivated this fix."""

    def _note(self, rhythm: str, position: str = "upper") -> EncodedSymbol:
        return EncodedSymbol(rhythm, "C4", nonote, nonote, nonote, position)

    def test_an_unrelated_group_inside_the_span_does_not_break_the_bracket(self) -> None:
        from homr.music_xml_generator import TupletParser

        groups = [
            SymbolChord([self._note("note_12")]),   # triplet note 1
            SymbolChord([self._note("note_4", "lower")]),  # other hand, no tuplet shape
            SymbolChord([self._note("note_12")]),   # triplet note 2
            SymbolChord([self._note("note_12")]),   # triplet note 3
        ]
        self.assertTrue(TupletParser.add_tuplets(groups))
        self.assertEqual(groups[0].tuplet_mark, "start")
        self.assertEqual(groups[1].tuplet_mark, "")
        self.assertEqual(groups[2].tuplet_mark, "")
        self.assertEqual(groups[3].tuplet_mark, "stop")

    def test_a_genuinely_mismatched_ratio_still_fails(self) -> None:
        from homr.music_xml_generator import TupletParser

        groups = [
            SymbolChord([self._note("note_12")]),  # 3:2 triplet
            SymbolChord([self._note("note_20")]),  # 5:4 quintuplet - a real mismatch
            SymbolChord([self._note("note_12")]),
        ]
        self.assertFalse(TupletParser.add_tuplets(groups))

    def test_running_out_of_groups_still_fails(self) -> None:
        from homr.music_xml_generator import TupletParser

        groups = [
            SymbolChord([self._note("note_12")]),
            SymbolChord([self._note("note_4", "lower")]),
        ]
        self.assertFalse(TupletParser.add_tuplets(groups))


class TestMultiCharacterClefSign(unittest.TestCase):
    """`clef_TAB5` is the only vocabulary clef whose sign is more than one letter.

    Splitting the token by character position wrote `<sign>T</sign><line>A</line>`, which
    is not a MusicXML clef and reparses as `clef_TA` - so every TAB staff homr renders,
    from a prediction or from ground truth, came out silently wrong.  PDMX carries enough
    of them that a 50-file roundtrip sample hit it (roundtrip_fidelity_corpora.py).
    """

    def _clef(self, rhythm: str) -> ET.Element:
        tokens = read_token_lines(
            [f"{rhythm} . . . . upper", "keySignature_0 . . . . .", "note_4 C4 _ _ _ upper"]
        )
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        clef = xml.find(".//clef")
        assert clef is not None
        return clef

    def test_tab_clef_keeps_its_whole_sign(self) -> None:
        clef = self._clef("clef_TAB5")
        self.assertEqual(clef.findtext("sign"), "TAB")
        self.assertEqual(clef.findtext("line"), "5")

    def test_single_letter_clefs_are_unchanged(self) -> None:
        for rhythm, sign, line in (("clef_G2", "G", "2"), ("clef_F4", "F", "4")):
            clef = self._clef(rhythm)
            self.assertEqual(clef.findtext("sign"), sign)
            self.assertEqual(clef.findtext("line"), line)


class TestTrailingRepeatDoesNotGrowAPhantomMeasure(unittest.TestCase):
    """A token stream ending on a bare repeatStart - a real crop-boundary shape, the
    source cut right where a repeat begins - used to render an extra empty measure
    holding only a forward-repeat barline on its RIGHT edge, a shape real engraving
    never produces. Confirmed on real PDMX crops via roundtrip_fidelity_corpora.py
    (1.6% of mismatched crops)."""

    def test_trailing_repeat_start_grows_no_extra_measure(self) -> None:
        tokens = """clef_G2 . . . . upper
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper
note_4 D4 _ _ _ upper
note_4 E4 _ _ _ upper
note_4 F4 _ _ _ upper
repeatStart . . . . ."""
        tokens_parsed = read_token_lines(tokens.splitlines())

        xml = generate_xml(XmlGeneratorArguments(), [tokens_parsed], "")

        measures = xml.find("part").findall("measure")
        self.assertEqual(len(measures), 1)
        self.assertEqual(len(_notes(measures[0])), 4)

    def test_trailing_repeat_start_is_not_lost(self) -> None:
        # No extra measure grows (above), but the repeat mark itself must not vanish
        # either - it attaches to the one real measure instead. Found via
        # roundtrip_fidelity.py on real Lieder ground truth (IMSLP16883): dropping the
        # phantom measure was silently dropping this token too.
        tokens = """clef_G2 . . . . upper
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper
note_4 D4 _ _ _ upper
note_4 E4 _ _ _ upper
note_4 F4 _ _ _ upper
repeatStart . . . . ."""
        tokens_parsed = read_token_lines(tokens.splitlines())

        xml = generate_xml(XmlGeneratorArguments(), [tokens_parsed], "")

        measures = xml.find("part").findall("measure")
        self.assertEqual(len(measures), 1)
        repeats = measures[0].findall(".//repeat")
        self.assertEqual(len(repeats), 1)
        self.assertEqual(repeats[0].get("direction"), "forward")

    def test_an_ordinary_final_barline_still_grows_no_extra_measure(self) -> None:
        # The common case this change must not touch: a normal ending already left the
        # post-loop current_measure genuinely empty (0 children), not barline-only.
        tokens = """clef_G2 . . . . upper
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper
note_4 D4 _ _ _ upper
note_4 E4 _ _ _ upper
note_4 F4 _ _ _ upper
barline . . . . ."""
        tokens_parsed = read_token_lines(tokens.splitlines())

        xml = generate_xml(XmlGeneratorArguments(), [tokens_parsed], "")

        measures = xml.find("part").findall("measure")
        self.assertEqual(len(measures), 1)

    def test_a_trailing_measure_with_real_content_is_kept(self) -> None:
        # A repeatStart followed by a real note must still get its own measure - only a
        # BARLINE-ONLY trailing measure is dropped.
        tokens = """clef_G2 . . . . upper
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper
repeatStart . . . . .
note_4 D4 _ _ _ upper"""
        tokens_parsed = read_token_lines(tokens.splitlines())

        xml = generate_xml(XmlGeneratorArguments(), [tokens_parsed], "")

        measures = xml.find("part").findall("measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual(len(_notes(measures[1])), 1)
