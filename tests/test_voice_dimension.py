"""The voice a note belongs to, from MusicXML through the sidecar to the repair passes.

Added because the representation carried `position` - upper or lower - and nothing else,
so two voices sharing a staff were indistinguishable. A tie joins one pitch within one
voice and a slur spans notes within one voice; flattened together, "the next chord on this
staff" is as likely to be the other voice's and the partner cannot be located.

Measured on the Lieder corpus, where 22% of sources carry a polyphonic staff: tie labels
are impossible on 8.8% of endpoints there against 0.0% in monophonic staves, and unpaired
slur endpoints run 13.3% against 1.6%.
"""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from homr.transformer.structured_notation import (
    MAX_VOICES_PER_STAFF,
    VOICE_CLASSES,
    NoteNotation,
    StemDirection,
    VoiceClass,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.notation_sidecar import (
    SCHEMA_VERSION,
    attach_sidecar,
    write_sidecar,
)
from training.omr_datasets.structured_notation_parser import NotationExtractor

TWO_VOICES = """<part><measure>
  <note><pitch><step>G</step><octave>4</octave></pitch><voice>1</voice><staff>1</staff></note>
  <note><pitch><step>C</step><octave>4</octave></pitch><voice>2</voice><staff>1</staff></note>
  <note><pitch><step>E</step><octave>3</octave></pitch><voice>5</voice><staff>2</staff></note>
  <note><pitch><step>C</step><octave>3</octave></pitch><voice>6</voice><staff>2</staff></note>
</measure></part>"""


def _extract(xml: str) -> list[VoiceClass]:
    extractor = NotationExtractor()
    return [extractor.extract(note).voice for note in ET.fromstring(xml).iter("note")]


class TestExtraction(unittest.TestCase):
    def test_voices_are_numbered_within_their_own_staff(self) -> None:
        """MusicXML numbers voices part-globally - 1-4 upper, 5-8 lower by convention.

        What matters for pairing is which line *within this staff*, so the lower staff's
        voices 5 and 6 are its first and second, not its fifth and sixth.
        """
        self.assertEqual(
            [VoiceClass.FIRST, VoiceClass.SECOND, VoiceClass.FIRST, VoiceClass.SECOND],
            _extract(TWO_VOICES),
        )

    def test_a_source_that_states_no_voice_says_unknown(self) -> None:
        """Not a claim that the note is in the first voice."""
        xml = (
            "<part><measure><note><pitch><step>G</step><octave>4</octave></pitch>"
            "</note></measure></part>"
        )
        self.assertEqual([VoiceClass.UNKNOWN], _extract(xml))

    def test_numbering_follows_first_appearance_not_sort_order(self) -> None:
        """A staff whose second voice enters first would otherwise have its lines swapped."""
        xml = """<part><measure>
          <note><pitch><step>C</step><octave>4</octave></pitch><voice>2</voice><staff>1</staff></note>
          <note><pitch><step>G</step><octave>4</octave></pitch><voice>1</voice><staff>1</staff></note>
        </measure></part>"""
        self.assertEqual([VoiceClass.FIRST, VoiceClass.SECOND], _extract(xml))

    def test_beyond_the_cap_is_unknown_rather_than_folded_in(self) -> None:
        """A wrong voice pairs a tie to the wrong note as surely as no voice at all."""
        notes = "".join(
            f"<note><pitch><step>G</step><octave>4</octave></pitch>"
            f"<voice>{n}</voice><staff>1</staff></note>"
            for n in range(1, MAX_VOICES_PER_STAFF + 2)
        )
        self.assertEqual(
            VoiceClass.UNKNOWN, _extract(f"<part><measure>{notes}</measure></part>")[-1]
        )


class TestTheSidecarCarriesIt(unittest.TestCase):
    def _symbol(self, voice: VoiceClass) -> EncodedSymbol:
        symbol = EncodedSymbol(rhythm="note_4", pitch="G4")
        symbol.notation = NoteNotation(beam_levels=(), stem=StemDirection.UP, slurs=(), voice=voice)
        return symbol

    def test_a_voice_survives_the_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tokens = Path(directory) / "x.tokens"
            tokens.write_text("note_4 G4 _ _ _ upper\nnote_4 C4 _ _ _ upper\n", encoding="utf-8")
            written = [self._symbol(VoiceClass.SECOND), self._symbol(VoiceClass.FIRST)]
            path = write_sidecar(tokens, written)
            self.assertIsNotNone(path)

            payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
            self.assertEqual(SCHEMA_VERSION, payload["schemaVersion"])
            self.assertEqual(["2", "1"], [r["voice"] for r in payload["notation"]])

            read_back = [self._symbol(VoiceClass.UNKNOWN), self._symbol(VoiceClass.UNKNOWN)]
            attach_sidecar(tokens, read_back)
            self.assertEqual(
                [VoiceClass.SECOND, VoiceClass.FIRST],
                [s.notation.voice for s in read_back],
            )

    def test_a_sidecar_written_before_voices_reads_as_unknown(self) -> None:
        """Every corpus on disk predates this, and must keep loading unchanged."""
        with tempfile.TemporaryDirectory() as directory:
            tokens = Path(directory) / "x.tokens"
            tokens.write_text("note_4 G4 _ _ _ upper\n", encoding="utf-8")
            path = Path(str(tokens) + ".notation.json")
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": "homr.notation-sidecar.v4",
                        "annotatedSymbols": 1,
                        "notation": [
                            {
                                "beams": [],
                                "stem": "up",
                                "slurs": [],
                                "tie": "none",
                                "dynamic": "none",
                                "advance": "not_applicable",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            symbols = [self._symbol(VoiceClass.SECOND)]
            attach_sidecar(tokens, symbols)
            self.assertEqual(VoiceClass.UNKNOWN, symbols[0].notation.voice)


class TestTheClassSet(unittest.TestCase):
    def test_unknown_is_not_a_predictable_class(self) -> None:
        """It marks a silent source; scoring it would teach silence as an answer - the
        same rule StemDirection.UNKNOWN and TieState.UNKNOWN already follow."""
        self.assertNotIn(VoiceClass.UNKNOWN, VOICE_CLASSES)
        self.assertEqual(MAX_VOICES_PER_STAFF, len(VOICE_CLASSES))


if __name__ == "__main__":
    unittest.main()


class TestTheOnsetIndex(unittest.TestCase):
    """Which simultaneity of its own voice a note belongs to.

    Recording the voice was not enough to recover adjacency, which was the point of
    recording it. A token line is a simultaneity across *all* voices - the converter
    merges whatever sounds together onto one line - so a voice's successive notes may
    share a line or sit several lines apart with other voices between. Measured on the
    rebuilt corpus with voices recorded and used: tie labels were still impossible on
    27.6% of endpoints in polyphonic files against 5.7% in monophonic ones.
    """

    CHORD_THEN_TWO_VOICES = """<part><measure>
      <note><pitch><step>G</step><octave>4</octave></pitch><voice>1</voice><staff>1</staff></note>
      <note><chord/><pitch><step>B</step><octave>4</octave></pitch><voice>1</voice><staff>1</staff></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><voice>1</voice><staff>1</staff></note>
      <backup><duration>4</duration></backup>
      <note><pitch><step>E</step><octave>3</octave></pitch><voice>2</voice><staff>1</staff></note>
      <note><pitch><step>F</step><octave>3</octave></pitch><voice>2</voice><staff>1</staff></note>
    </measure></part>"""

    def _indices(self, xml: str) -> list:
        extractor = NotationExtractor()
        return [extractor.extract(note).onset_index for note in ET.fromstring(xml).iter("note")]

    def test_a_chord_member_shares_its_chord_s_index(self) -> None:
        """So a member is never its own successor, which is what a tie search needs."""
        self.assertEqual([1, 1, 2, 1, 2], self._indices(self.CHORD_THEN_TWO_VOICES))

    def test_each_voice_counts_independently(self) -> None:
        """A backup returns to the same moment in another voice; both start at 1."""
        indices = self._indices(self.CHORD_THEN_TWO_VOICES)
        self.assertEqual(1, indices[0])
        self.assertEqual(1, indices[3])

    def test_each_staff_counts_independently(self) -> None:
        xml = """<part><measure>
          <note><pitch><step>G</step><octave>4</octave></pitch><voice>1</voice><staff>1</staff></note>
          <note><pitch><step>C</step><octave>3</octave></pitch><voice>5</voice><staff>2</staff></note>
          <note><pitch><step>A</step><octave>4</octave></pitch><voice>1</voice><staff>1</staff></note>
        </measure></part>"""
        self.assertEqual([1, 1, 2], self._indices(xml))

    def test_it_survives_the_sidecar_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tokens = Path(directory) / "x.tokens"
            tokens.write_text("note_4 G4 _ _ _ upper\nnote_4 C4 _ _ _ upper\n", encoding="utf-8")
            written = []
            for index in (3, 4):
                symbol = EncodedSymbol(rhythm="note_4", pitch="G4")
                symbol.notation = NoteNotation(
                    beam_levels=(), stem=StemDirection.UP, slurs=(), onset_index=index
                )
                written.append(symbol)
            write_sidecar(tokens, written)

            read_back = []
            for _ in range(2):
                symbol = EncodedSymbol(rhythm="note_4", pitch="G4")
                symbol.notation = NoteNotation(beam_levels=(), stem=StemDirection.UP, slurs=())
                read_back.append(symbol)
            attach_sidecar(tokens, read_back)
            self.assertEqual([3, 4], [s.notation.onset_index for s in read_back])

    def test_a_sidecar_written_before_v6_reads_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tokens = Path(directory) / "x.tokens"
            tokens.write_text("note_4 G4 _ _ _ upper\n", encoding="utf-8")
            Path(str(tokens) + ".notation.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": "homr.notation-sidecar.v5",
                        "annotatedSymbols": 1,
                        "notation": [
                            {
                                "beams": [],
                                "stem": "up",
                                "slurs": [],
                                "tie": "none",
                                "dynamic": "none",
                                "advance": "not_applicable",
                                "voice": "1",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            symbol = EncodedSymbol(rhythm="note_4", pitch="G4")
            symbol.notation = NoteNotation(
                beam_levels=(), stem=StemDirection.UP, slurs=(), onset_index=9
            )
            attach_sidecar(tokens, [symbol])
            self.assertIsNone(symbol.notation.onset_index)


class TestARestIsNeverTied(unittest.TestCase):
    """A rest is silence - there is nothing to sustain into the next note.

    25 of the rebuilt corpus's impossible tie labels were a `<tied>` element the source
    had attached to a rest, copied through because the extractor read `<tied>` from any
    note element at all. Such a label can never find a partner under any rule.
    """

    def _ties(self, xml: str) -> list:
        extractor = NotationExtractor()
        return [str(extractor.extract(note).tie) for note in ET.fromstring(xml).iter("note")]

    def test_a_tied_rest_records_no_tie(self) -> None:
        xml = """<part><measure>
          <note><rest/><voice>1</voice><staff>1</staff>
            <notations><tied type="start"/></notations></note>
        </measure></part>"""
        self.assertEqual(["none"], self._ties(xml))

    def test_a_pitched_note_still_records_its_tie(self) -> None:
        xml = """<part><measure>
          <note><pitch><step>G</step><octave>4</octave></pitch><voice>1</voice><staff>1</staff>
            <notations><tied type="start"/></notations></note>
        </measure></part>"""
        self.assertEqual(["start"], self._ties(xml))
