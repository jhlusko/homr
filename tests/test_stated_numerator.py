"""A stated numerator the label's own bars contradict must not be printed.

`timeSignatureBeats_n` is metadata and can go stale: IMSLP405017 changes metre
mid-score and the measure cutter carried the earlier 4 forward, so a system whose every
bar holds exactly three quarters states 4. Because a stated numerator won over an
inferred one, the review page rendered 4/4 over music plainly in 3 and a reviewer spent
their attention on a contradiction that was not in the music.

The override is deliberately narrow, and the second test is the one that matters: a
real metre change must still render its stated numerator, because there the label is
right and the rule would otherwise contradict it.
"""

import unittest
import xml.etree.ElementTree as ET
from fractions import Fraction

from homr.music_xml_generator import (
    XmlGeneratorArguments,
    add_tuplet_start_stop,
    generate_xml,
    group_into_chords,
    modal_measure_duration,
)
from homr.transformer.vocabulary import EncodedSymbol


def symbols(spec: str) -> list[EncodedSymbol]:
    """`rhythm` or `rhythm@pitch` - a note needs a pitch and a staff position or the
    generator has nothing to place."""
    out = []
    for token in spec.split():
        rhythm, _, pitch = token.partition("@")
        if pitch:
            out.append(EncodedSymbol(rhythm, pitch=pitch, position="upper"))
        else:
            out.append(EncodedSymbol(rhythm))
    return out


def bar(note: str, times: int) -> str:
    return " ".join([note] * times) + " barline"


def times_in(voice: list[EncodedSymbol]) -> list[tuple[str, str]]:
    xml = generate_xml(XmlGeneratorArguments(None, None, None), [voice], "")
    return [(t.findtext("beats"), t.findtext("beat-type")) for t in xml.iter("time")]


class TestStatedNumerator(unittest.TestCase):
    def test_a_stated_numerator_its_own_bars_contradict_is_not_printed(self) -> None:
        # Four bars of three quarters, labelled 4/4. The music is in 3.
        voice = symbols(
            "clef_G2 keySignature_0 timeSignatureBeats_4 timeSignature/4 "
            + " ".join(bar("note_4@C4", 3) for _ in range(4))
        )
        self.assertEqual({t for t in times_in(voice)}, {("3", "4")})

    def test_a_real_metre_change_still_renders_what_the_label_states(self) -> None:
        # 3/4, 3/4, 2/4, 2/4 - the shape of IMSLP632171-sys17-v0. No strict majority,
        # so there is no prevailing bar for the rule to rest on and the label wins.
        voice = symbols(
            "clef_G2 keySignature_0 timeSignatureBeats_2 timeSignature/4 "
            + bar("note_4@C4", 3) + " " + bar("note_4@C4", 3) + " "
            + bar("note_4@C4", 2) + " " + bar("note_4@C4", 2)
        )
        self.assertEqual({t for t in times_in(voice)}, {("2", "4")})

    def test_an_agreeing_stated_numerator_is_kept(self) -> None:
        voice = symbols(
            "clef_G2 keySignature_0 timeSignatureBeats_3 timeSignature/4 "
            + " ".join(bar("note_4@C4", 3) for _ in range(4))
        )
        self.assertEqual({t for t in times_in(voice)}, {("3", "4")})


class TestModalMeasureDuration(unittest.TestCase):
    def chords(self, spec: str) -> list:
        return add_tuplet_start_stop(group_into_chords(symbols(spec)))

    def test_a_clear_majority_is_the_modal_bar(self) -> None:
        spec = " ".join(bar("note_4@C4", 3) for _ in range(3)) + " " + bar("note_4@C4", 4)
        self.assertEqual(modal_measure_duration(self.chords(spec)), Fraction(3, 4))

    def test_a_tie_has_no_modal_bar(self) -> None:
        spec = (bar("note_4@C4", 3) + " " + bar("note_4@C4", 3) + " "
                + bar("note_4@C4", 2) + " " + bar("note_4@C4", 2))
        self.assertIsNone(modal_measure_duration(self.chords(spec)))

    def test_too_few_bars_has_no_modal_bar(self) -> None:
        spec = bar("note_4@C4", 3) + " " + bar("note_4@C4", 4)
        self.assertIsNone(modal_measure_duration(self.chords(spec)))


if __name__ == "__main__":
    unittest.main()
