import unittest
import xml.etree.ElementTree as ET
from fractions import Fraction

from homr.music_xml_generator import (
    ConversionState,
    XmlGeneratorArguments,
    build_time_signature,
    generate_xml,
)
from homr.transformer.vocabulary import (
    TIME_SIGNATURE_BEATS_PREFIX,
    VALID_TIME_SIGNATURE_NUMERATORS,
    EncodedSymbol,
    build_rhythm,
)


class TestVocabulary(unittest.TestCase):
    def test_numerator_tokens_exist(self) -> None:
        rhythm = build_rhythm()
        for n in (2, 3, 4, 6, 9, 12):
            self.assertIn(f"{TIME_SIGNATURE_BEATS_PREFIX}{n}", rhythm)

    def test_the_new_tokens_are_appended_last(self) -> None:
        """build_dict assigns indices in list order, so the numerator tokens must come
        after every pre-existing token; inserting them earlier would renumber the rest
        and silently invalidate trained checkpoints."""
        rhythm = build_rhythm()
        new = [rhythm[f"{TIME_SIGNATURE_BEATS_PREFIX}{n}"] for n in VALID_TIME_SIGNATURE_NUMERATORS]
        others = [i for t, i in rhythm.items() if not t.startswith(TIME_SIGNATURE_BEATS_PREFIX)]
        self.assertGreater(min(new), max(others))

    def test_denominator_tokens_are_unchanged(self) -> None:
        self.assertIn("timeSignature/4", build_rhythm())


class TestRendering(unittest.TestCase):
    def _time(self, symbols):
        xml = generate_xml(XmlGeneratorArguments(None, None, None), [symbols], "")
        times = list(xml.iter("time"))
        return [(t.findtext("beats"), t.findtext("beat-type")) for t in times]

    def test_a_stated_numerator_is_used(self) -> None:
        state = ConversionState(4, Fraction(1))
        state.stated_beats = 3
        attrs = ET.Element("attributes")
        build_time_signature(EncodedSymbol("timeSignature/4"), attrs, state)
        self.assertEqual(attrs.find("time/beats").text, "3")
        self.assertEqual(attrs.find("time/beat-type").text, "4")

    def test_the_stated_numerator_applies_once(self) -> None:
        """It is consumed by its own time signature, so a later one without a stated
        numerator falls back rather than inheriting a stale value."""
        state = ConversionState(4, Fraction(1))
        state.stated_beats = 3
        build_time_signature(EncodedSymbol("timeSignature/4"), ET.Element("attributes"), state)
        self.assertIsNone(state.stated_beats)

    def test_inference_still_runs_when_nothing_was_stated(self) -> None:
        """A checkpoint trained before these tokens existed never emits one, and must
        render exactly as it did before."""
        state = ConversionState(4, Fraction(1))
        attrs = ET.Element("attributes")
        build_time_signature(EncodedSymbol("timeSignature/4"), attrs, state)
        self.assertEqual(attrs.find("time/beats").text, "4")

    def test_three_four_survives_a_round_trip(self) -> None:
        symbols = [
            EncodedSymbol("clef_G2"),
            EncodedSymbol(f"{TIME_SIGNATURE_BEATS_PREFIX}3"),
            EncodedSymbol("timeSignature/4"),
            EncodedSymbol("note_4", "C4", "_", "_", "_", "upper"),
            EncodedSymbol("barline"),
        ]
        self.assertTrue(all(t == ("3", "4") for t in self._time(symbols)), self._time(symbols))

    def test_every_rendered_time_agrees_with_the_stated_numerator(self) -> None:
        """MusicXML wants a `<time>` in the opening attributes, and the generator adds
        one as a fallback when the explicit signature lands in a later attributes
        block. That fallback must not contradict what the label stated a moment
        later - it used to infer 1/4 in front of an explicit 3/4."""
        symbols = [
            EncodedSymbol("clef_G2"),
            EncodedSymbol(f"{TIME_SIGNATURE_BEATS_PREFIX}6"),
            EncodedSymbol("timeSignature/8"),
            EncodedSymbol("note_8", "C4", "_", "_", "_", "upper"),
            EncodedSymbol("barline"),
        ]
        times = self._time(symbols)
        self.assertTrue(times)
        for beats, beat_type in times:
            self.assertEqual(beats, "6", times)
            self.assertEqual(beat_type, "8", times)


if __name__ == "__main__":
    unittest.main()
