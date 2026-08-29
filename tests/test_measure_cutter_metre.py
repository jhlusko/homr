import unittest

from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.convert_lieder import MeasureCutter
from training.omr_datasets.convert_musetrainer import _context_at_measure


def sym(rhythm, position="upper"):
    return EncodedSymbol(rhythm, "_", "_", "_", "_", position)


def note(position="upper"):
    return EncodedSymbol("note_4", "C4", "_", "_", "_", position)


class TestNumeratorSurvivesSlicing(unittest.TestCase):
    """`"timeSignature" in symbol.rhythm` matches `timeSignatureBeats_3` as well as
    `timeSignature/4`. Because the numerator is emitted first, the denominator
    overwrote it in the carried state, so every restatement across a slice boundary
    lost it: 77 numerators against 416 time signatures in the corpus."""

    def _voice(self):
        first = [sym("clef_G2"), sym("keySignature_0"),
                 sym("timeSignatureBeats_3"), sym("timeSignature/4"),
                 note(), note(), note(), EncodedSymbol("barline")]
        rest = [note(), note(), note(), EncodedSymbol("barline")]
        return [first, list(rest), list(rest), list(rest)]

    def test_a_slice_that_does_not_redeclare_states_no_signature(self) -> None:
        """Deliberate: a courtesy signature is only visible where the source redeclares
        it, so a crop that does not show one must not be labelled with one."""
        cutter = MeasureCutter(self._voice())
        cutter.extract_measures(1)                 # consume the measure that declares it
        rhythms = [s.rhythm for s in cutter.extract_measures(2)]
        self.assertNotIn("timeSignature/4", rhythms)
        self.assertNotIn("timeSignatureBeats_3", rhythms)

    def test_the_numerator_precedes_its_denominator(self) -> None:
        rhythms = [s.rhythm for s in MeasureCutter(self._voice()).extract_measures(1)]
        self.assertLess(rhythms.index("timeSignatureBeats_3"),
                        rhythms.index("timeSignature/4"))

    def test_a_mid_slice_redeclaration_keeps_both(self) -> None:
        """Where the denominator survives, the numerator must too - the pairing the
        overwrite bug broke."""
        voice = [
            [sym("clef_G2"), sym("keySignature_0"), sym("timeSignatureBeats_4"),
             sym("timeSignature/4"), note(), EncodedSymbol("barline")],
            [sym("timeSignatureBeats_3"), sym("timeSignature/4"), note(),
             EncodedSymbol("barline")],
        ]
        rhythms = [s.rhythm for s in MeasureCutter(voice).extract_measures(2)]
        self.assertEqual(rhythms.count("timeSignature/4"), 2, rhythms)
        self.assertEqual(
            sum(1 for r in rhythms if r.startswith("timeSignatureBeats")), 2, rhythms)

    def test_the_declaring_slice_keeps_both(self) -> None:
        rhythms = [s.rhythm for s in MeasureCutter(self._voice()).extract_measures(1)]
        self.assertIn("timeSignatureBeats_3", rhythms)
        self.assertIn("timeSignature/4", rhythms)

    def test_a_voice_with_no_numerator_is_unchanged(self) -> None:
        """Sources predating the token, and any the parser could not read, must still
        slice exactly as before."""
        voice = [[sym("clef_G2"), sym("keySignature_0"), sym("timeSignature/4"),
                  note(), EncodedSymbol("barline")],
                 [note(), EncodedSymbol("barline")]]
        rhythms = [s.rhythm for s in MeasureCutter(voice).extract_measures(2)]
        self.assertNotIn("timeSignatureBeats_3", rhythms)
        self.assertIn("timeSignature/4", rhythms)



class TestNumeratorSurvivesWindowing(unittest.TestCase):
    """The same loss from the other side: a cutter built fresh for each window.

    The class above fixes one cutter walking a whole score, which is how Lieder is cut.
    pdmx and musetrainer instead build a new MeasureCutter per 8-measure window and seed
    its context from `_context_at_measure`, so a numerator stated before the window began
    was never handed over and the window opened on a bare `timeSignature/4` - 92.4% of
    pdmx's shipped token files (29,976 of 32,451) and 89.2% of musetrainer's, each paired
    with an image that does show a full signature.
    """

    def _voice(self):
        first = [sym("clef_G2"), sym("keySignature_0"),
                 sym("timeSignatureBeats_3"), sym("timeSignature/4"),
                 note(), note(), note(), EncodedSymbol("barline")]
        rest = [note(), note(), note(), EncodedSymbol("barline")]
        return [first] + [list(rest) for _ in range(5)]

    def _window(self, voice, start, count):
        """Exactly how convert_pdmx cuts a window that does not start the score."""
        clefs, key, time_sym, time_beats = _context_at_measure(voice, start, 1)
        cutter = MeasureCutter(list(voice[start : start + count]))
        cutter.clefs = clefs
        cutter.key = key
        cutter.time = time_sym
        cutter.time_beats = time_beats
        return [s.rhythm for s in cutter.extract_measures(count, always_include_time=True)]

    def test_a_later_window_restates_the_whole_signature(self) -> None:
        rhythms = self._window(self._voice(), 2, 3)
        self.assertIn("timeSignature/4", rhythms)
        self.assertLess(rhythms.index("timeSignatureBeats_3"),
                        rhythms.index("timeSignature/4"))

    def test_the_first_window_is_unchanged(self) -> None:
        rhythms = self._window(self._voice(), 0, 3)
        self.assertEqual(rhythms.count("timeSignatureBeats_3"), 1, rhythms)
        self.assertEqual(rhythms.count("timeSignature/4"), 1, rhythms)

    def test_the_numerator_does_not_overwrite_the_denominator(self) -> None:
        """`"timeSignature" in rhythm` matches the numerator too, so the carried
        denominator is only correct if the numerator is tested for first."""
        _clefs, _key, time_sym, time_beats = _context_at_measure(self._voice(), 3, 1)
        self.assertEqual(time_sym.rhythm, "timeSignature/4")
        self.assertEqual(time_beats.rhythm, "timeSignatureBeats_3")

    def test_a_metre_change_carries_the_latest_numerator(self) -> None:
        voice = self._voice()
        voice[2] = [sym("timeSignatureBeats_6"), sym("timeSignature/8"),
                    note(), EncodedSymbol("barline")]
        _clefs, _key, time_sym, time_beats = _context_at_measure(voice, 4, 1)
        self.assertEqual(time_sym.rhythm, "timeSignature/8")
        self.assertEqual(time_beats.rhythm, "timeSignatureBeats_6")

    def test_a_window_before_any_stated_metre_still_states_a_whole_one(self) -> None:
        """The denominator has always defaulted to /4; the numerator must default with
        it, or the fallback writes half a signature."""
        voice = [[sym("clef_G2"), note(), EncodedSymbol("barline")] for _ in range(3)]
        rhythms = self._window(voice, 1, 2)
        self.assertIn("timeSignatureBeats_4", rhythms)
        self.assertIn("timeSignature/4", rhythms)


if __name__ == "__main__":
    unittest.main()
