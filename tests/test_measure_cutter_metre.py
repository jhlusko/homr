import unittest

from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.convert_lieder import MeasureCutter


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


if __name__ == "__main__":
    unittest.main()
