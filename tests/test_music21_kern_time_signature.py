"""The music21 kern backend must state the metre numerator, like every other parser.

`validation/smb.py` defaults `--kern-parser` to `music21` while the prediction side is
parsed as MusicXML. A numerator missing on only one of those two sides is charged to the
tool as a recognition error it could not have avoided, so the two backends agreeing here
is what keeps that benchmark honest.

Skipped where music21 is absent - it is a declared dependency, but the GPU instance's
venv does not carry it, which is exactly how this defect survived unnoticed.
"""

import unittest

music21_available = True
try:
    import music21  # noqa: F401
except ImportError:
    music21_available = False


@unittest.skipUnless(music21_available, "music21 is not installed")
class TestMusic21TimeSignature(unittest.TestCase):
    def test_the_numerator_is_stated_and_precedes_the_denominator(self) -> None:
        from training.omr_datasets.music21_kern_parser import _time_signature_symbols

        symbols = [s.rhythm for s in _time_signature_symbols(3, 4)]

        self.assertEqual(symbols, ["timeSignatureBeats_3", "timeSignature/4"])

    def test_an_unusable_numerator_is_dropped_rather_than_invented(self) -> None:
        from training.omr_datasets.music21_kern_parser import _time_signature_symbols

        self.assertEqual(
            [s.rhythm for s in _time_signature_symbols(None, 4)], ["timeSignature/4"]
        )
        self.assertEqual(
            [s.rhythm for s in _time_signature_symbols(99, 4)], ["timeSignature/4"]
        )

    def test_both_kern_backends_state_the_same_metre(self) -> None:
        # The whole point: a cross-format comparison is only fair if the kern side can
        # express what the MusicXML side writes, whichever backend parses it.
        from validation.ned_score import _side_parts

        kern = "**kern\n*M3/4\n*clefG2\n4c\n4d\n4e\n*-\n"

        for backend in ("native", "music21"):
            rhythms = [s.rhythm for s in _side_parts(kern, backend, "native")[0]]
            with self.subTest(backend=backend):
                self.assertIn("timeSignatureBeats_3", rhythms)
                self.assertLess(
                    rhythms.index("timeSignatureBeats_3"), rhythms.index("timeSignature/4")
                )


if __name__ == "__main__":
    unittest.main()
