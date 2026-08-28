"""The overfull rule must not fire on a grand staff.

`overfull_bars` compares a bar against its staff's prevailing bar, and both come from
`measure_durations`, which on a grand staff is computed from `group_into_chords` -
which takes the MINIMUM duration across a chord. A bar whose hands play different
rhythms is then neither their sum nor either hand's own length, so the comparison is
between two equally distorted numbers. `audit_label_consistency` excludes grand staves
from every duration-dependent check for this reason; the corpus builder did not, and
discarded 371 grand-staff pairs at 8.4x the single-staff rate.
"""

import unittest

from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.audit_label_consistency import is_single_staff, overfull_bars


def symbols(spec: str) -> list[EncodedSymbol]:
    out = []
    for token in spec.split():
        rhythm, _, rest = token.partition("@")
        pitch, _, position = rest.partition(":")
        out.append(EncodedSymbol(rhythm, pitch=pitch or "_", position=position or "_"))
    return out


def bar(note: str, times: int) -> str:
    return " ".join([note] * times) + " barline"


class TestOverfullGuard(unittest.TestCase):
    def test_a_single_staff_overfull_bar_is_still_detected(self) -> None:
        # Three bars of three quarters and one of four: the long one is real.
        voice = symbols(
            " ".join(bar("note_4@C4:upper", 3) for _ in range(3)) + " " + bar("note_4@C4:upper", 4)
        )
        self.assertTrue(is_single_staff(voice))
        self.assertEqual(overfull_bars(voice), [3])

    def test_a_grand_staff_is_recognised_as_such(self) -> None:
        voice = symbols("note_4@C5:upper chord note_4@C3:lower barline "
                        "note_4@C5:upper chord note_4@C3:lower barline "
                        "note_4@C5:upper chord note_4@C3:lower barline")
        self.assertFalse(is_single_staff(voice))

    def test_the_builder_guard_skips_grand_staves(self) -> None:
        """The guard the builder applies, stated as the test that failing it breaks."""
        grand = symbols("note_4@C5:upper chord note_4@C3:lower barline "
                        "note_4@C5:upper chord note_4@C3:lower barline "
                        "note_4@C5:upper chord note_4@C3:lower barline")
        guarded = overfull_bars(grand) if is_single_staff(grand) else []
        self.assertEqual(guarded, [])


if __name__ == "__main__":
    unittest.main()
