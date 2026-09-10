"""`parse_staffs` applies the tier-1 cross-staff repairs, rather than logging them.

The proposers and appliers were unit-tested from the day they were written and nothing
called them, so a quartet whose cello opened on a different time signature from the
other three staves was detected, logged, and shipped. These tests exercise the wiring
itself - the part that was missing - not the proposal arithmetic, which
`test_cross_staff_repair.py` already covers.
"""

import unittest
from unittest.mock import patch

from homr.staff_parsing import _apply_cross_staff_repairs
from homr.transformer.vocabulary import EncodedSymbol


def _sym(rhythm: str) -> EncodedSymbol:
    return EncodedSymbol(rhythm)


class _Plan:
    """The two things `_apply_cross_staff_repairs` asks a SystemPlan for."""

    def __init__(self, systems: int, present: dict[tuple[int, int], bool] | None = None) -> None:
        self.systems = list(range(systems))
        self._present = present or {}

    def staff_for_voice(self, system: int, voice: int) -> int | None:
        if self._present and not self._present.get((system, voice), False):
            return None
        return voice


def _quartet(cello_time: str) -> dict[tuple[int, int], list[EncodedSymbol]]:
    """Four staves of one system; the fourth opens on `cello_time`."""
    return {
        (voice, 0): [
            _sym("clef_G2"),
            _sym("keySignature_0"),
            _sym("timeSignature_4/4" if voice != 3 else cello_time),
            _sym("note_4"),
        ]
        for voice in range(4)
    }


class TestCrossStaffRepairWiring(unittest.TestCase):
    def test_a_minority_time_signature_is_corrected_to_the_majority(self) -> None:
        decoded = _quartet("timeSignature_3/4")
        with patch("homr.staff_parsing.eprint"):
            _apply_cross_staff_repairs(_Plan(1), decoded, 4)
        self.assertEqual(
            [decoded[(voice, 0)][2].rhythm for voice in range(4)],
            ["timeSignature_4/4"] * 4,
        )

    def test_an_agreeing_system_is_left_exactly_as_decoded(self) -> None:
        decoded = _quartet("timeSignature_4/4")
        before = {key: list(value) for key, value in decoded.items()}
        with patch("homr.staff_parsing.eprint"):
            _apply_cross_staff_repairs(_Plan(1), decoded, 4)
        self.assertEqual(decoded, before)

    def test_a_tie_is_left_for_a_human(self) -> None:
        # Two staves each way is not a majority, and picking one would be a guess.
        decoded = _quartet("timeSignature_3/4")
        decoded[(2, 0)][2] = _sym("timeSignature_3/4")
        with patch("homr.staff_parsing.eprint"):
            _apply_cross_staff_repairs(_Plan(1), decoded, 4)
        self.assertEqual(
            [decoded[(voice, 0)][2].rhythm for voice in range(4)],
            ["timeSignature_4/4", "timeSignature_4/4", "timeSignature_3/4", "timeSignature_3/4"],
        )

    def test_two_staves_are_not_enough_to_outvote_anything(self) -> None:
        decoded = {
            (voice, 0): [_sym("clef_G2"), _sym("timeSignature_4/4" if voice == 0 else "timeSignature_3/4")]
            for voice in range(2)
        }
        with patch("homr.staff_parsing.eprint"):
            _apply_cross_staff_repairs(_Plan(1), decoded, 2)
        self.assertEqual(decoded[(1, 0)][1].rhythm, "timeSignature_3/4")

    def test_a_voice_absent_from_a_system_is_not_counted_or_corrected(self) -> None:
        decoded = _quartet("timeSignature_3/4")
        del decoded[(1, 0)]
        with patch("homr.staff_parsing.eprint"):
            _apply_cross_staff_repairs(_Plan(1, {(0, v): v != 1 for v in range(4)}), decoded, 4)
        # Three staves remain: two agree, so the odd one still moves.
        self.assertEqual(decoded[(3, 0)][2].rhythm, "timeSignature_4/4")
        self.assertNotIn((1, 0), decoded)

    def test_a_failure_leaves_the_system_untouched(self) -> None:
        decoded = _quartet("timeSignature_3/4")
        before = {key: list(value) for key, value in decoded.items()}
        with patch("homr.staff_parsing.propose_repairs", side_effect=RuntimeError("boom")):
            with patch("homr.staff_parsing.eprint"):
                _apply_cross_staff_repairs(_Plan(1), decoded, 4)
        self.assertEqual(decoded, before)


if __name__ == "__main__":
    unittest.main()
