import unittest

from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.fingerprint_measures import (
    align_to_ground_truth,
    measure_note_tokens,
    note_tokens,
)


def _note(pitch: str, lift: str = "_") -> EncodedSymbol:
    return EncodedSymbol("note_4", pitch, lift, "_", "_", "upper")


def _rest() -> EncodedSymbol:
    return EncodedSymbol("rest_4", "_", "_", "_", "_", "upper")


def _barline() -> EncodedSymbol:
    return EncodedSymbol("barline")


class TestNoteTokens(unittest.TestCase):
    def test_keeps_pitches_in_order(self) -> None:
        symbols = [_note("C4"), _note("D4"), _note("E4")]

        self.assertEqual(note_tokens(symbols), ["C4", "D4", "E4"])

    def test_carries_the_accidental_into_the_token(self) -> None:
        symbols = [_note("B4", "b"), _note("F5", "#")]

        self.assertEqual(note_tokens(symbols), ["B4b", "F5#"])

    def test_drops_rests_and_barlines(self) -> None:
        symbols = [_note("C4"), _rest(), _barline(), _note("D4")]

        self.assertEqual(note_tokens(symbols), ["C4", "D4"])

    def test_drops_notes_with_a_placeholder_pitch(self) -> None:
        symbols = [_note("C4"), _note("_"), _note("D4")]

        self.assertEqual(note_tokens(symbols), ["C4", "D4"])


class TestMeasureNoteTokens(unittest.TestCase):
    def test_flattens_measures_and_records_each_token_s_owner(self) -> None:
        measures = [[_note("C4"), _note("D4")], [_note("E4")]]

        flat, owner = measure_note_tokens(measures)

        self.assertEqual(flat, ["C4", "D4", "E4"])
        self.assertEqual(owner, [0, 0, 1])

    def test_an_empty_measure_contributes_nothing_but_does_not_shift_owners(self) -> None:
        measures = [[_note("C4")], [], [_note("E4")]]

        flat, owner = measure_note_tokens(measures)

        self.assertEqual(flat, ["C4", "E4"])
        self.assertEqual(owner, [0, 2])


class TestAlignToGroundTruth(unittest.TestCase):
    def _gt(self, measures: list[list[str]]) -> tuple[list[str], list[int]]:
        flat: list[str] = []
        owner: list[int] = []
        for i, m in enumerate(measures):
            for t in m:
                flat.append(t)
                owner.append(i)
        return flat, owner

    def test_finds_an_exact_passage_at_its_real_offset(self) -> None:
        gt_tokens, gt_owner = self._gt(
            [["C4", "D4"], ["E4", "F4"], ["G4", "A4"], ["B4", "C5"]]
        )

        result = align_to_ground_truth(["G4", "A4"], gt_tokens, gt_owner)

        self.assertEqual(result["start_measure"], 2)
        self.assertEqual(result["end_measure"], 3)
        self.assertTrue(result["trusted"])

    def test_recovers_a_shifted_passage_rather_than_its_assumed_position(self) -> None:
        # The whole point: the crop really contains measure 3's music, and that is
        # what it reports, regardless of which measure anyone expected.
        gt_tokens, gt_owner = self._gt(
            [["C4", "D4"], ["E4", "F4"], ["G4", "A4"], ["B4", "C5"]]
        )

        result = align_to_ground_truth(["B4", "C5"], gt_tokens, gt_owner)

        self.assertEqual(result["start_measure"], 3)
        self.assertEqual(result["end_measure"], 4)

    def test_spans_several_measures_when_the_crop_does(self) -> None:
        gt_tokens, gt_owner = self._gt(
            [["C4", "D4"], ["E4", "F4"], ["G4", "A4"], ["B4", "C5"]]
        )

        result = align_to_ground_truth(["E4", "F4", "G4", "A4"], gt_tokens, gt_owner)

        self.assertEqual(result["start_measure"], 1)
        self.assertEqual(result["end_measure"], 3)

    def test_tolerates_a_misread_note_inside_the_passage(self) -> None:
        gt_tokens, gt_owner = self._gt(
            [["C4", "D4", "E4"], ["F4", "G4", "A4"], ["B4", "C5", "D5"]]
        )

        # Middle note misread by the OMR - still the same passage.
        result = align_to_ground_truth(["F4", "X9", "A4"], gt_tokens, gt_owner)

        self.assertEqual(result["start_measure"], 1)
        self.assertTrue(result["coverage"] > 0)

    def test_an_unrelated_passage_is_not_trusted(self) -> None:
        gt_tokens, gt_owner = self._gt(
            [["C4", "D4", "E4"], ["F4", "G4", "A4"], ["B4", "C5", "D5"]]
        )

        result = align_to_ground_truth(["X1", "X2", "X3", "X4"], gt_tokens, gt_owner)

        self.assertTrue(result is None or not result["trusted"])

    def test_empty_crop_tokens_returns_none(self) -> None:
        gt_tokens, gt_owner = self._gt([["C4"]])

        self.assertIsNone(align_to_ground_truth([], gt_tokens, gt_owner))

    def test_empty_ground_truth_returns_none(self) -> None:
        self.assertIsNone(align_to_ground_truth(["C4"], [], []))


if __name__ == "__main__":
    unittest.main()
