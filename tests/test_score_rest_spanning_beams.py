import unittest

from training.transformer.score_rest_spanning_beams import (
    beam_groups,
    crop_key,
    score,
    spanned_rests,
)


def _record(kinds: str, predicted: list[str], reference: list[str], tokens: str = "sq1_0001_0002_3.txt") -> dict:
    return {"tokens": tokens, "kinds": kinds, "predicted_beam": [predicted], "reference_beam": [reference]}


class FakeTruth:
    def __init__(self, modes: dict[str, list[str | None]]) -> None:
        self.modes = modes

    def rests(self, token_path: str) -> list[str | None] | None:
        return self.modes.get(token_path)


NA = "not_applicable"


class TestGroups(unittest.TestCase):
    def test_a_group_runs_over_notes_in_token_order_and_skips_rests(self) -> None:
        groups, malformed = beam_groups("nrnon", ["begin", NA, "continue", NA, "end"])
        self.assertEqual(groups, [(0, 4)])
        self.assertEqual(malformed, 0)

    def test_an_unbeamed_note_inside_a_run_breaks_it_rather_than_being_repaired(self) -> None:
        groups, malformed = beam_groups("nnn", ["begin", "flag", "end"])
        self.assertEqual(groups, [])
        self.assertEqual(malformed, 2)  # the broken run, then the orphan end

    def test_spanned_rests_are_the_staffs_rest_ordinals_strictly_inside(self) -> None:
        kinds = "rnrnrn"
        self.assertEqual(spanned_rests(kinds, (1, 3)), [1])
        self.assertEqual(spanned_rests(kinds, (1, 5)), [1, 2])
        self.assertEqual(spanned_rests(kinds, (3, 3)), [])

    def test_crop_key_is_score_page_system_and_zero_based_part(self) -> None:
        self.assertEqual(crop_key("/x/sq10307350_0028_0002_1.txt"), ("sq10307350", 28, 2, 0))
        self.assertIsNone(crop_key("/x/nonsense.txt"))


class TestScore(unittest.TestCase):
    def test_precision_counts_only_predicted_groups_that_span_a_rest(self) -> None:
        records = [
            # Head beams across the rest exactly as the reference does: correct.
            _record("nrn", ["begin", NA, "end"], ["begin", NA, "end"], "a_0001_0001_1.txt"),
            # Head beams across a rest the reference leaves unbeamed: wrong.
            _record("nrn", ["begin", NA, "end"], ["flag", NA, "flag"], "b_0001_0001_1.txt"),
            # An ordinary group with no rest inside is not in the gate's population.
            _record("nnr", ["begin", "end", NA], ["begin", "end", NA], "c_0001_0001_1.txt"),
        ]
        report = score(records, None)
        self.assertEqual(report["counts"]["predicted_spanning"], 2)
        self.assertEqual(report["precision_vs_musicxml_reference"], 0.5)
        self.assertEqual(report["recall_vs_musicxml_reference"], 1.0)
        self.assertIsNone(report["precision_vs_mscx_beammode"])

    def test_mscx_support_needs_every_spanned_rest_inside_a_beam(self) -> None:
        records = [
            _record("nrrn", ["begin", NA, NA, "end"], ["begin", NA, NA, "end"], "a_0001_0001_1.txt"),
            _record("nrrn", ["begin", NA, NA, "end"], ["begin", NA, NA, "end"], "b_0001_0001_1.txt"),
            _record("nrn", ["begin", NA, "end"], ["begin", NA, "end"], "c_0001_0001_1.txt"),
        ]
        truth = FakeTruth({
            "a_0001_0001_1.txt": ["mid", "mid"],  # supported
            "b_0001_0001_1.txt": ["mid", None],   # one rest not in the beam: unsupported
            "c_0001_0001_1.txt": ["mid", "mid"],  # rest count disagrees with the crop: unjoined
        })
        report = score(records, truth)
        self.assertEqual(report["counts"]["staves_joined"], 2)
        self.assertEqual(report["counts"]["staves_unjoined"], 1)
        self.assertEqual(report["precision_vs_mscx_beammode"], 0.5)
        self.assertEqual(report["counts"]["mscx_beamed_rests"], 3)
        self.assertEqual(report["recall_vs_mscx_beammode"], 1.0)

    def test_a_rest_that_begins_a_beam_does_not_support_a_group_spanning_it(self) -> None:
        records = [_record("nrn", ["begin", NA, "end"], ["begin", NA, "end"], "a_0001_0001_1.txt")]
        report = score(records, FakeTruth({"a_0001_0001_1.txt": ["begin"]}))
        self.assertEqual(report["precision_vs_mscx_beammode"], 0.0)
        self.assertEqual(report["counts"]["mscx_beamed_rests"], 0)


if __name__ == "__main__":
    unittest.main()
