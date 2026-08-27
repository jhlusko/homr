import unittest

from training.omr_datasets.recover_by_fingerprint import (
    RECOVERABLE_STATUSES,
    ground_truth_stream,
    recoverable_systems,
)


class _Symbol:
    def __init__(self, rhythm: str, pitch: str = "", lift: str = "") -> None:
        self.rhythm = rhythm
        self.pitch = pitch
        self.lift = lift


class TestRecoverableSystems(unittest.TestCase):
    def test_aligned_systems_keep_their_model_free_label(self) -> None:
        self.assertNotIn("aligned", RECOVERABLE_STATUSES)
        report = {"systems": [{"system": 0, "status": "aligned", "detected_measures": 4}]}
        self.assertEqual(recoverable_systems(report), [])

    def test_ambiguous_and_count_mismatch_are_attempted(self) -> None:
        report = {
            "systems": [
                {"system": 0, "status": "ambiguous", "detected_measures": 4},
                {"system": 1, "status": "count_mismatch", "detected_measures": 3},
            ]
        }
        self.assertEqual([s["system"] for s in recoverable_systems(report)], [0, 1])

    def test_a_skipped_illustration_box_has_nothing_to_fingerprint(self) -> None:
        """Narrow ornament detections carry no measures; 352 of the 2478 skips on
        2026-08-27 were these, and they must not be fed to the model."""
        report = {
            "systems": [
                {"system": 0, "status": "skipped", "detected_measures": 0},
                {"system": 1, "status": "skipped", "detected_measures": 5},
            ]
        }
        self.assertEqual([s["system"] for s in recoverable_systems(report)], [1])


class TestGroundTruthStream(unittest.TestCase):
    def test_tokens_carry_their_measure_index_back(self) -> None:
        voice = [
            [_Symbol("note_4", "C4"), _Symbol("barline")],
            [_Symbol("note_4", "D4", "#"), _Symbol("rest_4", "_")],
        ]
        flat, owner = ground_truth_stream(voice)
        self.assertEqual(flat, ["C4", "D4#"])
        self.assertEqual(owner, [0, 1])


if __name__ == "__main__":
    unittest.main()
