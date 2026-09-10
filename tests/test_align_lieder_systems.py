import json
import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.align_lieder_systems import build_alignment_document


class TestBuildAlignmentDocument(unittest.TestCase):
    def test_uses_score_sequences_not_ordinal_ground_truth_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "A.json").write_text(json.dumps({"pages": [[3, 3, 4]]}))
            rows = [
                {"score_id": "A", "detected": 6, "ground_truth": 999,
                 "page_index": 0, "page_image": "p.png", "system_index": 0},
                {"score_id": "A", "detected": 4, "ground_truth": 999,
                 "page_index": 0, "page_image": "p.png", "system_index": 1},
            ]

            doc = build_alignment_document(rows, root, max_group=4, min_margin=1)

        systems = doc["scores"]["A"]["systems"]
        self.assertEqual((systems[0]["start_measure"], systems[0]["end_measure"]), (0, 6))
        self.assertTrue(all(item["status"] == "aligned" for item in systems))
        self.assertFalse(doc["model_predictions_used"])

    def test_narrow_illustration_detection_is_forced_to_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "A.json").write_text(json.dumps({"pages": [[3, 4]]}))
            rows = [
                {"score_id": "A", "detected": 7, "system_width_fraction": 0.1,
                 "page_index": 0, "page_image": "art.png", "system_index": 0},
                {"score_id": "A", "detected": 3, "system_width_fraction": 0.9,
                 "page_index": 1, "page_image": "score.png", "system_index": 0},
                {"score_id": "A", "detected": 4, "system_width_fraction": 0.9,
                 "page_index": 1, "page_image": "score.png", "system_index": 1},
            ]

            doc = build_alignment_document(rows, root, max_group=4, min_margin=1)

        systems = doc["scores"]["A"]["systems"]
        self.assertEqual(systems[0]["status"], "skipped")
        self.assertEqual(systems[0]["observed_measures"], 7)
        self.assertEqual(
            [(item["start_measure"], item["end_measure"]) for item in systems[1:]],
            [(0, 3), (3, 7)],
        )


if __name__ == "__main__":
    unittest.main()
