import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.staff_detection_agreement import (
    Agreement,
    count_detections,
    measure,
)

BBOX = "2.0 0 0 740 60 {confidence}\n"


def _bbox_file(directory: Path, confidences: list[str]) -> Path:
    path = directory / "system_yolo_bboxs.txt"
    path.write_text("".join(BBOX.format(confidence=c) for c in confidences), encoding="utf-8")
    return path


class TestCountingDetections(unittest.TestCase):
    def test_boxes_below_the_threshold_are_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _bbox_file(Path(tmp), ["0.95", "0.91", "0.42"])

            self.assertEqual(count_detections(path, 0.7), 2)

    def test_a_box_exactly_at_the_threshold_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _bbox_file(Path(tmp), ["0.70"])

            self.assertEqual(count_detections(path, 0.7), 1)

    def test_a_line_with_no_readable_confidence_still_counts(self) -> None:
        # It is a box the detector emitted. Dropping it would understate exactly the
        # disagreement this measures.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "system_yolo_bboxs.txt"
            path.write_text("2.0 0 0 740 60 nan-ish\n", encoding="utf-8")

            self.assertEqual(count_detections(path, 0.7), 1)

    def test_blank_lines_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "system_yolo_bboxs.txt"
            path.write_text("\n\n", encoding="utf-8")

            self.assertEqual(count_detections(path, 0.7), 0)


class TestAgreementCounts(unittest.TestCase):
    def test_a_match_and_a_mismatch_are_separated(self) -> None:
        agreement = Agreement()

        agreement.observe(parts=4, detected=4)
        agreement.observe(parts=4, detected=5)

        self.assertEqual((agreement.matched, agreement.mismatched), (1, 1))
        self.assertEqual(agreement.rate, 0.5)

    def test_the_direction_of_the_disagreement_is_kept(self) -> None:
        # Over-detection and under-detection need different fixes, so a single "mismatch"
        # count would not say which problem the track has.
        agreement = Agreement()

        agreement.observe(parts=4, detected=5)
        agreement.observe(parts=4, detected=3)

        self.assertEqual(agreement.delta[+1], 1)
        self.assertEqual(agreement.delta[-1], 1)

    def test_an_empty_measurement_does_not_divide_by_zero(self) -> None:
        self.assertEqual(Agreement().rate, 0.0)
        self.assertIn("0", Agreement().describe())


class TestMeasureOverATree(unittest.TestCase):
    def _tree(self, root: Path, confidences: list[str], parts: int) -> None:
        work = root / "scores" / "Composer" / "Work"
        systemwise = work / "images" / "synthetic" / "systemwise"
        systemwise.mkdir(parents=True)
        bodies = "".join(f'<part id="P{i + 1}"><measure number="1"/></part>' for i in range(parts))
        segments = work / "musicxml" / "unaligned"
        segments.mkdir(parents=True)
        (segments / "sq1:0001:0001.musicxml").write_text(
            f"<score-partwise>{bodies}</score-partwise>", encoding="utf-8"
        )
        (systemwise / "sq1:0001:0001_yolo_bboxs.txt").write_text(
            "".join(BBOX.format(confidence=c) for c in confidences), encoding="utf-8"
        )

    def test_detections_are_compared_against_the_matching_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root, ["0.9"] * 4, parts=4)

            agreement = measure(root, "synthetic", 0.7)

        self.assertEqual(agreement.matched, 1)
        self.assertEqual(agreement.pairs[(4, 4)], 1)

    def test_a_detection_file_with_no_segment_is_skipped(self) -> None:
        # Not a disagreement - there is simply nothing to compare it against, and
        # counting it as a mismatch would invent a failure.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root, ["0.9"] * 4, parts=4)
            stray = root / "scores" / "Composer" / "Work" / "images" / "synthetic" / "systemwise"
            (stray / "sq1:0009:0009_yolo_bboxs.txt").write_text("2.0 0 0 1 1 0.9\n", "utf-8")

            agreement = measure(root, "synthetic", 0.7)

        self.assertEqual(agreement.total, 1)


if __name__ == "__main__":
    unittest.main()
