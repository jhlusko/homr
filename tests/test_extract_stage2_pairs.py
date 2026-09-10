import unittest

from training.omr_datasets.extract_stage2_pairs import (
    eligible_system_positions,
    flat_detected_systems,
    flat_measure_ranges,
    group_staff_boxes_into_voices,
)


def _row(score_id: str, page_index: int, detected: int, ground_truth: int) -> dict:
    return {
        "score_id": score_id,
        "page_index": page_index,
        "system_index": 0,
        "detected": detected,
        "ground_truth": ground_truth,
    }


class TestEligibleSystemPositions(unittest.TestCase):
    def test_exact_bar_count_match_is_eligible(self) -> None:
        rows = [_row("A", 0, 4, 4)]

        result = eligible_system_positions(rows, {})

        self.assertEqual(result, {"A": {0}})

    def test_bar_count_mismatch_with_no_review_is_not_eligible(self) -> None:
        rows = [_row("A", 0, 4, 5)]

        result = eligible_system_positions(rows, {})

        self.assertEqual(result, {"A": set()})

    def test_mismatch_on_a_reviewed_matched_page_is_eligible(self) -> None:
        rows = [_row("A", 0, 4, 5)]
        review = {"A/0": {"score_id": "A", "page_index": 0, "judgment": "match"}}

        result = eligible_system_positions(rows, review)

        self.assertEqual(result, {"A": {0}})

    def test_different_layout_judgment_also_confers_eligibility(self) -> None:
        rows = [_row("A", 0, 4, 5)]
        review = {"A/0": {"score_id": "A", "page_index": 0, "judgment": "different_layout"}}

        result = eligible_system_positions(rows, review)

        self.assertEqual(result, {"A": {0}})

    def test_no_match_judgment_does_not_confer_eligibility(self) -> None:
        rows = [_row("A", 0, 4, 5)]
        review = {"A/0": {"score_id": "A", "page_index": 0, "judgment": "no_match"}}

        result = eligible_system_positions(rows, review)

        self.assertEqual(result, {"A": set()})

    def test_position_is_the_row_s_index_within_its_own_score(self) -> None:
        rows = [_row("A", 0, 4, 4), _row("A", 0, 3, 9), _row("A", 1, 5, 5)]

        result = eligible_system_positions(rows, {})

        self.assertEqual(result, {"A": {0, 2}})

    def test_a_review_judgment_only_applies_to_its_own_score(self) -> None:
        rows = [_row("A", 0, 4, 5), _row("B", 0, 4, 5)]
        review = {"A/0": {"score_id": "A", "page_index": 0, "judgment": "match"}}

        result = eligible_system_positions(rows, review)

        self.assertEqual(result, {"A": {0}, "B": set()})


class TestFlatMeasureRanges(unittest.TestCase):
    def test_one_entry_per_system_cumulative_across_pages(self) -> None:
        pages = [[3, 4], [5]]

        ranges = flat_measure_ranges(pages)

        self.assertEqual(ranges, [(0, 3), (3, 7), (7, 12)])

    def test_empty_pages_list(self) -> None:
        self.assertEqual(flat_measure_ranges([]), [])

    def test_a_page_with_no_systems_contributes_no_ranges(self) -> None:
        self.assertEqual(flat_measure_ranges([[3], [], [4]]), [(0, 3), (3, 7)])


class TestFlatDetectedSystems(unittest.TestCase):
    def test_flattens_pages_in_sorted_key_order_carrying_the_page_image(self) -> None:
        doc = {
            "pages": {
                2: {"image": "p2.png", "systems": [{"boundingBox": {"a": 1}}]},
                1: {"image": "p1.png", "systems": [{"boundingBox": {"a": 0}}, {"boundingBox": {"a": 2}}]},
            }
        }

        flat = flat_detected_systems(doc)

        self.assertEqual([s["page_image"] for s in flat], ["p1.png", "p1.png", "p2.png"])
        self.assertEqual([s["boundingBox"]["a"] for s in flat], [0, 2, 1])


def _box(left: int, top: int, width: int, height: int) -> dict:
    return {"left": left, "top": top, "width": width, "height": height}


class TestGroupStaffBoxesIntoVoices(unittest.TestCase):
    def test_one_box_per_voice_in_order(self) -> None:
        boxes = [_box(0, 0, 10, 5), _box(0, 10, 10, 5)]

        groups = group_staff_boxes_into_voices(boxes, [False, True])

        self.assertEqual(groups, boxes)

    def test_mismatched_box_count_returns_none(self) -> None:
        boxes = [_box(0, 0, 10, 5)]

        groups = group_staff_boxes_into_voices(boxes, [False, True])

        self.assertIsNone(groups)

    def test_too_many_boxes_also_returns_none(self) -> None:
        boxes = [_box(0, 0, 10, 5), _box(0, 10, 10, 5), _box(0, 20, 10, 5)]

        groups = group_staff_boxes_into_voices(boxes, [False, True])

        self.assertIsNone(groups)


if __name__ == "__main__":
    unittest.main()
