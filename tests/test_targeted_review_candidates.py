import unittest

from training.omr_datasets.targeted_review_candidates import targeted_candidates


def _row(score_id, page_index, system_index, detected, ground_truth, is_first=False, is_last=False):
    return {
        "score_id": score_id,
        "page_index": page_index,
        "page_image": f"{score_id}/p{page_index}.png",
        "system_index": system_index,
        "detected": detected,
        "ground_truth": ground_truth,
        "is_first_page": is_first,
        "is_last_page": is_last,
    }


class TestTargetedCandidates(unittest.TestCase):
    def test_surfaces_a_mismatch_from_an_otherwise_good_score(self) -> None:
        rows = [
            _row("A", 0, 0, 4, 4, is_first=True),
            _row("A", 1, 0, 5, 5),
            _row("A", 2, 0, 3, 5, is_last=True),  # the one bad page
        ]

        candidates = targeted_candidates(rows, min_score_exact_fraction=0.5)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["page_index"], 2)
        self.assertTrue(candidates[0]["is_last_page"])

    def test_drops_a_score_that_disagrees_too_often_overall(self) -> None:
        rows = [
            _row("B", 0, 0, 1, 4),
            _row("B", 1, 0, 2, 5),
            _row("B", 2, 0, 3, 6),
        ]

        candidates = targeted_candidates(rows, min_score_exact_fraction=0.7)

        self.assertEqual(candidates, [])

    def test_a_perfectly_matching_score_has_no_candidates(self) -> None:
        rows = [_row("C", 0, 0, 4, 4), _row("C", 1, 0, 5, 5)]

        candidates = targeted_candidates(rows, min_score_exact_fraction=0.7)

        self.assertEqual(candidates, [])

    def test_scores_are_evaluated_independently(self) -> None:
        rows = [
            _row("good", 0, 0, 4, 4),
            _row("good", 1, 0, 3, 5),  # one mismatch, still >= threshold overall
            _row("bad", 0, 0, 1, 9),
            _row("bad", 1, 0, 1, 9),
        ]

        candidates = targeted_candidates(rows, min_score_exact_fraction=0.5)

        score_ids = {c["score_id"] for c in candidates}
        self.assertEqual(score_ids, {"good"})


if __name__ == "__main__":
    unittest.main()
