import unittest

from training.omr_datasets.build_consensus_corpus import (
    ARBITRATED,
    CONSENSUS,
    PHANTOM,
    REJECTED,
    REVERSE_ONLY,
    UNARBITRATED,
    aligned_spans,
    classify_system,
    reverse_spans,
)


class TestClassifySystem(unittest.TestCase):
    def test_both_methods_on_the_same_measures_is_consensus(self) -> None:
        self.assertEqual(classify_system((4, 7), (4, 7, 1.0)), CONSENSUS)

    def test_disagreement_keeps_the_bar_count_label(self) -> None:
        """Human review of 33 disagreements: the bar-count label was right in 28, the
        content label in 5, and exactly one deserved rejecting.  Discarding both was
        wrong 32 times out of 33."""
        self.assertEqual(classify_system((4, 7), (1, 4, 1.0)), ARBITRATED)

    def test_a_phantom_needs_reverse_to_have_read_something(self) -> None:
        """IMSLP637441: reverse read notes off the crop and still placed it nowhere.
        That is evidence."""
        self.assertEqual(classify_system((0, 3), (0, 0, 0.0), crop_had_notes=True), PHANTOM)

    def test_an_unreadable_crop_is_an_abstention_not_a_phantom(self) -> None:
        """note_tokens drops rests, so a rest-heavy system yields no tokens and comes
        back empty.  Reading that as "no music" was wrong on 30 of 30 reviewed
        items - a third of them had no pitched note at all."""
        self.assertEqual(classify_system((0, 3), (0, 0, 0.0), crop_had_notes=False), UNARBITRATED)

    def test_reverse_alone_is_trainable_not_evaluable(self) -> None:
        self.assertEqual(classify_system(None, (4, 7, 0.95)), REVERSE_ONLY)

    def test_arbitration_is_not_evaluation_grade(self) -> None:
        """~15% of arbitrated pairs are still wrong, so they train but never evaluate."""
        self.assertNotEqual(classify_system((4, 7), (1, 4, 1.0)), CONSENSUS)

    def test_an_unsure_arbiter_confirms_nothing(self) -> None:
        """Below the threshold the system is left unarbitrated rather than counted
        as agreement - an unsure arbiter must never silently confirm a range."""
        self.assertEqual(classify_system((4, 7), (4, 7, 0.3)), UNARBITRATED)

    def test_an_unsure_arbiter_cannot_promote_an_unplaced_system(self) -> None:
        self.assertEqual(classify_system(None, (4, 7, 0.3)), REJECTED)

    def test_no_arbiter_at_all_leaves_a_placed_system_unarbitrated(self) -> None:
        self.assertEqual(classify_system((4, 7), None), UNARBITRATED)

    def test_the_threshold_is_inclusive(self) -> None:
        self.assertEqual(classify_system((4, 7), (4, 7, 0.8), 0.8), CONSENSUS)


class TestSpanExtraction(unittest.TestCase):
    def test_only_aligned_systems_contribute_count_spans(self) -> None:
        alignment = {"scores": {"A": {"systems": [
            {"system": 0, "status": "aligned", "start_measure": 0, "end_measure": 3},
            {"system": 1, "status": "ambiguous", "start_measure": 3, "end_measure": 6},
            {"system": 2, "status": "skipped"},
        ]}}}
        self.assertEqual(aligned_spans(alignment), {("A", 0): (0, 3)})

    def test_only_accepted_scores_contribute_reverse_spans(self) -> None:
        reports = [{"scores": [
            {"score_id": "A", "accepted": True,
             "assignments": [{"system": 0, "start_measure": 0, "end_measure": 3, "score": 0.9}]},
            {"score_id": "B", "accepted": False,
             "assignments": [{"system": 0, "start_measure": 0, "end_measure": 3, "score": 0.9}]},
        ]}]
        self.assertEqual(reverse_spans(reports), {("A", 0): (0, 3, 0.9)})


if __name__ == "__main__":
    unittest.main()
