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


class TestRestDominatedScores(unittest.TestCase):
    """A grand-staff accompaniment made of rests is a defect in the transcription
    itself, not in the alignment: IMSLP183806 carries 40 notes against 136 rests over
    66 measures while its scan shows dense piano, and every all-rest label human
    review rejected came from it.  1 of 234 scores trips this."""

    def _manifest(self, tmp, voice, notes, rests, pairs):
        from pathlib import Path
        out = {}
        for i in range(pairs):
            stem = f"IMSLPX-sys{i}-v{voice}"
            tok = Path(tmp) / f"{stem}.tokens"
            tok.write_text("\n".join(["note_4 C4 _ _ _ upper"] * notes
                                     + ["rest_4 _ _ _ _ upper"] * rests) + "\n",
                           encoding="utf-8")
            out[stem] = f"{tmp}/{stem}.png,{tok}"
        return out

    def test_a_rest_dominated_accompaniment_is_flagged(self) -> None:
        import tempfile
        from training.omr_datasets.build_consensus_corpus import rest_dominated_scores
        with tempfile.TemporaryDirectory() as tmp:
            m = self._manifest(tmp, voice=1, notes=1, rests=9, pairs=6)
            self.assertIn("IMSLPX", rest_dominated_scores(m))

    def test_a_normal_accompaniment_is_not(self) -> None:
        import tempfile
        from training.omr_datasets.build_consensus_corpus import rest_dominated_scores
        with tempfile.TemporaryDirectory() as tmp:
            m = self._manifest(tmp, voice=1, notes=9, rests=1, pairs=6)
            self.assertEqual(rest_dominated_scores(m), {})

    def test_a_resting_vocal_line_is_never_flagged(self) -> None:
        """Only the accompaniment is judged: a vocal line resting under a piano
        introduction is ordinary, and 56 of 56 such labels were correct on review."""
        import tempfile
        from training.omr_datasets.build_consensus_corpus import rest_dominated_scores
        with tempfile.TemporaryDirectory() as tmp:
            m = self._manifest(tmp, voice=0, notes=0, rests=10, pairs=6)
            self.assertEqual(rest_dominated_scores(m), {})

    def test_too_few_pairs_to_judge(self) -> None:
        import tempfile
        from training.omr_datasets.build_consensus_corpus import rest_dominated_scores
        with tempfile.TemporaryDirectory() as tmp:
            m = self._manifest(tmp, voice=1, notes=1, rests=9, pairs=2)
            self.assertEqual(rest_dominated_scores(m), {})
