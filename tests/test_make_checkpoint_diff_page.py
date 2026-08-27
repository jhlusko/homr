import unittest

from training.omr_datasets.make_checkpoint_diff_page import (
    PAD,
    accuracy,
    classify,
    diff_rows,
    show,
)


class TestClassify(unittest.TestCase):
    def test_new_right_old_wrong_is_a_gain(self) -> None:
        self.assertEqual(classify("note_4", "note_8", "note_4"), "gain")

    def test_old_right_new_wrong_is_a_regression(self) -> None:
        self.assertEqual(classify("note_4", "note_4", "note_8"), "regression")

    def test_both_right_is_agreement(self) -> None:
        self.assertEqual(classify("note_4", "note_4", "note_4"), "agree")

    def test_neither_right_is_flagged_separately(self) -> None:
        """Both wrong is usually a label problem, not a model one, and lumping it in
        with regressions would send a reviewer chasing the model for a data defect."""
        self.assertEqual(classify("note_4", "note_8", "note_2"), "both-wrong")


class TestShow(unittest.TestCase):
    def test_a_padded_position_reads_as_an_absence(self) -> None:
        self.assertEqual(show(PAD), "—")
        self.assertEqual(show(PAD + "p"), "—")

    def test_a_real_token_is_unchanged(self) -> None:
        self.assertEqual(show("note_4"), "note_4")


class TestAccuracy(unittest.TestCase):
    def _record(self, want, got):
        r = {"tokens": "t"}
        for b in ("rhythm", "pitch", "lift", "articulation", "slur", "position"):
            r[f"{b}_reference"] = want
            r[f"{b}_predicted"] = got
        return r

    def test_padding_counts_against_the_prediction(self) -> None:
        """A short prediction must be penalised: that padding rule is what made
        truncated labels dangerous, and relaxing it here would hide the effect."""
        hit, total = accuracy(self._record(["a", "b"], ["a", PAD + "p"]))
        self.assertEqual((hit, total), (6, 12))

    def test_a_perfect_prediction_scores_everything(self) -> None:
        hit, total = accuracy(self._record(["a", "b"], ["a", "b"]))
        self.assertEqual(hit, total)


class TestDiffRows(unittest.TestCase):
    def test_only_rhythm_and_pitch_are_diffed_position_by_position(self) -> None:
        rec = {"tokens": "t"}
        for b in ("rhythm", "pitch", "lift", "articulation", "slur", "position"):
            rec[f"{b}_reference"] = ["a"]
            rec[f"{b}_predicted"] = ["a"]
        self.assertEqual([r["branch"] for r in diff_rows(rec, rec)], ["rhythm", "pitch"])


if __name__ == "__main__":
    unittest.main()


class TestSymbolsFrom(unittest.TestCase):
    """The scored records store each decoder branch as a parallel array, which is the
    six fields an EncodedSymbol carries - so the stream can be rebuilt without
    re-running the model."""

    def _record(self, ref, pred):
        r = {"tokens": "t"}
        for b in ("rhythm", "pitch", "lift", "articulation", "slur", "position"):
            r[f"{b}_reference"] = ref
            r[f"{b}_predicted"] = pred
        return r

    def test_a_symbol_is_rebuilt_per_position(self) -> None:
        from training.omr_datasets.make_checkpoint_diff_page import symbols_from
        got = symbols_from(self._record(["note_4"], ["note_8"]), "predicted")
        self.assertEqual([s.rhythm for s in got], ["note_8"])

    def test_the_reference_side_is_selectable(self) -> None:
        from training.omr_datasets.make_checkpoint_diff_page import symbols_from
        got = symbols_from(self._record(["note_4"], ["note_8"]), "reference")
        self.assertEqual([s.rhythm for s in got], ["note_4"])

    def test_padded_positions_are_dropped(self) -> None:
        """A padded slot marks an absence; engraving it would put a nonsense symbol
        into the rendered score."""
        from training.omr_datasets.make_checkpoint_diff_page import PAD, symbols_from
        rec = self._record(["note_4", PAD], ["note_4", PAD + "p"])
        self.assertEqual(len(symbols_from(rec, "predicted")), 1)
        self.assertEqual(len(symbols_from(rec, "reference")), 1)
