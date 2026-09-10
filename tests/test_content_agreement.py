import unittest

from training.omr_datasets.content_agreement import PAD, content_agreement


def record(ref_pitch, got_pitch, ref_rhythm=None, got_rhythm=None):
    n = len(ref_pitch)
    return {
        "tokens": "t",
        "pitch_reference": ref_pitch, "pitch_predicted": got_pitch,
        "rhythm_reference": ref_rhythm or ["note_4"] * n,
        "rhythm_predicted": got_rhythm or ["note_4"] * n,
    }


class TestContentAgreement(unittest.TestCase):
    def test_identical_content_agrees_completely(self) -> None:
        self.assertEqual(content_agreement(record(["C4", "D4"], ["C4", "D4"])), (1.0, 2))

    def test_a_wrong_pitch_counts_against(self) -> None:
        agreement, notes = content_agreement(record(["C4", "D4"], ["C4", "E4"]))
        self.assertEqual((agreement, notes), (0.5, 2))

    def test_the_right_pitch_with_the_wrong_duration_is_still_an_error(self) -> None:
        """A pitch-only comparison would call this perfect."""
        agreement, _ = content_agreement(
            record(["C4"], ["C4"], ["note_4"], ["note_8"]))
        self.assertEqual(agreement, 0.0)

    def test_padded_positions_are_excluded(self) -> None:
        """A padded slot is a LENGTH disagreement - a range question, which already has
        its own gates - so counting it here would double-count what range checks do."""
        agreement, notes = content_agreement(
            record(["C4", PAD], ["C4", PAD + "p"], ["note_4", PAD], ["note_4", PAD + "p"]))
        self.assertEqual((agreement, notes), (1.0, 1))

    def test_a_stave_with_nothing_comparable_does_not_read_as_disagreement(self) -> None:
        agreement, notes = content_agreement(
            record([PAD], [PAD + "p"], [PAD], [PAD + "p"]))
        self.assertEqual((agreement, notes), (1.0, 0))


if __name__ == "__main__":
    unittest.main()
