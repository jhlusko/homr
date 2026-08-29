import unittest

from training.transformer import train_scans_with_synthetic as run


class TestMixture(unittest.TestCase):
    def test_both_ossq_tracks_are_clef_corrected(self) -> None:
        # Adding the synthetic track uncorrected would reintroduce the 2.4% missing-clef
        # bug that was just removed from the scanned side.
        self.assertIn("phase2clef", run.OSSQ_SYNTHETIC_INDEX)
        self.assertIn("phase7num", run.OSSQ_SCANNED_INDEX)
        self.assertNotIn("phase7clef", run.OSSQ_SCANNED_INDEX)

    def test_the_two_tracks_are_different_corpora(self) -> None:
        self.assertNotEqual(run.OSSQ_SCANNED_INDEX, run.OSSQ_SYNTHETIC_INDEX)

    def test_validation_is_scans_only(self) -> None:
        # The question is what synthetic data does to *scan* accuracy. A validation set
        # that grew easier alongside the training set could not answer it.
        self.assertIn("mixed_valid_clef", run.MIXED_VAL_INDEX)
        self.assertNotIn("phase2clef", run.MIXED_VAL_INDEX)

    def test_replay_is_retained(self) -> None:
        self.assertGreater(run.PDMX_REPLAY_COUNT, 0)


if __name__ == "__main__":
    unittest.main()
