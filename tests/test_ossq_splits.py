import unittest

from training.omr_datasets.ossq_splits import SPLIT_NAMES, load_split_manifest


class TestOssqSplitManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_split_manifest()

    def test_covers_the_whole_corpus(self) -> None:
        self.assertEqual(len(self.manifest.scores), 122)

    def test_segment_counts_match_the_published_split(self) -> None:
        self.assertEqual(
            self.manifest.segment_counts,
            {"train": 13480, "valid": 1610, "test_synth": 945, "test_scanned": 959},
        )

    def test_no_score_leaks_across_train_valid_test(self) -> None:
        self.manifest.check_no_leakage()

    def test_split_sizes(self) -> None:
        self.assertEqual(len(self.manifest.scores_in("train")), 100)
        self.assertEqual(len(self.manifest.scores_in("valid")), 11)
        self.assertEqual(len(self.manifest.scores_in("test_synth")), 11)
        self.assertEqual(len(self.manifest.scores_in("test_scanned")), 10)

    def test_test_sets_are_per_track_not_leakage(self) -> None:
        # The two test sets share scores on purpose: the same held-out works, seen once
        # per track. What must not happen is one of them also being in train or valid.
        shared = self.manifest.scores_in("test_synth") & self.manifest.scores_in("test_scanned")
        self.assertEqual(len(shared), 10)
        train_and_valid = self.manifest.scores_in("train") | self.manifest.scores_in("valid")
        self.assertEqual(shared & train_and_valid, set())

    def test_track_filtering(self) -> None:
        self.assertEqual(self.manifest.scores_in("test_synth", "scanned"), set())
        self.assertEqual(len(self.manifest.scores_in("test_scanned", "scanned")), 10)

    def test_split_for_a_known_score(self) -> None:
        # Andrée's quartet, the score B0's layout failures were first traced on.
        self.assertEqual(self.manifest.split_for("sq7313978", "synthetic"), "train")

    def test_split_for_an_unknown_score_is_none(self) -> None:
        self.assertIsNone(self.manifest.split_for("sq0", "synthetic"))

    def test_every_assignment_names_a_real_split(self) -> None:
        for tracks in self.manifest.scores.values():
            for split in tracks.values():
                self.assertIn(split, SPLIT_NAMES)

    def test_digest_is_stable_and_recorded(self) -> None:
        self.assertEqual(len(self.manifest.digest), 64)
        self.assertEqual(self.manifest.digest, load_split_manifest().digest)

    def test_provenance_is_recorded(self) -> None:
        # The split is the published one; the commit it came from is part of the record.
        self.assertIn("sqomr", self.manifest.source["repository"])
        self.assertEqual(len(self.manifest.source["commit"]), 40)


if __name__ == "__main__":
    unittest.main()
