"""Replay draws from several corpora, not just PDMX.

Replay exists so a small domain corpus does not specialise the model away from
everything else. PDMX alone was a narrow reading of that, and it is also the measured
metre gap: the numerators v4 never predicts are the ones a 1,300-pair PDMX replay showed
it about twenty times (RUNLOG IV.15.1).
"""

import argparse
import unittest

from training.transformer import train_lieder_only
from training.transformer.train_lieder_only import REPLAY_CORPORA, _replay_pair


class TestReplayPair(unittest.TestCase):
    def test_it_parses_a_corpus_and_count(self) -> None:
        self.assertEqual(_replay_pair("grandstaff=1300"), ("grandstaff", 1300))

    def test_a_missing_or_bad_count_is_rejected(self) -> None:
        for bad in ("grandstaff", "grandstaff=", "grandstaff=0", "grandstaff=-5", "grandstaff=x"):
            with self.subTest(bad=bad), self.assertRaises(argparse.ArgumentTypeError):
                _replay_pair(bad)


class TestReplayCorpora(unittest.TestCase):
    def test_every_corpus_the_project_converts_is_offered(self) -> None:
        # A corpus missing from this map cannot be replayed at all, and the omission is
        # invisible - there is no error, just a narrower mixture than intended.
        self.assertEqual(
            sorted(REPLAY_CORPORA), ["grandstaff", "musetrainer", "pdmx", "primus"]
        )

    def test_pdmx_stays_the_default(self) -> None:
        # Existing invocations must keep meaning what they meant.
        self.assertEqual(train_lieder_only.PDMX_REPLAY_COUNT, 600)
        self.assertIn("pdmx", REPLAY_CORPORA)


if __name__ == "__main__":
    unittest.main()


class TestGrandstaffIndexIsTheCurrentOne(unittest.TestCase):
    """The token-only reconversion writes a second index; they must not silently diverge."""

    def test_the_converter_warns_that_the_tmp_index_supersedes(self) -> None:
        from pathlib import Path

        source = Path("training/omr_datasets/convert_grandstaff.py").read_text(encoding="utf-8")

        self.assertIn("supersedes", source)
        # The note has to be tied to the token-only path; printing it unconditionally
        # would tell a full rebuild to promote an index it never wrote.
        self.assertIn("if only_recreate_token_files:", source)


class TestScansUsesTheV4Corpus(unittest.TestCase):
    """`train_scans` mixes OSSQ with Lieder; the Lieder half must be the corrected one."""

    def test_it_reads_the_boundary_safe_lieder_index(self) -> None:
        from training.transformer import train_scans

        # The original index is what v4 replaces - IV.10 found grouped-boundary
        # displacement fabricating cuts in it. Mixing corrected OSSQ with those labels
        # would restore on one side what was removed from the other.
        self.assertIn("v4_boundary_safe", train_scans.IMSLP_TRAIN_INDEX)
        self.assertNotEqual(train_scans.IMSLP_TRAIN_INDEX, "/workspace/b0/imslp_train_index.txt")

    def test_its_lieder_count_matches_the_v4_split(self) -> None:
        from training.transformer import train_scans

        self.assertEqual(train_scans.IMSLP_COUNT, 3622)

    def test_it_can_replay_from_several_corpora(self) -> None:
        from training.transformer import train_scans

        self.assertIs(train_scans.REPLAY_CORPORA, REPLAY_CORPORA)
