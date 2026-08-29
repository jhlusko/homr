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
