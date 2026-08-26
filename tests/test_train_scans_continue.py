import unittest
from pathlib import Path

import training.transformer.train as train_module
from training.transformer import train_scans_continue


class TestContinuationConfig(unittest.TestCase):
    def test_it_reads_the_clef_corrected_corpus(self) -> None:
        # Pointing at the previous corpus would train more epochs on exactly the labels
        # this run exists to replace, and nothing in the output would say so.
        self.assertIn("phase7clef", train_scans_continue.OSSQ_SCANNED_INDEX)
        self.assertNotIn("phase7fix", train_scans_continue.OSSQ_SCANNED_INDEX)

    def test_validation_spans_both_domains(self) -> None:
        # A Lieder-only held-out set never measures the OSSQ half, which is how 56.7%
        # of it being mislabeled stayed invisible.
        self.assertIn("mixed", train_scans_continue.MIXED_VAL_INDEX)

    def test_replay_is_retained(self) -> None:
        self.assertGreater(train_scans_continue.PDMX_REPLAY_COUNT, 0)


class TestCheckpointOverride(unittest.TestCase):
    def test_train_transformer_accepts_an_explicit_checkpoint(self) -> None:
        # The override has to reach the Config that train_transformer builds itself;
        # setting it on a Config in the caller has no effect at all.
        import inspect

        signature = inspect.signature(train_module.train_transformer)

        self.assertIn("checkpoint", signature.parameters)
        self.assertIsNone(signature.parameters["checkpoint"].default)

    def test_the_pinned_checkpoint_is_not_modified_on_disk(self) -> None:
        # Continuing one experiment must not redirect production and every other run.
        source = Path(train_module.__file__).read_text(encoding="utf-8")

        self.assertIn("config.filepaths.checkpoint = checkpoint", source)
        self.assertNotIn("shutil.copy(checkpoint", source)


if __name__ == "__main__":
    unittest.main()
