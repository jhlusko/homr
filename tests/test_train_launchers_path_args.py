"""E1 needs every corpus path in train_scans/train_rare_numerators to come from the
caller, not a `/workspace/b0` constant - otherwise a run against public, pinned
corpora silently trains on this instance's private working copies instead.
"""

import unittest
from unittest import mock

import training.transformer.train_rare_numerators as train_rare_numerators
import training.transformer.train_scans as train_scans


class TestTrainScansPathArgs(unittest.TestCase):
    def test_ossq_index_argument_overrides_the_default_constant(self) -> None:
        custom = "/tmp/public-ossq/index_train.txt"
        with mock.patch.object(train_scans, "train_transformer") as fake_train:
            with mock.patch.object(train_scans, "write_recipe_manifest"):
                with mock.patch.object(train_scans.Path, "exists", return_value=True):
                    with mock.patch(
                        "sys.argv",
                        ["train_scans.py", "--ossq-index", custom, "--epochs", "1"],
                    ):
                        train_scans.main()

        used_dataset_index = fake_train.call_args.kwargs["dataset_index"]
        self.assertIn(custom, used_dataset_index)
        self.assertNotIn(train_scans.OSSQ_SCANNED_INDEX, used_dataset_index)


class TestTrainRareNumeratorsPathArgs(unittest.TestCase):
    def test_every_corpus_path_argument_overrides_its_default_constant(self) -> None:
        overrides = {
            "--ossq-index": "/tmp/public-ossq/index_train.txt",
            "--lieder-train-index": "/tmp/public-lieder/train.txt",
            "--lieder-val-index": "/tmp/public-lieder/valid.txt",
            "--pdmx-index": "/tmp/public-pdmx/index_train.txt",
            "--rare-index": "/tmp/public-rare/index.txt",
        }
        argv = ["train_rare_numerators.py", "--checkpoint", "/tmp/ckpt.pth"]
        for flag, value in overrides.items():
            argv += [flag, value]

        with mock.patch.object(train_rare_numerators, "train_transformer") as fake_train:
            with mock.patch.object(train_rare_numerators, "write_recipe_manifest"):
                with mock.patch.object(train_rare_numerators.Path, "is_file", return_value=True):
                    with mock.patch("sys.argv", argv):
                        train_rare_numerators.main()

        used_kwargs = fake_train.call_args.kwargs
        used_dataset_index = used_kwargs["dataset_index"]
        for flag, value in overrides.items():
            if flag == "--lieder-val-index":
                self.assertEqual(used_kwargs["validation_index"], value)
            else:
                self.assertIn(value, used_dataset_index)
        self.assertNotIn(train_rare_numerators.OSSQ_SCANNED_INDEX, used_dataset_index)
        self.assertNotIn(train_rare_numerators.IMSLP_TRAIN_INDEX, used_dataset_index)
        self.assertNotEqual(used_kwargs["validation_index"], train_rare_numerators.IMSLP_VAL_INDEX)


if __name__ == "__main__":
    unittest.main()
