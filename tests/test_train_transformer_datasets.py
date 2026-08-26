"""The dataset-selection guard in `train_transformer`.

The behaviour under test is a safety property, not a convenience: the default
five-corpus path calls `_check_datasets_are_present`, which *downloads and
converts* any missing corpus (`convert_lieder`'s own docstring warns it "can take
up to several hours"). That is right for an interactive full training run and
badly wrong for a targeted run launched unattended, so naming indexes explicitly
must never reach that path - it must verify and refuse instead.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from training.transformer import train as train_module


class TestExplicitDatasetIndexes(unittest.TestCase):
    def test_a_missing_named_index_exits_without_downloading_anything(self) -> None:
        with (
            patch.object(train_module, "Distribute") as distribute,
            patch.object(train_module, "_check_datasets_are_present") as check,
        ):
            distribute.return_value.is_rank0.return_value = True

            with self.assertRaises(SystemExit):
                train_module.train_transformer(
                    dataset_index=["/definitely/not/a/real/index.txt"],
                )

            check.assert_not_called()

    def test_mismatched_index_and_weight_lengths_exit(self) -> None:
        with TemporaryDirectory() as tmp:
            index = Path(tmp) / "index.txt"
            index.write_text("a.png,a.tokens\n")

            with patch.object(train_module, "Distribute") as distribute:
                distribute.return_value.is_rank0.return_value = True

                with self.assertRaises(SystemExit):
                    train_module.train_transformer(
                        dataset_index=[str(index)],
                        dataset_weights=[1.0, 1.0],
                    )

    def test_the_default_path_still_uses_the_auto_convert_check(self) -> None:
        # The unattended guard must not have changed how a normal interactive
        # full training run behaves.
        with (
            patch.object(train_module, "Distribute") as distribute,
            patch.object(train_module, "_check_datasets_are_present") as check,
            patch.object(train_module, "load_and_mix_training_sets") as mix,
        ):
            distribute.return_value.is_rank0.return_value = True
            check.return_value = []
            mix.side_effect = RuntimeError("stop here - past the dataset check")

            with self.assertRaises(RuntimeError):
                train_module.train_transformer()

            check.assert_called_once()


if __name__ == "__main__":
    unittest.main()
