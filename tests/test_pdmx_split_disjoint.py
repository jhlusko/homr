import unittest
from pathlib import Path

from training.omr_datasets.convert_pdmx import (
    pdmx_index,
    pdmx_train_index,
    pdmx_valid_index,
)


class TestPdmxSplits(unittest.TestCase):
    """`index.txt` holds every converted row, validation included. Replay pointed at
    it, so every run - 426, 447 and 448 alike - drew its 1,300 replay pairs from a pool
    containing the whole 3,349-row validation set, and every PDMX figure this project
    has quoted was measured against rows the model may have trained on. A clean split
    already existed beside it and nothing used it."""

    def test_training_does_not_read_the_combined_index(self) -> None:
        self.assertNotEqual(pdmx_train_index, pdmx_index)
        self.assertTrue(pdmx_train_index.endswith("index_train.txt"))

    def test_the_validation_split_is_named_separately(self) -> None:
        self.assertTrue(pdmx_valid_index.endswith("index_valid.txt"))
        self.assertNotEqual(pdmx_valid_index, pdmx_train_index)

    def test_the_splits_are_disjoint_on_disk(self) -> None:
        train, valid = Path(pdmx_train_index), Path(pdmx_valid_index)
        if not train.is_file() or not valid.is_file():
            self.skipTest("PDMX not present in this environment")
        a = {line.strip() for line in train.read_text().splitlines() if line.strip()}
        b = {line.strip() for line in valid.read_text().splitlines() if line.strip()}
        self.assertTrue(a and b)
        self.assertEqual(a & b, set(), "train and validation rows must not overlap")


if __name__ == "__main__":
    unittest.main()
