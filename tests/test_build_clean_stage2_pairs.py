import json
import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.build_clean_stage2_pairs import quarantine_recovered


class TestQuarantineRecovered(unittest.TestCase):
    def test_preserves_audit_entries_without_deleting_pair_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "x.png"
            tokens = root / "x.tokens"
            image.write_bytes(b"png")
            tokens.write_text("tokens")
            source = root / "recovered.txt"
            source.write_text(f"{image},{tokens}\n")
            destination = root / "quarantine.txt"
            report = root / "quarantine.json"

            count = quarantine_recovered(source, destination, report)

            self.assertEqual(count, 1)
            self.assertEqual(destination.read_text(), source.read_text())
            self.assertTrue(image.exists())
            self.assertTrue(tokens.exists())
            self.assertFalse(json.loads(report.read_text())["recoverable_files_deleted"])


if __name__ == "__main__":
    unittest.main()
