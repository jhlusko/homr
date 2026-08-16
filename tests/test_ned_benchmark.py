import sqlite3
import tempfile
import unittest
from pathlib import Path

from validation.ned_benchmark import Sample, run_benchmark

_KERN = "**kern\n*M4/4\n*clefG2\n4c\n4d\n4e\n4f\n*-\n"


def _samples(count: int = 2) -> list[Sample]:
    return [Sample(f"s{i}", _KERN, Path(f"/nonexistent/{i}.png")) for i in range(count)]


def _rows(db_path: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT sample_id, actual_text, ned, error FROM samples ORDER BY sample_id"
        ).fetchall()
    finally:
        conn.close()


class TestSingleSampleMode(unittest.TestCase):
    """Tools without batch_run (music21, hum2xml, oemer, transcoda) return their output
    rather than writing it onto the sample; run_benchmark has to store it."""

    def test_plain_tool_output_is_scored(self) -> None:
        def perfect(kern_text: str, image: Path | None) -> str:
            return kern_text

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "out.db")
            run_benchmark(_samples(), perfect, num_workers=1, output_db=db_path)
            rows = _rows(db_path)

        self.assertEqual([r[0] for r in rows], ["s0", "s1"])
        for _, actual_text, ned, error in rows:
            self.assertEqual(actual_text, _KERN)
            self.assertEqual(ned, 0.0)
            self.assertIsNone(error)

    def test_a_raising_tool_is_recorded_as_a_failure_with_no_output(self) -> None:
        def broken(kern_text: str, image: Path | None) -> str:
            raise RuntimeError("tool exploded")

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "out.db")
            run_benchmark(_samples(1), broken, num_workers=1, output_db=db_path)
            rows = _rows(db_path)

        self.assertEqual(len(rows), 1)
        _, actual_text, _, error = rows[0]
        # NULL, not "", so --update skips it instead of rescoring an empty output.
        self.assertIsNone(actual_text)
        self.assertIn("tool exploded", error)

    def test_one_failure_does_not_leak_output_into_the_next_sample(self) -> None:
        seen: list[str] = []

        def flaky(kern_text: str, image: Path | None) -> str:
            seen.append("call")
            if len(seen) == 1:
                return kern_text
            raise RuntimeError("second call fails")

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "out.db")
            run_benchmark(_samples(2), flaky, num_workers=1, output_db=db_path)
            rows = _rows(db_path)

        self.assertEqual(rows[0][1], _KERN)
        self.assertIsNone(rows[1][1])


class TestBatchMode(unittest.TestCase):
    def test_batch_tool_results_are_scored(self) -> None:
        # Dual-mode, like HomrTool and SmtTool: callable for one sample, batch_run for many.
        class PerfectBatchTool:
            def __call__(self, kern_text: str, image: Path | None) -> str:
                return kern_text

            def batch_run(self, batch: list[Sample]) -> None:
                for sample in batch:
                    sample.set_success(sample.kern_text)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "out.db")
            run_benchmark(_samples(3), PerfectBatchTool(), num_workers=1, output_db=db_path)
            rows = _rows(db_path)

        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row[2] == 0.0 for row in rows))

    def test_batch_tool_that_marks_nothing_fails_cleanly(self) -> None:
        class SilentBatchTool:
            def __call__(self, kern_text: str, image: Path | None) -> str:
                return kern_text

            def batch_run(self, batch: list[Sample]) -> None:
                return

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "out.db")
            run_benchmark(_samples(1), SilentBatchTool(), num_workers=1, output_db=db_path)
            rows = _rows(db_path)

        # Scored as an empty prediction rather than blowing up on a missing attribute.
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
