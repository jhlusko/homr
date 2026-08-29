import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.omr_datasets import compare_bar_counts
from training.omr_datasets.compare_bar_counts import measure_count_from_barline_centers


class TestMeasureCountFromBarlineCenters(unittest.TestCase):
    def test_counts_interior_dividers_plus_one(self) -> None:
        self.assertEqual(
            measure_count_from_barline_centers([25, 25.5, 50, 50.5, 75, 75.5], 0, 100),
            4,
        )

    def test_edge_barlines_do_not_add_measures(self) -> None:
        self.assertEqual(
            measure_count_from_barline_centers(
                [0, 0.5, 25, 25.5, 50, 50.5, 75, 75.5, 99.5, 100], 0, 100
            ),
            4,
        )

    def test_nearby_staff_barlines_are_clustered(self) -> None:
        self.assertEqual(
            measure_count_from_barline_centers([24.5, 25.0, 25.5, 75, 75.5], 0, 100), 3
        )

    def test_a_lone_stem_like_vertical_is_not_a_measure_divider(self) -> None:
        self.assertEqual(
            measure_count_from_barline_centers([25, 25.5, 50, 75, 75.5], 0, 100), 3
        )

    def test_no_detection_is_unknown_not_one_measure(self) -> None:
        self.assertEqual(measure_count_from_barline_centers([], 0, 100), 0)


class TestFailuresAreNotSilent(unittest.TestCase):
    """A shard that loses scores must say so - see the 2026-08-27 pid-ceiling run,
    where 130 of 330 scores died in the per-score handler and two shards still wrote
    an empty rows file and exited 0."""

    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        gt_dir = root / "gt"
        systems_dir = root / "systems"
        png_dir = root / "pngs"
        gt_dir.mkdir()
        systems_dir.mkdir()
        (png_dir / "IMSLP1").mkdir(parents=True)
        # The page named by the systems file must really exist: the root selector
        # requires it, precisely so a root that merely has a directory of the right
        # name cannot be chosen (IMSLP621830/IMSLP622484 exist under both roots with
        # different file naming).
        (png_dir / "IMSLP1" / "IMSLP1-p001.png").write_bytes(b"png")
        (gt_dir / "IMSLP1.json").write_text(json.dumps({"pages": []}), encoding="utf-8")
        (systems_dir / "IMSLP1.yaml").write_text(
            "pages:\n  1:\n    image: IMSLP1/IMSLP1-p001.png\n    systems: []\n",
            encoding="utf-8",
        )
        return gt_dir, systems_dir, png_dir

    def _run(
        self, root: Path, rows_out: Path, failed_out: Path, coverage_out: Path | None = None
    ) -> int:
        gt_dir, systems_dir, png_dir = self._fixture(root)
        argv = [
            "compare_bar_counts",
            "--ground-truth", str(gt_dir),
            "--systems", str(systems_dir),
            "--pngs", str(png_dir),
            "--rows-out", str(rows_out),
            "--failed-out", str(failed_out),
        ]
        if coverage_out:
            argv.extend(["--coverage-out", str(coverage_out)])
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            compare_bar_counts, "compare_one_score", side_effect=RuntimeError("pthread_create failed")
        ):
            try:
                compare_bar_counts.main()
            except SystemExit as e:
                return int(e.code or 0)
        return 0

    def test_a_failed_score_is_recorded_and_the_run_exits_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed_out = root / "failed.json"
            code = self._run(root, root / "rows.json", failed_out)
            self.assertEqual(code, 1)
            recorded = json.loads(failed_out.read_text(encoding="utf-8"))
            self.assertEqual([entry["score_id"] for entry in recorded], ["IMSLP1"])
            self.assertIn("pthread_create", recorded[0]["reason"])

    def test_an_all_failed_run_writes_no_rows_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_out = root / "rows.json"
            self._run(root, rows_out, root / "failed.json")
            self.assertFalse(rows_out.exists())

    def test_coverage_records_the_failed_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coverage_out = root / "coverage.json"
            self._run(root, root / "rows.json", root / "failed.json", coverage_out)
            coverage = json.loads(coverage_out.read_text(encoding="utf-8"))
            self.assertEqual(coverage["expected_score_ids"], ["IMSLP1"])
            self.assertEqual(coverage["completed_score_ids"], [])
            self.assertEqual(coverage["failed_score_ids"], ["IMSLP1"])


if __name__ == "__main__":
    unittest.main()
