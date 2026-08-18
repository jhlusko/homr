import sqlite3
import tempfile
import unittest
from pathlib import Path

from validation.base_domain_gap import Pair, describe, pair_up, read_scores

SCHEMA = """
CREATE TABLE samples (
    sample_id TEXT PRIMARY KEY, ned REAL, rhythm_ned REAL, pitch_ned REAL,
    lift_ned REAL, articulation_ned REAL, slur_ned REAL, error TEXT
)
"""


def _db(path: Path, rows: list[tuple]) -> Path:
    connection = sqlite3.connect(str(path))
    connection.execute(SCHEMA)
    for sample_id, ned, error in rows:
        connection.execute(
            "INSERT INTO samples (sample_id, ned, rhythm_ned, pitch_ned, lift_ned, "
            "articulation_ned, slur_ned, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (sample_id, ned, ned, ned, ned, ned, ned, error),
        )
    connection.commit()
    connection.close()
    return path


class TestReadScores(unittest.TestCase):
    def test_ned_is_normalised_from_percent_to_a_0_1_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _db(Path(tmp) / "a.db", [("s1_0001", 5.5, None)])

            self.assertAlmostEqual(read_scores(path)["s1_0001"]["ned"], 0.055)

    def test_a_failed_sample_with_no_ned_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _db(Path(tmp) / "a.db", [("s1_0001", None, "timeout")])

            self.assertEqual(read_scores(path), {})

    def test_every_component_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _db(Path(tmp) / "a.db", [("s1_0001", 2.0, None)])

            row = read_scores(path)["s1_0001"]

        self.assertIn("rhythm_ned", row)
        self.assertIn("slur_ned", row)


class TestPairUp(unittest.TestCase):
    def test_only_shared_ids_are_paired(self) -> None:
        pairs = pair_up(
            {"a": {"ned": 0.1}, "b": {"ned": 0.1}}, {"a": {"ned": 0.2}}, "ned"
        )

        self.assertEqual([p.sample_id for p in pairs], ["a"])

    def test_degradation_is_scanned_minus_synthetic(self) -> None:
        pairs = pair_up({"a": {"ned": 0.05}}, {"a": {"ned": 0.20}}, "ned")

        self.assertAlmostEqual(pairs[0].degradation, 0.15)

    def test_a_page_that_improved_on_scans_has_negative_degradation(self) -> None:
        # Rare but possible - synthetic is not always easier for every single page.
        pairs = pair_up({"a": {"ned": 0.20}}, {"a": {"ned": 0.05}}, "ned")

        self.assertLess(pairs[0].degradation, 0)


class TestDescribe(unittest.TestCase):
    def test_nothing_in_common_is_reported_not_crashed(self) -> None:
        self.assertIn("no pages in common", describe([], "ned"))

    def test_mean_ned_for_both_renderings_is_reported(self) -> None:
        pairs = [Pair("a_0001", 0.05, 0.15), Pair("a_0002", 0.03, 0.10)]

        report = describe(pairs, "ned")

        self.assertIn("synthetic 4.0%", report)
        self.assertIn("scanned 12.5%", report)

    def test_a_single_score_does_not_print_a_by_score_breakdown(self) -> None:
        # Nothing to compare a single score's rate against.
        pairs = [Pair("a_0001", 0.05, 0.15)]

        self.assertNotIn("by score", describe(pairs, "ned"))

    def test_multiple_scores_get_a_breakdown(self) -> None:
        pairs = [Pair("a_0001", 0.05, 0.15), Pair("b_0001", 0.05, 0.06)]

        report = describe(pairs, "ned")

        self.assertIn("by score (2 scores)", report)
        self.assertIn("a", report)
        self.assertIn("b", report)

    def test_negative_degradation_pages_do_not_go_negative_in_the_worst_decile_share(self) -> None:
        # An improved page contributing negative "degradation" to the worst-decile sum
        # would be nonsensical - clamped to zero instead.
        pairs = [Pair(f"a_{i:04d}", 0.05, 0.20 if i == 0 else 0.03) for i in range(10)]

        report = describe(pairs, "ned")

        self.assertIn("100.0%", report)  # all real degradation sits on the one bad page


if __name__ == "__main__":
    unittest.main()
