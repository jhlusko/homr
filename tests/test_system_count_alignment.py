import unittest

from training.omr_datasets.system_count_alignment import (
    align_system_counts,
    aligned_ranges,
)


class TestSystemCountAlignment(unittest.TestCase):
    def test_one_to_one_alignment(self) -> None:
        report = align_system_counts([3, 4, 5], [3, 4, 5])

        self.assertEqual(aligned_ranges(report), {0: (0, 3), 1: (3, 7), 2: (7, 12)})

    def test_one_scan_line_can_cover_two_reference_lines(self) -> None:
        report = align_system_counts([3, 7, 2], [3, 3, 4, 2])

        self.assertEqual(aligned_ranges(report)[1], (3, 10))
        match = next(move for move in report["moves"] if move["scan_start"] == 1)
        self.assertEqual((match["source_start"], match["source_end"]), (1, 3))

    def test_two_scan_lines_can_split_one_reference_line(self) -> None:
        report = align_system_counts([3, 2, 2, 4], [3, 4, 4])

        # The grouped total proves the two scan lines contain four measures together,
        # but it does not independently prove their internal 2+2 boundary.  A one-bar
        # detector error here formerly emitted two shifted labels as ``aligned``.
        self.assertEqual(aligned_ranges(report), {0: (0, 3), 3: (7, 11)})
        self.assertEqual(report["systems"][1]["status"], "boundary_ambiguous")
        self.assertEqual(report["systems"][2]["status"], "boundary_ambiguous")

    def test_equal_group_total_does_not_certify_shifted_internal_boundaries(self) -> None:
        report = align_system_counts([3, 2], [2, 3], min_margin=0)

        self.assertEqual(aligned_ranges(report), {})
        self.assertTrue(
            all(item["status"] == "boundary_ambiguous" for item in report["systems"])
        )

    def test_false_positive_scan_system_is_skipped_without_shifting_the_score(self) -> None:
        report = align_system_counts([8, 3, 4], [3, 4])

        self.assertEqual(report["systems"][0]["status"], "skipped")
        self.assertEqual(aligned_ranges(report), {1: (0, 3), 2: (3, 7)})

    def test_missing_scan_system_skips_source_without_shifting_later_music(self) -> None:
        report = align_system_counts([3, 5], [3, 4, 5])

        self.assertEqual(aligned_ranges(report), {0: (0, 3), 1: (7, 12)})

    def test_repeated_counts_with_two_equal_paths_are_quarantined(self) -> None:
        report = align_system_counts([3], [3, 3])

        self.assertEqual(report["systems"][0]["status"], "ambiguous")
        self.assertEqual(aligned_ranges(report), {})

    def test_zero_detection_is_never_emitted_as_a_measure_range(self) -> None:
        report = align_system_counts([0, 4], [4])

        self.assertEqual(report["systems"][0]["status"], "skipped")
        self.assertEqual(aligned_ranges(report), {1: (0, 4)})

    def test_one_bar_miss_is_quarantined_without_shifting_the_suffix(self) -> None:
        report = align_system_counts([5, 5, 6, 7], [5, 6, 6, 7], min_margin=1)

        self.assertEqual(report["systems"][1]["status"], "count_mismatch")
        self.assertEqual(
            (report["systems"][2]["start_measure"], report["systems"][2]["end_measure"]),
            (11, 17),
        )
        self.assertEqual(aligned_ranges(report), {3: (17, 24)})


if __name__ == "__main__":
    unittest.main()
