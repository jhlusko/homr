import unittest
from unittest.mock import MagicMock

from homr.system_grouping import GroupingResult, SystemPartition, find_system_grouping


def _staffs(spans: list[tuple[float, float]], unit_size: float = 10.0) -> list[MagicMock]:
    """Build staffs from (min_y, max_y) pairs; only geometry matters here."""
    staffs = []
    for min_y, max_y in spans:
        staff = MagicMock()
        staff.min_y = min_y
        staff.max_y = max_y
        staff.average_unit_size = unit_size
        staffs.append(staff)
    return staffs


def _from_gaps(gaps: list[float], height: float = 45.0, unit_size: float = 10.0) -> list[MagicMock]:
    """Build staffs whose successive gaps (in unit sizes) are exactly `gaps`."""
    spans = [(0.0, height)]
    for gap in gaps:
        min_y = spans[-1][1] + gap * unit_size
        spans.append((min_y, min_y + height))
    return _staffs(spans, unit_size)


def _require(result: GroupingResult | None) -> GroupingResult:
    if result is None:
        raise AssertionError("expected a grouping result")
    return result


def _require_partition(partition: SystemPartition | None) -> SystemPartition:
    if partition is None:
        raise AssertionError("expected a competing partition")
    return partition


# Measured from homr's own staff detection on sq7313978:0001.png (Andrée, String Quartet
# in A major, page 1) - the page whose bracket detection produced the inconsistent rows
# [3, 4, 3, 1, 4, 4]. Ground truth is five 4-staff systems; detection found 19 of the 20
# staves, missing one inside the first system, which is what leaves the 14.80 gap.
_ANDREE_PAGE_1_GAPS = [
    4.87, 14.80,             # system 1: 3 staves detected of 4, the 14.80 is the missed one
    8.70, 4.48, 4.74, 6.51,  # cut, then system 2
    9.12, 3.65, 4.96, 3.76,  # cut, then system 3
    9.04, 4.68, 5.92, 6.37,  # cut, then system 4
    8.82, 5.41, 4.69, 6.65,  # cut, then system 5
]  # fmt: skip


class TestRealQuartetPage(unittest.TestCase):
    def test_recovers_five_systems_from_the_page_that_collapsed(self) -> None:
        result = _require(find_system_grouping(_from_gaps(_ANDREE_PAGE_1_GAPS), set()))

        self.assertTrue(result.confident)
        self.assertEqual(result.best.staves_per_system, 4)
        self.assertEqual([len(group) for group in result.best.groups], [3, 4, 4, 4, 4])

    def test_the_missed_staff_does_not_move_a_boundary(self) -> None:
        # The 14.80 gap is the largest on the page but sits *inside* system 1. A rule
        # that cut at the largest gaps would put a system boundary there.
        result = _require(find_system_grouping(_from_gaps(_ANDREE_PAGE_1_GAPS), set()))

        self.assertEqual(result.best.groups[0], (0, 1, 2))

    def test_bracket_evidence_that_agrees_leaves_the_answer_alone(self) -> None:
        # The rows the bracket detector did produce, as adjacent-index pairs.
        connected = {(0, 1), (1, 2), (3, 4), (4, 5), (5, 6), (7, 8), (8, 9)}
        connected |= {(11, 12), (12, 13), (13, 14), (15, 16), (16, 17), (17, 18)}

        result = _require(find_system_grouping(_from_gaps(_ANDREE_PAGE_1_GAPS), connected))

        self.assertTrue(result.confident)
        self.assertEqual(result.best.broken_connections, 0)
        self.assertEqual([len(g) for g in result.best.groups], [3, 4, 4, 4, 4])


class TestPagesThatMustNotBeRegrouped(unittest.TestCase):
    def _assert_not_regrouped(self, gaps: list[float]) -> None:
        result = find_system_grouping(_from_gaps(gaps), set())
        self.assertTrue(result is None or not result.confident)

    def test_evenly_spaced_single_staves(self) -> None:
        # A solo part: every gap is a system gap, so there is no split to find.
        self._assert_not_regrouped([7.0] * 15)

    def test_slightly_irregular_single_staves(self) -> None:
        self._assert_not_regrouped([7.0, 7.6, 6.8, 7.2, 7.9, 6.6, 7.4, 7.1, 6.9, 7.7, 7.3, 6.7])

    def test_too_few_systems_to_read(self) -> None:
        # Two 4-staff systems: real, but not enough repetition for geometry to carry it.
        self._assert_not_regrouped([4.0, 4.0, 4.0, 9.0, 4.0, 4.0, 4.0])

    def test_a_single_staff_page_returns_nothing(self) -> None:
        self.assertIsNone(find_system_grouping(_from_gaps([]), set()))


class TestOtherLayouts(unittest.TestCase):
    def test_piano_grand_staff_pages_group_in_twos(self) -> None:
        gaps = [3.5, 10.0, 3.6, 9.8, 3.4, 10.2, 3.5, 9.9, 3.6]
        result = _require(find_system_grouping(_from_gaps(gaps), set()))

        self.assertTrue(result.confident)
        self.assertEqual(result.best.staves_per_system, 2)
        self.assertEqual([len(g) for g in result.best.groups], [2, 2, 2, 2, 2])

    def test_voice_plus_piano_groups_in_threes(self) -> None:
        gaps = [4.0, 3.8, 10.5, 4.1, 3.9, 10.2, 4.0, 3.7, 10.4, 4.2, 3.8]
        result = _require(find_system_grouping(_from_gaps(gaps), set()))

        self.assertTrue(result.confident)
        self.assertEqual(result.best.staves_per_system, 3)

    def test_an_incomplete_final_system_is_allowed(self) -> None:
        gaps = [4.0, 4.0, 4.0, 9.0] * 4 + [4.0]
        result = _require(find_system_grouping(_from_gaps(gaps), set()))

        self.assertTrue(result.confident)
        self.assertEqual([len(g) for g in result.best.groups], [4, 4, 4, 4, 2])

    def test_a_system_short_in_the_middle_of_the_page_is_allowed(self) -> None:
        # Staff detection missing one staff out of a complete system leaves a short
        # system nowhere near a page edge. Observed on consecutive pages of the same
        # quartet, where the bracket rows read [4, 4, 3, 4, 4] and [4, 3, 4, 4, 4].
        gaps = (
            [4.0, 4.0, 4.0, 9.0]  # system 1
            + [4.0, 4.0, 4.0, 9.0]  # system 2
            + [4.0, 4.0, 9.0]  # system 3, one staff missing
            + [4.0, 4.0, 4.0, 9.0]  # system 4
            + [4.0, 4.0, 4.0]  # system 5
        )
        result = _require(find_system_grouping(_from_gaps(gaps), set()))

        self.assertTrue(result.confident)
        self.assertEqual(result.best.staves_per_system, 4)
        self.assertEqual([len(g) for g in result.best.groups], [4, 4, 3, 4, 4])

    def test_a_system_short_at_the_front_is_allowed(self) -> None:
        gaps = [4.0, 4.0, 9.0] + [4.0, 4.0, 4.0, 9.0] * 3 + [4.0, 4.0, 4.0]
        result = _require(find_system_grouping(_from_gaps(gaps), set()))

        self.assertTrue(result.confident)
        self.assertEqual([len(g) for g in result.best.groups], [3, 4, 4, 4, 4])

    def test_a_partition_that_splits_bracketed_staves_is_refused(self) -> None:
        # Claim a bracket across every boundary the geometric answer wants to cut.
        hostile = {(2, 3), (6, 7), (10, 11), (14, 15)}
        result = _require(find_system_grouping(_from_gaps(_ANDREE_PAGE_1_GAPS), hostile))

        self.assertFalse(result.confident)

    def test_an_unambiguous_page_has_no_competing_partition(self) -> None:
        # Only one candidate survives the per-cut ordering gate on the quartet page, so
        # the absence of a runner-up is the signal, not a gap in the result.
        result = _require(find_system_grouping(_from_gaps(_ANDREE_PAGE_1_GAPS), set()))

        self.assertIsNone(result.runner_up)

    def test_a_page_that_reads_two_ways_keeps_the_competitor(self) -> None:
        # Gaps alternating small/large at two scales: readable as systems of 2 or of 4.
        gaps = [3.0, 9.0, 3.0, 12.0, 3.0, 9.0, 3.0, 12.0, 3.0, 9.0, 3.0]
        result = _require(find_system_grouping(_from_gaps(gaps), set()))

        runner_up = _require_partition(result.runner_up)
        self.assertLessEqual(runner_up.score, result.best.score)
        self.assertNotEqual(runner_up.staves_per_system, result.best.staves_per_system)


if __name__ == "__main__":
    unittest.main()
