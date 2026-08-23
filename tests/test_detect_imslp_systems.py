import unittest
from dataclasses import dataclass

from homr import constants
from training.omr_datasets.detect_imslp_systems import GENERAL_PADDING_UNITS, _group_bounds


@dataclass
class _FakeStaff:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    average_unit_size: float


class _FakeGroup:
    def __init__(self, staffs: list[_FakeStaff]) -> None:
        self.staffs = staffs


class TestGroupBounds(unittest.TestCase):
    def test_extends_beyond_the_staff_lines_by_the_ledger_line_margin_and_padding(self) -> None:
        staff = _FakeStaff(min_x=10, max_x=110, min_y=100, max_y=140, average_unit_size=4)
        padding = GENERAL_PADDING_UNITS * 4
        ledger_margin = constants.max_number_of_ledger_lines * 4

        left, top, right, bottom = _group_bounds(_FakeGroup([staff]))

        self.assertEqual(left, 10 - padding)
        self.assertEqual(right, 110 + padding)
        self.assertEqual(top, 100 - ledger_margin - padding)
        self.assertEqual(bottom, 140 + ledger_margin + padding)

    def test_uses_each_staff_s_own_unit_size_for_its_own_margin(self) -> None:
        small = _FakeStaff(min_x=0, max_x=100, min_y=100, max_y=140, average_unit_size=2)
        large = _FakeStaff(min_x=0, max_x=100, min_y=300, max_y=340, average_unit_size=10)

        _left, top, _right, bottom = _group_bounds(_FakeGroup([small, large]))

        self.assertEqual(
            top, 100 - constants.max_number_of_ledger_lines * 2 - GENERAL_PADDING_UNITS * 2
        )
        self.assertEqual(
            bottom, 340 + constants.max_number_of_ledger_lines * 10 + GENERAL_PADDING_UNITS * 10
        )

    def test_a_zero_margin_would_regress_to_the_bare_staff_line_extent(self) -> None:
        # Documents the bug this exists to fix: without any margin, the box would be
        # exactly the staff lines' own extent, cutting off ledger lines and barlines.
        staff = _FakeStaff(min_x=0, max_x=100, min_y=100, max_y=140, average_unit_size=4)

        left, top, right, bottom = _group_bounds(_FakeGroup([staff]))

        self.assertLess(left, staff.min_x)
        self.assertGreater(right, staff.max_x)
        self.assertLess(top, staff.min_y)
        self.assertGreater(bottom, staff.max_y)


if __name__ == "__main__":
    unittest.main()
