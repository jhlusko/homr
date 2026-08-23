import unittest

import numpy as np

from training.omr_datasets.olimpic_repair import (
    SAFETY_MARGIN,
    Box,
    coverage,
    extend_upward,
    repair_document,
    trim_to_gutter,
)


def _page_with_ink(rows: list[tuple[int, int]], height: int = 3000, width: int = 2000):
    """A white page with black bands over the given row ranges."""
    page = np.full((height, width, 3), 255, dtype=np.uint8)
    for top, bottom in rows:
        page[top:bottom, 100:1900] = 0
    return page


def _page(tops: list[int], height: int = 400) -> list[Box]:
    return [Box(left=60, top=top, width=1800, height=height) for top in tops]


class TestExtendUpward(unittest.TestCase):
    """A Lieder system is voice, then lyrics, then piano. The annotated box bounds only
    the piano, so the voice and its lyrics are in the gap above it."""

    def test_a_box_grows_into_the_space_above_it(self) -> None:
        boxes = _page([500, 1500, 2500])

        grown = extend_upward(boxes)

        self.assertLess(grown[1].top, boxes[1].top)
        self.assertEqual(grown[1].bottom, boxes[1].bottom)

    def test_it_stops_short_of_the_system_above(self) -> None:
        # Stems, ledger lines and slurs from the piano above overhang its box. Taking
        # them in would put one system's ink in another's image.
        boxes = _page([500, 1500])

        grown = extend_upward(boxes)

        self.assertEqual(grown[1].top, boxes[0].bottom + SAFETY_MARGIN)
        self.assertGreater(grown[1].top, boxes[0].bottom)

    def test_the_bottom_edge_never_moves(self) -> None:
        # Only the top is wrong; the piano's lower edge was annotated by hand.
        boxes = _page([500, 1500, 2500])

        for original, grown in zip(boxes, extend_upward(boxes)):
            self.assertEqual(original.bottom, grown.bottom)

    def test_the_first_system_uses_the_pages_own_gap(self) -> None:
        # It has nothing above it to stop at, so the page's typical spacing is the guide.
        boxes = _page([500, 1500, 2500])

        grown = extend_upward(boxes)

        self.assertLess(grown[0].top, 500)
        self.assertGreaterEqual(grown[0].top, 0)

    def test_nothing_extends_above_the_page(self) -> None:
        boxes = _page([50, 1500])

        grown = extend_upward(boxes)

        self.assertGreaterEqual(grown[0].top, 0)

    def test_boxes_out_of_order_are_still_handled(self) -> None:
        # The yaml is not guaranteed sorted, and reading it in the wrong order would
        # extend each box into the wrong neighbour's space.
        boxes = [Box(60, 2500, 1800, 400), Box(60, 500, 1800, 400), Box(60, 1500, 1800, 400)]

        grown = extend_upward(boxes)

        self.assertEqual([b.bottom for b in grown], [900, 1900, 2900])

    def test_a_single_system_page_still_grows(self) -> None:
        boxes = _page([1000])

        grown = extend_upward(boxes)

        self.assertLess(grown[0].top, 1000)


class TestCoverage(unittest.TestCase):
    """The measure 27.39 diagnosed with, so the repair is checked in the same terms."""

    def test_the_piano_only_case_reads_low(self) -> None:
        # Height 412 against a pitch of 1013 is what the published boxes look like.
        boxes = [Box(60, 0, 1800, 412), Box(60, 1013, 1800, 412)]

        self.assertAlmostEqual(coverage(boxes), 0.41, places=2)

    def test_extending_raises_it_towards_a_whole_system(self) -> None:
        boxes = [Box(60, 0, 1800, 412), Box(60, 1013, 1800, 412), Box(60, 2026, 1800, 412)]

        after = coverage(extend_upward(boxes))

        self.assertGreater(after, 0.9)

    def test_a_page_with_one_box_has_no_coverage_to_report(self) -> None:
        self.assertEqual(coverage([Box(60, 0, 1800, 412)]), 0.0)


class TestRepairDocument(unittest.TestCase):
    def _document(self) -> dict:
        return {
            "pages": {
                4: {
                    "height": 3069,
                    "systems": [
                        {"boundingBox": {"left": 66, "top": 482, "width": 1853, "height": 251}},
                        {"boundingBox": {"left": 69, "top": 1025, "width": 1856, "height": 258}},
                        {"boundingBox": {"left": 69, "top": 1564, "width": 1856, "height": 239}},
                    ],
                }
            }
        }

    def test_coverage_improves(self) -> None:
        _, before, after = repair_document(self._document())

        self.assertLess(before[0], 0.55)
        self.assertGreater(after[0], 0.9)

    def test_the_yaml_keeps_its_shape(self) -> None:
        # It is fed back into OLiMPiC's own cropping step, which expects this structure.
        document, _, _ = repair_document(self._document())

        box = document["pages"][4]["systems"][0]["boundingBox"]
        self.assertEqual(sorted(box), ["height", "left", "top", "width"])

    def test_a_page_with_one_system_is_left_alone(self) -> None:
        # Nothing to measure a gap against, and guessing would be worse than declining.
        document = {"pages": {1: {"height": 3069, "systems": [
            {"boundingBox": {"left": 66, "top": 482, "width": 1853, "height": 251}}]}}}

        repaired, before, after = repair_document(document)

        self.assertEqual(before, [])
        self.assertEqual(repaired["pages"][1]["systems"][0]["boundingBox"]["top"], 482)


class TestTrimToGutter(unittest.TestCase):
    """A fixed margin guesses how far the system above overhangs its box, and inspection
    found it guessing low. The page knows: the overhang ends where the ink stops."""

    def test_the_top_lands_below_the_overhang(self) -> None:
        # Ink to row 620 though the box above ends at 600 - a stem hanging past its box.
        page = _page_with_ink([(500, 620)])
        box = Box(left=60, top=1100, width=1800, height=400)

        trimmed = trim_to_gutter(page, box, ceiling=600)

        self.assertGreaterEqual(trimmed.top, 620)
        self.assertLess(trimmed.top, 700)

    def test_a_clear_ceiling_is_taken_as_given(self) -> None:
        page = _page_with_ink([])
        box = Box(left=60, top=1100, width=1800, height=400)

        self.assertEqual(trim_to_gutter(page, box, ceiling=600).top, 600)

    def test_the_bottom_still_never_moves(self) -> None:
        page = _page_with_ink([(500, 620)])
        box = Box(left=60, top=1100, width=1800, height=400)

        self.assertEqual(trim_to_gutter(page, box, ceiling=600).bottom, box.bottom)

    def test_solid_ink_all_the_way_down_falls_back_to_the_geometric_ceiling(self) -> None:
        # No clear band exists anywhere in the search window - falls back to the same
        # safe bound extend_upward already uses without a page image, not to zero
        # recovery (leaving the box unchanged would mean no lyrics recovered at all).
        page = _page_with_ink([(600, 1100)])
        box = Box(left=60, top=1100, width=1800, height=400)

        trimmed = trim_to_gutter(page, box, ceiling=600)

        self.assertEqual(trimmed.top, 600)
        self.assertEqual(trimmed.bottom, box.bottom)

    def test_extending_with_a_page_trims_where_geometry_would_not(self) -> None:
        page = _page_with_ink([(500, 700)])
        boxes = [Box(60, 100, 1800, 500), Box(60, 1100, 1800, 400)]

        geometric = extend_upward(boxes)[1]
        informed = extend_upward(boxes, page)[1]

        self.assertLess(geometric.top, 700)
        self.assertGreaterEqual(informed.top, 700)


if __name__ == "__main__":
    unittest.main()
