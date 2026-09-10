import unittest
import xml.etree.ElementTree as ET

from training.omr_datasets.barline_placement import apply_barlines, part_barlines

REPEAT_END = (
    "<barline location='right'><bar-style>light-heavy</bar-style>"
    "<repeat direction='backward'/></barline>"
)
REPEAT_START = (
    "<barline location='left'><bar-style>heavy-light</bar-style>"
    "<repeat direction='forward'/></barline>"
)


def _part(*measures: str) -> ET.Element:
    body = "".join(f"<measure number='{i}'>{m}</measure>" for i, m in enumerate(measures, 1))
    return ET.fromstring(f"<part id='P1'>{body}</part>")


class TestPartBarlines(unittest.TestCase):
    def test_it_reports_one_entry_per_measure(self) -> None:
        part = _part("", REPEAT_END, "")

        self.assertEqual([len(m) for m in part_barlines(part)], [0, 1, 0])

    def test_a_measure_with_two_barlines_keeps_both(self) -> None:
        # A measure can be both the start and the end of a repeated section.
        part = _part(REPEAT_START + REPEAT_END)

        self.assertEqual(len(part_barlines(part)[0]), 2)

    def test_the_elements_are_copies_not_references(self) -> None:
        # The index is reused across every segment of a score; handing out references
        # would let one segment's edits mutate another's.
        part = _part(REPEAT_END)

        extracted = part_barlines(part)[0][0]
        extracted.set("location", "left")

        self.assertEqual(part.find("measure/barline").get("location"), "right")

    def test_a_part_with_no_barlines_gives_empty_lists(self) -> None:
        self.assertEqual(part_barlines(_part("", "")), [[], []])


class TestApplyBarlines(unittest.TestCase):
    def test_a_right_barline_lands_at_the_end_of_its_measure(self) -> None:
        part = _part("<note><rest/></note>")

        applied = apply_barlines(part, [[ET.fromstring(REPEAT_END)]])

        self.assertEqual(applied, 1)
        self.assertEqual(list(part.find("measure"))[-1].tag, "barline")

    def test_a_left_barline_lands_at_the_head_of_its_measure(self) -> None:
        part = _part("<note><rest/></note>")

        apply_barlines(part, [[ET.fromstring(REPEAT_START)]])

        self.assertEqual(list(part.find("measure"))[0].tag, "barline")

    def test_a_left_barline_goes_after_attributes_not_before(self) -> None:
        # <attributes> must be the first child; a barline ahead of it makes the file
        # invalid, and MuseScore's reaction to that is not something to rely on.
        part = _part("<attributes><divisions>1</divisions></attributes><note><rest/></note>")

        apply_barlines(part, [[ET.fromstring(REPEAT_START)]])

        children = [child.tag for child in part.find("measure")]
        self.assertEqual(children[0], "attributes")
        self.assertEqual(children[1], "barline")

    def test_barlines_land_in_the_measure_they_belong_to(self) -> None:
        part = _part("", "", "")

        apply_barlines(part, [[], [ET.fromstring(REPEAT_END)], []])

        measures = part.findall("measure")
        self.assertEqual(len(measures[0].findall("barline")), 0)
        self.assertEqual(len(measures[1].findall("barline")), 1)
        self.assertEqual(len(measures[2].findall("barline")), 0)

    def test_the_repeat_direction_survives(self) -> None:
        # The whole point: a repeat changes how the music is played, so the direction
        # has to arrive intact rather than as a bare double bar.
        part = _part("")

        apply_barlines(part, [[ET.fromstring(REPEAT_END)]])

        self.assertEqual(part.find("measure/barline/repeat").get("direction"), "backward")

    def test_fewer_slices_than_measures_is_not_an_error(self) -> None:
        # A short slice leaves the remaining measures untouched rather than raising:
        # partial recovery is still useful, and the alignment gate upstream is what
        # decides whether to trust the part at all.
        part = _part("", "", "")

        applied = apply_barlines(part, [[ET.fromstring(REPEAT_END)]])

        self.assertEqual(applied, 1)

    def test_it_refuses_anything_that_is_not_a_part(self) -> None:
        with self.assertRaises(ValueError):
            apply_barlines(ET.Element("measure"), [])

    def test_applying_twice_would_duplicate_so_callers_apply_once(self) -> None:
        # Documents the contract rather than guarding it: this is called once per
        # generated segment, on a freshly extracted part.
        part = _part("")

        apply_barlines(part, [[ET.fromstring(REPEAT_END)]])
        apply_barlines(part, [[ET.fromstring(REPEAT_END)]])

        self.assertEqual(len(part.findall("measure/barline")), 2)


if __name__ == "__main__":
    unittest.main()
