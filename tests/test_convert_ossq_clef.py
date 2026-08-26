import unittest
import xml.etree.ElementTree as ET

from training.omr_datasets.convert_ossq import clef_of, ensure_clef, extract_part

BASS = "<clef><sign>F</sign><line>4</line></clef>"
TREBLE = "<clef><sign>G</sign><line>2</line></clef>"


def _segment(*parts: str) -> ET.Element:
    body = "".join(f"<part id='P{i}'>{p}</part>" for i, p in enumerate(parts, 1))
    return ET.fromstring(f"<score-partwise>{body}</score-partwise>")


def _measure(attributes: str = "", notes: str = "") -> str:
    attrs = f"<attributes>{attributes}</attributes>" if attributes else ""
    return f"<measure number='1'>{attrs}{notes}</measure>"


def _single(part_body: str) -> ET.Element:
    return extract_part(_segment(part_body), 0)


class TestClefOf(unittest.TestCase):
    def test_it_finds_the_clef(self) -> None:
        found = clef_of(_single(_measure(BASS)))

        self.assertEqual(found.findtext("sign"), "F")

    def test_the_last_clef_wins_when_a_part_changes_clef(self) -> None:
        # What carries to the next system is the clef in effect at the end, not the one
        # the system started in.
        body = _measure(BASS) + _measure(TREBLE)

        self.assertEqual(clef_of(_single(body)).findtext("sign"), "G")

    def test_no_clef_gives_none(self) -> None:
        self.assertIsNone(clef_of(_single(_measure("<key><fifths>-3</fifths></key>"))))


class TestEnsureClef(unittest.TestCase):
    def test_a_missing_clef_is_carried_in(self) -> None:
        # 2.4% of staves in both tracks arrive with no clef at all. The crop shows one,
        # so the label and the image disagree with nothing in either file to say why.
        single = _single(_measure("<key><fifths>-3</fifths></key>"))

        self.assertTrue(ensure_clef(single, ET.fromstring(BASS)))
        self.assertEqual(next(single.iter("clef")).findtext("sign"), "F")

    def test_an_existing_clef_is_never_overwritten(self) -> None:
        # The segment's own clef is the truth whenever it has one; carrying over it
        # would turn a correct label into a wrong one at a genuine clef change.
        single = _single(_measure(TREBLE))

        self.assertFalse(ensure_clef(single, ET.fromstring(BASS)))
        self.assertEqual(next(single.iter("clef")).findtext("sign"), "G")

    def test_nothing_to_carry_leaves_it_alone(self) -> None:
        single = _single(_measure("<key><fifths>0</fifths></key>"))

        self.assertFalse(ensure_clef(single, None))
        self.assertIsNone(next(single.iter("clef"), None))

    def test_it_creates_attributes_when_the_measure_has_none(self) -> None:
        single = _single(_measure(notes="<note><rest/></note>"))

        self.assertTrue(ensure_clef(single, ET.fromstring(BASS)))
        self.assertEqual(next(single.iter("clef")).findtext("sign"), "F")

    def test_the_clef_lands_in_the_first_measure(self) -> None:
        single = _single(_measure("<key><fifths>0</fifths></key>") + _measure())

        ensure_clef(single, ET.fromstring(BASS))
        first = single.find("part/measure")

        self.assertIsNotNone(first.find("attributes/clef"))

    def test_a_part_with_no_measures_is_left_alone(self) -> None:
        single = _single("")

        self.assertFalse(ensure_clef(single, ET.fromstring(BASS)))

    def test_the_carried_clef_is_copied_not_shared(self) -> None:
        # The same carried element is used for every following segment; inserting it
        # by reference would let one segment's tree mutate another's.
        carried = ET.fromstring(BASS)
        first = _single(_measure("<key><fifths>0</fifths></key>"))
        second = _single(_measure("<key><fifths>0</fifths></key>"))

        ensure_clef(first, carried)
        ensure_clef(second, carried)
        next(first.iter("clef")).find("sign").text = "C"

        self.assertEqual(next(second.iter("clef")).findtext("sign"), "F")


if __name__ == "__main__":
    unittest.main()
