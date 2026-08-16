import unittest
import xml.etree.ElementTree as ET

from training.omr_datasets.slur_placement import (
    Alignment,
    is_visible,
    note_signature,
    part_placements,
    part_signature,
)

VISIBLE = """
  <note><pitch><step>C</step><octave>5</octave></pitch>
    <duration>1</duration><type>eighth</type>{notations}</note>
"""
INVISIBLE = """
  <note print-object="no"><pitch><step>D</step><octave>5</octave></pitch>
    <duration>1</duration><type>eighth</type></note>
"""
GRACE = """
  <note><grace/><pitch><step>E</step><octave>5</octave></pitch><type>16th</type></note>
"""


def _part(notes: str) -> ET.Element:
    return ET.fromstring(f"<part><measure>{notes}</measure></part>")


class TestVisibility(unittest.TestCase):
    def test_an_invisible_note_is_excluded(self) -> None:
        # The segmentation drops these, so a whole-score walk that keeps them runs ahead
        # and every placement after the first lands on the wrong note.
        signatures = part_signature(_part(VISIBLE.format(notations="") + INVISIBLE))

        self.assertEqual(len(signatures), 1)

    def test_a_grace_note_is_kept(self) -> None:
        # Grace notes were the other suspect and survive segmentation intact; dropping
        # them would break the alignment this exists to provide.
        signatures = part_signature(_part(VISIBLE.format(notations="") + GRACE))

        self.assertEqual(len(signatures), 2)

    def test_visibility_defaults_to_printed(self) -> None:
        note = ET.fromstring("<note><type>eighth</type></note>")

        self.assertTrue(is_visible(note))


class TestSignature(unittest.TestCase):
    def test_pitch_and_type_distinguish_notes(self) -> None:
        # A count alone cannot: a dropped note and an added one cancel out.
        first = note_signature(
            ET.fromstring(
                "<note><pitch><step>C</step><octave>5</octave></pitch>"
                "<type>eighth</type></note>"
            )
        )
        second = note_signature(
            ET.fromstring(
                "<note><pitch><step>D</step><octave>5</octave></pitch>"
                "<type>eighth</type></note>"
            )
        )

        self.assertNotEqual(first, second)

    def test_a_rest_is_distinguishable_from_a_note(self) -> None:
        rest = note_signature(ET.fromstring("<note><rest/><type>eighth</type></note>"))
        note = note_signature(ET.fromstring("<note><type>eighth</type></note>"))

        self.assertNotEqual(rest, note)


class TestPlacements(unittest.TestCase):
    def test_a_stated_placement_is_read(self) -> None:
        notations = '<notations><slur type="start" number="1" placement="above"/></notations>'

        placements = part_placements(_part(VISIBLE.format(notations=notations)))

        self.assertEqual(placements, [{"1": "above"}])

    def test_orientation_is_accepted_where_placement_is_absent(self) -> None:
        notations = '<notations><slur type="start" number="1" orientation="under"/></notations>'

        placements = part_placements(_part(VISIBLE.format(notations=notations)))

        self.assertEqual(placements, [{"1": "under"}])

    def test_a_slur_with_no_direction_contributes_nothing(self) -> None:
        # Nothing to transfer, which is different from there being no slur - both are
        # empty here because both mean the label cannot be improved.
        notations = '<notations><slur type="start" number="1"/></notations>'

        placements = part_placements(_part(VISIBLE.format(notations=notations)))

        self.assertEqual(placements, [{}])

    def test_placements_line_up_with_the_signature(self) -> None:
        # The two walks must agree on which notes exist, or a placement transfers onto
        # its neighbour.
        notations = '<notations><slur type="start" number="1" placement="below"/></notations>'
        notes = VISIBLE.format(notations=notations) + INVISIBLE + VISIBLE.format(notations="")

        self.assertEqual(
            len(part_placements(_part(notes))), len(part_signature(_part(notes)))
        )


class TestAlignmentReport(unittest.TestCase):
    def test_the_rate_is_over_parts_checked(self) -> None:
        alignment = Alignment(parts_checked=4, parts_aligned=3)

        self.assertEqual(alignment.rate, 0.75)

    def test_an_empty_report_does_not_divide_by_zero(self) -> None:
        self.assertEqual(Alignment().rate, 0.0)
        self.assertIn("0", Alignment().describe())


if __name__ == "__main__":
    unittest.main()
