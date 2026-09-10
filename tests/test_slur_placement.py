import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

from training.omr_datasets.slur_placement import (
    Alignment,
    PlacementIndex,
    apply_placements,
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




SEG_TEMPLATE = (
    '<score-partwise><part id="P1"><measure number="1">{notes}</measure></part>'
    "</score-partwise>"
)


def _slurred(step: str, number: str = "1", kind: str = "start") -> str:
    return (
        f'<note><pitch><step>{step}</step><octave>5</octave></pitch>'
        f"<duration>1</duration><type>eighth</type>"
        f'<notations><slur type="{kind}" number="{number}"/></notations></note>'
    )


class TestPlacementIndex(unittest.TestCase):
    """Slices are precomputed per segment rather than consumed as a running cursor.

    A cursor would have to advance for every segment whether or not it gets converted - a
    system skipped for a crop mismatch still occupies notes in the score - and one missed
    advance would shift every later placement onto the wrong note.
    """

    def _score(self, root: Path, aligned: bool = True) -> Path:
        work = root / "scores" / "C" / "W"
        segments = work / "musicxml" / "unaligned"
        segments.mkdir(parents=True)
        whole_notes = _slurred("C") + _slurred("D", kind="stop") + _slurred("E")
        placement = whole_notes.replace('type="start" number="1"', 'type="start" number="1" placement="above"')
        (work / "sq1.musicxml").write_text(SEG_TEMPLATE.format(notes=placement), encoding="utf-8")

        first = _slurred("C") + _slurred("D", kind="stop")
        second = _slurred("E") if aligned else _slurred("G")
        (segments / "sq1:0001:0001.musicxml").write_text(
            SEG_TEMPLATE.format(notes=first), encoding="utf-8"
        )
        (segments / "sq1:0001:0002.musicxml").write_text(
            SEG_TEMPLATE.format(notes=second), encoding="utf-8"
        )
        return work

    def test_each_segment_gets_its_own_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = self._score(Path(tmp))

            index = PlacementIndex(work, "sq1", work / "sq1.musicxml")

            self.assertEqual(index.aligned_parts, 1)
            self.assertEqual(len(index.for_segment(1, 1, 0)), 2)
            self.assertEqual(len(index.for_segment(1, 2, 0)), 1)

    def test_the_slice_carries_the_placement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = self._score(Path(tmp))

            index = PlacementIndex(work, "sq1", work / "sq1.musicxml")

            self.assertEqual(index.for_segment(1, 1, 0)[0], {"1": "above"})

    def test_a_part_that_does_not_align_contributes_nothing(self) -> None:
        # A wrong placement is worse than none: it trains a head to read direction off
        # the wrong note.
        with tempfile.TemporaryDirectory() as tmp:
            work = self._score(Path(tmp), aligned=False)

            index = PlacementIndex(work, "sq1", work / "sq1.musicxml")

            self.assertEqual(index.aligned_parts, 0)
            self.assertEqual(index.skipped_parts, 1)
            self.assertIsNone(index.for_segment(1, 1, 0))


class TestApplyPlacements(unittest.TestCase):
    def test_placement_is_written_onto_the_matching_slur(self) -> None:
        part = ET.fromstring(f"<part><measure>{_slurred('C')}</measure></part>")

        applied = apply_placements(part, [{"1": "below"}])

        self.assertEqual(applied, 1)
        self.assertEqual(part.find(".//slur").get("placement"), "below")

    def test_a_slur_number_that_does_not_match_is_left_alone(self) -> None:
        # Numbering survives segmentation, so a mismatch means the join is wrong, not
        # that the placement belongs to whichever slur happens to be there.
        part = ET.fromstring(f"<part><measure>{_slurred('C', number='2')}</measure></part>")

        applied = apply_placements(part, [{"1": "below"}])

        self.assertEqual(applied, 0)
        self.assertIsNone(part.find(".//slur").get("placement"))

    def test_a_length_disagreement_applies_nothing(self) -> None:
        part = ET.fromstring(f"<part><measure>{_slurred('C')}</measure></part>")

        self.assertEqual(apply_placements(part, [{}, {"1": "above"}]), 0)

    def test_an_existing_placement_is_not_overwritten(self) -> None:
        notes = _slurred("C").replace('number="1"', 'number="1" placement="above"')
        part = ET.fromstring(f"<part><measure>{notes}</measure></part>")

        apply_placements(part, [{"1": "below"}])

        self.assertEqual(part.find(".//slur").get("placement"), "above")




class TestApplyPlacementsRefusesTheWrongElement(unittest.TestCase):
    """The failure that made the first re-conversion produce no placement at all.

    extract_part returns a <score-partwise> root; apply_placements walks <measure>
    children and so found none, dropped every placement, and reported it as an ordinary
    length mismatch. 457 slices carried placement and not one landed.
    """

    def test_a_score_element_is_refused_rather_than_silently_dropped(self) -> None:
        score = ET.fromstring(
            f'<score-partwise><part id="P1"><measure>{_slurred("C")}</measure></part>'
            "</score-partwise>"
        )

        with self.assertRaises(ValueError):
            apply_placements(score, [{"1": "above"}])

    def test_its_part_child_works(self) -> None:
        score = ET.fromstring(
            f'<score-partwise><part id="P1"><measure>{_slurred("C")}</measure></part>'
            "</score-partwise>"
        )

        applied = apply_placements(score.find("part"), [{"1": "above"}])

        self.assertEqual(applied, 1)


if __name__ == "__main__":
    unittest.main()
