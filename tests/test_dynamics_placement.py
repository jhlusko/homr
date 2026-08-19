import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from homr.transformer.structured_notation import DynamicMark
from training.omr_datasets.dynamics_placement import (
    DynamicsPlacementIndex,
    apply_dynamics,
    part_dynamics,
)

SEG_TEMPLATE = (
    '<score-partwise><part id="P1"><measure number="1">{notes}</measure></part>'
    "</score-partwise>"
)


def _note(step: str = "C") -> str:
    return (
        f"<note><pitch><step>{step}</step><octave>5</octave></pitch>"
        "<duration>1</duration><type>eighth</type></note>"
    )


def _invisible(step: str = "D") -> str:
    return (
        f'<note print-object="no"><pitch><step>{step}</step><octave>5</octave></pitch>'
        "<duration>1</duration><type>eighth</type></note>"
    )


def _direction(mark: str) -> str:
    return f"<direction><direction-type><dynamics><{mark}/></dynamics></direction-type></direction>"


def _part(notes: str) -> ET.Element:
    return ET.fromstring(f"<part><measure>{notes}</measure></part>")  # noqa: S314


class TestPartDynamics(unittest.TestCase):
    def test_a_dynamic_attaches_to_the_next_visible_note(self) -> None:
        marks = part_dynamics(_part(_direction("f") + _note()))

        self.assertEqual(marks, [DynamicMark.F])

    def test_an_invisible_note_is_excluded_from_the_returned_list(self) -> None:
        # Must match part_signature's alignment unit or the positional join misaligns:
        # one direction and two notes (one invisible), but only one visible note, so the
        # returned list has exactly one entry.
        marks = part_dynamics(_part(_direction("f") + _invisible() + _note()))

        self.assertEqual(len(marks), 1)

    def test_an_invisible_note_can_still_claim_the_pending_mark(self) -> None:
        # The invisible note is real (not a rest or chord member) and still claims the
        # dynamic in document order, exactly as NotationExtractor would for the segment
        # itself - so the visible note after it must not also claim it.
        marks = part_dynamics(_part(_direction("f") + _invisible() + _note()))

        self.assertEqual(marks, [DynamicMark.NONE])

    def test_no_direction_is_none(self) -> None:
        marks = part_dynamics(_part(_note()))

        self.assertEqual(marks, [DynamicMark.NONE])


class TestDynamicsPlacementIndex(unittest.TestCase):
    def _score(self, root: Path, aligned: bool = True) -> Path:
        work = root / "scores" / "C" / "W"
        segments = work / "musicxml" / "unaligned"
        segments.mkdir(parents=True)
        whole_notes = _direction("f") + _note("C") + _note("D") + _direction("pp") + _note("E")
        (work / "sq1.musicxml").write_text(
            SEG_TEMPLATE.format(notes=whole_notes), encoding="utf-8"
        )

        first = _note("C") + _note("D")
        second = _note("E") if aligned else _note("G")
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

            index = DynamicsPlacementIndex(work, "sq1", work / "sq1.musicxml")

            self.assertEqual(index.aligned_parts, 1)
            self.assertEqual(len(index.for_segment(1, 1, 0)), 2)
            self.assertEqual(len(index.for_segment(1, 2, 0)), 1)

    def test_the_slice_carries_the_mark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = self._score(Path(tmp))

            index = DynamicsPlacementIndex(work, "sq1", work / "sq1.musicxml")

            self.assertEqual(index.for_segment(1, 1, 0), [DynamicMark.F, DynamicMark.NONE])
            self.assertEqual(index.for_segment(1, 2, 0), [DynamicMark.PP])

    def test_a_part_that_does_not_align_contributes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = self._score(Path(tmp), aligned=False)

            index = DynamicsPlacementIndex(work, "sq1", work / "sq1.musicxml")

            self.assertEqual(index.aligned_parts, 0)
            self.assertEqual(index.skipped_parts, 1)
            self.assertIsNone(index.for_segment(1, 1, 0))


class TestApplyDynamics(unittest.TestCase):
    def test_a_direction_is_inserted_before_the_marked_note(self) -> None:
        part = ET.fromstring(  # noqa: S314
            f"<part><measure>{_note('C')}{_note('D')}</measure></part>"
        )

        applied = apply_dynamics(part, [DynamicMark.NONE, DynamicMark.MF])

        self.assertEqual(applied, 1)
        measure = part.find("measure")
        children = list(measure)
        self.assertEqual(children[0].tag, "note")
        self.assertEqual(children[1].tag, "direction")
        self.assertEqual(children[2].tag, "note")
        self.assertEqual(children[1].find(".//dynamics")[0].tag, "mf")

    def test_the_reinjected_direction_is_read_back_correctly(self) -> None:
        # End to end: written back into the XML the same way, then read by the ordinary
        # extractor - the whole point of writing a <direction> rather than annotating the
        # note object directly.
        from training.omr_datasets.structured_notation_parser import parse_part  # noqa: PLC0415

        part = ET.fromstring(  # noqa: S314
            f"<part><measure>{_note('C')}{_note('D')}</measure></part>"
        )
        apply_dynamics(part, [DynamicMark.NONE, DynamicMark.MF])

        notes, _ = parse_part(part)

        self.assertEqual(notes[0].dynamic, DynamicMark.NONE)
        self.assertEqual(notes[1].dynamic, DynamicMark.MF)

    def test_none_marks_insert_nothing(self) -> None:
        part = ET.fromstring(f"<part><measure>{_note('C')}</measure></part>")  # noqa: S314

        applied = apply_dynamics(part, [DynamicMark.NONE])

        self.assertEqual(applied, 0)
        self.assertEqual(len(list(part.find("measure"))), 1)

    def test_a_score_element_is_refused_rather_than_silently_dropped(self) -> None:
        score = ET.fromstring(  # noqa: S314
            f'<score-partwise><part id="P1"><measure>{_note("C")}</measure></part>'
            "</score-partwise>"
        )

        with self.assertRaises(ValueError):
            apply_dynamics(score, [DynamicMark.F])


if __name__ == "__main__":
    unittest.main()
