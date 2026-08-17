import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from training.omr_datasets.musescore_boxes import (
    VERIFIABLE_CLASSES,
    Box,
    Unrenderable,
    _lines,
    _scale,
    boxes_of_class,
    pair,
    source_dynamics,
    source_syllables,
    without_part_names,
    typed_boxes,
)

SCORE = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
  <part id="P1"><measure number="1">
    {notes}
  </measure></part>
</score-partwise>
"""


def _note(lyrics: str) -> str:
    return (
        "<note><pitch><step>C</step><octave>4</octave></pitch>"
        f"<duration>1</duration><type>quarter</type>{lyrics}</note>"
    )


def _lyric(text: str, verse: str = "1", syllabic: str = "single") -> str:
    return f'<lyric number="{verse}"><syllabic>{syllabic}</syllabic><text>{text}</text></lyric>'


def _svg(paths: str, width: float = 1000, height: float = 500) -> str:
    return f'<svg viewBox="0 0 {width} {height}">{paths}</svg>'


def _lyric_path(left: float, top: float, right: float, bottom: float) -> str:
    return f'<path class="Lyrics" d="M{left},{top} L{right},{bottom} L{left},{bottom}" />'


class TestScale(unittest.TestCase):
    def test_the_viewbox_maps_onto_the_raster(self) -> None:
        # MuseScore writes the SVG in its own units and the PNG at a chosen dpi; the boxes
        # are only usable if one maps onto the other.
        self.assertEqual(_scale(_svg("", 2000, 1000), 1000, 500), (0.5, 0.5))

    def test_an_svg_without_a_viewbox_is_refused(self) -> None:
        with self.assertRaises(Unrenderable):
            _scale("<svg></svg>", 100, 100)


class TestLines(unittest.TestCase):
    """Verse 1 and verse 2 are separate lines of text under one staff, and the page reads
    the first entire before the second."""

    def test_boxes_on_one_line_stay_together_despite_uneven_tops(self) -> None:
        # A capital or an accent raises the top by several pixels. Bucketing on top was
        # the first attempt and split a line in two at the bucket boundary.
        boxes = [Box(10, 100, 30, 120), Box(40, 94, 60, 120), Box(70, 101, 90, 120)]

        self.assertEqual(len(_lines(boxes)), 1)

    def test_a_second_verse_becomes_a_second_line(self) -> None:
        boxes = [Box(10, 100, 30, 120), Box(10, 140, 30, 160)]

        self.assertEqual([len(line) for line in _lines(boxes)], [1, 1])

    def test_the_top_line_comes_first_and_reads_left_to_right(self) -> None:
        boxes = [Box(70, 140, 90, 160), Box(10, 140, 30, 160), Box(40, 100, 60, 120)]

        lines = _lines(boxes)

        self.assertEqual(lines[0][0].top, 100)
        self.assertEqual([box.left for box in lines[1]], [10, 70])

    def test_no_boxes_is_no_lines(self) -> None:
        self.assertEqual(_lines([]), [])


class TestBoxesOfClass(unittest.TestCase):
    def test_paths_become_boxes_in_image_pixels(self) -> None:
        svg = _svg(_lyric_path(100, 200, 140, 220), width=1000, height=500)

        boxes = boxes_of_class(svg, "Lyrics", (0.5, 0.5))

        self.assertEqual((boxes[0].left, boxes[0].top), (50, 100))

    def test_only_the_asked_for_class_is_returned(self) -> None:
        svg = _svg(_lyric_path(10, 10, 20, 20) + '<path class="Note" d="M0,0 L5,5" />')

        self.assertEqual(len(boxes_of_class(svg, "Lyrics", (1.0, 1.0))), 1)


class TestSourceSyllables(unittest.TestCase):
    def test_verses_come_out_in_reading_order_not_document_order(self) -> None:
        # MusicXML interleaves - note 1 verse 1, note 1 verse 2, note 2 verse 1 - but the
        # engraving puts each verse on its own line, so the page reads verse 1 entire.
        notes = _note(_lyric("one", "1") + _lyric("ONE", "2")) + _note(
            _lyric("two", "1") + _lyric("TWO", "2")
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.musicxml"
            path.write_text(SCORE.format(notes=notes), encoding="utf-8")

            texts = [text for text, _, _ in source_syllables(path)]

        self.assertEqual(texts, ["one", "two", "ONE", "TWO"])

    def test_empty_lyrics_are_not_syllables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.musicxml"
            path.write_text(SCORE.format(notes=_note(_lyric("  "))), encoding="utf-8")

            self.assertEqual(source_syllables(path), [])


class TestTypedBoxes(unittest.TestCase):
    """A page carries far more text than lyrics, and an OCR pass cropping a band under a
    staff meets all of it. MuseScore already says which is which."""

    def test_text_is_grouped_by_what_musescore_calls_it(self) -> None:
        svg = _svg(
            _lyric_path(10, 200, 40, 220)
            + '<path class="Dynamic" d="M10,300 L30,320" />'
            + '<path class="Tempo" d="M10,10 L60,30" />'
        )

        found = typed_boxes(svg, (1.0, 1.0))

        self.assertEqual(sorted(found), ["Dynamic", "Lyrics", "Tempo"])

    def test_classes_with_nothing_drawn_are_absent_rather_than_empty(self) -> None:
        found = typed_boxes(_svg(_lyric_path(10, 200, 40, 220)), (1.0, 1.0))

        self.assertEqual(list(found), ["Lyrics"])

    def test_notes_are_not_text(self) -> None:
        found = typed_boxes(_svg('<path class="Note" d="M0,0 L5,5" />'), (1.0, 1.0))

        self.assertEqual(found, {})

    def test_only_countable_classes_are_marked_verifiable(self) -> None:
        # MuseScore decides whether a <words> renders as Tempo, StaffText or Expression and
        # the MusicXML does not record which, so those cannot be joined to a string.
        self.assertEqual(sorted(VERIFIABLE_CLASSES), ["Dynamic", "Lyrics"])


class TestSourceDynamics(unittest.TestCase):
    def test_each_marking_is_read_as_its_glyph_name(self) -> None:
        notes = (
            '<direction><direction-type><dynamics><p/></dynamics></direction-type></direction>'
            + _note(_lyric("x"))
            + '<direction><direction-type><dynamics><f/><f/></dynamics></direction-type></direction>'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.musicxml"
            path.write_text(SCORE.format(notes=notes), encoding="utf-8")

            self.assertEqual(source_dynamics(path), ["p", "ff"])

    def test_a_score_without_dynamics_has_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.musicxml"
            path.write_text(SCORE.format(notes=_note(_lyric("x"))), encoding="utf-8")

            self.assertEqual(source_dynamics(path), [])


class TestWithoutPartNames(unittest.TestCase):
    """Every system renders as a score of its own, so MuseScore prints the instrument name
    on all of them. Real engraving prints it once."""

    def test_the_name_is_emptied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            score = directory / "s.musicxml"
            score.write_text(SCORE.format(notes=_note(_lyric("x"))), encoding="utf-8")

            stripped = without_part_names(score, directory / "r.musicxml")
            names = [e.text for e in ET.parse(stripped).getroot().iter("part-name")]

        # ElementTree writes an empty string as <part-name /> and reads it back as None,
        # so the assertion is that nothing is left to print, not that it is exactly "".
        self.assertEqual([name for name in names if name], [])

    def test_the_original_is_not_touched(self) -> None:
        # It is the label source as well as the render source.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            score = directory / "s.musicxml"
            score.write_text(SCORE.format(notes=_note(_lyric("x"))), encoding="utf-8")

            without_part_names(score, directory / "r.musicxml")

            self.assertIn("<part-name>Voice</part-name>", score.read_text(encoding="utf-8"))

    def test_the_music_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            score = directory / "s.musicxml"
            score.write_text(SCORE.format(notes=_note(_lyric("va"))), encoding="utf-8")

            stripped = without_part_names(score, directory / "r.musicxml")

            self.assertEqual(source_syllables(stripped), source_syllables(score))


class TestPair(unittest.TestCase):
    def _fixture(self, directory: Path, notes: str, paths: str) -> tuple[Path, Path, Path]:
        score = directory / "s.musicxml"
        score.write_text(SCORE.format(notes=notes), encoding="utf-8")
        svg = directory / "s-1.svg"
        svg.write_text(_svg(paths, 1000, 500), encoding="utf-8")
        png = directory / "s-1.png"
        cv2.imwrite(str(png), np.full((500, 1000, 3), 255, dtype=np.uint8))
        return score, svg, png

    def test_each_box_gets_the_syllable_that_drew_it(self) -> None:
        notes = _note(_lyric("va", syllabic="begin")) + _note(_lyric("gues", syllabic="end"))
        paths = _lyric_path(100, 200, 140, 220) + _lyric_path(200, 200, 260, 220)

        with tempfile.TemporaryDirectory() as tmp:
            syllables = pair(*self._fixture(Path(tmp), notes, paths))

        self.assertEqual([s.text for s in syllables], ["va", "gues"])
        self.assertEqual([s.syllabic for s in syllables], ["begin", "end"])

    def test_boxes_are_paired_by_position_not_by_the_order_they_appear_in_the_svg(self) -> None:
        notes = _note(_lyric("first")) + _note(_lyric("second"))
        # The rightmost path is written first in the file.
        paths = _lyric_path(200, 200, 260, 220) + _lyric_path(100, 200, 140, 220)

        with tempfile.TemporaryDirectory() as tmp:
            syllables = pair(*self._fixture(Path(tmp), notes, paths))

        self.assertEqual(syllables[0].text, "first")
        self.assertLess(syllables[0].box.left, syllables[1].box.left)

    def test_a_count_mismatch_is_refused_rather_than_paired_short(self) -> None:
        # The whole join rests on the counts agreeing. One box too few would shift every
        # syllable after it onto its neighbour's box, and nothing downstream could tell.
        notes = _note(_lyric("one")) + _note(_lyric("two"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Unrenderable):
                pair(*self._fixture(Path(tmp), notes, _lyric_path(100, 200, 140, 220)))

    def test_an_unreadable_raster_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            score, svg, _ = self._fixture(directory, _note(_lyric("x")), _lyric_path(1, 1, 2, 2))

            with self.assertRaises(Unrenderable):
                pair(score, svg, directory / "absent.png")


if __name__ == "__main__":
    unittest.main()
