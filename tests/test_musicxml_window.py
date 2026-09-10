import unittest
import xml.etree.ElementTree as ET

from training.omr_datasets.convert_pdmx import (
    has_empty_final_measure,
    has_too_few_notes,
    sounding_notes,
)
from training.omr_datasets.musicxml_window import (
    extract_window,
    measure_count,
    prevailing_attributes,
)

NOTE = "<note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration></note>"


def _measure(number: int, attributes: str = "") -> str:
    return f'<measure number="{number}">{attributes}{NOTE}</measure>'


def _part(measures: str) -> ET.Element:
    return ET.fromstring(f'<part id="P1">{measures}</part>')


FULL = (
    "<attributes><divisions>2</divisions>"
    "<key><fifths>3</fifths></key>"
    "<time><beats>4</beats><beat-type>4</beat-type></time>"
    '<clef number="1"><sign>G</sign><line>2</line></clef></attributes>'
)


class TestPrevailingAttributes(unittest.TestCase):
    def test_attributes_carry_forward_to_a_later_window(self) -> None:
        # A window starting at measure 40 inherits its clef from measure 1 and would
        # otherwise render with no clef at all.
        part = _part(_measure(1, FULL) + _measure(2) + _measure(3))

        carried = prevailing_attributes(part, upto=2)

        self.assertIsNotNone(carried)
        self.assertEqual(carried.find("clef/sign").text, "G")
        self.assertEqual(carried.find("divisions").text, "2")

    def test_a_partial_later_block_does_not_erase_the_clef(self) -> None:
        # The reason attributes are merged element by element rather than by taking the
        # most recent block whole: a mid-score key change does not restate the clef.
        later = "<attributes><key><fifths>0</fifths></key></attributes>"
        part = _part(_measure(1, FULL) + _measure(2, later) + _measure(3))

        carried = prevailing_attributes(part, upto=3)

        self.assertEqual(carried.find("clef/sign").text, "G")
        self.assertEqual(carried.find("key/fifths").text, "0")

    def test_both_clefs_of_a_grand_staff_survive(self) -> None:
        two = (
            '<attributes><divisions>2</divisions>'
            '<clef number="1"><sign>G</sign></clef>'
            '<clef number="2"><sign>F</sign></clef></attributes>'
        )
        part = _part(_measure(1, two) + _measure(2))

        carried = prevailing_attributes(part, upto=2)

        self.assertEqual([c.find("sign").text for c in carried.findall("clef")], ["G", "F"])

    def test_nothing_before_the_start_yields_nothing(self) -> None:
        part = _part(_measure(1, FULL))

        self.assertIsNone(prevailing_attributes(part, upto=0))


class TestExtractWindow(unittest.TestCase):
    def test_the_window_holds_only_its_measures(self) -> None:
        part = _part("".join(_measure(n) for n in range(1, 6)))

        window = extract_window(part, 1, 3)

        self.assertEqual(measure_count(window.find("part")), 2)

    def test_the_first_measure_gains_the_inherited_context(self) -> None:
        part = _part(_measure(1, FULL) + _measure(2) + _measure(3))

        window = extract_window(part, 1, 3)

        first = window.find("part/measure")
        self.assertEqual(first.find("attributes/clef/sign").text, "G")

    def test_a_window_starting_on_a_clef_change_keeps_the_new_clef(self) -> None:
        # Inherited context must not overwrite what the measure itself states.
        change = '<attributes><clef number="1"><sign>F</sign><line>4</line></clef></attributes>'
        part = _part(_measure(1, FULL) + _measure(2, change))

        window = extract_window(part, 1, 2)

        signs = [c.text for c in window.findall("part/measure/attributes/clef/sign")]
        self.assertEqual(signs, ["F"])

    def test_the_inherited_divisions_still_arrive_on_a_clef_change(self) -> None:
        change = '<attributes><clef number="1"><sign>F</sign></clef></attributes>'
        part = _part(_measure(1, FULL) + _measure(2, change))

        window = extract_window(part, 1, 2)

        self.assertEqual(window.find("part/measure/attributes/divisions").text, "2")

    def test_the_source_part_is_not_mutated(self) -> None:
        # The window is rendered per voice and per window; mutating the part would leak
        # one window's injected context into the next.
        part = _part(_measure(1, FULL) + _measure(2))

        extract_window(part, 1, 2)

        self.assertIsNone(part.findall("measure")[1].find("attributes"))

    def test_an_empty_range_yields_nothing(self) -> None:
        # Never a blank score: it would render a blank image and be paired with a
        # non-empty token sequence.
        part = _part(_measure(1))

        self.assertIsNone(extract_window(part, 5, 7))




class TestEmptyFinalMeasure(unittest.TestCase):
    """A trailing bar with nothing sounding is what a truncated score looks like.

    A bar of rests counts as empty too. In a hand-engraved edition that could be a real
    ending, but PDMX is user-submitted MuseScore files, and in that population a trailing
    bar of rests is far more likely to be an abandoned edit than a deliberate ending on
    silence. The filter follows the corpus rather than the engraving convention.
    """

    def test_a_part_ending_on_an_empty_bar_is_caught(self) -> None:
        part = _part(_measure(1) + '<measure number="2"/>')

        self.assertTrue(has_empty_final_measure([part]))

    def test_a_part_ending_on_notes_is_not(self) -> None:
        part = _part(_measure(1) + _measure(2))

        self.assertFalse(has_empty_final_measure([part]))

    def test_a_final_bar_of_rests_is_also_excluded(self) -> None:
        rest = '<measure number="2"><note><rest/><duration>4</duration></note></measure>'
        part = _part(_measure(1) + rest)

        self.assertTrue(has_empty_final_measure([part]))

    def test_a_final_bar_mixing_rests_and_notes_is_kept(self) -> None:
        # Something still sounds in it, so the score reached its end.
        mixed = (
            '<measure number="2"><note><rest/><duration>2</duration></note>'
            "<note><pitch><step>C</step><octave>5</octave></pitch>"
            "<duration>2</duration></note></measure>"
        )
        part = _part(_measure(1) + mixed)

        self.assertFalse(has_empty_final_measure([part]))

    def test_any_part_ending_empty_condemns_the_score(self) -> None:
        # Parts are rendered independently, so one truncated part means the score was
        # exported mid-edit and the others are suspect too.
        good = _part(_measure(1) + _measure(2))
        bad = _part(_measure(1) + '<measure number="2"/>')

        self.assertTrue(has_empty_final_measure([good, bad]))

    def test_a_part_with_no_measures_is_not_flagged(self) -> None:
        self.assertFalse(has_empty_final_measure([_part("")]))




class TestMinimumNotes(unittest.TestCase):
    """A fragment yields a window or two of very sparse staves.

    The threshold comes from the distribution: over 1,200 sampled scores the median holds
    227 sounding notes, and 96 drops 20.8% of scores holding 3.0% of them. Set for quality
    rather than volume, which is the right trade in a corpus of 254,035 scores.
    """

    def _part_with(self, notes: int, rests: int = 0) -> ET.Element:
        note = "<note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration></note>"
        rest = "<note><rest/><duration>1</duration></note>"
        return _part(f'<measure number="1">{note * notes}{rest * rests}</measure>')

    def test_a_fragment_is_rejected(self) -> None:
        self.assertTrue(has_too_few_notes([self._part_with(10)]))

    def test_a_real_score_is_kept(self) -> None:
        self.assertFalse(has_too_few_notes([self._part_with(200)]))

    def test_a_score_just_under_the_bar_goes(self) -> None:
        # 80 notes is a third of the median score and would have survived a lower bar.
        self.assertTrue(has_too_few_notes([self._part_with(80)]))

    def test_rests_do_not_count_towards_the_threshold(self) -> None:
        # A score padded out with rests is exactly the fragment this catches, so counting
        # them would let it through.
        self.assertTrue(has_too_few_notes([self._part_with(10, rests=200)]))

    def test_notes_are_counted_across_parts(self) -> None:
        # Two staves of 60 is a real piece; either alone would look like a fragment.
        parts = [self._part_with(60), self._part_with(60)]

        self.assertFalse(has_too_few_notes(parts))
        self.assertEqual(sounding_notes(parts), 120)

    def test_the_boundary_is_inclusive(self) -> None:
        self.assertFalse(has_too_few_notes([self._part_with(96)]))
        self.assertTrue(has_too_few_notes([self._part_with(95)]))


if __name__ == "__main__":
    unittest.main()
