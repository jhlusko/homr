import unittest
import xml.etree.ElementTree as ET

from homr.transformer.vocabulary import EncodedSymbol
from validation.ned_score import (
    _align_parts,
    _ned_from_parts,
    _split_grand_staff,
    _strip_articulation_from_parts,
    _without_unreliable_articulation,
    compute_ned,
)


def _sym(rhythm: str) -> EncodedSymbol:
    return EncodedSymbol(rhythm)


_GRAND_STAFF_PART = """
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch>
        <duration>1</duration><type>quarter</type><voice>1</voice><staff>1</staff></note>
      <note><pitch><step>C</step><octave>3</octave></pitch>
        <duration>1</duration><type>quarter</type><voice>2</voice><staff>2</staff></note>
    </measure>
  </part>
"""

_VOICE_PART = """
  <part id="P0">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>E</step><octave>5</octave></pitch>
        <duration>1</duration><type>quarter</type><voice>1</voice><staff>1</staff></note>
    </measure>
  </part>
"""


def _wrap(parts_xml: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P0" /><score-part id="P1" /></part-list>
  {parts_xml}
</score-partwise>"""


class TestSplitGrandStaff(unittest.TestCase):
    def test_lone_grand_staff_part_still_splits(self) -> None:
        xml_text = _wrap(_GRAND_STAFF_PART)

        result = _split_grand_staff(xml_text)

        root = ET.fromstring(result)  # noqa: S314
        parts = root.findall("part")
        self.assertEqual(2, len(parts))
        self.assertEqual(1, len(parts[0].findall(".//note")))
        self.assertEqual(1, len(parts[1].findall(".//note")))
        part_ids = {p.get("id") for p in parts}
        score_part_ids = {sp.get("id") for sp in root.findall(".//score-part")}
        self.assertEqual(part_ids, score_part_ids)

    def test_grand_staff_alongside_other_part_also_splits(self) -> None:
        # A solo voice part plus a piano grand staff: two <part> elements in
        # the document, only one of which has <staves>2</staves>. This used
        # to be left untouched because the old check required exactly one
        # <part> in the whole document.
        xml_text = _wrap(_VOICE_PART + _GRAND_STAFF_PART)

        result = _split_grand_staff(xml_text)

        root = ET.fromstring(result)  # noqa: S314
        parts = root.findall("part")
        self.assertEqual(3, len(parts))
        note_counts = [len(p.findall(".//note")) for p in parts]
        self.assertEqual([1, 1, 1], note_counts)
        part_ids = [p.get("id") for p in parts]
        self.assertEqual(len(part_ids), len(set(part_ids)))
        score_part_ids = {sp.get("id") for sp in root.findall(".//score-part")}
        self.assertEqual(set(part_ids), score_part_ids)

    def test_no_grand_staff_returns_input_unchanged(self) -> None:
        xml_text = _wrap(_VOICE_PART)

        result = _split_grand_staff(xml_text)

        self.assertEqual(xml_text, result)

    def test_invalid_xml_returns_input_unchanged(self) -> None:
        self.assertEqual("not xml", _split_grand_staff("not xml"))


class TestAlignParts(unittest.TestCase):
    def test_reorders_reversed_parts_by_content(self) -> None:
        # kern lists [bass, treble] but the tool's split-grand-staff output
        # lists [treble, bass] for the same piece - the exact reversal found
        # in polish-scores samples like 43 and 52 (see memory).
        bass = [_sym("note_4"), _sym("note_8"), _sym("note_4")]
        treble = [_sym("note_8"), _sym("note_16"), _sym("note_16"), _sym("note_4")]
        kern_parts = [bass, treble]
        xml_parts = [treble, bass]

        aligned_kern, aligned_xml = _align_parts(kern_parts, xml_parts)

        self.assertEqual(aligned_kern, [bass, treble])
        self.assertEqual(aligned_xml, [bass, treble])

    def test_already_aligned_parts_are_unchanged(self) -> None:
        part_a = [_sym("note_4"), _sym("note_8")]
        part_b = [_sym("note_16"), _sym("note_4"), _sym("note_4")]

        aligned_kern, aligned_xml = _align_parts([part_a, part_b], [part_a, part_b])

        self.assertEqual(aligned_kern, [part_a, part_b])
        self.assertEqual(aligned_xml, [part_a, part_b])

    def test_ned_from_parts_ignores_part_order(self) -> None:
        # Same content as test_reorders_reversed_parts_by_content: a perfect,
        # zero-distance match once parts are correctly paired up. Before the
        # fix, comparing positionally (kern's bass against the tool's treble)
        # would have produced a large, spurious NED for identical content.
        bass = [_sym("note_4"), _sym("note_8"), _sym("note_4")]
        treble = [_sym("note_8"), _sym("note_16"), _sym("note_16"), _sym("note_4")]

        result = _ned_from_parts([bass, treble], [treble, bass])

        self.assertEqual(0, result.distance)
        self.assertEqual(0.0, result.ned)


class TestUnreliableArticulation(unittest.TestCase):
    def test_removes_only_unreliable_components(self) -> None:
        self.assertEqual(_without_unreliable_articulation("staccato"), "_")
        self.assertEqual(_without_unreliable_articulation("staccatissimo"), "_")
        self.assertEqual(_without_unreliable_articulation("turn"), "_")
        self.assertEqual(_without_unreliable_articulation("staccatissimo_staccato"), "_")

    def test_keeps_reliable_components_of_compound_values(self) -> None:
        self.assertEqual(_without_unreliable_articulation("accent_staccato"), "accent")
        self.assertEqual(_without_unreliable_articulation("fermata_turn"), "fermata")
        self.assertEqual(
            _without_unreliable_articulation("accent_staccato_tenuto"), "accent_tenuto"
        )

    def test_leaves_reliable_only_values_unchanged(self) -> None:
        self.assertEqual(_without_unreliable_articulation("accent"), "accent")

    def test_passes_through_nonote_and_empty(self) -> None:
        self.assertEqual(_without_unreliable_articulation("."), ".")
        self.assertEqual(_without_unreliable_articulation("_"), "_")

    def test_strip_articulation_from_parts_clears_only_unreliable_symbols(self) -> None:
        part = [
            EncodedSymbol("note_4", "C5", articulation="staccato"),
            EncodedSymbol("note_4", "D5", articulation="accent"),
            EncodedSymbol("note_4", "E5", articulation="accent_staccato"),
        ]

        result = _strip_articulation_from_parts([part])

        self.assertEqual(
            [s.articulation for s in result[0]],
            ["_", "accent", "accent"],
        )
        # Original symbols must not be mutated - only copies are edited.
        self.assertEqual(part[0].articulation, "staccato")

    def test_compute_ned_ignores_staccato_mismatch_when_flag_set(self) -> None:
        kern = "**kern\n4c\n4d\n4e\n*-\n"
        pred = "**kern\n4c'\n4d\n4e'\n*-\n"

        without_flag = compute_ned(kern, pred)
        with_flag = compute_ned(kern, pred, ignore_unreliable_articulation=True)

        self.assertGreater(without_flag.articulation_ned, 0.0)
        self.assertEqual(with_flag.articulation_ned, 0.0)
        self.assertEqual(with_flag.ned, 0.0)

    def test_compute_ned_still_scores_accent_mismatch_when_flag_set(self) -> None:
        kern = "**kern\n4c\n4d^\n4e\n*-\n"
        pred = "**kern\n4c\n4d\n4e\n*-\n"

        with_flag = compute_ned(kern, pred, ignore_unreliable_articulation=True)

        self.assertGreater(with_flag.articulation_ned, 0.0)


def _one_part_xml(steps: str) -> str:
    notes = "".join(
        f"<note><pitch><step>{step}</step><octave>4</octave></pitch>"
        f"<duration>1</duration><type>quarter</type><voice>1</voice><staff>1</staff></note>"
        for step in steps
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<score-partwise version="3.1">'
        '<part-list><score-part id="P1"><part-name>P1</part-name></score-part></part-list>'
        '<part id="P1"><measure number="1">'
        "<attributes><divisions>1</divisions><key><fifths>0</fifths></key>"
        "<time><beats>4</beats><beat-type>4</beat-type></time>"
        "<clef><sign>G</sign><line>2</line></clef></attributes>"
        f"{notes}</measure></part></score-partwise>"
    )


class TestMusicXmlGroundTruth(unittest.TestCase):
    """The ground-truth side accepts MusicXML, not only **kern (used by validation/ossq)."""

    def test_identical_musicxml_on_both_sides_scores_zero(self) -> None:
        xml = _one_part_xml("CDEF")

        result = compute_ned(xml, xml)

        self.assertEqual(result.ned, 0.0)
        self.assertEqual(result.distance, 0)
        self.assertGreater(result.kern_len, 0)

    def test_musicxml_ground_truth_still_penalises_a_wrong_pitch(self) -> None:
        result = compute_ned(_one_part_xml("CDEF"), _one_part_xml("CDEG"))

        self.assertGreater(result.ned, 0.0)
        self.assertGreater(result.pitch_ned, 0.0)
        self.assertEqual(result.rhythm_ned, 0.0)

    def test_repeated_clef_and_key_can_be_collapsed_on_both_sides(self) -> None:
        # The reference restates clef and key at every system start; homr reports state
        # changes only. Same music, different convention.
        restated = _one_part_xml("CD") + ""
        reference = restated.replace(
            "</measure></part>",
            "</measure><measure number='2'>"
            "<attributes><key><fifths>0</fifths></key>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
            "<note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration>"
            "<type>quarter</type><voice>1</voice><staff>1</staff></note>"
            "</measure></part>",
        )
        prediction = restated.replace(
            "</measure></part>",
            "</measure><measure number='2'>"
            "<note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration>"
            "<type>quarter</type><voice>1</voice><staff>1</staff></note>"
            "</measure></part>",
        )

        as_written = compute_ned(reference, prediction)
        collapsed = compute_ned(reference, prediction, collapse_repeated_attributes=True)

        self.assertGreater(as_written.ned, 0.0)
        self.assertEqual(collapsed.ned, 0.0)

    def test_collapsing_still_penalises_a_wrong_clef(self) -> None:
        reference = _one_part_xml("CD")
        prediction = reference.replace(
            "<sign>G</sign><line>2</line>", "<sign>F</sign><line>4</line>"
        )

        collapsed = compute_ned(reference, prediction, collapse_repeated_attributes=True)

        self.assertGreater(collapsed.ned, 0.0)

    def test_collapsing_keeps_a_genuine_clef_change(self) -> None:
        # A mid-part clef change is a change, not a restatement, and must survive.
        reference = _one_part_xml("CD").replace(
            "</measure></part>",
            "</measure><measure number='2'>"
            "<attributes><clef><sign>F</sign><line>4</line></clef></attributes>"
            "<note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration>"
            "<type>quarter</type><voice>1</voice><staff>1</staff></note>"
            "</measure></part>",
        )
        prediction = _one_part_xml("CD").replace(
            "</measure></part>",
            "</measure><measure number='2'>"
            "<note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration>"
            "<type>quarter</type><voice>1</voice><staff>1</staff></note>"
            "</measure></part>",
        )

        collapsed = compute_ned(reference, prediction, collapse_repeated_attributes=True)

        self.assertGreater(collapsed.ned, 0.0)

    def test_kern_ground_truth_against_musicxml_prediction_is_unaffected(self) -> None:
        # The mixed kern/MusicXML case is what smb and polish-scores rely on; detecting
        # the format per side must not have changed it.
        result = compute_ned("**kern\n*M4/4\n*clefG2\n4c\n4d\n4e\n4f\n*-\n", _one_part_xml("CDEF"))

        self.assertEqual(result.pitch_ned, 0.0)
        self.assertEqual(result.rhythm_ned, 0.0)
