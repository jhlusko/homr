"""Independent source matching must not manufacture agreement from ambiguous notes."""

import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.source_tie_label_audit import source_notes, token_notes


def test_source_identity_distinguishes_staff_and_rhythm_and_keeps_ambiguity() -> None:
    part = ET.fromstring("""
        <part><measure number="27">
          <note><pitch><step>C</step><octave>5</octave></pitch><type>half</type>
            <staff>1</staff><voice>1</voice><notations><tied type="start"/></notations></note>
          <note><pitch><step>C</step><octave>5</octave></pitch><type>eighth</type>
            <staff>1</staff><voice>2</voice><notations><slur type="start"/></notations></note>
          <note><pitch><step>C</step><octave>5</octave></pitch><type>eighth</type>
            <staff>1</staff><voice>2</voice></note>
          <note><pitch><step>C</step><octave>5</octave></pitch><type>half</type>
            <staff>2</staff><voice>5</voice></note>
        </measure></part>
    """)
    notes = source_notes(part, 0, 1)
    assert notes[(0, "upper", "C5", "note_2")][0]["tie"] == "start"
    assert notes[(0, "lower", "C5", "note_2")][0]["tie"] == "none"
    assert len(notes[(0, "upper", "C5", "note_8")]) == 2
    assert all(n["tie"] == "none" for n in notes[(0, "upper", "C5", "note_8")])


def test_octave_shift_from_before_crop_excludes_start_stop_and_active_measures() -> None:
    part = ET.fromstring("""
        <part>
          <measure><direction><direction-type><octave-shift type="down" size="8"/>
            </direction-type><staff>1</staff></direction></measure>
          <measure><note><pitch><step>C</step><octave>6</octave></pitch><type>half</type>
            <staff>1</staff></note></measure>
          <measure><direction><direction-type><octave-shift type="stop" size="8"/>
            </direction-type><staff>1</staff></direction>
            <note><pitch><step>C</step><octave>6</octave></pitch><type>half</type>
            <staff>1</staff></note></measure>
          <measure><note><pitch><step>C</step><octave>6</octave></pitch><type>half</type>
            <staff>1</staff></note></measure>
        </part>
    """)
    notes = source_notes(part, 1, 4)
    assert (1, "upper", "C6", "unsupported_octave_shift") in notes
    assert (2, "upper", "C6", "unsupported_octave_shift") in notes
    assert (3, "upper", "C6", "note_2") in notes


def test_token_indices_include_rests_and_measure_indices_include_double_barlines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.tokens"
    path.write_text(
        "rest_8 _ _ _ _ upper&note_2 C5 _ _ _ upper&note_2 D3 _ _ _ lower\n"
        "doublebarline . . . . .\n"
        "note_4 C5 _ _ _ upper\n"
        "repeatEnd . . . . .\n"
        "note_4 E5 _ _ _ upper\n"
    )
    notes, size = token_notes(path, 27)
    assert size == 5
    assert notes[(27, "upper", "C5", "note_2")] == [1]
    assert notes[(27, "lower", "D3", "note_2")] == [2]
    assert notes[(28, "upper", "C5", "note_4")] == [3]
    assert notes[(29, "upper", "E5", "note_4")] == [4]
