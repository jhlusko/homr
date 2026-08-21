import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.ossq_measure_length_audit import audit_file

_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Violin I</part-name></score-part>
    <score-part id="P2"><part-name>Violin II</part-name></score-part>
  </part-list>
"""

_AGREEING = (
    _HEADER
    + """
  <part id="P1"><measure number="1">
    <attributes><divisions>2</divisions></attributes>
    <note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><type>quarter</type></note>
    <note><pitch><step>D</step><octave>5</octave></pitch><duration>2</duration><type>quarter</type></note>
  </measure></part>
  <part id="P2"><measure number="1">
    <attributes><divisions>2</divisions></attributes>
    <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><type>half</type></note>
  </measure></part>
</score-partwise>"""
)

_DISAGREEING = (
    _HEADER
    + """
  <part id="P1"><measure number="1">
    <attributes><divisions>2</divisions></attributes>
    <note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><type>quarter</type></note>
    <note><pitch><step>D</step><octave>5</octave></pitch><duration>2</duration><type>quarter</type></note>
  </measure></part>
  <part id="P2"><measure number="1">
    <attributes><divisions>2</divisions></attributes>
    <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
  </measure></part>
</score-partwise>"""
)

# Two voices sharing a part via <backup> - voice 1 is a quarter+eighth chord, voice 2
# is a dotted-quarter; both should reach the same peak (3 eighths) if the ground truth
# is internally consistent.
_BACKUP_AGREEING = (
    _HEADER
    + """
  <part id="P1"><measure number="1">
    <attributes><divisions>2</divisions></attributes>
    <note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><type>quarter</type><voice>1</voice></note>
    <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>eighth</type><voice>1</voice></note>
    <backup><duration>3</duration></backup>
    <note><pitch><step>E</step><octave>4</octave></pitch><duration>3</duration><type>quarter</type><voice>2</voice></note>
  </measure></part>
  <part id="P2"><measure number="1">
    <attributes><divisions>2</divisions></attributes>
    <note><pitch><step>F</step><octave>4</octave></pitch><duration>3</duration><type>quarter</type></note>
  </measure></part>
</score-partwise>"""
)


def _audit(xml: str) -> list:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "score.musicxml"
        path.write_text(xml, encoding="utf-8")
        return audit_file(path)


class TestAuditFile(unittest.TestCase):
    def test_agreeing_parts_produce_no_finding(self) -> None:
        self.assertEqual(_audit(_AGREEING), [])

    def test_disagreeing_parts_are_flagged(self) -> None:
        findings = _audit(_DISAGREEING)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["measure_index"], 0)
        lengths = {name.split(" (")[0]: length for name, (_, length) in findings[0]["per_part"].items()}
        self.assertEqual(lengths["P1"], "2")
        self.assertEqual(lengths["P2"], "1")

    def test_backup_rewinds_the_cursor_not_the_peak(self) -> None:
        # Voice 1 alone reaches a peak of 1.5 quarters (quarter + eighth); voice 2
        # (after backup) reaches 1.5 quarters too (a dotted quarter) - same peak,
        # should agree with the other part's plain dotted quarter.
        self.assertEqual(_audit(_BACKUP_AGREEING), [])

    def test_a_single_part_has_nothing_to_compare(self) -> None:
        single_part = (
            _HEADER
            + """
  <part id="P1"><measure number="1">
    <attributes><divisions>2</divisions></attributes>
    <note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><type>quarter</type></note>
  </measure></part>
</score-partwise>"""
        )
        self.assertEqual(_audit(single_part), [])


if __name__ == "__main__":
    unittest.main()
