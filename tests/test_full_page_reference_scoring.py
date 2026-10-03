import unittest
import xml.etree.ElementTree as ET

from training.onnx.audit_full_page_references import certified_page
from training.onnx.score_full_page_musicxml import compare, staff_events


class TestPageReferenceScoring(unittest.TestCase):
    def test_page_boundaries_must_be_complete_and_contiguous(self):
        row = dict(status='aligned', boundary_certified=True, count_exact=True,
                   system_index=0, start_measure=0, end_measure=3)
        self.assertTrue(certified_page([row]))
        self.assertFalse(certified_page([dict(row, boundary_certified=False)]))
        self.assertFalse(certified_page([row, dict(row, system_index=2, start_measure=4, end_measure=7)]))

    def test_divisions_before_page_and_polyphonic_onsets_are_preserved(self):
        root = ET.fromstring('''<score-partwise><part id="p"><measure>
          <attributes><divisions>12</divisions><staves>2</staves></attributes></measure>
          <measure><note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><staff>1</staff></note>
          <backup><duration>4</duration></backup>
          <note><pitch><step>G</step><octave>3</octave></pitch><duration>12</duration><staff>2</staff></note>
          <note print-object="no"><rest/><duration>12</duration><staff>2</staff></note>
          </measure></part></score-partwise>''')
        result = staff_events(root, 1, 2)
        self.assertEqual(result[0][0]['duration'], '1/3')
        self.assertEqual(result[1][0]['onset'], '0')
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(compare(result, result)['event']['errors'], 0)
        self.assertEqual(compare(result, result[:1])['event']['errors'], 1)

    def test_extra_predicted_staff_counts_as_insertions(self):
        note = dict(pitch=('C', 4, '0'), duration='1', grace=False)
        result = compare([[note]], [[note], [note]])
        self.assertEqual(result['event']['errors'], 1)
        self.assertEqual(result['event']['support'], 1)
