import unittest
import xml.etree.ElementTree as ET

from training.onnx.ablate_cached_page_xml import full_measure_rest_hypothesis


def example(extra='', note='<rest/><duration>16</duration><type>whole</type>', beats=12, denominator=8):
    return ET.fromstring(f'''<score-partwise><part><measure><attributes>
      <divisions>4</divisions><time><beats>{beats}</beats><beat-type>{denominator}</beat-type></time>
      </attributes><note>{note}</note>{extra}</measure></part></score-partwise>''')


class TestRestDiagnostic(unittest.TestCase):
    def test_uses_predicted_meter_and_preserves_other_events(self):
        root = example()
        self.assertEqual(len(full_measure_rest_hypothesis(root)), 1)
        self.assertEqual(root.findtext('.//duration'), '24')
        self.assertEqual(root.find('.//rest').get('measure'), 'yes')
        self.assertEqual(full_measure_rest_hypothesis(root), [])
        root = example(beats=3, denominator=4)
        full_measure_rest_hypothesis(root)
        self.assertEqual(root.findtext('.//duration'), '12')

    def test_declines_chords_polyphony_and_already_correct_duration(self):
        for root in (example(extra='<note><rest/><duration>4</duration><type>quarter</type></note>'),
                     example(extra='<backup><duration>16</duration></backup>'),
                     example(note='<rest/><duration>24</duration><type>whole</type>')):
            self.assertEqual(full_measure_rest_hypothesis(root), [])

    def test_declines_missing_meter_dots_and_invisible_rest(self):
        for root in (example(note='<rest/><duration>16</duration><type>whole</type><dot/>'),
                     example(note='<rest/><duration>16</duration><type>whole</type><time-modification/>')):
            self.assertEqual(full_measure_rest_hypothesis(root), [])
        root = example()
        root.find('.//note').set('print-object', 'no')
        self.assertEqual(full_measure_rest_hypothesis(root), [])
        root = example()
        root.find('.//attributes').remove(root.find('.//time'))
        self.assertEqual(full_measure_rest_hypothesis(root), [])
