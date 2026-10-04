import copy
import unittest
import xml.etree.ElementTree as ET

from training.onnx.audit_lieder_rest_stretch import find_conflicts, native_evidence


def fixtures():
    xml = ET.fromstring('''<score-partwise><part><measure number="8"><attributes><divisions>4</divisions>
        <time><beats>3</beats><beat-type>8</beat-type></time></attributes>
        <note><rest/><duration>3</duration><type>eighth</type></note>
        <note><rest/><duration>3</duration><type>eighth</type></note>
        </measure></part></score-partwise>''')
    native = ET.fromstring('''<museScore><Score><Part><Staff id="1"/></Part><Staff id="1"><Measure><voice>
        <TimeSig><sigN>3</sigN><sigD>8</sigD><stretchN>2</stretchN><stretchD>3</stretchD></TimeSig>
        <Rest><durationType>eighth</durationType></Rest><Rest><durationType>eighth</durationType></Rest>
        </voice></Measure></Staff></Score></museScore>''')
    return xml, native


class TestRestStretchAudit(unittest.TestCase):
    def test_visible_rest_scope_location_and_explicit_native_factor(self):
        xml, native = fixtures()
        cases = find_conflicts(xml)
        self.assertEqual([c['rest_ordinal'] for c in cases], [1, 2])
        self.assertEqual([c['source_onset_quarters'] for c in cases], ['0', '3/4'])
        self.assertEqual(cases[0]['printed_measure'], '8')
        evidence = native_evidence(native, xml, cases[1])
        self.assertEqual(evidence['stretch'], '2/3')
        self.assertTrue(evidence['exactly_explained_by_native_stretch'])
        self.assertEqual(cases[1]['timing']['elapsed_quarters'], '3/4')

    def test_invisible_rest_is_not_counted_but_preserves_rest_ordinal(self):
        xml, native = fixtures()
        xml.find('./part/measure/note').set('print-object', 'no')
        cases = find_conflicts(xml)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]['rest_ordinal'], 2)
        self.assertTrue(native_evidence(native, xml, cases[0])['exactly_explained_by_native_stretch'])

    def test_time_signature_without_stretch_resets_inherited_factor(self):
        xml, native = fixtures()
        xml.find('part').append(copy.deepcopy(xml.find('./part/measure')))
        staff = native.find('./Score/Staff')
        measure = copy.deepcopy(staff.find('Measure'))
        signature = measure.find('./voice/TimeSig')
        signature.remove(signature.find('stretchN'));signature.remove(signature.find('stretchD'))
        staff.append(measure)
        evidence = native_evidence(native, xml, find_conflicts(xml)[2])
        self.assertEqual(evidence['stretch'], '1')
        self.assertFalse(evidence['exactly_explained_by_native_stretch'])

    def test_mismatched_native_measure_inventory_is_refused(self):
        xml, native = fixtures()
        case = find_conflicts(xml)[0]
        native.find('./Score/Staff').append(ET.Element('Measure'))
        with self.assertRaisesRegex(ValueError, 'physical measure inventory'):
            native_evidence(native, xml, case)

    def test_written_dot_mismatch_and_real_tuplet_are_not_auto_explained(self):
        for tag in ('dots', 'Tuplet'):
            with self.subTest(tag=tag):
                xml, native = fixtures()
                ET.SubElement(native.find('./Score/Staff/Measure/voice/Rest'), tag).text = '1'
                with self.assertRaises(ValueError):
                    native_evidence(native, xml, find_conflicts(xml)[0])


if __name__ == '__main__':
    unittest.main()
