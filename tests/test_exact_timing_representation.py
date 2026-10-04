from fractions import Fraction
import unittest
import json
import xml.etree.ElementTree as ET

from training.omr_datasets.exact_timing_labels import extract_exact_timing, apply_exact_timing, semantic_timing

from training.onnx.audit_exact_timing_representation import duration_candidates


class TestExactTimingRepresentation(unittest.TestCase):
    def test_duplet_elapsed_time_has_dotted_alias_but_not_duplet_notation(self):
        candidates = duration_candidates('note', Fraction(3, 4))
        dotted = next(c for c in candidates if c['token'] == 'note_8.')
        self.assertEqual(dotted['written_base_quarters'], '1/2')
        self.assertEqual((dotted['actual'], dotted['normal'], dotted['dots']), (1, 1, 1))
        self.assertFalse(any(c['actual'] == 2 and c['normal'] == 3 for c in candidates))

    def test_six_quarter_rest_duration_is_already_representable(self):
        candidates = duration_candidates('rest', Fraction(6))
        self.assertTrue(any(c['token'] == 'rest_1.' for c in candidates))
        self.assertFalse(any(c['token'].endswith('m') for c in candidates))

    def test_irregular_measure_length_has_no_exact_legacy_token(self):
        self.assertEqual(duration_candidates('rest', Fraction(79, 16)), [])

    def test_sidecar_preserves_duplet_and_nested_brackets_across_divisions(self):
        note = ET.fromstring('''<note><pitch><step>E</step><octave>5</octave></pitch>
            <duration>3</duration><type>eighth</type><time-modification>
            <actual-notes>2</actual-notes><normal-notes>3</normal-notes>
            <normal-type>eighth</normal-type><normal-dot/></time-modification>
            <notations><slur type="start"/><tuplet type="start" number="1"/>
            <tuplet type="start" number="2"/></notations></note>''')
        payload = json.loads(json.dumps(extract_exact_timing(note, 4)))
        apply_exact_timing(note, payload, 12)
        self.assertEqual(note.findtext('duration'), '9')
        self.assertEqual(semantic_timing(extract_exact_timing(note, 12)), semantic_timing(payload))
        self.assertEqual(payload['source_divisions'], '4')
        self.assertEqual(payload['source_duration_ticks'], '3')
        self.assertEqual(extract_exact_timing(note, 12)['source_duration_ticks'], '9')
        self.assertEqual(note.findtext('type'), 'eighth')
        self.assertIsNotNone(note.find('./notations/slur'))
        self.assertEqual(len(note.findall('./notations/tuplet')), 2)

    def test_full_rest_sidecar_preserves_glyph_and_arbitrary_duration(self):
        note = ET.fromstring('<note><rest measure="yes"/><duration>79</duration></note>')
        payload = extract_exact_timing(note, 16)
        apply_exact_timing(note, payload, 32)
        self.assertEqual(note.findtext('duration'), '158')
        self.assertEqual(note.find('rest').get('measure'), 'yes')
        self.assertIsNone(note.find('type'))

    def test_sidecar_refuses_rounding_to_an_inadequate_tick_grid(self):
        note = ET.fromstring('<note><rest measure="yes"/><duration>9</duration></note>')
        with self.assertRaisesRegex(ValueError, 'exactly'):
            apply_exact_timing(note, extract_exact_timing(note, 2), 1)
