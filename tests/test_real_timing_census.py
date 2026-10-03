import unittest
import xml.etree.ElementTree as ET

from training.onnx.census_real_timing_labels import inspect_label_text, inspect_reference
from training.onnx.roundtrip_real_timing_labels import measure_timing


def score(notes, divisions=4):
    return ET.fromstring('<score-partwise><part><measure number="1"><attributes>'
                        f'<divisions>{divisions}</divisions></attributes>' + notes + '</measure></part></score-partwise>')


class TestTimingCensus(unittest.TestCase):
    def test_measure_rest_loss_uses_canonical_duration(self):
        counts, issues = inspect_reference(score('''<note><rest measure="yes"/><duration>24</duration></note>
            <note><rest measure="yes"/><duration>12</duration></note>'''))
        self.assertEqual(counts['full_measure_rests'], 2)
        self.assertEqual(counts['full_measure_rest_duration_loss'], 1)
        self.assertEqual(issues[0]['expected_quarters'], '6')
        self.assertEqual(issues[0]['emitted_quarters'], '4')

    def test_duplet_loss_and_integral_quintuplet_are_distinguished(self):
        duplet = '''<note><pitch><step>C</step><octave>4</octave></pitch><duration>3</duration>
            <type>eighth</type><time-modification><actual-notes>2</actual-notes>
            <normal-notes>3</normal-notes></time-modification></note>'''
        quintuplet = '''<note><rest/><duration>8</duration><type>eighth</type>
            <time-modification><actual-notes>5</actual-notes><normal-notes>4</normal-notes>
            </time-modification></note>'''
        counts, issues = inspect_reference(score(duplet.replace('<duration>3</duration>', '<duration>15</duration>') + quintuplet, divisions=20))
        self.assertEqual(counts['explicit_time_modification_events'], 2)
        self.assertEqual(counts['non_integral_reciprocal_events'], 1)
        self.assertEqual(counts['confirmed_reciprocal_rounding_duration_loss'], 1)
        self.assertEqual(issues[0]['token'], 'note_5')
        self.assertEqual(issues[0]['emitted_quarters'], '4/5')
        counts, _ = inspect_reference(score(duplet.replace('<duration>3</duration>', '<duration>2</duration>')))
        self.assertNotIn('confirmed_reciprocal_rounding_duration_loss', counts)
        self.assertEqual(counts['explicit_ratio_source_duration_disagreement'], 1)

    def test_label_census_does_not_count_multi_event_bar_as_sole_rest(self):
        counts = inspect_label_text('''timeSignatureBeats_12 . . . . .
timeSignature/8 . . . . .
rest_1 _ _ _ _ upper
barline . . . . .
rest_1 _ _ _ _ upper&note_8 C4 _ _ _ lower
barline . . . . .''')
        self.assertEqual(counts['sole_rest_1_in_explicit_12_8_bars'], 1)
        self.assertEqual(counts['note_rest_tokens'], 3)

    def test_order_and_onset_changes_are_conditional_on_inventory(self):
        a = dict(measure=0, onset='0', duration='1', pitch=('C', 4, '0'), grace=False)
        b = dict(measure=0, onset='1', duration='1', pitch=('D', 4, '0'), grace=False)
        result = measure_timing([[a, b]], [[b, dict(a, onset='2')]])
        self.assertEqual(result['order_changed_inventory_preserved_staff_measures'], 1)
        self.assertEqual(result['onset_moved_inventory_preserved_events'], 1)
        result = measure_timing([[a, b]], [[a]])
        self.assertEqual(result['changed_inventory_staff_measures'], 1)
        self.assertNotIn('onset_moved_inventory_preserved_events', result)

    def test_equivalent_polyphonic_inventory_is_an_exact_control(self):
        event = dict(measure=0, onset='0', duration='1', pitch=('C', 4, '0'), grace=False)
        result = measure_timing([[event]], [[dict(event)]])
        self.assertEqual(result['inventory_preserved_staff_measures'], 1)
        self.assertNotIn('onset_changed_inventory_preserved_staff_measures', result)
