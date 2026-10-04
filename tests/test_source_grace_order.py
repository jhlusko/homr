import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from training.omr_datasets import music_xml_parser as parser
from training.onnx.audit_source_grace_order import check


class TestSourceGraceOrder(unittest.TestCase):
    def fixture(self):
        return ET.fromstring('''<score-partwise><part id="P1"><measure number="1">
        <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef></attributes>
        <note><grace/><pitch><step>E</step><octave>5</octave></pitch><type>16th</type></note>
        <note><grace/><pitch><step>G</step><octave>5</octave></pitch><type>16th</type></note>
        <note><chord/><grace/><pitch><step>F</step><octave>5</octave></pitch><type>16th</type></note>
        <note><pitch><step>C</step><octave>5</octave></pitch><duration>16</duration><type>whole</type></note>
        </measure></part></score-partwise>''')

    def test_source_sequence_and_explicit_chord_pass(self):
        result=check(self.fixture())
        self.assertEqual(result['grace_notes'],3)
        self.assertEqual(result['grace_groups'],2)
        self.assertEqual(result['explicit_grace_chords'],1)
        self.assertEqual(result['mismatches'],[])

    def test_unrelated_unsupported_duration_does_not_block_grace_group_audit(self):
        root = self.fixture()
        principal = root.find('./part/measure').findall('note')[-1]
        principal.find('type').text = '16th'
        ET.SubElement(principal, 'dot')
        principal.append(ET.fromstring('''<time-modification><actual-notes>3</actual-notes>
            <normal-notes>2</normal-notes></time-modification>'''))
        self.assertEqual(check(root)['mismatches'], [])

    def test_grace_groups_have_zero_advance_and_principal_keeps_real_gap(self):
        from homr.transformer.structured_notation import AdvanceClass
        from homr.transformer.vocabulary import sort_token_chords
        root = self.fixture()
        measure = root.find('./part/measure')
        principal = measure.findall('note')[-1]
        principal.find('duration').text = '4'
        principal.find('type').text = 'quarter'
        measure.append(ET.fromstring('''<note><pitch><step>D</step><octave>5</octave></pitch>
            <duration>4</duration><type>quarter</type></note>'''))
        voices = parser.music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
        notes = [s for voice in voices for bar in voice for s in bar
                 if s.rhythm.startswith('note')]
        groups = [g for voice in voices for bar in voice for g in sort_token_chords(bar)
                  if any(s.rhythm.startswith('note') and s.rhythm.endswith('G') for s in g)]
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(any(s.notation.advance == AdvanceClass.ZERO for s in g) for g in groups))
        self.assertEqual(next(s for s in notes if s.pitch == 'C5').notation.advance,
                         AdvanceClass.QUARTER)

    def test_audit_detects_collapsing_successive_graces(self):
        original=parser.merge_upper_and_lower_staff
        def collapse(voices, **kwargs):
            for voice in voices:
                for entry in voice:
                    entry.grace_order=0
            return original(voices, **kwargs)
        with patch.object(parser,'merge_upper_and_lower_staff',collapse):
            result=check(self.fixture())
        self.assertEqual(len(result['mismatches']),1)
        self.assertEqual(result['mismatches'][0]['expected'],[[1],[2,3]])
        self.assertEqual(result['mismatches'][0]['actual'],[[1,2,3]])

    def test_voice_backup_does_not_require_document_order_across_onsets(self):
        root = self.fixture()
        measure = root.find('./part/measure')
        for xml in [
            '<note><grace/><pitch><step>D</step><octave>6</octave></pitch><type>16th</type></note>',
            '<backup><duration>16</duration></backup>',
            '<note><grace/><pitch><step>A</step><octave>3</octave></pitch><type>16th</type></note>',
        ]:
            measure.append(ET.fromstring(xml))
        self.assertEqual(check(root)['mismatches'], [])
