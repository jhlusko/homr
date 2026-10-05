import unittest
import xml.etree.ElementTree as ET

from training.onnx.audit_crop_onsets import source_events, traced_parse, group_audit, serialized_copy, export_audit


def fixture():
    return ET.fromstring('''<score-partwise><part id="P1"><measure number="1">
    <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time>
    <staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef>
    <clef number="2"><sign>F</sign><line>4</line></clef></attributes>
    <note><grace/><pitch><step>D</step><octave>5</octave></pitch><type>16th</type><staff>1</staff></note>
    <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><type>quarter</type><staff>1</staff></note>
    <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><type>quarter</type><staff>1</staff></note>
    <backup><duration>8</duration></backup>
    <note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration><type>quarter</type><staff>2</staff></note>
    <note><pitch><step>D</step><octave>3</octave></pitch><duration>4</duration><type>quarter</type><staff>2</staff></note>
    </measure></part></score-partwise>''')


class TestCropOnsets(unittest.TestCase):
    def test_independent_onsets_correct_grace_and_voice_backup_grouping(self):
        root=fixture();events=source_events(root);symbols=traced_parse(root)[0][0]
        result=group_audit(symbols,events)
        self.assertEqual(result['mixed_onsets'],[])
        self.assertEqual(result['split_onsets'],[])
        self.assertEqual(len(result['groups']),2)
        self.assertEqual(events['p1-m1-n4']['onset_quarters'],'0')

    def test_mixed_and_split_onsets_are_detected(self):
        from homr.transformer.vocabulary import EncodedSymbol
        root=fixture();events=source_events(root);symbols=traced_parse(root)[0][0]
        notes=[s for s in symbols if s.rhythm.startswith('note') and not s.rhythm.endswith('G')]
        bad=[notes[0],EncodedSymbol('chord'),notes[-1],notes[1],notes[2]]
        result=group_audit(bad,events)
        self.assertTrue(result['mixed_onsets'])
        self.assertTrue(result['split_onsets'])

    def test_legacy_export_preserves_simple_onsets_and_note_tags(self):
        root=fixture();events=source_events(root);symbols=traced_parse(root)[0][0]
        result=export_audit(serialized_copy(symbols),events)
        self.assertNotIn('error',result)
        self.assertEqual(result['onset_errors'],[])
        self.assertEqual(result['missing_events'],[])
        self.assertEqual(result['duplicate_events'],{})

    def test_export_reports_slur_endpoint_moved_within_chord(self):
        root=fixture();measure=root.find('./part/measure')
        second=measure.findall('note')[1]
        extra=ET.fromstring('''<note><chord/><pitch><step>E</step><octave>5</octave></pitch>
        <duration>4</duration><type>quarter</type><staff>1</staff></note>''')
        measure.insert(list(measure).index(second)+1,extra)
        ET.SubElement(ET.SubElement(second,'notations'),'slur',type='start',number='1')
        events=source_events(root);symbols=traced_parse(root)[0][0]
        result=export_audit(serialized_copy(symbols),events)
        self.assertTrue(result['slur_errors'])
        anchored=export_audit(serialized_copy(symbols,preserve_annotations=True),events)
        self.assertEqual(anchored['slur_errors'],[])
        self.assertEqual(anchored['missing_events'],[])

    def test_export_measure_anchor_uses_crop_offset_not_printed_number(self):
        root=fixture();part=root.find('part');measure=part.find('measure')
        part.remove(measure)
        part.append(ET.Element('measure',number='10'))
        part.append(ET.Element('measure',number='11'))
        measure.set('number','12');part.append(measure)
        events=source_events(root);symbols=traced_parse(root)[0][-1]
        correct=export_audit(serialized_copy(symbols),events,start_measure=2)
        wrong=export_audit(serialized_copy(symbols),events,start_measure=0)
        self.assertEqual(correct['measure_errors'],[])
        self.assertTrue(wrong['measure_errors'])

    def test_staff_specific_time_signature_is_inherited(self):
        root=fixture();attributes=root.find('./part/measure/attributes')
        attributes.find('time').set('number','1')
        attributes.append(ET.fromstring('<time number="2"><beats>3</beats><beat-type>8</beat-type></time>'))
        events=source_events(root)
        self.assertEqual(events['p1-m1-n2']['time_signature'],dict(beats='4',beat_type='4'))
        self.assertEqual(events['p1-m1-n4']['time_signature'],dict(beats='3',beat_type='8'))

    def test_tie_alias_is_separate_from_literal_phrase_slur_fidelity(self):
        root=fixture();note=root.find('./part/measure').findall('note')[1]
        ET.SubElement(ET.SubElement(note,'notations'),'tied',type='start')
        events=source_events(root);symbols=traced_parse(root)[0][0]
        exported=export_audit(serialized_copy(symbols,preserve_annotations=True),events)
        self.assertTrue(exported['slur_errors'])
        self.assertEqual(exported['legacy_endpoint_errors'],[])
