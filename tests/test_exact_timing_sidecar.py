from pathlib import Path
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET

from training.omr_datasets.exact_timing_labels import extract_exact_timing, apply_exact_timing
from training.omr_datasets.exact_timing_sidecar import ExactTimingMismatch, timing_sidecar_path
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.notation_sidecar import write_sidecar, attach_sidecar
from training.transformer.training_vocabulary import read_tokens, token_lines_to_str
from training.architecture.transformer.exact_timing_targets import build_timing_targets
from homr.transformer.vocabulary import EncodedSymbol


def fixture():
    return '''<score-partwise><part id="P1"><measure number="1"><attributes>
      <divisions>4</divisions><time><beats>6</beats><beat-type>4</beat-type></time>
      <clef><sign>G</sign><line>2</line></clef></attributes>
      <note><rest measure="yes"/><duration>24</duration></note>
      </measure><measure number="2">
      <note><grace/><pitch><step>D</step><octave>5</octave></pitch><type>eighth</type></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>3</duration><type>eighth</type>
      <time-modification><actual-notes>2</actual-notes><normal-notes>3</normal-notes></time-modification>
      <notations print-object="no"><tuplet type="start" number="1"/></notations>
      <notations print-object="yes"><tuplet type="start" number="2"/></notations></note>
      <note><chord/><pitch><step>C</step><octave>5</octave></pitch><duration>3</duration><type>eighth</type>
      </note></measure></part></score-partwise>'''


def symbols():
    return [s for measure in music_xml_string_to_tokens(fixture())[0] for s in measure]


class TestExactTimingSidecar(unittest.TestCase):
    def test_tick_grid_survives_serialization_and_normalizes_explicit_nested_ratio(self):
        note = ET.fromstring('''<note><pitch><step>E</step><octave>5</octave></pitch>
            <duration>53</duration><type>16th</type><time-modification>
            <actual-notes>9</actual-notes><normal-notes>4</normal-notes>
            </time-modification></note>''')
        symbol = EncodedSymbol('note_16', 'E5')
        symbol.exact_timing = extract_exact_timing(note, 480)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crop.tokens'
            path.write_text(token_lines_to_str([symbol]))
            write_sidecar(path, [symbol])
            restored = read_tokens(str(path))
            attach_sidecar(path, restored)
            classes = {'timing.elapsed': ['1/9'], 'timing.written': ['16th:0'],
                       'timing.ratio': ['9:4'], 'timing.full_rest': ['no', 'yes']}
            targets, diagnostics = build_timing_targets([restored], classes, duration_policy='tick-quantized')
            self.assertEqual(targets['timing.elapsed'].item(), 0)
            self.assertEqual(targets['timing.ratio'].item(), 0)
            self.assertEqual(diagnostics['masked_reasons'], {'explicit_ratio_tick_quantization_normalized': 1})
            self.assertEqual(restored[0].exact_timing['elapsed_quarters'], '53/480')
            self.assertEqual(restored[0].exact_timing['source_duration_ticks'], '53')

    def test_v2_labels_remain_readable_without_guessed_source_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crop.tokens'
            path.write_text(token_lines_to_str(symbols()))
            write_sidecar(path, symbols())
            payload = json.loads(timing_sidecar_path(path).read_text())
            for record in payload['symbols']:
                if record['timing'] is not None:
                    record['timing']['schema'] = 'homr.exact-timing-label.v2'
                    record['timing'].pop('source_divisions')
                    record['timing'].pop('source_duration_ticks')
            timing_sidecar_path(path).write_text(json.dumps(payload))
            restored = read_tokens(str(path))
            attach_sidecar(path, restored)
            duplet = next(s for s in restored if s.pitch == 'E5')
            self.assertNotIn('source_divisions', duplet.exact_timing)
            shell = ET.fromstring('<note><pitch><step>E</step><octave>5</octave></pitch></note>')
            apply_exact_timing(shell, duplet.exact_timing, 4)
            self.assertEqual([n.get('print-object') for n in shell.findall('notations')], ['no', 'yes'])

    def test_inconsistent_source_grid_refused_before_any_attachment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crop.tokens'
            path.write_text(token_lines_to_str(symbols()))
            write_sidecar(path, symbols())
            payload = json.loads(timing_sidecar_path(path).read_text())
            payload['symbols'][-1]['timing']['source_duration_ticks'] = '999'
            timing_sidecar_path(path).write_text(json.dumps(payload))
            restored = read_tokens(str(path))
            with self.assertRaisesRegex(ExactTimingMismatch, 'source tick grid'):
                attach_sidecar(path, restored)
            self.assertTrue(all(s.exact_timing is None for s in restored))

    def test_crop_roundtrip_preserves_rest_duplet_and_grace_alignment(self):
        parsed = symbols()
        # Crop selection does not reconstruct symbols or detach their metadata.
        cropped = parsed[next(i for i, s in enumerate(parsed) if s.rhythm.startswith('rest')):]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crop.tokens'
            text = token_lines_to_str(cropped)
            path.write_text(text)
            write_sidecar(path, cropped)
            restored = read_tokens(str(path))
            attach_sidecar(path, restored)
            self.assertEqual(token_lines_to_str(restored), text)
            notes = [s for s in restored if s.rhythm.startswith(('note', 'rest'))]
            self.assertEqual(sum(s.exact_timing is not None for s in notes), 3)
            self.assertTrue(notes[0].exact_timing['full_measure_rest'])
            self.assertEqual(notes[0].exact_timing['elapsed_quarters'], '6')
            self.assertIsNone(next(s for s in notes if 'G' in s.rhythm).exact_timing)
            duplet = next(s for s in notes if s.pitch == 'E5')
            self.assertEqual(duplet.exact_timing['elapsed_quarters'], '3/4')
            shell = ET.fromstring('<note><pitch><step>E</step><octave>5</octave></pitch></note>')
            apply_exact_timing(shell, duplet.exact_timing, 4)
            self.assertEqual([n.get('print-object') for n in shell.findall('notations')], ['no', 'yes'])
            self.assertEqual(extract_exact_timing(shell, 4), duplet.exact_timing)

    def test_same_count_token_edit_is_refused_by_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crop.tokens'
            path.write_text(token_lines_to_str(symbols()))
            write_sidecar(path, symbols())
            path.write_text(path.read_text().replace('E5', 'F5'))
            with self.assertRaisesRegex(ExactTimingMismatch, 'hash'):
                attach_sidecar(path, read_tokens(str(path)))

    def test_same_count_wrong_symbol_order_is_refused_before_attachment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crop.tokens'
            path.write_text(token_lines_to_str(symbols()))
            write_sidecar(path, symbols())
            restored = read_tokens(str(path))
            restored.reverse()
            with self.assertRaisesRegex(ExactTimingMismatch, 'identity/order'):
                attach_sidecar(path, restored)
            self.assertTrue(all(s.exact_timing is None for s in restored))

    def test_rerun_without_reference_metadata_removes_stale_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'crop.tokens'
            path.write_text(token_lines_to_str(symbols()))
            write_sidecar(path, symbols())
            self.assertTrue(timing_sidecar_path(path).exists())
            write_sidecar(path, read_tokens(str(path)))
            self.assertFalse(timing_sidecar_path(path).exists())
