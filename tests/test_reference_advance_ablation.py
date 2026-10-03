import os
import unittest
from unittest.mock import patch

from homr.transformer.structured_notation import AdvanceClass, NoteNotation, StemDirection
from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.ablate_real_reference_advance import export_with_counts, paired_timing
from training.onnx.score_full_page_musicxml import staff_events


def event(onset='0', duration='1'):
    return dict(measure=0, onset=onset, duration=duration, pitch=('C', 4, '0'), grace=False)


def notation(advance):
    return NoteNotation(beam_levels=(), stem=StemDirection.NOT_APPLICABLE, slurs=(), advance=advance)


class TestReferenceAdvanceAblation(unittest.TestCase):
    def test_advance_preserves_printed_triplet_duration(self):
        voice = [EncodedSymbol('timeSignatureBeats_4'), EncodedSymbol('timeSignature/4'),
                 EncodedSymbol('note_12', 'C4', '_', '_', '_', 'upper'),
                 EncodedSymbol('chord'),
                 EncodedSymbol('note_4', 'E4', '_', '_', '_', 'upper',
                               notation=notation(AdvanceClass.EIGHTH)),
                 EncodedSymbol('note_8', 'G4', '_', '_', '_', 'upper'),
                 EncodedSymbol('barline')]
        with patch.dict(os.environ):
            baseline, _, _ = export_with_counts([voice], False)
            advanced, _, _ = export_with_counts([voice], True)
        a, b = staff_events(baseline)[0], staff_events(advanced)[0]
        self.assertEqual([e['duration'] for e in a], [e['duration'] for e in b])
        self.assertEqual(b[0]['duration'], '1/3')
        self.assertEqual(b[-1]['onset'], '1/2')

    def test_common_onset_gain_cannot_hide_new_inventory_loss(self):
        result, examples = paired_timing([[event()]], [[event(onset='1')]], [[event(duration='1/2')]])
        self.assertEqual(result['advance_lost_inventory_staff_measures'], 1)
        self.assertNotIn('onset_improved_common_measures', result)
        self.assertIn('new_inventory_loss', examples)

    def test_common_inventory_counts_improvement_and_regression(self):
        result, _ = paired_timing([[event()]], [[event(onset='1')]], [[event()]])
        self.assertEqual(result['onset_improved_common_measures'], 1)
        self.assertEqual(result['baseline_moved_events_on_common_measures'], 1)
        self.assertEqual(result['advance_moved_events_on_common_measures'], 0)
        result, _ = paired_timing([[event()]], [[event()]], [[event(onset='1')]])
        self.assertEqual(result['onset_regressed_common_measures'], 1)

    def test_export_changes_cursor_with_supported_reference_advance(self):
        voice = [EncodedSymbol('note_2', 'C4', '_', '_', '_', 'upper'),
                 EncodedSymbol('chord'),
                 EncodedSymbol('note_4', 'E4', '_', '_', '_', 'upper',
                               notation=notation(AdvanceClass.EIGHTH)),
                 EncodedSymbol('note_8', 'G4', '_', '_', '_', 'upper'),
                 EncodedSymbol('barline')]
        with patch.dict(os.environ):
            baseline, _, _ = export_with_counts([voice], False)
            advanced, counts, _ = export_with_counts([voice], True)
        a, b = staff_events(baseline)[0], staff_events(advanced)[0]
        self.assertEqual([e['duration'] for e in a], [e['duration'] for e in b])
        self.assertEqual(a[-1]['onset'], '1')
        self.assertEqual(b[-1]['onset'], '1/2')
        self.assertEqual(counts['advance_changed_consumed_duration_groups'], 1)

    def test_unpromoted_advance_keeps_the_current_fallback(self):
        voice = [EncodedSymbol('note_4', 'C4', '_', '_', '_', 'upper',
                               notation=notation(AdvanceClass.WHOLE)),
                 EncodedSymbol('note_4', 'D4', '_', '_', '_', 'upper'), EncodedSymbol('barline')]
        with patch.dict(os.environ):
            baseline, _, _ = export_with_counts([voice], False)
            advanced, _, _ = export_with_counts([voice], True)
        self.assertEqual(staff_events(baseline), staff_events(advanced))
