import copy
from fractions import Fraction
import unittest
import xml.etree.ElementTree as ET

import torch

from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.exact_timing_labels import extract_exact_timing
from training.architecture.transformer.exact_timing_targets import HEADS, timing_values, build_timing_targets
from training.architecture.transformer.exact_timing_heads import initialize_timing_bundle, load_timing_bundle, timing_loss
from training.architecture.transformer.structured_targets import align_to_decoder_output
from training.architecture.transformer.structured_losses import IGNORE_INDEX


def duplet_symbol():
    note = ET.fromstring('''<note><pitch><step>C</step><octave>5</octave></pitch>
        <duration>3</duration><type>eighth</type><time-modification>
        <actual-notes>2</actual-notes><normal-notes>3</normal-notes></time-modification></note>''')
    symbol = EncodedSymbol('note_5', 'C5')
    symbol.exact_timing = extract_exact_timing(note, 4)
    return symbol


def manifest():
    return dict(schemaVersion='homr.exact-timing-targets.v1', classes={
        'timing.elapsed': ['1', '3/4'], 'timing.written': ['eighth:0', 'quarter:0'],
        'timing.ratio': ['1:1', '2:3'], 'timing.full_rest': ['no', 'yes']})


class TestTimingTargets(unittest.TestCase):
    def test_duplet_uses_canonical_duration_and_written_type_not_rounded_token(self):
        symbol = duplet_symbol()
        values, flags = timing_values(symbol.exact_timing, False)
        self.assertEqual(values, {'timing.elapsed': '3/4', 'timing.written': 'eighth:0', 'timing.ratio': '2:3'})
        self.assertEqual(flags, [])
        self.assertEqual(symbol.rhythm, 'note_5')

    def test_full_rest_only_supervises_duration_and_rest_identity(self):
        payload = extract_exact_timing(ET.fromstring('<note><rest measure="yes"/><duration>24</duration></note>'), 4)
        self.assertEqual(timing_values(payload, True)[0], {'timing.elapsed': '6', 'timing.full_rest': 'yes'})

    def test_inconsistent_source_ratio_is_masked_without_losing_written_target(self):
        payload = duplet_symbol().exact_timing
        payload['elapsed_quarters'] = '1/2'
        values, flags = timing_values(payload, False)
        self.assertNotIn('timing.ratio', values)
        self.assertEqual(values['timing.written'], 'eighth:0')
        self.assertEqual(flags, ['ratio_source_duration_disagreement'])

    def test_bos_eos_padding_and_decoder_shift_do_not_supply_free_targets(self):
        sequences = [[EncodedSymbol('BOS'), duplet_symbol(), EncodedSymbol('EOS')]]
        targets, diagnostic = build_timing_targets(sequences, manifest()['classes'], sequence_length=4)
        shifted = align_to_decoder_output(targets)
        self.assertEqual(shifted['timing.elapsed'].tolist(), [[1, IGNORE_INDEX, IGNORE_INDEX]])
        self.assertTrue((targets['timing.full_rest'] == IGNORE_INDEX).all())
        self.assertEqual(sum(diagnostic['unseen'].values()), 0)

    def test_unseen_duration_is_reported_and_not_folded_to_a_common_class(self):
        symbol = duplet_symbol();symbol.exact_timing['elapsed_quarters'] = '79/16'
        targets, diagnostic = build_timing_targets([[symbol]], manifest()['classes'])
        self.assertEqual(targets['timing.elapsed'].item(), IGNORE_INDEX)
        self.assertEqual(diagnostic['unseen']['timing.elapsed'], 1)

    def test_nested_tuplet_floor_tick_is_classified_as_quantization(self):
        payload = duplet_symbol().exact_timing
        payload.update(elapsed_quarters='53/480', written_type='16th',
                       time_modification_xml='<time-modification><actual-notes>9</actual-notes>'
                           '<normal-notes>4</normal-notes></time-modification>')
        values, flags = timing_values(payload, False, tick_quarters=Fraction(1, 480))
        self.assertEqual(values['timing.elapsed'], '1/9')
        self.assertEqual(values['timing.ratio'], '9:4')
        self.assertEqual(flags, ['explicit_ratio_tick_quantization_normalized'])
        payload['elapsed_quarters'] = '52/480'
        values, flags = timing_values(payload, False, tick_quarters=Fraction(1, 480))
        self.assertNotIn('timing.elapsed', values)
        self.assertEqual(flags, ['ratio_source_duration_disagreement'])

    def test_quantized_contract_refuses_sidecars_without_source_tick_resolution(self):
        with self.assertRaisesRegex(ValueError, 'source-divisions'):
            build_timing_targets([[duplet_symbol()]], manifest()['classes'], duration_policy='tick-quantized')


class TestTimingMigration(unittest.TestCase):
    def test_auxiliary_initialization_preserves_core_parameters_logits_and_rng(self):
        core = torch.nn.Linear(4, 3)
        hidden = torch.randn(2, 3, 4)
        before = core(hidden).detach().clone()
        weights = {k: v.clone() for k, v in core.state_dict().items()}
        rng = torch.get_rng_state().clone()
        bundle = initialize_timing_bundle(4, manifest(), 'a'*64, seed=20261004)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        heads = load_timing_bundle(bundle, manifest(), 'a'*64, allow_untrained=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        logits = heads(hidden.detach())
        self.assertEqual(set(logits), set(HEADS))
        self.assertTrue(torch.equal(before, core(hidden)))
        for k, v in core.state_dict().items():
            self.assertTrue(torch.equal(weights[k], v))

    def test_wrong_base_or_same_size_reordered_classes_are_refused(self):
        bundle = initialize_timing_bundle(4, manifest(), 'a'*64, seed=0)
        with self.assertRaisesRegex(ValueError, 'base hash'):
            load_timing_bundle(bundle, manifest(), 'b'*64, allow_untrained=True)
        reordered = copy.deepcopy(manifest())
        reordered['classes']['timing.ratio'].reverse()
        with self.assertRaisesRegex(ValueError, 'class order'):
            load_timing_bundle(bundle, reordered, 'a'*64, allow_untrained=True)
        with self.assertRaisesRegex(ValueError, 'Untrained'):
            load_timing_bundle(bundle, manifest(), 'a'*64)

    def test_explicit_loss_weights_and_masking_allow_head_only_gradients(self):
        bundle = initialize_timing_bundle(4, manifest(), 'a'*64, seed=0)
        heads = load_timing_bundle(bundle, manifest(), 'a'*64, allow_untrained=True)
        hidden = torch.randn(1, 1, 4, requires_grad=True)
        targets, _ = build_timing_targets([[duplet_symbol()]], manifest()['classes'])
        logits = heads(hidden.detach())
        with self.assertRaisesRegex(ValueError, 'four explicit'):
            timing_loss(logits, targets, weights={})
        loss = timing_loss(logits, targets, weights={h: 1.0 for h in HEADS})
        loss.total.backward()
        self.assertIsNone(hidden.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in heads.parameters()))
        self.assertEqual(next(h for h in loss.heads if h.name == 'timing.full_rest').supervised, 0)
