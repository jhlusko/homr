"""Output-only timing targets; legacy recognition targets stay unchanged."""
from fractions import Fraction
import xml.etree.ElementTree as ET

import torch

from training.architecture.transformer.structured_losses import IGNORE_INDEX

HEADS = ('timing.elapsed', 'timing.written', 'timing.ratio', 'timing.full_rest')
TYPE_QUARTERS = {
    'maxima': Fraction(32), 'long': Fraction(16), 'breve': Fraction(8),
    'whole': Fraction(4), 'half': Fraction(2), 'quarter': Fraction(1),
    'eighth': Fraction(1, 2), '16th': Fraction(1, 4), '32nd': Fraction(1, 8),
    '64th': Fraction(1, 16), '128th': Fraction(1, 32), '256th': Fraction(1, 64),
    '512th': Fraction(1, 128), '1024th': Fraction(1, 256),
}


def timing_values(payload, is_rest, *, tick_quarters=None):
    """Available targets and source-inconsistent ratio flags, never guessed OTHER."""
    if payload is None:
        return {}, []
    elapsed = Fraction(payload['elapsed_quarters'])
    if elapsed <= 0:
        raise ValueError('Nonpositive timing target')
    full_rest = payload['full_measure_rest']
    values = {'timing.elapsed': str(elapsed)}
    if is_rest:
        values['timing.full_rest'] = 'yes' if full_rest else 'no'
    if full_rest:
        return values, []
    written = payload['written_type']
    dots = payload['dots']
    if written is None:
        return values, ['missing_written_type']
    values['timing.written'] = f'{written}:{dots}'
    if written not in TYPE_QUARTERS:
        return values, ['unsupported_written_type']
    modification = payload['time_modification_xml']
    ratio = Fraction(1)
    if modification is not None:
        root = ET.fromstring(modification)
        actual = int(root.findtext('actual-notes'))
        normal = int(root.findtext('normal-notes'))
        if min(actual, normal) <= 0:
            raise ValueError('Nonpositive timing ratio')
        ratio = Fraction(actual, normal)
    written_duration = TYPE_QUARTERS[written] * sum(
        (Fraction(1, 2**i) for i in range(dots + 1)), Fraction(0))
    if written_duration / ratio != elapsed:
        if tick_quarters is not None and modification is not None:
            tick = Fraction(tick_quarters)
            if tick <= 0:
                raise ValueError('Nonpositive source tick grid')
            exact_ticks = (written_duration / ratio) / tick
            stored_ticks = elapsed / tick
            floor = exact_ticks.numerator // exact_ticks.denominator
            if (exact_ticks.denominator != 1 and stored_ticks.denominator == 1
                    and stored_ticks.numerator in (floor, floor + 1)):
                values['timing.elapsed'] = str(written_duration / ratio)
                values['timing.ratio'] = f'{ratio.numerator}:{ratio.denominator}'
                return values, ['explicit_ratio_tick_quantization_normalized']
        if tick_quarters is not None:
            # Candidate strict-quality policy: preserve raw metadata, but do
            # not supervise sounding duration on unresolved source conflicts.
            values.pop('timing.elapsed')
        return values, ['ratio_source_duration_disagreement']
    values['timing.ratio'] = f'{ratio.numerator}:{ratio.denominator}'
    return values, []


def build_timing_targets(sequences, classes, *, sequence_length=None, duration_policy='strict'):
    """Full token-position targets, including BOS/EOS; caller shifts once for decode.

    Missing metadata and unseen train-only classes stay masked. Diagnostics retain
    source-disagreement and unseen counts instead of silently relabelling rare notes.
    """
    if duration_policy not in ('strict', 'tick-quantized'):
        raise ValueError('Unknown timing duration policy')
    length = max((len(s) for s in sequences), default=0) if sequence_length is None else sequence_length
    if any(len(s) > length for s in sequences):
        raise ValueError('Cannot truncate timing target sequence')
    targets = {h: torch.full((len(sequences), length), IGNORE_INDEX, dtype=torch.long) for h in HEADS}
    lookup = {h: {v: i for i, v in enumerate(classes[h])} for h in HEADS}
    diagnostics = {'unseen': {h: 0 for h in HEADS}, 'masked_reasons': {}}
    for b, sequence in enumerate(sequences):
        for t, symbol in enumerate(sequence):
            tick = None
            if duration_policy == 'tick-quantized' and symbol.exact_timing is not None:
                divisions = symbol.exact_timing.get('source_divisions')
                if divisions is None:
                    raise ValueError('Tick-quantized targets require source-divisions metadata')
                tick = Fraction(1) / Fraction(divisions)
            values, reasons = timing_values(symbol.exact_timing, symbol.rhythm.startswith('rest'), tick_quarters=tick)
            for reason in reasons:
                diagnostics['masked_reasons'][reason] = diagnostics['masked_reasons'].get(reason, 0) + 1
            for h, value in values.items():
                if value in lookup[h]:
                    targets[h][b, t] = lookup[h][value]
                else:
                    diagnostics['unseen'][h] += 1
    return targets, diagnostics
