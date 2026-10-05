"""Replay cached page symbols and isolate late repairs without rerunning recognition.

The full-measure-rest XML edit is a diagnostic hypothesis, not a runtime policy.
It uses the predicted meter and sole-note structure, never reference note labels.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from training.onnx.qualify_joint_export import setup, sha, write
from training.onnx.score_full_page_musicxml import compare, staff_events
from training.onnx.score_paired_page_diagnostic import read_reference


def full_measure_rest_hypothesis(root):
    """Change only sole, visible, undotted whole rests in an explicitly stated meter.

    A lone whole rest can also represent an incomplete bar. This is deliberately an
    ablation rather than evidence that such a rewrite is safe to enable generally.
    """
    changes = []
    for part_no, part in enumerate(root.findall('part'), 1):
        divisions = None
        beats = denominator = None
        for measure_no, measure in enumerate(part.findall('measure'), 1):
            for attributes in measure.findall('attributes'):
                if attributes.find('divisions') is not None:
                    divisions = Fraction(attributes.findtext('divisions'))
                time = attributes.find('time')
                if time is not None:
                    numerator = time.findtext('beats', '')
                    denom = time.findtext('beat-type', '')
                    beats = int(numerator) if numerator.isdigit() else None
                    denominator = int(denom) if denom.isdigit() else None
            notes = measure.findall('note')
            if len(notes) != 1 or measure.find('backup') is not None or measure.find('forward') is not None:
                continue
            note = notes[0]
            if (note.find('rest') is None or note.get('print-object') == 'no'
                    or note.findtext('type') != 'whole' or note.find('dot') is not None
                    or note.find('time-modification') is not None or note.find('grace') is not None
                    or divisions is None or divisions <= 0 or not beats or not denominator):
                continue
            duration = note.find('duration')
            if duration is None or Fraction(duration.text) != 4 * divisions:
                continue
            target = Fraction(4 * beats, denominator) * divisions
            if target.denominator != 1 or target == Fraction(duration.text):
                continue
            changes.append(dict(part=part_no, measure=measure_no,
                                before_quarters=str(Fraction(duration.text) / divisions),
                                after_quarters=str(target / divisions)))
            duration.text = str(target.numerator)
            note.find('rest').set('measure', 'yes')
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'report.json').exists():
        raise ValueError('Refusing to overwrite completed ablation')
    setup()
    from homr import music_xml_generator as generator
    from homr.beam_repair import repair_beams
    from homr.tie_repair import repair_ties
    from homr.slur_crossing import repair_crossings
    from homr.tuplet_repair import repair_symbols
    from homr.transformer.vocabulary import EncodedSymbol, kern_to_symbol_duration
    from training.transformer.review_cached_joint_decoding import restore
    from training.transformer.qualify_joint_holdout import serialize

    original_report = json.loads((args.run / 'comparison.json').read_text())
    page = original_report['page']
    if sha(page['reference']) != page['reference_sha256']:
        raise ValueError('Reference changed')
    reference = staff_events(read_reference(page['reference']), page['start_measure'], page['end_measure'])
    code_root = Path(__file__).resolve().parents[2]
    source_files = [str(Path(__file__).resolve().relative_to(code_root)),
        'homr/music_xml_generator.py', 'homr/transformer/vocabulary.py',
        'homr/beam_repair.py', 'homr/tie_repair.py', 'homr/slur_crossing.py',
        'homr/tuplet_repair.py', 'training/transformer/review_cached_joint_decoding.py',
        'training/onnx/score_full_page_musicxml.py']
    sources = {name: sha(code_root / name) for name in source_files}
    arms = {}
    passes = {'tuplet': repair_symbols, 'beam': repair_beams,
              'tie': repair_ties, 'slur_crossing': repair_crossings}
    variants = ['baseline', 'no_tuplet', 'no_beam', 'no_tie', 'no_slur_crossing',
                'no_late_repairs', 'sole_whole_rest_hypothesis', 'no_meter_inference']
    for arm in ('accepted', 'candidate'):
        identity = json.loads((args.run / arm / 'identity.json').read_text())
        for name, expected in identity['code_sha256'].items():
            if name in sources and sources[name] != expected:
                raise ValueError('Page source changed: ' + name)
        policy = identity['policy']
        if policy['HOMR_STEM_ARBITRATION'] != '0' or policy['HOMR_SLUR_SIDE'] != '0':
            raise ValueError('Replay does not implement stem/slur-side arbitration')
        os.environ.update(policy)
        cached = args.run / arm / 'decoded-staves.json'
        rows = json.loads(cached.read_text())
        output = ET.parse(args.run / arm / 'page.musicxml').getroot()
        if sha(args.run / arm / 'page.musicxml') != original_report['arms'][arm]['xml_sha256']:
            raise ValueError('Original XML changed')
        raw_counts = [dict(Counter(row['rhythm'] for row in voice)) for voice in rows]
        results = {}
        for variant in variants:
            voices = [restore(voice) for voice in rows]
            rewrites = []
            for name, function in passes.items():
                if variant in ('no_' + name, 'no_late_repairs'):
                    continue
                for voice_no, voice in enumerate(voices):
                    before = [serialize(symbol) for symbol in voice]
                    if name == 'tuplet':
                        voices[voice_no], _ = function(voice)
                    else:
                        function(voice)
                    after = [serialize(symbol) for symbol in voices[voice_no]]
                    rewrites.extend(dict(stage=name, staff=voice_no+1, symbol=i,
                                         before=a, after=b)
                                    for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b)
            original_meter_function = generator.stated_numerator_contradicts_bars
            try:
                if variant == 'no_meter_inference':
                    generator.stated_numerator_contradicts_bars = lambda voice, modal: False
                root = generator.generate_xml(generator.XmlGeneratorArguments(), voices, '')
            finally:
                generator.stated_numerator_contradicts_bars = original_meter_function
            rest_changes = full_measure_rest_hypothesis(root) if variant == 'sole_whole_rest_hypothesis' else []
            events = staff_events(root)
            metrics = compare(reference, events)
            exact_replay = ET.tostring(root) == ET.tostring(output)
            if variant == 'baseline' and not exact_replay:
                raise ValueError('Baseline replay does not reproduce original XML: ' + arm)
            if variant == 'baseline' and metrics != original_report['arms'][arm]['metrics']:
                raise ValueError('Baseline metrics changed')
            directory = args.out / arm / variant
            directory.mkdir(parents=True, exist_ok=True)
            xml = directory / 'page.musicxml'
            ET.ElementTree(root).write(xml, encoding='unicode', xml_declaration=True)
            write(directory / 'post-repair-symbols.json', [[serialize(s) for s in voice] for voice in voices])
            results[variant] = dict(metrics=metrics, xml_sha256=sha(xml),
                exact_original_xml=exact_replay, same_events_as_original=events == staff_events(output),
                rewrites=rewrites, rest_changes=rest_changes,
                meters=[(t.findtext('beats'), t.findtext('beat-type')) for t in root.findall('.//time')])
        arms[arm] = dict(cached_sha256=sha(cached), raw_rhythm_counts=raw_counts, variants=results)
    probe = generator.ConversionState(32, Fraction(3, 2))
    probe.beats = 12
    rest = generator.build_note_or_rest(EncodedSymbol('rest_0', '_', '_', '_', '_', 'upper'),
                                       0, False, probe, '')
    report = dict(page=page, arms=arms, threads=1, code_sha256=sources,
        rest_zero_probe=dict(parsed_fraction=str(kern_to_symbol_duration('0').fraction),
                             xml=ET.tostring(rest, encoding='unicode')),
        limitations='Previously evaluated single-page diagnostic. No recognition rerun. '
            'Cached symbols already include parse_staffs repairs; only late passes are ablated. '
            'Sole-rest modification is an unsafe-to-promote hypothesis, not a general policy. '
            'No_meter_inference only disables rejection of a stated numerator; fallback remains. '
            'No onset/barline/notation/lyrics accuracy or crop-token promotion gate.')
    write(args.out / 'report.json', report)
    lines = ['# Cached page XML ablations', '', report['limitations'], '',
             '| Arm | Variant | Event edits | Pitch edits | Rhythm edits | Changes |',
             '|---|---|---:|---:|---:|---:|']
    for arm, data in arms.items():
        for variant, result in data['variants'].items():
            metrics = result['metrics']
            lines.append(f"| {arm} | {variant} | {metrics['event']['errors']} | "
                         f"{metrics['pitch']['errors']} | {metrics['rhythm']['errors']} | "
                         f"{len(result['rewrites']) + len(result['rest_changes'])} |")
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
