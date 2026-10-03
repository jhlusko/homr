"""Isolate predicted advance on cached page decodes with exact archived control replay.

Previously evaluated single-page diagnostic; no recognition rerun or promotion gate.
"""
import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from training.onnx.ablate_real_reference_advance import export_with_counts, paired_timing
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write
from training.onnx.qualify_reference_timing_fixes import archived_function
from training.onnx.roundtrip_real_timing_labels import measure_timing
from training.onnx.score_full_page_musicxml import compare, staff_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--archived-generator', type=Path, required=True)
    parser.add_argument('--cached-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'report.json').exists():
        raise ValueError('Refusing to overwrite diagnostic')
    from homr import music_xml_generator as generator
    from homr.beam_repair import repair_beams
    from homr.tie_repair import repair_ties
    from homr.slur_crossing import repair_crossings
    from homr.tuplet_repair import repair_symbols
    from training.transformer.review_cached_joint_decoding import restore
    original = json.loads((args.run / 'comparison.json').read_text())
    receipt = json.loads(args.cached_receipt.read_text())
    page = original['page']
    if sha(page['reference']) != page['reference_sha256']:
        raise ValueError('Reference changed')
    reference = staff_events(read_mxl(page['reference']), page['start_measure'], page['end_measure'])
    division = generator.find_division_and_time_signature_nominator
    old_division = archived_function(args.archived_generator,
        'find_division_and_time_signature_nominator', vars(generator))
    arms = {}
    code_root = Path(__file__).resolve().parents[2]
    names = ['homr/music_xml_generator.py', 'homr/beam_repair.py', 'homr/tie_repair.py',
             'homr/slur_crossing.py', 'homr/tuplet_repair.py', 'homr/transformer/vocabulary.py',
             'training/transformer/review_cached_joint_decoding.py',
             'training/onnx/score_full_page_musicxml.py',
             str(Path(__file__).resolve().relative_to(code_root))]
    for arm in ('accepted', 'candidate'):
        identity = json.loads((args.run / arm / 'identity.json').read_text())
        if sha(args.archived_generator) != identity['code_sha256']['homr/music_xml_generator.py']:
            raise ValueError('Wrong archived generator for ' + arm)
        for name in names:
            if name != 'homr/music_xml_generator.py' and name in identity['code_sha256']:
                if sha(code_root / name) != identity['code_sha256'][name]:
                    raise ValueError('Non-generator source changed: ' + name)
        os.environ.update(identity['policy'])
        if identity['policy']['HOMR_STEM_ARBITRATION'] != '0' or identity['policy']['HOMR_SLUR_SIDE'] != '0':
            raise ValueError('Unsupported arbitration replay')
        cached = args.run / arm / 'decoded-staves.json'
        if sha(cached) != receipt['arms'][arm]['cached_sha256']:
            raise ValueError('Cached predictions changed: ' + arm)
        rows = json.loads(cached.read_text())
        baseline_path = args.run / arm / 'page.musicxml'
        if sha(baseline_path) != original['arms'][arm]['xml_sha256']:
            raise ValueError('Original XML changed')
        baseline_xml = ET.parse(baseline_path).getroot()
        events = {}
        results = {}
        for variant, function, enabled in [('archived_control', old_division, False),
                                            ('fixed_control', division, False),
                                            ('fixed_advance', division, True)]:
            voices = [restore(voice) for voice in rows]
            for index, voice in enumerate(voices):
                voices[index], _ = repair_symbols(voice)
                repair_beams(voices[index])
                repair_ties(voices[index])
                repair_crossings(voices[index])
            try:
                generator.find_division_and_time_signature_nominator = function
                root, usage, classes = export_with_counts(voices, enabled)
            finally:
                generator.find_division_and_time_signature_nominator = division
            events[variant] = staff_events(root)
            metrics = compare(reference, events[variant])
            if variant == 'archived_control' and (
                    ET.tostring(root) != ET.tostring(baseline_xml)
                    or metrics != original['arms'][arm]['metrics']):
                raise ValueError('Original page does not replay exactly: ' + arm)
            path = args.out / arm / (variant + '.musicxml')
            path.parent.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(root).write(path, encoding='unicode', xml_declaration=True)
            results[variant] = dict(metrics=metrics, timing=measure_timing(reference, events[variant]),
                xml_sha256=sha(path), group_usage=usage, classes=classes,
                zero_duration_non_grace_notes=sum(1 for note in root.findall('.//note')
                    if note.find('grace') is None and note.findtext('duration') == '0'))
        paired, examples = paired_timing(reference, events['fixed_control'], events['fixed_advance'])
        arms[arm] = dict(cached_sha256=sha(cached), identity_sha256=sha(args.run / arm / 'identity.json'),
                        variants=results, paired=paired, examples=examples)
    write(args.out / 'report.json', dict(page=page, arms=arms, threads=1,
        code_sha256={name: sha(code_root / name) for name in names},
        archived_generator_sha256=sha(args.archived_generator),
        cached_receipt_sha256=sha(args.cached_receipt),
        limitation='Previously bar-count evaluated Lorelei page, no recognition rerun. '
            'Cached symbols include upstream repairs; late repairs held constant. '
            'Fixed staff/measure inventory diagnostics ignore voice identity. '
            'This is not untouched-page generalization or a model promotion gate.'))


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    main()
