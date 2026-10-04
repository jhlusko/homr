"""Compare rest preservation on hash-verified cached model predictions."""
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
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--archived-generator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'report.json').exists():
        raise ValueError('Refusing to overwrite model diagnostic')
    from homr import music_xml_generator as generator
    from homr.beam_repair import repair_beams
    from homr.tie_repair import repair_ties
    from homr.slur_crossing import repair_crossings
    from homr.tuplet_repair import repair_symbols
    from training.transformer.review_cached_joint_decoding import restore
    old = json.loads((args.baseline / 'report.json').read_text())
    code_root = Path(__file__).resolve().parents[2]
    for name, expected_hash in old['code_sha256'].items():
        path = args.archived_generator if name == 'homr/music_xml_generator.py' else code_root / name
        if sha(path) != expected_hash:
            raise ValueError('Uncontrolled source change: ' + name)
    page = old['page']
    if sha(page['reference']) != page['reference_sha256']:
        raise ValueError('Reference changed')
    reference = staff_events(read_mxl(page['reference']), page['start_measure'], page['end_measure'])
    original = generator.build_note_chord
    legacy = archived_function(args.archived_generator, 'build_note_chord', vars(generator))
    arms = {}
    for arm in ('accepted', 'candidate'):
        directory = args.run / arm
        identity_path = directory / 'identity.json'
        cached = directory / 'decoded-staves.json'
        if sha(cached) != old['arms'][arm]['cached_sha256'] or sha(identity_path) != old['arms'][arm]['identity_sha256']:
            raise ValueError('Cached predictions or identity changed')
        identity = json.loads(identity_path.read_text())
        if identity['policy']['HOMR_STEM_ARBITRATION'] != '0' or identity['policy']['HOMR_SLUR_SIDE'] != '0':
            raise ValueError('Unsupported arbitration replay')
        os.environ.update(identity['policy'])
        rows = json.loads(cached.read_text())
        events, results = {}, {}
        for prefix, function in [('old', legacy), ('new', original)]:
            for enabled in (False, True):
                variant = prefix + ('_advance' if enabled else '_control')
                voices = [restore(voice) for voice in rows]
                for index, voice in enumerate(voices):
                    voices[index], _ = repair_symbols(voice)
                    repair_beams(voices[index])
                    repair_ties(voices[index])
                    repair_crossings(voices[index])
                try:
                    generator.build_note_chord = function
                    xml, _, _ = export_with_counts(voices, enabled)
                finally:
                    generator.build_note_chord = original
                events[variant] = staff_events(xml)
                metrics = compare(reference, events[variant])
                timing = measure_timing(reference, events[variant])
                if prefix == 'old':
                    baseline_variant = 'fixed_advance' if enabled else 'fixed_control'
                    previous = old['arms'][arm]['variants'][baseline_variant]
                    previous_path = args.baseline / arm / (baseline_variant + '.musicxml')
                    if (sha(previous_path) != previous['xml_sha256']
                            or ET.tostring(xml) != ET.tostring(ET.parse(previous_path).getroot())
                            or metrics != previous['metrics'] or timing != previous['timing']):
                        raise ValueError('Cached baseline does not replay exactly: ' + arm)
                output = args.out / arm / (variant + '.musicxml')
                output.parent.mkdir(parents=True, exist_ok=True)
                ET.ElementTree(xml).write(output, encoding='unicode', xml_declaration=True)
                results[variant] = dict(metrics=metrics, timing=timing, xml_sha256=sha(output),
                    rest_events=sum(e['pitch'] is None for staff in events[variant] for e in staff))
        pairs = {}
        for a, b in [('old_control', 'new_control'), ('old_advance', 'new_advance')]:
            counts, examples = paired_timing(reference, events[a], events[b])
            pairs[a + '_vs_' + b] = dict(counts=counts, examples=examples)
        arms[arm] = dict(variants=results, paired=pairs, cached_sha256=sha(cached),
                        identity_sha256=sha(identity_path))
    names = list(old['code_sha256']) + [str(Path(__file__).resolve().relative_to(code_root))]
    write(args.out / 'report.json', dict(page=page, arms=arms, threads=1,
        code_sha256={name: sha(code_root / name) for name in names},
        archived_generator_sha256=sha(args.archived_generator),
        baseline_report_sha256=sha(args.baseline / 'report.json'),
        limitation='Previously bar-count evaluated Lorelei page; no recognition rerun. '
            'Cached hashes and original off/on XML/metrics replay exactly. '
            'Not untouched-page generalization or a model promotion gate.'))


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    main()
