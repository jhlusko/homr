"""Replay archived controls and isolate timing corrections on real references."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from training.onnx.ablate_real_reference_advance import export_with_counts, paired_timing
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write
from training.onnx.roundtrip_real_timing_labels import measure_timing
from training.onnx.score_full_page_musicxml import compare, staff_events


def archived_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'scores.jsonl').exists():
        raise ValueError('Refusing to overwrite qualification')
    from training.omr_datasets import music_xml_parser as source
    from homr import music_xml_generator as generator
    snapshot = args.baseline / 'source-snapshot'
    old_report = json.loads((args.baseline / 'report.json').read_text())
    for name, expected in old_report['code_sha256'].items():
        if sha(snapshot / name) != expected:
            raise ValueError('Archived source mismatch: ' + name)
    old_append = archived_function(snapshot / 'training/omr_datasets/music_xml_parser.py',
        'append_note', vars(source), 'TokensMeasure')
    old_division = archived_function(snapshot / 'homr/music_xml_generator.py',
        'find_division_and_time_signature_nominator', vars(generator))
    new_append = source.TokensMeasure.append_note
    new_division = generator.find_division_and_time_signature_nominator
    rows = [json.loads(line) for line in (args.baseline / 'scores.jsonl').read_text().splitlines()]
    totals = defaultdict(lambda: defaultdict(Counter))
    tick = time.monotonic()
    with (args.out / 'scores.jsonl').open('w', buffering=1) as output:
        for index, row in enumerate(rows, 1):
            record = {key: row[key] for key in ('source', 'score', 'reference', 'reference_sha256')}
            if sha(row['reference']) != row['reference_sha256']:
                raise ValueError('Reference changed')
            reference = Path(row['reference'])
            root = read_mxl(reference) if reference.suffix == '.mxl' else ET.parse(reference).getroot()
            expected = staff_events(root)
            events = {}
            record['arms'] = {}
            for arm, append, division, enabled in [
                ('old_control', old_append, old_division, False),
                ('old_advance', old_append, old_division, True),
                ('precision_only', old_append, new_division, True),
                ('fixed_control', new_append, new_division, False),
                ('fixed_advance', new_append, new_division, True)]:
                stage = 'parser'
                try:
                    source.TokensMeasure.append_note = append
                    generator.find_division_and_time_signature_nominator = division
                    parts = source.music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
                    voices = [[s for m in part for s in m] for part in parts]
                    stage = 'export'
                    xml, _, _ = export_with_counts(voices, enabled)
                    xml_hash = hashlib.sha256(ET.tostring(xml)).hexdigest()
                    events[arm] = staff_events(xml)
                    metrics = compare(expected, events[arm])
                    if arm.startswith('old_'):
                        hash_key = 'control_xml_sha256' if arm == 'old_control' else 'advance_xml_sha256'
                        metric_key = 'baseline_metrics' if arm == 'old_control' else 'advance_metrics'
                        if row['status'] != 'ok' or xml_hash != row[hash_key] or metrics != row[metric_key]:
                            raise RuntimeError('Archived control mismatch: ' + row['score'])
                    record['arms'][arm] = dict(status='ok', xml_sha256=xml_hash,
                        metrics=metrics, timing=measure_timing(expected, events[arm]))
                except RuntimeError:
                    raise
                except Exception as error:
                    message = type(error).__name__ + ': ' + str(error)
                    if arm.startswith('old_') and (row['status'] != 'failed'
                            or stage != row['failure_stage'] or message != row['error']):
                        raise RuntimeError('Archived failure mismatch') from error
                    record['arms'][arm] = dict(status='failed', stage=stage, error=message)
                finally:
                    source.TokensMeasure.append_note = new_append
                    generator.find_division_and_time_signature_nominator = new_division
                result = record['arms'][arm]
                totals[row['source']][arm]['success' if result['status'] == 'ok' else 'failed'] += 1
                if result['status'] == 'ok':
                    totals[row['source']][arm].update(result['timing'])
                    totals[row['source']][arm]['event_errors'] += result['metrics']['event']['errors']
            record['paired'] = {}
            for a, b in [('old_advance', 'precision_only'), ('old_advance', 'fixed_advance'),
                          ('fixed_control', 'fixed_advance')]:
                if a in events and b in events:
                    key = a + '_vs_' + b
                    paired, examples = paired_timing(expected, events[a], events[b])
                    record['paired'][key] = dict(counts=paired, examples=examples)
                    totals[row['source']][key].update(paired)
            output.write(json.dumps(record) + '\n')
            if index % 10 == 0:
                print(f'{index}/{len(rows)} in {time.monotonic()-tick:.1f}s', flush=True)
    code_root = Path(__file__).resolve().parents[2]
    names = ['training/omr_datasets/music_xml_parser.py', 'homr/music_xml_generator.py',
             str(Path(__file__).resolve().relative_to(code_root))]
    write(args.out / 'report.json', dict(sources=totals, seconds=time.monotonic()-tick,
        threads=1, baseline_report_sha256=sha(args.baseline / 'report.json'),
        code_sha256={name: sha(code_root / name) for name in names},
        limitation='Canonical reference roundtrips, fixed staff/measure inventory comparisons; '
            'not model predictions, voice-aware equivalence or promotion.'))


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    main()
