"""Qualify ending-marker separation against exact prior OSSQ/Lieder controls."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from training.onnx.ablate_real_reference_advance import export_with_counts, paired_timing
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write
from training.onnx.qualify_reference_timing_fixes import archived_function
from training.onnx.roundtrip_real_timing_labels import measure_timing
from training.onnx.score_full_page_musicxml import compare, staff_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--archived-merger', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scores', nargs='*', help='Optional bounded preflight score identities')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'scores.jsonl').exists():
        raise ValueError('Refusing to overwrite qualification')
    from training.omr_datasets import music_xml_parser as source
    from training.omr_datasets import staff_merging as merging
    root_path = Path(__file__).resolve().parents[2]
    baseline_report = json.loads((args.baseline / 'report.json').read_text())
    for name, expected_hash in baseline_report['code_sha256'].items():
        if sha(root_path / name) != expected_hash:
            raise ValueError('Non-merger source changed: ' + name)
    old_merge = archived_function(args.archived_merger, 'create_chord_over_two_staffs', vars(merging))
    new_merge = merging.create_chord_over_two_staffs
    rows = [json.loads(line) for line in (args.baseline / 'scores.jsonl').read_text().splitlines()]
    if args.scores:
        rows = [r for r in rows if r['score'] in args.scores]
        if {r['score'] for r in rows} != set(args.scores):
            raise ValueError('Missing preflight score')
    totals = defaultdict(lambda: defaultdict(Counter))
    tick = time.monotonic()
    with (args.out / 'scores.jsonl').open('w', buffering=1) as output:
        for index, row in enumerate(rows, 1):
            record = {key: row[key] for key in ('source', 'score', 'reference', 'reference_sha256')}
            if sha(row['reference']) != row['reference_sha256']:
                raise ValueError('Reference changed')
            reference_path = Path(row['reference'])
            root = read_mxl(reference_path) if reference_path.suffix == '.mxl' else ET.parse(reference_path).getroot()
            expected = staff_events(root)
            events = {}
            record['arms'] = {}
            for prefix, function in [('old', old_merge), ('new', new_merge)]:
                stage = 'parser'
                try:
                    merging.create_chord_over_two_staffs = function
                    parts = source.music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
                    voices = [[s for m in part for s in m] for part in parts]
                    stage = 'export'
                    for enabled in (False, True):
                        arm = prefix + ('_advance' if enabled else '_control')
                        xml, usage, _ = export_with_counts(copy.deepcopy(voices), enabled)
                        xml_hash = hashlib.sha256(ET.tostring(xml)).hexdigest()
                        events[arm] = staff_events(xml)
                        metrics = compare(expected, events[arm])
                        if prefix == 'old':
                            old = row['arms']['fixed_advance' if enabled else 'fixed_control']
                            if (old['status'] != 'ok' or xml_hash != old['xml_sha256']
                                    or metrics != old['metrics'] or measure_timing(expected, events[arm]) != old['timing']):
                                raise RuntimeError('Prior control mismatch: ' + row['score'])
                        record['arms'][arm] = dict(status='ok', xml_sha256=xml_hash,
                            metrics=metrics, timing=measure_timing(expected, events[arm]),
                            grace_events=sum(e['grace'] for staff in events[arm] for e in staff),
                            group_usage=usage)
                        if row['score'] in ('sq9396439', 'IMSLP89026'):
                            directory = args.out / 'examples' / row['score']
                            directory.mkdir(parents=True, exist_ok=True)
                            ET.ElementTree(xml).write(directory / (arm + '.musicxml'), encoding='unicode')
                except RuntimeError:
                    raise
                except Exception as error:
                    message = type(error).__name__ + ': ' + str(error)
                    for enabled in (False, True):
                        arm = prefix + ('_advance' if enabled else '_control')
                        if arm in record['arms']:
                            continue
                        old = row['arms']['fixed_advance' if enabled else 'fixed_control']
                        if prefix == 'old' and (old['status'] != 'failed' or stage != old['stage'] or message != old['error']):
                            raise RuntimeError('Prior failure mismatch') from error
                        record['arms'][arm] = dict(status='failed', stage=stage, error=message)
                finally:
                    merging.create_chord_over_two_staffs = new_merge
            record['paired'] = {}
            for a, b in [('old_control', 'new_control'), ('old_advance', 'new_advance'),
                          ('new_control', 'new_advance')]:
                if a in events and b in events:
                    key = a + '_vs_' + b
                    counts, examples = paired_timing(expected, events[a], events[b])
                    record['paired'][key] = dict(counts=counts, examples=examples)
                    totals[row['source']][key].update(counts)
            for arm, result in record['arms'].items():
                summary = totals[row['source']][arm]
                summary[result['status']] += 1
                if result['status'] == 'ok':
                    summary.update(result['timing'])
                    summary['grace_events'] += result['grace_events']
                    for metric in ('event', 'pitch', 'rhythm', 'accidental'):
                        summary[metric + '_errors'] += result['metrics'][metric]['errors']
            output.write(json.dumps(record) + '\n')
            if index % 10 == 0:
                print(f'{index}/{len(rows)} in {time.monotonic()-tick:.1f}s', flush=True)
    names = list(baseline_report['code_sha256']) + ['training/omr_datasets/staff_merging.py',
             str(Path(__file__).resolve().relative_to(root_path))]
    write(args.out / 'report.json', dict(sources=totals, seconds=time.monotonic()-tick,
        threads=1, records=len(rows), bounded_preflight=bool(args.scores),
        code_sha256={name: sha(root_path / name) for name in names},
        archived_merger_sha256=sha(args.archived_merger),
        baseline_report_sha256=sha(args.baseline / 'report.json'),
        baseline_records_sha256=sha(args.baseline / 'scores.jsonl'),
        limitation='Canonical reference-label/XML timing; exact prior controls, '
            'fixed staff/measure inventory comparisons, not model predictions or promotion.'))


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    main()
