"""Qualify simultaneous-rest preservation with exact OSSQ/Lieder export controls."""
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
    parser.add_argument('--archived-generator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scores', nargs='*', help='Optional bounded preflight')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'scores.jsonl').exists():
        raise ValueError('Refusing to overwrite qualification')
    from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
    from homr import music_xml_generator as generator
    code_root = Path(__file__).resolve().parents[2]
    baseline_report = json.loads((args.baseline / 'report.json').read_text())
    for name, expected_hash in baseline_report['code_sha256'].items():
        path = args.archived_generator if name == 'homr/music_xml_generator.py' else code_root / name
        if sha(path) != expected_hash:
            raise ValueError('Uncontrolled source change: ' + name)
    old_render = archived_function(args.archived_generator, 'build_note_chord', vars(generator))
    new_render = generator.build_note_chord
    rows = [json.loads(line) for line in (args.baseline / 'scores.jsonl').read_text().splitlines()]
    if args.scores:
        rows = [r for r in rows if r['score'] in args.scores]
        if {r['score'] for r in rows} != set(args.scores):
            raise ValueError('Missing preflight reference')
    totals = defaultdict(lambda: defaultdict(Counter))
    tick = time.monotonic()
    with (args.out / 'scores.jsonl').open('w', buffering=1) as output:
        for index, row in enumerate(rows, 1):
            record = {key: row[key] for key in ('source', 'score', 'reference', 'reference_sha256')}
            if sha(row['reference']) != row['reference_sha256']:
                raise ValueError('Reference changed')
            path = Path(row['reference'])
            root = read_mxl(path) if path.suffix == '.mxl' else ET.parse(path).getroot()
            expected = staff_events(root)
            events, xmls = {}, {}
            record['arms'] = {}
            stage = 'parser'
            try:
                parts = music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
                voices = [[s for m in part for s in m] for part in parts]
                stage = 'export'
                for prefix, function in [('old', old_render), ('new', new_render)]:
                    for enabled in (False, True):
                        arm = prefix + ('_advance' if enabled else '_control')
                        try:
                            generator.build_note_chord = function
                            xml, usage, _ = export_with_counts(copy.deepcopy(voices), enabled)
                            xml_hash = hashlib.sha256(ET.tostring(xml)).hexdigest()
                            events[arm] = staff_events(xml)
                            xmls[arm] = xml
                            metrics = compare(expected, events[arm])
                            previous = row['arms']['new_advance' if enabled else 'new_control']
                            timing = measure_timing(expected, events[arm])
                            if prefix == 'old' and (previous['status'] != 'ok'
                                    or xml_hash != previous['xml_sha256'] or metrics != previous['metrics']
                                    or timing != previous['timing']):
                                raise RuntimeError('Prior control mismatch: ' + row['score'])
                            record['arms'][arm] = dict(status='ok', xml_sha256=xml_hash,
                                metrics=metrics, timing=timing, group_usage=usage,
                                rest_events=sum(e['pitch'] is None for staff in events[arm] for e in staff),
                                zero_duration_non_grace_notes=sum(1 for note in xml.findall('.//note')
                                    if note.find('grace') is None and note.findtext('duration') == '0'))
                        except RuntimeError:
                            raise
                        except Exception as error:
                            message = type(error).__name__ + ': ' + str(error)
                            previous = row['arms']['new_advance' if enabled else 'new_control']
                            if prefix == 'old' and (previous['status'] != 'failed'
                                    or stage != previous['stage'] or message != previous['error']):
                                raise RuntimeError('Prior failure mismatch') from error
                            record['arms'][arm] = dict(status='failed', stage=stage, error=message)
                        finally:
                            generator.build_note_chord = new_render
            except RuntimeError:
                raise
            except Exception as error:
                message = type(error).__name__ + ': ' + str(error)
                for prefix in ('old', 'new'):
                    for enabled in (False, True):
                        previous = row['arms']['new_advance' if enabled else 'new_control']
                        if previous['status'] != 'failed' or stage != previous['stage'] or message != previous['error']:
                            raise RuntimeError('Parser failure does not replay') from error
                        record['arms'][prefix + ('_advance' if enabled else '_control')] = dict(
                            status='failed', stage=stage, error=message)
            record['paired'] = {}
            for a, b in [('old_control', 'new_control'), ('old_advance', 'new_advance'),
                          ('new_control', 'new_advance')]:
                if a in events and b in events:
                    key = a + '_vs_' + b
                    counts, examples = paired_timing(expected, events[a], events[b])
                    record['paired'][key] = dict(counts=counts, examples=examples)
                    totals[row['source']][key].update(counts)
            if args.scores or row['score'] == 'IMSLP89026':
                directory = args.out / 'examples' / row['score']
                directory.mkdir(parents=True, exist_ok=True)
                for arm, xml in xmls.items():
                    ET.ElementTree(xml).write(directory / (arm + '.musicxml'), encoding='unicode')
            for arm, result in record['arms'].items():
                summary = totals[row['source']][arm]
                summary[result['status']] += 1
                if result['status'] == 'ok':
                    summary.update(result['timing'])
                    summary['rest_events'] += result['rest_events']
                    summary['zero_duration_non_grace_notes'] += result['zero_duration_non_grace_notes']
                    for metric in ('event', 'pitch', 'rhythm', 'accidental'):
                        summary[metric + '_errors'] += result['metrics'][metric]['errors']
            output.write(json.dumps(record) + '\n')
            if index % 10 == 0:
                print(f'{index}/{len(rows)} in {time.monotonic()-tick:.1f}s', flush=True)
    names = list(baseline_report['code_sha256']) + [str(Path(__file__).resolve().relative_to(code_root))]
    write(args.out / 'report.json', dict(sources=totals, seconds=time.monotonic()-tick,
        records=len(rows), bounded_preflight=bool(args.scores), threads=1,
        code_sha256={name: sha(code_root / name) for name in names},
        archived_generator_sha256=sha(args.archived_generator),
        baseline_report_sha256=sha(args.baseline / 'report.json'),
        baseline_records_sha256=sha(args.baseline / 'scores.jsonl'),
        limitation='Reference-label/XML roundtrips, exact prior controls; '
            'fixed staff/measure inventory diagnostics, not model recognition or promotion.'))


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    main()
