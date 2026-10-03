"""CPU-only real-reference parser/export roundtrip, preserving failure coverage."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_full_page_musicxml import compare, project, staff_events


def measure_timing(reference, predicted):
    """Isolate order/onset changes where a physical measure's inventory survives.

    Fixed physical staff and measure index; no best-match reassignment. Lossy measures
    remain excluded from this conditional diagnostic, not from the overall metrics.
    """
    counts = Counter()
    for staff in range(max(len(reference), len(predicted))):
        a = reference[staff] if staff < len(reference) else []
        b = predicted[staff] if staff < len(predicted) else []
        by_measure_a, by_measure_b = defaultdict(list), defaultdict(list)
        for event in a:
            by_measure_a[event['measure']].append(event)
        for event in b:
            by_measure_b[event['measure']].append(event)
        for measure in sorted(set(by_measure_a) | set(by_measure_b)):
            x, y = by_measure_a[measure], by_measure_b[measure]
            counts['staff_measures'] += 1
            if Counter(project(x, 'event')) != Counter(project(y, 'event')):
                counts['changed_inventory_staff_measures'] += 1
                continue
            counts['inventory_preserved_staff_measures'] += 1
            counts['inventory_preserved_events'] += len(x)
            if project(x, 'event') != project(y, 'event'):
                counts['order_changed_inventory_preserved_staff_measures'] += 1
            timed_x = Counter((event['onset'], key) for event, key in zip(x, project(x, 'event'), strict=True))
            timed_y = Counter((event['onset'], key) for event, key in zip(y, project(y, 'event'), strict=True))
            moved = sum((timed_x - timed_y).values())
            if moved:
                counts['onset_changed_inventory_preserved_staff_measures'] += 1
                counts['onset_moved_inventory_preserved_events'] += moved
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'report.json').exists():
        raise ValueError('Refusing to overwrite completed audit')
    os.environ['HOMR_STRUCTURED_EXPORT'] = 'beam,stem,tie,slur'
    from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
    from homr.music_xml_generator import generate_xml, XmlGeneratorArguments
    rows = [json.loads(line) for line in (args.census / 'references.jsonl').read_text().splitlines()]
    aggregates = defaultdict(lambda: dict(counts=Counter(), timings=Counter(), event_errors=0,
                                         event_support=0, score_status=Counter(), affected=defaultdict(set)))
    tick = time.monotonic()
    with (args.out / 'scores.jsonl').open('w', buffering=1) as output:
        for number, row in enumerate(rows, 1):
            record = dict(source=row['source'], score=row['score'], reference=row['reference'],
                          reference_sha256=row['sha256'], indexed_score=row['indexed_score'])
            stage = 'reference'
            try:
                if sha(row['reference']) != row['sha256']:
                    raise ValueError('Reference changed')
                path = Path(row['reference'])
                root = read_mxl(path) if path.suffix == '.mxl' else ET.parse(path).getroot()
                expected = staff_events(root)
                stage = 'parser'
                parts = music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
                voices = [[symbol for measure in part for symbol in measure] for part in parts]
                stage = 'export'
                result = generate_xml(XmlGeneratorArguments(), voices, '')
                actual = staff_events(result)
                stage = 'score'
                metrics = compare(expected, actual)
                timings = measure_timing(expected, actual)
                counts = dict(reference_events=sum(map(len, expected)),
                              reference_staves=len(expected), predicted_staves=len(actual))
                record.update(status='ok', metrics=metrics, timing=timings, counts=counts,
                              xml_sha256=__import__('hashlib').sha256(ET.tostring(result)).hexdigest())
            except Exception as error:
                record.update(status='failed', failure_stage=stage,
                              error=type(error).__name__ + ': ' + str(error))
            output.write(json.dumps(record) + '\n')
            for scope in ('all_available_references', 'indexed_score_references'):
                if scope == 'indexed_score_references' and not row['indexed_score']:
                    continue
                group = aggregates[(row['source'], scope)]
                group['score_status'][record['status']] += 1
                if record['status'] == 'ok':
                    group['counts'].update(record['counts'])
                    group['timings'].update(record['timing'])
                    group['event_errors'] += record['metrics']['event']['errors']
                    group['event_support'] += record['metrics']['event']['support']
                    for key, value in record['timing'].items():
                        if value and ('changed' in key or 'moved' in key):
                            group['affected'][key].add(row['score'])
            if number % 10 == 0:
                print(f'{number}/{len(rows)} references in {time.monotonic()-tick:.1f}s', flush=True)
    summary = defaultdict(dict)
    for (source, scope), values in aggregates.items():
        summary[source][scope] = dict(counts=dict(values['counts']), timing=dict(values['timings']),
            event_errors=values['event_errors'], event_support=values['event_support'],
            score_status=dict(values['score_status']),
            affected_score_counts={key: len(scores) for key, scores in values['affected'].items()},
            affected_scores={key: sorted(scores) for key, scores in values['affected'].items()})
    source_root = Path(__file__).resolve().parents[2]
    files = [str(Path(__file__).resolve().relative_to(source_root)),
        'training/onnx/score_full_page_musicxml.py', 'training/omr_datasets/music_xml_parser.py',
        'training/omr_datasets/staff_merging.py', 'homr/music_xml_generator.py',
        'homr/transformer/vocabulary.py']
    write(args.out / 'report.json', dict(sources=dict(summary), threads=1,
        seconds=time.monotonic()-tick, input_sha256=sha(args.census / 'references.jsonl'),
        code_sha256={name: sha(source_root / name) for name in files},
        limitations='No PDMX, model inference or repairs. Current-source whole-score roundtrip. '
            'Failures retained; aggregate accuracy and onset counts conditional on successful scores. '
            'Fixed staff/physical-measure indexing; not voice-aware, no best alignment. '
            'Onset/order measures condition on preserved pitch/duration/grace inventory; '
            'other measures can also have onset errors. Matching note inventory does not establish '
            'engraving/voice correctness. Full score coverage exceeds frozen crop coverage. '
            'Does not change original model evaluation or published labels.'))
    print(json.dumps(dict(summary), indent=2, default=str), flush=True)


if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    main()
