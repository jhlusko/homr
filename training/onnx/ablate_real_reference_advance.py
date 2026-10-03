"""Paired ground-truth advance A/B on indexed OSSQ/Lieder references, never models.

Regenerate the disabled-advance control and require exact previous XML/metric
equality. Compare onsets on common inventory-preserved staff measures and retain
new inventory losses. Enabling a reference sidecar does not qualify predictions.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write
from training.onnx.roundtrip_real_timing_labels import measure_timing
from training.onnx.score_full_page_musicxml import compare, project, staff_events


def groups_by_measure(staves):
    groups = defaultdict(list)
    for staff, events in enumerate(staves, 1):
        for event in events:
            groups[(staff, event['measure'])].append(event)
    return groups


def moved_onsets(reference, predicted):
    a = Counter((event['onset'], key) for event, key in zip(reference, project(reference, 'event'), strict=True))
    b = Counter((event['onset'], key) for event, key in zip(predicted, project(predicted, 'event'), strict=True))
    return sum((a - b).values())


def paired_timing(reference, baseline, advanced):
    expected, before, after = map(groups_by_measure, (reference, baseline, advanced))
    counts = Counter()
    examples = defaultdict(list)
    for key in sorted(set(expected) | set(before) | set(after)):
        x, a, b = expected[key], before[key], after[key]
        inventory = Counter(project(x, 'event'))
        old_inventory, new_inventory = Counter(project(a, 'event')), Counter(project(b, 'event'))
        old_valid, new_valid = old_inventory == inventory, new_inventory == inventory
        if old_inventory != new_inventory:
            counts['advance_changed_inventory_staff_measures'] += 1
        if old_valid and not new_valid:
            counts['advance_lost_inventory_staff_measures'] += 1
            kind = 'new_inventory_loss'
        elif new_valid and not old_valid:
            counts['advance_recovered_inventory_staff_measures'] += 1
            kind = 'recovered_inventory'
        elif old_valid and new_valid:
            counts['common_inventory_preserved_staff_measures'] += 1
            counts['common_inventory_preserved_events'] += len(x)
            old_moved, new_moved = moved_onsets(x, a), moved_onsets(x, b)
            counts['baseline_moved_events_on_common_measures'] += old_moved
            counts['advance_moved_events_on_common_measures'] += new_moved
            counts['baseline_onset_changed_common_measures'] += bool(old_moved)
            counts['advance_onset_changed_common_measures'] += bool(new_moved)
            counts['baseline_order_changed_common_measures'] += project(x, 'event') != project(a, 'event')
            counts['advance_order_changed_common_measures'] += project(x, 'event') != project(b, 'event')
            if new_moved < old_moved:
                counts['onset_improved_common_measures'] += 1
                kind = 'onset_improved'
            elif new_moved > old_moved:
                counts['onset_regressed_common_measures'] += 1
                kind = 'onset_regressed'
            else:
                continue
        else:
            continue
        if len(examples[kind]) < 3:
            examples[kind].append(dict(staff=key[0], physical_measure_index=key[1],
                reference=x, baseline=a, advance=b))
    return dict(counts), dict(examples)


def export_with_counts(voices, enabled):
    from homr import music_xml_generator as generator
    from homr.transformer.structured_notation import AdvanceClass
    os.environ['HOMR_STRUCTURED_EXPORT'] = 'beam,stem,tie,slur' + (',advance' if enabled else '')
    original = generator.SymbolChord.get_render_duration
    seen = set()
    counts, classes = Counter(), Counter()
    def tracked(group):
        duration = original(group)
        if group not in seen:
            seen.add(group)
            if any(symbol.rhythm.startswith(('note', 'rest')) for symbol in group.symbols):
                counts['rhythmic_groups'] += 1
                advance = getattr(getattr(group.advance_symbol, 'notation', None),
                                  'advance', AdvanceClass.NOT_APPLICABLE)
                classes[str(advance)] += 1
                supported = advance in generator._RENDERABLE_ADVANCE_DURATIONS or (
                    advance == AdvanceClass.ZERO and any('G' in symbol.rhythm for symbol in group.symbols))
                counts['supported_class_groups' if supported else 'fallback_groups'] += 1
                if enabled and duration != group.get_duration():
                    counts['advance_changed_consumed_duration_groups'] += 1
        return duration
    try:
        generator.SymbolChord.get_render_duration = tracked
        root = generator.generate_xml(generator.XmlGeneratorArguments(), voices, '')
    finally:
        generator.SymbolChord.get_render_duration = original
    return root, dict(counts), dict(classes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'report.json').exists():
        raise ValueError('Refusing to overwrite completed A/B')
    from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
    baseline_report = json.loads((args.baseline / 'report.json').read_text())
    source_root = Path(__file__).resolve().parents[2]
    # The old driver is unchanged too; this A/B is implemented in a new file.
    for name, digest in baseline_report['code_sha256'].items():
        if sha(source_root / name) != digest:
            raise ValueError('Baseline implementation changed: ' + name)
    rows = [json.loads(line) for line in (args.baseline / 'scores.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['indexed_score']]
    aggregates = defaultdict(lambda: dict(status=Counter(), paired=Counter(),
        advanced_timing=Counter(), classes=Counter(), group_usage=Counter(),
        baseline_event_errors=0, advance_event_errors=0, event_support=0))
    tick = time.monotonic()
    with (args.out / 'scores.jsonl').open('w', buffering=1) as output:
        for number, row in enumerate(rows, 1):
            record = {key: row[key] for key in ('source', 'score', 'reference', 'reference_sha256')}
            stage = 'reference'
            try:
                if sha(row['reference']) != row['reference_sha256']:
                    raise ValueError('Reference changed')
                path = Path(row['reference'])
                root = read_mxl(path) if path.suffix == '.mxl' else ET.parse(path).getroot()
                expected = staff_events(root)
                stage = 'parser'
                parts = music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
                voices = [[symbol for measure in part for symbol in measure] for part in parts]
                stage = 'export'
                control, _, _ = export_with_counts(voices, False)
                baseline_hash = hashlib.sha256(ET.tostring(control)).hexdigest()
                control_events = staff_events(control)
                control_metrics = compare(expected, control_events)
                if (row['status'] != 'ok' or baseline_hash != row['xml_sha256']
                        or control_metrics != row['metrics'] or measure_timing(expected, control_events) != row['timing']):
                    raise RuntimeError('Control does not exactly reproduce completed baseline: ' + row['score'])
                advance, usage, classes = export_with_counts(voices, True)
                advanced_events = staff_events(advance)
                metrics = compare(expected, advanced_events)
                paired, examples = paired_timing(expected, control_events, advanced_events)
                record.update(status='ok', control_exact=True, baseline_metrics=control_metrics,
                    advance_metrics=metrics, paired_timing=paired,
                    advance_timing=measure_timing(expected, advanced_events),
                    group_usage=usage, advance_classes=classes,
                    control_xml_sha256=baseline_hash,
                    advance_xml_sha256=hashlib.sha256(ET.tostring(advance)).hexdigest(),
                    meter_changed=[ET.tostring(t) for t in control.findall('.//time')] !=
                                  [ET.tostring(t) for t in advance.findall('.//time')], examples=examples)
                # Keep exact reconstruction evidence for any new losses and the
                # first deterministic example per source, not just aggregate counts.
                if paired.get('advance_lost_inventory_staff_measures') or number == 1 or (
                        row['source'] == 'lieder' and not aggregates['lieder']['status']['ok']):
                    directory = args.out / 'examples' / row['score']
                    directory.mkdir(parents=True, exist_ok=True)
                    ET.ElementTree(control).write(directory / 'baseline.musicxml', encoding='unicode')
                    ET.ElementTree(advance).write(directory / 'advance.musicxml', encoding='unicode')
            except RuntimeError:
                raise
            except Exception as error:
                message = type(error).__name__ + ': ' + str(error)
                if row['status'] == 'failed' and (stage != row['failure_stage'] or message != row['error']):
                    raise RuntimeError('Previous failure does not replay identically') from error
                record.update(status='failed', failure_stage=stage, error=message,
                              same_failure_as_baseline=row['status'] == 'failed')
            output.write(json.dumps(record) + '\n')
            group = aggregates[row['source']]
            group['status'][record['status']] += 1
            if record['status'] == 'ok':
                group['paired'].update(record['paired_timing'])
                group['advanced_timing'].update(record['advance_timing'])
                group['classes'].update(record['advance_classes'])
                group['group_usage'].update(record['group_usage'])
                group['baseline_event_errors'] += record['baseline_metrics']['event']['errors']
                group['advance_event_errors'] += record['advance_metrics']['event']['errors']
                group['event_support'] += record['advance_metrics']['event']['support']
            if number % 10 == 0:
                print(f'{number}/{len(rows)} indexed references in {time.monotonic()-tick:.1f}s', flush=True)
    summary = {source: {key: dict(value) if isinstance(value, Counter) else value
                        for key, value in group.items()} for source, group in aggregates.items()}
    files = list(baseline_report['code_sha256']) + [str(Path(__file__).resolve().relative_to(source_root))]
    write(args.out / 'report.json', dict(sources=summary, threads=1, seconds=time.monotonic()-tick,
        policy='Ground-truth sidecar; current advance allowlist; beam/stem/tie/slur; no late repairs',
        baseline_report_sha256=sha(args.baseline / 'report.json'),
        baseline_records_sha256=sha(args.baseline / 'scores.jsonl'),
        code_sha256={name: sha(source_root / name) for name in files},
        limitations='Indexed OSSQ/Lieder only, no PDMX or model predictions. '
            'Controls exactly replay baseline XML, metrics and failures. '
            'Fixed staff and physical-measure indexing, not voice-aware equivalence. '
            'Common onset comparisons require reference inventory preserved in both arms. '
            'New inventory losses are counted and examples retained; other lossy measures '
            'may also have onset errors. Source sidecar is not a trained-head gate.'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    main()
