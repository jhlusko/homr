"""Qualify additive dataset timing metadata against exact real-source controls."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from homr.transformer.vocabulary import serialized_chords
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.exact_timing_targets import HEADS, build_timing_targets
from training.architecture.transformer.structured_targets import align_to_decoder_output
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.notation_sidecar import write_sidecar, attach_sidecar
from training.transformer.training_vocabulary import read_tokens, token_lines_to_str
from training.onnx.ablate_real_reference_advance import export_with_counts
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write
from training.onnx.roundtrip_real_timing_labels import measure_timing
from training.onnx.score_full_page_musicxml import compare, staff_events


def check_crops(out, parts, manifest=None):
    counts, failures = Counter(), []
    for part_no, measures in enumerate(parts, 1):
        starts = {0, max(0, len(measures)-3)}
        for kind in ('full_measure_rest', 'time_modification_xml'):
            index = next((i for i, measure in enumerate(measures)
                if any(s.exact_timing and s.exact_timing[kind] for s in measure)), None)
            if index is not None:
                starts.add(index)
        for start in sorted(starts):
            symbols = [s for measure in measures[start:start+3] for s in measure]
            path = out / f'part-{part_no}-measure-index-{start}.tokens'
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                text = token_lines_to_str(symbols)
            except KeyError as error:
                # Full canonical scores can contain legacy tokens excluded by
                # actual corpus crop filters. Keep this gap visible.
                failures.append(dict(part=part_no, measure_index=start,
                                     unsupported_legacy_token=str(error)))
                continue
            path.write_text(text)
            write_sidecar(path, symbols)
            restored = read_tokens(str(path))
            attach_sidecar(path, restored)
            expected = [s.exact_timing for chord in serialized_chords(symbols) for s in chord
                        if s.rhythm.startswith(('note', 'rest'))]
            actual = [s.exact_timing for s in restored if s.rhythm.startswith(('note', 'rest'))]
            if expected != actual or token_lines_to_str(restored) != text:
                raise RuntimeError('Crop timing/text roundtrip mismatch: ' + str(path))
            counts['verified_crops'] += 1
            counts['timed_symbols'] += sum(x is not None for x in actual)
            if manifest is not None:
                sequence = [EncodedSymbol('BOS'), *restored, EncodedSymbol('EOS')]
                targets, diagnostics = build_timing_targets([sequence], manifest['classes'],
                    sequence_length=len(sequence)+1, duration_policy='tick-quantized')
                shifted = align_to_decoder_output(targets)
                for head in HEADS:
                    if (targets[head][0, 0].item() != -100 or
                            not (targets[head][0, -2:] == -100).all()):
                        raise RuntimeError('Control/padding timing target was supervised')
                    if not (shifted[head] == targets[head][:, 1:]).all():
                        raise RuntimeError('Decoder target shift mismatch')
                    counts[head + '.supervised'] += (shifted[head] != -100).sum().item()
                    counts[head + '.unseen'] += diagnostics['unseen'][head]
                for reason, count in diagnostics['masked_reasons'].items():
                    counts['target_reason.' + reason] += count
                counts['verified_target_windows'] += 1
    return dict(counts), failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scores', nargs='*')
    parser.add_argument('--target-manifest', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.target_manifest.read_text()) if args.target_manifest else None
    if manifest is not None and manifest.get('durationPolicy') != 'tick-quantized':
        raise ValueError('Requires the frozen tick-quantized target manifest')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'scores.jsonl').exists():
        raise ValueError('Refusing to overwrite qualification')
    root_path = Path(__file__).resolve().parents[2]
    old = json.loads((args.baseline / 'report.json').read_text())
    for file, expected_hash in old['code_sha256'].items():
        if file != 'training/omr_datasets/music_xml_parser.py' and sha(root_path / file) != expected_hash:
            raise ValueError('Uncontrolled source change: ' + file)
    rows = [json.loads(x) for x in (args.baseline / 'scores.jsonl').read_text().splitlines()]
    if args.scores:
        rows = [r for r in rows if r['score'] in args.scores]
        if {r['score'] for r in rows} != set(args.scores):
            raise ValueError('Missing requested score')
    tick = time.monotonic()
    totals = defaultdict(Counter)
    with (args.out / 'scores.jsonl').open('w', buffering=1) as output:
        for index, row in enumerate(rows, 1):
            if sha(row['reference']) != row['reference_sha256']:
                raise ValueError('Reference changed')
            path = Path(row['reference'])
            root = read_mxl(path) if path.suffix == '.mxl' else ET.parse(path).getroot()
            expected = staff_events(root)
            record = {k: row[k] for k in ('source', 'score', 'reference', 'reference_sha256')}
            try:
                parts = music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
            except Exception as error:
                message = type(error).__name__ + ': ' + str(error)
                previous = row['arms']['new_control']
                if previous['status'] != 'failed' or previous['stage'] != 'parser' or previous['error'] != message:
                    raise RuntimeError('Parser failure changed') from error
                record.update(status='failed', stage='parser', error=message)
                totals[row['source']]['unchanged_parser_failures'] += 1
            else:
                voices = [[s for measure in part for s in measure] for part in parts]
                record['arms'] = {}
                for enabled in (False, True):
                    arm = 'new_advance' if enabled else 'new_control'
                    previous = row['arms'][arm]
                    try:
                        xml, _, _ = export_with_counts(copy.deepcopy(voices), enabled)
                    except Exception as error:
                        message = type(error).__name__ + ': ' + str(error)
                        if (previous['status'] != 'failed' or previous['stage'] != 'export'
                                or previous['error'] != message):
                            raise RuntimeError('Export failure changed') from error
                        record['arms'][arm] = dict(status='failed', stage='export', error=message)
                        continue
                    metrics = compare(expected, staff_events(xml))
                    timing = measure_timing(expected, staff_events(xml))
                    xml_sha = hashlib.sha256(ET.tostring(xml)).hexdigest()
                    if (previous['status'] != 'ok' or xml_sha != previous['xml_sha256']
                            or metrics != previous['metrics'] or timing != previous['timing']):
                        raise RuntimeError('Legacy control mismatch: ' + row['score'] + ' ' + arm)
                    record['arms'][arm] = dict(status='ok', xml_sha256=xml_sha, metrics=metrics, timing=timing)
                crops, gaps = check_crops(args.out / 'crops' / row['score'], parts, manifest)
                unchanged_failure = any(a['status'] == 'failed' for a in record['arms'].values())
                record.update(status='failed' if unchanged_failure else 'ok', crops=crops, crop_gaps=gaps)
                totals[row['source']].update(crops)
                totals[row['source']]['unsupported_legacy_crop_gaps'] += len(gaps)
                totals[row['source']]['unchanged_export_failures' if unchanged_failure
                                      else 'exact_legacy_controls'] += 1
            output.write(json.dumps(record) + '\n')
            print(f'{index}/{len(rows)} {row["source"]} {row["score"]} {record["status"]} '
                  f'in {time.monotonic()-tick:.1f}s', flush=True)
    files = list(old['code_sha256']) + ['homr/transformer/vocabulary.py',
        'training/omr_datasets/exact_timing_labels.py', 'training/omr_datasets/exact_timing_sidecar.py',
        'training/omr_datasets/notation_sidecar.py', 'training/transformer/training_vocabulary.py',
        'training/architecture/transformer/exact_timing_targets.py',
        'training/architecture/transformer/structured_targets.py',
        str(Path(__file__).resolve().relative_to(root_path))]
    write(args.out / 'report.json', dict(sources=totals, seconds=time.monotonic()-tick,
        records=len(rows), threads=1, bounded=bool(args.scores),
        code_sha256={f: sha(root_path/f) for f in files},
        baseline_report_sha256=sha(args.baseline/'report.json'),
        baseline_records_sha256=sha(args.baseline/'scores.jsonl'),
        target_manifest=str(args.target_manifest) if args.target_manifest else None,
        target_manifest_sha256=sha(args.target_manifest) if args.target_manifest else None,
        limitation='Additive metadata; exact old XML/metrics/timing, selected real measure windows '
            'through shared label writer/reader, no image rendering or full crop rebuild. '
            'Unsupported legacy crop tokens reported as gaps. No model prediction/loss changes.'))


if __name__ == '__main__':
    main()
