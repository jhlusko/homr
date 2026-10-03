"""Census timing representation risks in OSSQ/Lieder references and frozen labels.

No PDMX, inference, rendering, corpus mutation or training. Explicit-ratio census
is distinct from a full parser roundtrip: bracket visibility/state may suppress a
ratio, and polyphonic alignment/merging is not measured here.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write


def inspect_reference(root):
    from training.omr_datasets.music_xml_parser import DURATION_NUMBER, _measure_rest_rhythm, _rhythm_token
    from homr.transformer.vocabulary import kern_to_symbol_duration
    counts = Counter()
    issues = []
    for part_no, part in enumerate(root.findall('part'), 1):
        divisions = Fraction(1)
        for measure_no, measure in enumerate(part.findall('measure'), 1):
            note_no = 0
            for child in measure:
                if child.tag == 'attributes' and child.find('divisions') is not None:
                    divisions = Fraction(child.findtext('divisions'))
                if child.tag != 'note':
                    continue
                note_no += 1
                if child.get('print-object') == 'no' or child.findtext('notehead') == 'none':
                    counts['invisible_notes_excluded'] += 1
                    continue
                if child.find('grace') is not None:
                    counts['grace_notes_excluded'] += 1
                    continue
                counts['printed_timed_events'] += 1
                rest = child.find('rest')
                counts['rests' if rest is not None else 'pitched_or_unpitched_notes'] += 1
                location = dict(part=part_no, physical_measure=measure_no,
                    printed_measure=measure.get('number'), note_index=note_no,
                    staff=child.findtext('staff', '1'), voice=child.findtext('voice', '1'))
                if divisions <= 0 or child.find('duration') is None:
                    counts['invalid_or_missing_duration'] += 1
                    continue
                expected = Fraction(child.findtext('duration')) / divisions
                if rest is not None and rest.get('measure') == 'yes':
                    counts['full_measure_rests'] += 1
                    if expected == 6:
                        counts['six_quarter_full_measure_rests'] += 1
                    token = _measure_rest_rhythm(int(child.findtext('duration')), int(divisions))
                    rendered = 4 * kern_to_symbol_duration(token.split('_', 1)[1]).fraction
                    if rendered != expected:
                        counts['full_measure_rest_duration_loss'] += 1
                        issues.append(dict(location=location, kind='full_measure_rest_duration_loss',
                            expected_quarters=str(expected), token=token, emitted_quarters=str(rendered)))
                    continue
                modification = child.find('time-modification')
                if modification is None:
                    continue
                counts['explicit_time_modification_events'] += 1
                try:
                    actual = int(modification.findtext('actual-notes'))
                    normal = int(modification.findtext('normal-notes'))
                    if min(actual, normal) <= 0:
                        raise ValueError('Nonpositive ratio')
                    base = DURATION_NUMBER[child.findtext('type', 'eighth')]
                    if base <= 0:
                        counts['unsupported_tuplet_base'] += 1
                        continue
                except (TypeError, ValueError, KeyError):
                    counts['invalid_tuplet_metadata'] += 1
                    continue
                dots = len(child.findall('dot'))
                dot_factor = sum((Fraction(1, 2**i) for i in range(dots+1)), Fraction(0))
                ratio_duration = Fraction(4, base) * Fraction(normal, actual) * dot_factor
                reciprocal = Fraction(base * actual, normal)
                non_integral = reciprocal.denominator != 1
                if non_integral:
                    counts['non_integral_reciprocal_events'] += 1
                # Match the current parser's explicit-ratio arithmetic, including
                # Python rounding. This does not claim its bracket state applies it.
                token = _rhythm_token('rest' if rest is not None else 'note',
                                      round(base * (actual / normal)), dots, False)
                rendered = 4 * kern_to_symbol_duration(token.split('_', 1)[1]).fraction
                consistent = ratio_duration == expected
                if not consistent:
                    counts['explicit_ratio_source_duration_disagreement'] += 1
                if non_integral and consistent and rendered != expected:
                    counts['confirmed_reciprocal_rounding_duration_loss'] += 1
                    issues.append(dict(location=location, kind='confirmed_reciprocal_rounding_duration_loss',
                        ratio=f'{actual}:{normal}', base_type=child.findtext('type', 'eighth'),
                        dots=dots, exact_reciprocal=str(reciprocal), expected_quarters=str(expected),
                        token=token, emitted_quarters=str(rendered),
                        bracket_markers=[dict(t.attrib) for t in child.findall('./notations/tuplet')]))
    return dict(counts), issues


def inspect_label_text(text):
    counts = Counter()
    numerator = denominator = None
    current = []
    def close():
        if len(current) == 1 and current[0] == 'rest_1' and (numerator, denominator) == (12, 8):
            counts['sole_rest_1_in_explicit_12_8_bars'] += 1
        current.clear()
    for line in text.splitlines():
        for piece in line.split('&'):
            fields = piece.split()
            if not fields:
                continue
            rhythm = fields[0]
            if rhythm.startswith('timeSignatureBeats_'):
                numerator = int(rhythm.split('_')[1])
            elif rhythm.startswith('timeSignature/'):
                denominator = int(rhythm.split('/')[1])
            elif rhythm in ('barline', 'doublebarline', 'bolddoublebarline', 'repeatStart', 'repeatEnd', 'repeatBoth'):
                close()
            elif rhythm.startswith(('note_', 'rest_')):
                counts['note_rest_tokens'] += 1
                counts['rhythm:' + rhythm] += 1
                current.append(rhythm)
    close()
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--reference-audit', type=Path, required=True)
    parser.add_argument('--indexes', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'report.json').exists():
        raise ValueError('Refusing to overwrite completed census')
    records = defaultdict(dict)
    for line in (args.reference_audit / 'lieder-page-audit.jsonl').read_text().splitlines():
        row = json.loads(line)
        records['lieder'][row['score']] = dict(score=row['score'], reference=row.get('reference'))
    for row in json.loads((args.reference_audit / 'ossq-score-audit.json').read_text()):
        records['ossq'][row['score']] = dict(score=row['score'], reference=row.get('reference'))
    all_results = {}
    with (args.out / 'reference-issues.jsonl').open('w') as issue_file, \
            (args.out / 'frozen-labels.jsonl').open('w') as label_file, \
            (args.out / 'references.jsonl').open('w') as reference_file:
        for source in ('ossq', 'lieder'):
            index_directory = args.indexes / (source + '-omr-data')
            indexed_scores = set()
            labels = {}
            input_indexes = []
            for split in ('train', 'valid'):
                index = index_directory / f'index_{split}.published.txt'
                input_indexes.append(dict(path=str(index), sha256=sha(index)))
                for line in index.read_text().splitlines():
                    if not line.strip():
                        continue
                    relative = line.split(',')[-1]
                    score = re.search(r'(?:sq|IMSLP)\d+', relative)[0]
                    indexed_scores.add(score)
                    labels.setdefault(relative, dict(score=score, splits=[]))['splits'].append(split)
            label_totals = Counter()
            label_scores = defaultdict(set)
            missing = []
            for relative, row in sorted(labels.items()):
                root = (args.workspace / 'homr-artifacts/lieder-v8' if source == 'lieder' else
                    args.workspace / 'homr-artifacts/gpu-roadmap-20260919/ossq-scanned-v1/dataset')
                path = root / relative
                if not path.exists():
                    missing.append(str(path))
                    continue
                try:
                    counts = inspect_label_text(path.read_text())
                    label_totals.update(counts)
                    for kind, count in counts.items():
                        if count:
                            label_scores[kind].add(row['score'])
                    label_file.write(json.dumps(dict(source=source, path=str(path), relative=relative,
                        **row, sha256=sha(path), counts=counts)) + '\n')
                except Exception as error:
                    missing.append(dict(path=str(path), error=str(error)))
            totals = {'all_available_references': Counter(), 'indexed_score_references': Counter()}
            affected = {scope: defaultdict(set) for scope in totals}
            failures = []
            successful = set()
            for position, (score, row) in enumerate(sorted(records[source].items()), 1):
                reference = row['reference']
                try:
                    if not reference:
                        raise ValueError('Missing independent reference')
                    path = Path(reference)
                    root = read_mxl(path) if path.suffix == '.mxl' else ET.parse(path).getroot()
                    if root.tag != 'score-partwise':
                        raise ValueError('Unsupported reference structure')
                    counts, issues = inspect_reference(root)
                    successful.add(score)
                    for scope in totals:
                        if scope == 'indexed_score_references' and score not in indexed_scores:
                            continue
                        totals[scope].update(counts)
                        for issue in issues:
                            affected[scope][issue['kind']].add(score)
                    for issue in issues:
                        issue_file.write(json.dumps(dict(source=source, score=score, reference=reference,
                            indexed_score=score in indexed_scores, **issue)) + '\n')
                    reference_file.write(json.dumps(dict(source=source, score=score, reference=reference,
                        sha256=sha(path), indexed_score=score in indexed_scores, counts=counts)) + '\n')
                except Exception as error:
                    failures.append(dict(score=score, reference=reference, error=str(error)))
                if position % 25 == 0:
                    print(f'{source}: {position}/{len(records[source])} references; '
                          f'{len(labels)-len(missing)}/{len(labels)} frozen files', flush=True)
            all_results[source] = dict(reference_scores_requested=len(records[source]),
                reference_scores_read=len(successful), indexed_scores=len(indexed_scores),
                indexed_reference_scores_read=len(successful & indexed_scores),
                indexed_scores_without_read_reference=sorted(indexed_scores-successful),
                references={scope: dict(counts=dict(counter),
                    affected_score_counts={kind: len(scores) for kind, scores in affected[scope].items()},
                    affected_scores={kind: sorted(scores) for kind, scores in affected[scope].items()})
                    for scope, counter in totals.items()}, reference_failures=failures,
                frozen_labels=dict(requested_files=len(labels), read_files=len(labels)-len(missing),
                    missing_or_failed=missing, counts=dict(label_totals),
                    score_counts={kind: len(scores) for kind, scores in label_scores.items()},
                    indexes=input_indexes))
    source_root = Path(__file__).resolve().parents[2]
    source_files = [str(Path(__file__).resolve().relative_to(source_root)),
        'training/omr_datasets/music_xml_parser.py', 'homr/transformer/vocabulary.py',
        'training/onnx/audit_full_page_references.py']
    report = dict(sources=all_results, threads=1,
        code_sha256={name: sha(source_root / name) for name in source_files},
        input_sha256={str(p): sha(p) for p in [
            args.reference_audit / 'lieder-page-audit.jsonl', args.reference_audit / 'ossq-score-audit.json',
            args.workspace / 'ossq-omr-data/sources.lock.json', args.workspace / 'lieder-omr-data/sources.lock.json']},
        limitations='No PDMX. Canonical events and crop-window tokens have different denominators. '
            'Overlapping crop windows repeat events; token spellings alone do not identify erroneous tuplets. '
            'Explicit-ratio arithmetic is not a full parser/alignment roundtrip; hidden/state-suppressed '
            'ratios may be ignored. Confirmed rounding losses require source duration consistency. '
            'Frozen labels use recovered public-core indexes: local Lieder-v8 files and OSSQ-v1 token files '
            '(OSSQ-v2 build receipt reports zero token-file changes). Local file hashes are recorded; '
            'this does not newly verify every file against published archive manifests. '
            'Current-parser findings do not establish every historical builder version. '
            'No onset/voice-order census or model error measurement.')
    write(args.out / 'report.json', report)
    lines = ['# OSSQ / Lieder timing representation census', '', report['limitations'], '',
        '| Source / canonical scope | Scores read | Timed events | Full-rest losses | Rounding losses |',
        '|---|---:|---:|---:|---:|']
    for source, result in all_results.items():
        for scope, data in result['references'].items():
            counts = data['counts']
            score_count = result['reference_scores_read' if scope == 'all_available_references' else 'indexed_reference_scores_read']
            lines.append(f"| {source} / {scope} | {score_count} | {counts.get('printed_timed_events', 0)} | "
                f"{counts.get('full_measure_rest_duration_loss', 0)} | "
                f"{counts.get('confirmed_reciprocal_rounding_duration_loss', 0)} |")
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    # The scanner is pure Python/XML; cap any libraries loaded by helpers as well.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    main()
