"""Train-only timing target inventory with separate OSSQ/Lieder held-out coverage."""
import argparse
from collections import Counter, defaultdict
import json
from fractions import Fraction
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from training.architecture.transformer.exact_timing_targets import HEADS, timing_values
from training.omr_datasets.exact_timing_labels import extract_exact_timing
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write


def scan(root, duration_policy):
    counts, heads, reasons = Counter(), {h: Counter() for h in HEADS}, Counter()
    for part in root.findall('part'):
        divisions = 1
        for measure in part.findall('measure'):
            for child in measure:
                if child.tag == 'attributes' and child.find('divisions') is not None:
                    divisions = child.findtext('divisions')
                if child.tag != 'note':
                    continue
                if child.get('print-object') == 'no' or child.findtext('notehead') == 'none':
                    counts['excluded_invisible'] += 1
                    continue
                if child.find('grace') is not None:
                    counts['excluded_grace'] += 1
                    continue
                is_rest = child.find('rest') is not None
                if not is_rest and child.find('pitch') is None:
                    counts['excluded_unpitched'] += 1
                    continue
                try:
                    payload = extract_exact_timing(child, divisions)
                    tick = Fraction(1) / Fraction(divisions) if duration_policy == 'tick-quantized' else None
                    values, flags = timing_values(payload, is_rest, tick_quarters=tick)
                except (ValueError, TypeError, ZeroDivisionError) as error:
                    reasons[type(error).__name__ + ': ' + str(error)] += 1
                    counts['invalid_timing_events'] += 1
                    continue
                counts['timed_events'] += 1
                counts['rests' if is_rest else 'pitched_notes'] += 1
                reasons.update(flags)
                for head, value in values.items():
                    heads[head][value] += 1
    return dict(counts=counts, heads=heads, masked_reasons=reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--split-labels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--duration-policy', choices=('strict', 'tick-quantized'), default='strict')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'report.json').exists() or (args.out/'scores.jsonl').exists():
        raise ValueError('Refusing to overwrite target inventory')
    splits = defaultdict(set)
    for line in args.split_labels.read_text().splitlines():
        row = json.loads(line)
        splits[row['source'], row['score']].update(row['splits'])
    if any(v not in ({'train'}, {'valid'}) for v in splits.values()):
        raise ValueError('Expected score-disjoint train/valid memberships')
    rows = [json.loads(x) for x in args.references.read_text().splitlines()]
    totals = defaultdict(lambda: dict(counts=Counter(), heads={h: Counter() for h in HEADS}, masked_reasons=Counter()))
    tick = time.monotonic()
    with (args.out/'scores.jsonl').open('w', buffering=1) as output:
        for index, row in enumerate(rows, 1):
            key = row['source'], row['score']
            split = next(iter(splits[key]))
            path = Path(row['reference'])
            if sha(path) != row['reference_sha256']:
                raise ValueError('Reference hash mismatch')
            root = read_mxl(path) if path.suffix == '.mxl' else ET.parse(path).getroot()
            stats = scan(root, args.duration_policy)
            group = totals[row['source'] + '/' + split]
            group['counts'].update(stats['counts'])
            group['counts']['scores'] += 1
            group['masked_reasons'].update(stats['masked_reasons'])
            for h in HEADS:
                group['heads'][h].update(stats['heads'][h])
            output.write(json.dumps(dict(source=row['source'], score=row['score'], split=split,
                reference=row['reference'], reference_sha256=row['reference_sha256'], **stats))+'\n')
            if index % 10 == 0:
                print(f'{index}/{len(rows)} in {time.monotonic()-tick:.1f}s', flush=True)
    classes = {h: sorted({v for group, stats in totals.items() if group.endswith('/train')
                         for v in stats['heads'][h]}) for h in HEADS}
    # Binary flag's semantics are predefined, not inferred from validation support.
    classes['timing.full_rest'] = ['no', 'yes']
    manifest = dict(schemaVersion='homr.exact-timing-targets.v1', classes=classes,
        durationPolicy=args.duration_policy,
        classSource='canonical events in train-only scores from audited crop indexes',
        maskPolicy='no metadata/grace/control/unseen values; full_rest only on rests; '
            'ratio only with source-duration consistency or explicit floor/ceil tick quantization; '
            'tick-quantized policy masks unresolved elapsed-duration conflicts',
        referenceInputSha256=sha(args.references), splitInputSha256=sha(args.split_labels))
    write(args.out/'target-manifest.json', manifest)
    unseen = {}
    for group, stats in totals.items():
        unseen[group] = {h: {v: n for v, n in stats['heads'][h].items() if v not in classes[h]} for h in HEADS}
    code_root = Path(__file__).resolve().parents[2]
    files = [str(Path(__file__).resolve().relative_to(code_root)),
        'training/architecture/transformer/exact_timing_targets.py',
        'training/omr_datasets/exact_timing_labels.py']
    write(args.out/'report.json', dict(groups=totals, unseen=unseen, seconds=time.monotonic()-tick,
        records=len(rows), threads=1, target_manifest_sha256=sha(args.out/'target-manifest.json'),
        code_sha256={f: sha(code_root/f) for f in files},
        limitation='Canonical train-score inventory, not actual training crop frequency/support. '
            'No model fitting or inference; class discovery excludes validation scores. '
            'Score membership comes from recovered public-core indexes, not a new split decision. '
            'Invalid timing and inconsistent ratio supervision are counted explicitly.'))


if __name__ == '__main__':
    main()
