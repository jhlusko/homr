"""Class support on actual replayed crops, separated by split and token parity."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path

from training.architecture.transformer.exact_timing_targets import HEADS, timing_values
from training.onnx.qualify_joint_export import sha, write


def inventory(report, manifest):
    groups = defaultdict(lambda: dict(crops=0, events=0, heads={h: Counter() for h in HEADS},
        unseen={h: Counter() for h in HEADS}, reasons=Counter()))
    for score in report['scores']:
        for crop in score['crops']:
            if len(crop['splits']) != 1 or crop['splits'][0] not in ('train', 'valid'):
                raise ValueError('Expected a single frozen split')
            key = crop['splits'][0] + ('/identical' if crop['exact_legacy_tokens'] else '/changed')
            group = groups[key];group['crops'] += 1
            path = Path(crop['replayed_label'])
            payload = json.loads(Path(str(path)+'.timing.json').read_text())
            if sha(path) != payload['tokenSha256']:
                raise ValueError('Replayed tokens changed')
            for record in payload['symbols']:
                timing = record['timing']
                if timing is None:
                    continue
                group['events'] += 1
                values, reasons = timing_values(timing, record['identity'][0].startswith('rest'),
                    tick_quarters=Fraction(1)/Fraction(timing['source_divisions']))
                group['reasons'].update(reasons)
                for head, value in values.items():
                    group['heads'][head][value] += 1
                    if value not in manifest['classes'][head]:
                        group['unseen'][head][value] += 1
    eligible_train = groups.get('train/identical', dict(heads={h:{} for h in HEADS}))
    unsupported = {h:[v for v in manifest['classes'][h] if not eligible_train['heads'][h].get(v)] for h in HEADS}
    return dict(groups=groups, classes_without_identical_lieder_train_crop_support=unsupported)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-report', type=Path, required=True)
    parser.add_argument('--target-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite crop inventory')
    report=json.loads(args.replay_report.read_text());manifest=json.loads(args.target_manifest.read_text())
    if manifest.get('durationPolicy') != 'tick-quantized':
        raise ValueError('Requires frozen normalized duration policy')
    result=inventory(report,manifest)
    result.update(replay_report_sha256=sha(args.replay_report), target_manifest_sha256=sha(args.target_manifest),
        code_sha256=sha(Path(__file__)), limitation='Actual indexed Lieder crop occurrences, not unique canonical events. '
        'Changed tokens kept separate from crops eligible for metadata-only migration. '
        'Frozen mixed-source training classes are not expanded from validation. '
        'OSSQ crop support is not measured by this report.')
    write(args.out,result)
    print(json.dumps({k:{'crops':v['crops'],'events':v['events'],'reasons':v['reasons']} for k,v in result['groups'].items()}))


if __name__ == '__main__':
    main()
