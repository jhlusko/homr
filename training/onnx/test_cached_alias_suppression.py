"""Predeclared confidence-grid diagnostic; reference categories only score choices."""
import argparse
import json
from collections import Counter
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def suppress(first, last, threshold):
    for payload, expected in ((first, 'start'), (last, 'stop')):
        if payload is None or len(payload['final_notation_slurs']) != 2:
            return False
        if any(event != 'none' for event, _ in payload['final_notation_slurs']):
            return False
        choices = {c['head']: c for c in payload['head_choices']}
        for name in ('slur.slot.1.event', 'slur.slot.2.event'):
            c = choices.get(name)
            if c is None or c['value'] != 'none' or c['probability'] < threshold:
                return False
        tie = choices.get('tie.state')
        if tie is None or tie['value'] not in (expected, 'start_and_stop'):
            return False
        if tie['probability'] < threshold:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aliases', 'endpoints', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output must be new')
    aliases = json.loads(args.aliases.read_text())['results']
    endpoints = json.loads(args.endpoints.read_text())['results']
    if not all(r['exact_xml'] for r in endpoints):
        raise ValueError('Endpoint replay must be exact')
    results = []
    for record in aliases:
        saved = next(r for r in endpoints
                     if (r['score'], r['arm']) == (record['score'], record['arm']))
        lookup = {tuple(r['anchor']): r['provenance'] for r in saved['all_endpoints'] if r['unique']}
        for threshold in (.8, .9, .95, .98):
            counts, cases = Counter(), []
            for case in record['cases']:
                first, last = [lookup.get(tuple(k)) for k in case['endpoints']]
                decision = suppress(first, last, threshold)
                counts[('remove:' if decision else 'retain:') + case['category']] += case['multiplicity']
                cases.append(dict(case, suppress=decision, endpoint_provenance=[first, last]))
            results.append({'score': record['score'], 'source': record['source'], 'arm': record['arm'],
                            'threshold': threshold, 'counts': dict(counts), 'cases': cases})
    args.out.mkdir(parents=True)
    write(args.out / 'report.json', {'results': results, 'source_sha256': sha(Path(__file__)),
          'aliases_sha256': sha(args.aliases), 'endpoints_sha256': sha(args.endpoints),
          'scope': 'Exploratory cached complete-span deletion diagnostic on conditional duplicate '
          'support only. Decision uses endpoint predictions/confidence, never reference categories. '
          'No XML mutation, runtime rule, calibration/promotion or new recognition.'})
    print(json.dumps([{k: v for k, v in r.items() if k != 'cases'} for r in results], indent=2))


if __name__ == '__main__':
    main()
