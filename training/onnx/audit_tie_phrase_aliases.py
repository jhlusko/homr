"""Identify exact tie/phrase duplicate spans, keeping simultaneous source phrases."""
import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from training.onnx.audit_notation_spans import extract
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_paired_page_diagnostic import read_reference


def classify_pairs(reference, predicted, common):
    phrases = Counter({k: v for k, v in predicted['phrase'].items()
                       if all(a in common for a in k)})
    duplicates = phrases & predicted['sounding_tie'] & predicted['engraved_tie']
    cases = []
    for pair, count in duplicates.items():
        truth_phrase = reference['phrase'][pair]
        truth_tie = reference['sounding_tie'][pair]
        category = ('source_phrase_and_tie' if truth_phrase and truth_tie else
                    'source_phrase_only' if truth_phrase else
                    'source_tie_only' if truth_tie else 'neither_source_span')
        cases.append({'endpoints': pair, 'multiplicity': count, 'category': category})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output must be new')
    results = []
    for directory in sorted(args.run.iterdir()):
        path = directory / 'comparison.json'
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        page = manifest['page']
        if sha(page['reference']) != page['reference_sha256']:
            raise ValueError('Reference changed')
        rows, truth, issues = extract(read_reference(page['reference']),
                                      page['start_measure'], page['end_measure'])
        common = {k for k, v in rows.items() if len(v) == 1}
        predictions = {}
        for arm in ('accepted', 'candidate'):
            xml = directory / arm / 'page.musicxml'
            if sha(xml) != manifest['arms'][arm]['xml_sha256']:
                raise ValueError('Prediction changed')
            rows, spans, _ = extract(ET.parse(xml).getroot())
            common &= {k for k, v in rows.items() if len(v) == 1}
            predictions[arm] = spans
        for arm, spans in predictions.items():
            cases = classify_pairs(truth, spans, common)
            counts = Counter()
            for case in cases:
                counts[case['category']] += case['multiplicity']
                case['physical_measures'] = [a[1] + page['start_measure'] + 1
                                             for a in case['endpoints']]
            results.append({'score': page['score'], 'source': page['source'], 'arm': arm,
                            'counts': dict(counts), 'cases': cases,
                            'reference_pairing_issues': issues})
    args.out.mkdir(parents=True)
    write(args.out / 'report.json', {'results': results, 'source_sha256': sha(Path(__file__)),
          'scope': 'Source-assisted diagnostic only; duplicate complete spans on unique common '
          'anchors. No oracle-based runtime filtering. Simultaneous phrases protected explicitly.'})
    print(json.dumps([{k: v for k, v in r.items() if k not in ('cases', 'reference_pairing_issues')}
                      for r in results], indent=2))


if __name__ == '__main__':
    main()
