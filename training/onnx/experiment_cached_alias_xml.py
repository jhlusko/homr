"""Isolated source-free XML deletion experiment; never edits saved/runtime outputs."""
import argparse
import copy
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from training.onnx.audit_page_notation_policy import anchors
from training.onnx.audit_notation_spans import extract, compare
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_paired_page_diagnostic import read_reference
from training.onnx.test_cached_alias_suppression import suppress


def phrase_elements(root):
    provenance = {id(n): {'id': id(n)} for n in root.iter('note')}
    rows = anchors(root, provenance=provenance)
    inverse = {r['provenance']['id']: k for k, values in rows.items()
               if len(values) == 1 for r in values}
    pairs = defaultdict(list)
    for part in root.findall('part'):
        active = {}
        for note in part.iter('note'):
            parent = note.find('notations')
            if parent is None:
                continue
            for tag in sorted(parent.findall('slur'), key=lambda t: t.get('type') != 'stop'):
                number, kind = tag.get('number', '1'), tag.get('type')
                key = inverse.get(id(note))
                if kind == 'start':
                    active[number] = None if number in active else (key, parent, tag)
                elif kind == 'stop':
                    first = active.pop(number, None)
                    if first is not None and first[0] is not None and key is not None:
                        pairs[(first[0], key)].append((first[1], first[2], parent, tag))
                elif number in active:
                    active[number] = None
    return pairs


def without_slurs(root):
    clone = copy.deepcopy(root)
    for parent in clone.iter('notations'):
        for tag in parent.findall('slur'):
            parent.remove(tag)
    return ET.tostring(clone)


def apply(root, endpoint_lookup, threshold=.9):
    rows, spans, issues = extract(root)
    candidates = spans['phrase'] & spans['sounding_tie'] & spans['engraved_tie']
    locations = phrase_elements(root)
    invariant = without_slurs(root)
    original_slurs = [ET.tostring(s) for s in root.iter('slur')]
    removed = Counter()
    removed_tags = []
    for pair in candidates:
        if spans['phrase'][pair] != 1 or len(locations.get(pair, [])) != 1:
            continue
        if not suppress(endpoint_lookup.get(pair[0]), endpoint_lookup.get(pair[1]), threshold):
            continue
        first_parent, first, last_parent, last = locations[pair][0]
        removed_tags += [ET.tostring(first), ET.tostring(last)]
        first_parent.remove(first)
        last_parent.remove(last)
        removed[pair] += 1
    _, after, after_issues = extract(root)
    if without_slurs(root) != invariant:
        raise ValueError('Non-slur XML changed')
    if after['sounding_tie'] != spans['sounding_tie'] or after['engraved_tie'] != spans['engraved_tie']:
        raise ValueError('Ties changed')
    if after['phrase'] != spans['phrase'] - removed or after_issues != issues:
        raise ValueError('Unowned phrase or boundary changed')
    if Counter(ET.tostring(s) for s in root.iter('slur')) != Counter(original_slurs) - Counter(removed_tags):
        raise ValueError('Unrelated slur elements changed')
    return rows, spans, after, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'endpoints', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output must be new')
    records = json.loads(args.endpoints.read_text())['results']
    args.out.mkdir(parents=True)
    results = []
    for folder in sorted(args.run.iterdir()):
        manifest = folder / 'comparison.json'
        if not manifest.exists():
            continue
        saved = json.loads(manifest.read_text())
        page = saved['page']
        variants = {}
        for arm in ('accepted', 'candidate'):
            source = folder / arm / 'page.musicxml'
            record = next(r for r in records if (r['score'], r['arm']) == (page['score'], arm))
            if not record['exact_xml'] or sha(source) != record['xml_sha256'] or sha(source) != saved['arms'][arm]['xml_sha256']:
                raise ValueError('Endpoint/XML identity changed')
            lookup = {tuple(r['anchor']): r['provenance'] for r in record['all_endpoints'] if r['unique']}
            root = ET.parse(source).getroot()
            rows, before, after, removed = apply(root, lookup)
            target = args.out / (page['score'] + '-' + arm + '.musicxml')
            ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
            variants[arm] = (rows, before, after, removed, target)
        # References enter only after both source-free XML decisions and invariants pass.
        if sha(page['reference']) != page['reference_sha256']:
            raise ValueError('Reference changed')
        truth_rows, truth, _ = extract(read_reference(page['reference']), page['start_measure'], page['end_measure'])
        common = {k for k, v in truth_rows.items() if len(v) == 1}
        for rows, *_ in variants.values():
            common &= {k for k, v in rows.items() if len(v) == 1}
        for arm, (rows, before, after, removed, target) in variants.items():
            results.append({'score': page['score'], 'source': page['source'], 'arm': arm,
                            'removed_all_unique_predicted_spans': sum(removed.values()),
                            'before': compare(truth, before, common), 'after': compare(truth, after, common),
                            'removed_pairs': [[pair, n] for pair, n in removed.items()],
                            'invariants_pass': True, 'output_sha256': sha(target)})
    write(args.out / 'report.json', {'results': results, 'threshold': .9,
          'endpoints_sha256': sha(args.endpoints), 'source_sha256': sha(Path(__file__)),
          'scope': 'Isolated XML experiment, selection uses predictions only; no runtime/default changes. '
          'Source-assisted conditional scoring is not a broad promotion gate.'})
    print(json.dumps([{k: v for k, v in r.items() if k not in ('before', 'after', 'removed_pairs')}
                      for r in results], indent=2))


if __name__ == '__main__':
    main()
