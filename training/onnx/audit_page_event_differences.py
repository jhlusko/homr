"""Explain full-page edit counts without treating missing notes as wrong head labels."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from rapidfuzz.distance import Levenshtein

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_full_page_musicxml import project, staff_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / 'comparison.json').read_text())
    page = report['page']
    if sha(page['reference']) != page['reference_sha256']:
        raise ValueError('Reference changed')
    root = read_mxl(page['reference'])
    reference = staff_events(root, page['start_measure'], page['end_measure'])
    measure_numbers = [m.get('number') for m in root.find('part').findall('measure')]
    arms = {}
    for arm in ('accepted', 'candidate'):
        xml = args.run / arm / 'page.musicxml'
        if sha(xml) != report['arms'][arm]['xml_sha256']:
            raise ValueError('XML changed')
        predicted = staff_events(ET.parse(xml).getroot())
        summaries = {}
        examples = []
        for metric in ('event', 'pitch', 'rhythm', 'accidental'):
            counts = Counter()
            for staff in range(max(len(reference), len(predicted))):
                a = reference[staff] if staff < len(reference) else []
                b = predicted[staff] if staff < len(predicted) else []
                a_events = [e for e in a if e['pitch'] is not None] if metric in ('pitch', 'accidental') else a
                b_events = [e for e in b if e['pitch'] is not None] if metric in ('pitch', 'accidental') else b
                for operation in Levenshtein.editops(project(a, metric), project(b, metric)):
                    counts[operation.tag] += 1
                    if metric == 'event':
                        expected = a_events[operation.src_pos] if operation.src_pos < len(a_events) else None
                        actual = b_events[operation.dest_pos] if operation.dest_pos < len(b_events) else None
                        anchor = expected
                        location = None if anchor is None else dict(
                            reference_measure=measure_numbers[page['start_measure'] + anchor['measure']],
                            quarter_note_beat=str(Fraction(anchor['onset']) + 1))
                        examples.append(dict(staff=staff+1, operation=operation.tag, location=location,
                                             reference=expected if operation.tag != 'insert' else None,
                                             predicted=actual if operation.tag != 'delete' else None))
            if sum(counts.values()) != report['arms'][arm]['metrics'][metric]['errors']:
                raise ValueError('Edit count audit mismatch')
            summaries[metric] = dict(counts)
        arms[arm] = dict(edit_types=summaries, event_differences=examples)
    write(args.run / 'edit-audit.json', dict(arms=arms, scorer_sha256=sha(__file__),
        limitation='Locations follow one optimal sequence alignment and are not uniquely adjudicated. '
                   'Beat unit is a quarter note, not necessarily the notated meter beat. '
                   'Pitch/accidental deletions and insertions are coverage errors, not head substitutions.'))
    print(json.dumps({arm: data['edit_types'] for arm, data in arms.items()}, indent=2))


if __name__ == '__main__':
    main()
