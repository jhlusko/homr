"""Measure canonical page-reference timing lost through current training labels.

Parse parts separately and report failures, rather than silently dropping them.
This is a diagnostic of current source, not proof which historical parser built a pin.
"""
import argparse
import copy
from collections import Counter
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from training.onnx.score_paired_page_diagnostic import read_reference
from training.onnx.qualify_joint_export import setup, sha, write
from training.onnx.score_full_page_musicxml import compare, project, staff_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.out / 'report.json').exists():
        raise ValueError('Refusing to overwrite completed label diagnostic')
    args.out.mkdir(parents=True, exist_ok=True)
    setup()
    os.environ['HOMR_STRUCTURED_EXPORT'] = 'beam,stem,tie,slur'
    from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
    from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
    from training.transformer.qualify_joint_holdout import serialize
    from rapidfuzz.distance import Levenshtein
    page = json.loads((args.run / 'comparison.json').read_text())['page']
    if sha(page['reference']) != page['reference_sha256']:
        raise ValueError('Reference changed')
    if page['start_measure'] != 0:
        raise ValueError('This bounded diagnostic requires a first page to inherit attributes correctly')
    root = read_reference(page['reference'])
    results = []
    for index, part in enumerate(root.findall('part')):
        reference = copy.deepcopy(root)
        for other in reference.findall('part'):
            reference.remove(other)
        selected = copy.deepcopy(part)
        for measure in selected.findall('measure')[page['end_measure']:]:
            selected.remove(measure)
        reference.append(selected)
        directory = args.out / f'part-{index+1}'
        directory.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(reference).write(directory / 'reference.musicxml', encoding='unicode')
        try:
            parts = music_xml_string_to_tokens(ET.tostring(reference, encoding='unicode'))
            if len(parts) != 1:
                raise ValueError('Expected exactly one parsed part')
            voice = [symbol for measure in parts[0] for symbol in measure]
            write(directory / 'labels.json', [serialize(s) for s in voice])
            rendered = generate_xml(XmlGeneratorArguments(), [voice], '')
            ET.ElementTree(rendered).write(directory / 'roundtrip.musicxml', encoding='unicode')
            expected, actual = staff_events(reference), staff_events(rendered)
            differences = []
            for staff in range(max(len(expected), len(actual))):
                a = expected[staff] if staff < len(expected) else []
                b = actual[staff] if staff < len(actual) else []
                for operation in Levenshtein.editops(project(a, 'event'), project(b, 'event')):
                    differences.append(dict(staff=staff+1, operation=operation.tag,
                        reference=a[operation.src_pos] if operation.tag != 'insert' else None,
                        roundtrip=b[operation.dest_pos] if operation.tag != 'delete' else None))
            results.append(dict(part=index+1, status='ok',
                metrics=compare(expected, actual), event_differences=differences,
                same_unordered_events_per_staff=len(expected) == len(actual) and all(
                    Counter(project(a, 'event')) == Counter(project(b, 'event'))
                    for a, b in zip(expected, actual, strict=True)),
                labels_sha256=sha(directory / 'labels.json'),
                xml_sha256=sha(directory / 'roundtrip.musicxml')))
        except ValueError as error:
            results.append(dict(part=index+1, status='failed', error=str(error)))
    source_root = Path(__file__).resolve().parents[2]
    names = [str(Path(__file__).resolve().relative_to(source_root)),
             'training/omr_datasets/music_xml_parser.py',
             'training/omr_datasets/staff_merging.py',
             'homr/music_xml_generator.py', 'homr/transformer/vocabulary.py']
    report = dict(page=page, threads=1, parts=results,
        code_sha256={name: sha(source_root / name) for name in names},
        limitation='Current-parser diagnostic on one previously evaluated page. '
                   'Does not establish parser version used by published corpora; '
                   'failures are retained, not filtered out of model evaluation.')
    write(args.out / 'report.json', report)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
