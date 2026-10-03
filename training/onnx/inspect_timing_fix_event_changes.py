"""Reproduce selected OSSQ/Lieder event changes with located inventory evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from training.onnx.ablate_real_reference_advance import export_with_counts, groups_by_measure
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write
from training.onnx.qualify_reference_timing_fixes import archived_function
from training.onnx.score_full_page_musicxml import compare, project, staff_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'report.json').exists():
        raise ValueError('Refusing to overwrite inspection')
    from training.omr_datasets import music_xml_parser as source
    from homr import music_xml_generator as generator
    snapshot = args.baseline / 'source-snapshot'
    old_append = archived_function(snapshot / 'training/omr_datasets/music_xml_parser.py',
                                   'append_note', vars(source), 'TokensMeasure')
    old_division = archived_function(snapshot / 'homr/music_xml_generator.py',
        'find_division_and_time_signature_nominator', vars(generator))
    new_append, new_division = source.TokensMeasure.append_note, generator.find_division_and_time_signature_nominator
    selected = {'sq12536479', 'sq8913219', 'sq9396439', 'IMSLP89026'}
    rows = [json.loads(line) for line in (args.qualification / 'scores.jsonl').read_text().splitlines()]
    results = {}
    for row in rows:
        if row['score'] not in selected:
            continue
        if sha(row['reference']) != row['reference_sha256']:
            raise ValueError('Reference changed')
        reference_path = Path(row['reference'])
        root = read_mxl(reference_path) if reference_path.suffix == '.mxl' else ET.parse(reference_path).getroot()
        expected = staff_events(root)
        cases, events = {}, {}
        for arm, append, division in [('old_advance', old_append, old_division),
                                      ('precision_only', old_append, new_division),
                                      ('fixed_advance', new_append, new_division)]:
            try:
                source.TokensMeasure.append_note = append
                generator.find_division_and_time_signature_nominator = division
                parts = source.music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
                voices = [[s for m in part for s in m] for part in parts]
                xml, _, _ = export_with_counts(voices, True)
            finally:
                source.TokensMeasure.append_note = new_append
                generator.find_division_and_time_signature_nominator = new_division
            events[arm] = staff_events(xml)
            metrics = compare(expected, events[arm])
            if (metrics != row['arms'][arm]['metrics'] or
                    hashlib.sha256(ET.tostring(xml)).hexdigest() != row['arms'][arm]['xml_sha256']):
                raise ValueError('Qualification does not replay exactly')
            directory = args.out / row['score']
            directory.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(xml).write(directory / (arm + '.musicxml'), encoding='unicode')
            cases[arm] = dict(metrics=metrics, zero_duration_non_grace_notes=sum(
                1 for note in xml.findall('.//note')
                if note.find('grace') is None and note.findtext('duration') == '0'))
        gt = groups_by_measure(expected)
        old = groups_by_measure(events['old_advance'])
        new = groups_by_measure(events['fixed_advance'])
        changes = []
        for key in sorted(set(old) | set(new)):
            a, b, reference = old[key], new[key], gt[key]
            if project(a, 'event') != project(b, 'event'):
                changes.append(dict(staff=key[0], measure_index=key[1],
                    old_inventory_correct=Counter(project(a, 'event')) == Counter(project(reference, 'event')),
                    fixed_inventory_correct=Counter(project(b, 'event')) == Counter(project(reference, 'event')),
                    reference=reference, old=a, fixed=b))
        results[row['score']] = dict(cases=cases, changed_event_measures=changes,
            reference=row['reference'], reference_sha256=row['reference_sha256'])
    if set(results) != selected:
        raise ValueError('Incomplete selected inspection')
    code_root = Path(__file__).resolve().parents[2]
    names = [str(Path(__file__).resolve().relative_to(code_root)),
             'homr/music_xml_generator.py', 'training/omr_datasets/music_xml_parser.py']
    write(args.out / 'report.json', dict(scores=results,
        code_sha256={name: sha(code_root / name) for name in names}, threads=1,
        qualification_records_sha256=sha(args.qualification / 'scores.jsonl'),
        limitation='Three selected OSSQ scores and one Lieder score; '
            'not complete voice-aware musical equivalence.'))


if __name__ == '__main__':
    main()
