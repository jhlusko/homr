"""Diagnose two real advance regressions with isolated cursor hypotheses.

MusicXML chord members do not advance the cursor. This temporary parser wrapper
restores the pre-call cursor for a chord or grace note in separate probes.
It does not alter production source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.ablate_real_reference_advance import export_with_counts, paired_timing
from training.onnx.qualify_joint_export import sha, write
from training.onnx.roundtrip_real_timing_labels import measure_timing
from training.onnx.score_full_page_musicxml import compare, staff_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    from training.omr_datasets import music_xml_parser as source_parser
    selected = {}
    for line in (args.run / 'scores.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['score'] in ('IMSLP154220', 'IMSLP22647'):
            selected[row['score']] = row
    if len(selected) != 2:
        raise ValueError('Missing selected regression scores')
    results = {}
    original = source_parser.TokensMeasure.append_note
    def keep_chord_cursor(measure, staff, is_chord, duration, invisible, symbol):
        cursor = measure.current_position
        original(measure, staff, is_chord, duration, invisible, symbol)
        if is_chord:
            measure.current_position = cursor
    def keep_grace_cursor(measure, staff, is_chord, duration, invisible, symbol):
        cursor = measure.current_position
        original(measure, staff, is_chord, duration, invisible, symbol)
        if 'G' in symbol.rhythm:
            measure.current_position = cursor
    for score, row in selected.items():
        if sha(row['reference']) != row['reference_sha256']:
            raise ValueError('Reference changed')
        root = read_mxl(row['reference'])
        expected = staff_events(root)
        cases = {}
        directory = args.run / 'chord-cursor-probe' / score
        directory.mkdir(parents=True, exist_ok=True)
        for name, function in [('current_parser', original), ('chord_cursor_hypothesis', keep_chord_cursor),
                               ('grace_cursor_hypothesis', keep_grace_cursor)]:
            try:
                source_parser.TokensMeasure.append_note = function
                parts = source_parser.music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
            finally:
                source_parser.TokensMeasure.append_note = original
            voices = [[symbol for measure in part for symbol in measure] for part in parts]
            a, _, _ = export_with_counts(voices, False)
            b, _, _ = export_with_counts(voices, True)
            before, after = staff_events(a), staff_events(b)
            metrics_a, metrics_b = compare(expected, before), compare(expected, after)
            if name == 'current_parser' and (metrics_a != row['baseline_metrics'] or metrics_b != row['advance_metrics']
                    or hashlib.sha256(ET.tostring(a)).hexdigest() != row['control_xml_sha256']
                    or hashlib.sha256(ET.tostring(b)).hexdigest() != row['advance_xml_sha256']):
                raise ValueError('Selected A/B does not replay exactly')
            paired, examples = paired_timing(expected, before, after)
            cases[name] = dict(baseline_metrics=metrics_a, advance_metrics=metrics_b,
                baseline_timing=measure_timing(expected, before),
                advance_timing=measure_timing(expected, after), paired=paired, examples=examples)
            ET.ElementTree(a).write(directory / (name + '-baseline.musicxml'), encoding='unicode')
            ET.ElementTree(b).write(directory / (name + '-advance.musicxml'), encoding='unicode')
        results[score] = cases
    code_root = Path(__file__).resolve().parents[2]
    names = [str(Path(__file__).resolve().relative_to(code_root)),
        'training/onnx/ablate_real_reference_advance.py',
        'training/omr_datasets/music_xml_parser.py', 'homr/music_xml_generator.py']
    write(args.run / 'chord-cursor-probe.json', dict(scores=results, threads=1,
        code_sha256={name: sha(code_root / name) for name in names},
        limitation='Two selected regression scores, no corpus or production mutation. '
            'Current-parser A/B replays exactly; cursor changes are isolated hypotheses, '
            'not a complete validated parser patch or trained-head gate.'))
    print(json.dumps({score: {name: case['paired'] for name, case in cases.items()}
                      for score, cases in results.items()}, indent=2), flush=True)


if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    main()
