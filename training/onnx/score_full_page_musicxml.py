"""Matched page-level MusicXML event-edit diagnostics; not crop token SER."""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from rapidfuzz.distance import Levenshtein

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write


def staff_events(root, start=0, end=None):
    """Read written pitches/durations without recognition-vocabulary filtering.

    Keep physical staff order fixed. Do not optimize staff assignment by predictions.
    Invisible notes still advance time but are not printed recognition targets.
    """
    if root.tag != 'score-partwise':
        raise ValueError('Only score-partwise MusicXML is supported')
    streams = []
    for part in root.findall('part'):
        divisions = Fraction(1)
        staff_count = 1
        notes = defaultdict(list)
        measures = part.findall('measure')
        stop = len(measures) if end is None else end
        if not 0 <= start < stop <= len(measures):
            raise ValueError('Invalid page reference measure window')
        for index, measure in enumerate(measures[:stop]):
            cursor, previous_onset = Fraction(0), Fraction(0)
            for element in measure:
                if element.tag == 'attributes':
                    if element.find('divisions') is not None:
                        divisions = Fraction(element.findtext('divisions'))
                        if divisions <= 0:
                            raise ValueError('Nonpositive MusicXML divisions')
                    staff_count = max(staff_count, int(element.findtext('staves', '1')))
                elif element.tag in ('backup', 'forward'):
                    delta = Fraction(element.findtext('duration', '0')) / divisions
                    cursor += delta if element.tag == 'forward' else -delta
                elif element.tag == 'note':
                    staff = int(element.findtext('staff', '1'))
                    staff_count = max(staff_count, staff)
                    duration = Fraction(element.findtext('duration', '0')) / divisions
                    chord = element.find('chord') is not None
                    onset = previous_onset if chord else cursor
                    if not chord:
                        previous_onset = onset
                        cursor += duration
                    if index < start or element.get('print-object') == 'no':
                        continue
                    pitch = element.find('pitch')
                    if pitch is None and element.find('rest') is None:
                        raise ValueError('Unsupported unpitched note')
                    event = dict(measure=index-start, onset=str(onset), duration=str(duration),
                                 grace=element.find('grace') is not None,
                                 pitch=None if pitch is None else (
                                     pitch.findtext('step'), int(pitch.findtext('octave')),
                                     str(Fraction(pitch.findtext('alter', '0')))))
                    notes[staff].append(event)
        for staff in range(1, staff_count+1):
            events = sorted(notes[staff], key=lambda e: (
                e['measure'], Fraction(e['onset']), str(e['pitch']), Fraction(e['duration'])))
            streams.append(events)
    return streams


def project(events, metric):
    if metric == 'pitch':
        return [tuple(e['pitch']) for e in events if e['pitch'] is not None]
    if metric == 'accidental':
        return [e['pitch'][2] for e in events if e['pitch'] is not None]
    if metric == 'rhythm':
        return [(e['duration'], e['grace'], e['pitch'] is None) for e in events]
    return [(tuple(e['pitch']) if e['pitch'] is not None else None,
             e['duration'], e['grace']) for e in events]


def compare(reference, predicted):
    result = {}
    for metric in ('event', 'pitch', 'rhythm', 'accidental'):
        errors = support = 0
        for index in range(max(len(reference), len(predicted))):
            a = project(reference[index] if index < len(reference) else [], metric)
            b = project(predicted[index] if index < len(predicted) else [], metric)
            support += len(a)
            errors += Levenshtein.distance(a, b)
        result[metric] = dict(errors=errors, support=support,
                              error_percent=100 * errors / support if support else None)
    result['reference_staves'] = len(reference)
    result['predicted_staves'] = len(predicted)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages', type=Path, required=True)
    parser.add_argument('--page-index', type=int, default=0)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    page = json.loads(args.pages.read_text())[args.page_index]
    if sha(page['reference']) != page['reference_sha256']:
        raise ValueError('Ground-truth source changed')
    root = read_mxl(page['reference'])
    if any('homr' in (s.text or '').lower() for s in root.findall('.//software')):
        raise ValueError('Cannot use a HOMR output as reference')
    reference = staff_events(root, page['start_measure'], page['end_measure'])
    results = {}
    for arm in ('accepted', 'candidate'):
        directory = args.run / arm
        receipt = json.loads((directory / 'result.json').read_text())
        if receipt['status'] != 'ok':
            raise ValueError('Page decode failed: ' + arm)
        xml = directory / 'page.musicxml'
        if sha(xml) != receipt['xml_sha256']:
            raise ValueError('Predicted XML changed')
        events = staff_events(ET.parse(xml).getroot())
        results[arm] = dict(metrics=compare(reference, events), seconds=receipt['seconds'],
                            xml_sha256=receipt['xml_sha256'], counts=receipt['counts'])
    report = dict(page=page, arms=results, scorer_sha256=sha(__file__), threads=1,
        protocol='Per-physical-staff ordered MusicXML event Levenshtein edits; fixed staff order; '
                 'extra/missing staves count as insertions/deletions; pitches include alter; durations in quarters; '
                 'invisible notes omitted; polyphonic/chord events ordered by onset; no vocabulary filtering',
        limitations='One page, one score, previously used for bar-count evaluation. Not untouched-test evidence. '
                    'Event edits are not crop token SER and do not implement the +0.25pp Lieder token gate. '
                    'No onset/barline/notation/lyrics accuracy gate or significance claim. '
                    'Matched float32 exports, not the deployed quantized/provider pipeline.')
    write(args.run / 'comparison.json', report)
    lines = ['# Lorelei full-page diagnostic', '', report['limitations'], '',
             '| Metric | Accepted | Candidate |', '|---|---:|---:|']
    for metric in ('event', 'pitch', 'rhythm', 'accidental'):
        a = results['accepted']['metrics'][metric]['error_percent']
        b = results['candidate']['metrics'][metric]['error_percent']
        lines.append(f'| {metric} edit error | {a:.3f}% | {b:.3f}% |')
    lines += ['', 'Reference source: `' + page['reference'] + '`',
              'Reference physical measure window (zero-based, end excluded): '
              f"{page['start_measure']}–{page['end_measure']}", '', report['protocol'], '']
    (args.run / 'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
