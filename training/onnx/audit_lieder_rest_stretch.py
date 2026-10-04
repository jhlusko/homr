"""Locate unresolved rest timing, inspect native staff stretch and crop exposure.

No corpus repair or new target classes. Diagnostic event IDs are added only to
an in-memory source copy so they can be traced through the actual crop cutter.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

from homr.circle_of_fifths import strip_naturals
from homr.music_xml_generator import add_tuplet_start_stop, group_into_chords, modal_measure_duration, stated_numerator_contradicts_bars
from homr.transformer.vocabulary import TIME_SIGNATURE_BEATS_PREFIX
from training.architecture.transformer.exact_timing_targets import HEADS, timing_values, build_timing_targets
from training.omr_datasets import exact_timing_labels
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.notation_sidecar import write_sidecar, attach_sidecar
from training.omr_datasets.recover_excluded_pairs import slice_voice_measures
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write
from training.transformer.training_vocabulary import token_lines_to_str, read_tokens


def find_conflicts(root):
    """Same visible pitched/rest scope as the frozen canonical inventory."""
    cases = []
    for part_no, part in enumerate(root.findall('part'), 1):
        divisions, meter = Fraction(1), {}
        for measure_no, measure in enumerate(part.findall('measure'), 1):
            cursor, onset, note_index = Fraction(0), Fraction(0), 0
            rest_orders = Counter()
            for child in measure:
                if child.tag == 'attributes':
                    if child.find('divisions') is not None:
                        divisions = Fraction(child.findtext('divisions'))
                    for signature in child.findall('time'):
                        meter[signature.get('number', 'all')] = dict(beats=signature.findtext('beats'), beat_type=signature.findtext('beat-type'))
                if child.tag in ('backup', 'forward'):
                    cursor += (1 if child.tag == 'forward' else -1) * Fraction(child.findtext('duration')) / divisions
                if child.tag != 'note':
                    continue
                note_index += 1
                if child.find('grace') is not None:
                    continue
                if child.find('chord') is None:
                    onset = cursor
                elapsed = Fraction(child.findtext('duration')) / divisions
                if child.find('chord') is None:
                    cursor += elapsed
                rest, staff, voice = child.find('rest') is not None, child.findtext('staff', '1'), child.findtext('voice', '1')
                if rest:
                    rest_orders[staff, voice] += 1
                if child.get('print-object') == 'no' or child.findtext('notehead') == 'none':
                    continue
                if not rest and child.find('pitch') is None:
                    continue
                payload = exact_timing_labels.extract_exact_timing(child, divisions)
                _, reasons = timing_values(payload, rest, tick_quarters=1/divisions)
                if 'ratio_source_duration_disagreement' not in reasons:
                    continue
                event_id = f'p{part_no}-m{measure_no}-n{note_index}'
                child.set('data-timing-audit-id', event_id)
                cases.append(dict(id=event_id, part=part_no, physical_measure=measure_no,
                    printed_measure=measure.get('number'), note_index=note_index,
                    staff=staff, voice=voice, rest=rest, rest_ordinal=rest_orders[staff, voice],
                    source_onset_quarters=str(onset), meter=meter.get(staff, meter.get('all')),
                    timing=payload, source_note_xml=ET.tostring(child, encoding='unicode'), crop_exposure=[]))
    return cases


def native_evidence(native, xml, case):
    """Require native part/staff/measure/rest mapping and written value agreement."""
    parts, native_parts = xml.findall('part'), native.findall('./Score/Part')
    if len(parts) != len(native_parts):
        raise ValueError('Native part inventory differs')
    ids = [s.get('id') for s in native_parts[case['part']-1].findall('Staff')]
    staff_id = ids[int(case['staff'])-1]
    staff = next(s for s in native.findall('./Score/Staff') if s.get('id') == staff_id)
    measures = staff.findall('Measure')
    if len(measures) != len(parts[case['part']-1].findall('measure')):
        raise ValueError('Native physical measure inventory differs')
    stretch, signature, changed_at = Fraction(1), None, None
    for index, measure in enumerate(measures[:case['physical_measure']], 1):
        for node in measure.findall('./voice/TimeSig'):
            stretch = Fraction(int(node.findtext('stretchN', '1')), int(node.findtext('stretchD', '1')))
            signature, changed_at = ET.tostring(node, encoding='unicode'), index
    measure = measures[case['physical_measure']-1]
    voice = measure.findall('voice')[(int(case['voice'])-1) % 4]
    rests = voice.findall('Rest')
    xml_measure = parts[case['part']-1].findall('measure')[case['physical_measure']-1]
    xml_rests = [n for n in xml_measure.findall('note') if n.find('rest') is not None
                 and n.findtext('staff', '1') == case['staff'] and n.findtext('voice', '1') == case['voice']]
    if len(rests) != len(xml_rests) or not case['rest']:
        raise ValueError('Native rest inventory differs')
    rest = rests[case['rest_ordinal']-1]
    payload = case['timing']
    if rest.findtext('durationType') != payload['written_type'] or int(rest.findtext('dots', '0')) != payload['dots']:
        raise ValueError('Native written rest differs')
    if rest.find('Tuplet') is not None:
        raise ValueError('Native rest has a separate tuplet requiring further investigation')
    from training.architecture.transformer.exact_timing_targets import TYPE_QUARTERS
    written = TYPE_QUARTERS[payload['written_type']] * sum((Fraction(1, 2**i) for i in range(payload['dots']+1)), Fraction(0))
    raw = Fraction(payload['elapsed_quarters'])
    return dict(native_staff_id=staff_id, stretch=str(stretch), signature_changed_at=changed_at,
        native_time_signature_xml=signature, native_rest_xml=ET.tostring(rest, encoding='unicode'),
        written_quarters=str(written), raw_times_stretch=str(raw*stretch),
        exactly_explained_by_native_stretch=stretch != 1 and raw*stretch == written)


def clean_crop(voices, start, end, voice_index):
    symbols = strip_naturals(slice_voice_measures(voices[voice_index], start, end))
    modal = modal_measure_duration(add_tuplet_start_stop(group_into_chords(symbols)))
    if stated_numerator_contradicts_bars(symbols, modal):
        symbols = [s for s in symbols if not s.rhythm.startswith(TIME_SIGNATURE_BEATS_PREFIX)]
    return symbols


def audit_score(row, labels, alignment, out, manifest=None):
    path = Path(row['reference'])
    if sha(path) != row['reference_sha256']:
        raise ValueError('Canonical source changed')
    root = read_mxl(path)
    cases = find_conflicts(root)
    if len(cases) != row['masked_reasons'].get('ratio_source_duration_disagreement', 0):
        raise ValueError('Conflict census changed')
    native_path = path.with_suffix('.mscx')
    native = ET.parse(native_path).getroot() if cases else None
    for case in cases:
        case['native'] = native_evidence(native, root, case)
    original_extract = exact_timing_labels.extract_exact_timing
    def traced_extract(note, divisions):
        payload = original_extract(note, divisions)
        if note.get('data-timing-audit-id'):
            payload['audit_source_event'] = note.get('data-timing-audit-id')
        return payload
    with patch.object(exact_timing_labels, 'extract_exact_timing', traced_extract):
        voices = music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
    mapped = {c['id']: c for c in cases}
    counts, crop_rows = Counter(), []
    assignments = {int(s['system']): s for s in alignment['systems']}
    for label in labels:
        token = Path(label['path'])
        if sha(token) != label['sha256']:
            raise ValueError('Frozen crop label changed')
        match = re.fullmatch(r'IMSLP\d+-sys(\d+)-v(\d+)', token.stem)
        if match is None:
            raise ValueError('Unknown real crop identity')
        system, voice = map(int, match.groups())
        assignment = assignments[system]
        if assignment['status'] != 'aligned' or not assignment.get('boundary_certified') or not assignment.get('count_exact'):
            raise ValueError('Frozen crop lacks certified source boundaries')
        symbols = clean_crop(voices, assignment['start_measure'], assignment['end_measure'], voice)
        text = token_lines_to_str(symbols)
        same = text.encode() == token.read_bytes()
        counts['indexed_crops'] += 1
        counts['identical_legacy_tokens' if same else 'legacy_token_differences'] += 1
        new_path = out/'crops'/token.name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(text)
        write_sidecar(new_path, symbols)
        restored = read_tokens(str(new_path)); attach_sidecar(new_path, restored)
        target_stats = {}
        if manifest is not None:
            targets, diagnostics = build_timing_targets([restored], manifest['classes'], duration_policy='tick-quantized')
            for head in HEADS:
                target_stats[head+'.supervised'] = (targets[head] != -100).sum().item()
                target_stats[head+'.unseen'] = diagnostics['unseen'][head]
            for reason, count in diagnostics['masked_reasons'].items():
                target_stats['target_reason.'+reason] = count
            counts.update(target_stats)
        image = token.with_suffix('.png')
        if not image.is_file():
            raise ValueError('Frozen crop image missing')
        ids = [s.exact_timing['audit_source_event'] for s in restored
               if s.exact_timing and 'audit_source_event' in s.exact_timing]
        exposure = dict(token=str(token), image=str(image), image_sha256=sha(image), system=system,
            voice_index=voice, splits=label['splits'], start_measure=assignment['start_measure'],
            end_measure=assignment['end_measure'], exact_legacy_tokens=same,
            replayed_label=str(new_path), event_ids=ids, target_support=target_stats)
        crop_rows.append(exposure)
        for event_id in ids:
            mapped[event_id]['crop_exposure'].append(exposure)
            counts['conflict_crop_occurrences'] += 1
    counts['canonical_conflicts'] = len(cases)
    counts['native_stretch_explains'] = sum(c['native']['exactly_explained_by_native_stretch'] for c in cases)
    counts['distinct_conflicts_in_indexed_crops'] = sum(bool(c['crop_exposure']) for c in cases)
    return dict(score=row['score'], reference=str(path), reference_sha256=sha(path),
        native=str(native_path), native_sha256=sha(native_path) if native is not None else None,
        native_version=native.findtext('programVersion') if native is not None else None,
        xml_software=root.findtext('./identification/encoding/software'), counts=counts, cases=cases, crops=crop_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--all-indexed-scores', action='store_true')
    parser.add_argument('--target-manifest', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'report.json').exists():
        raise ValueError('Refusing to overwrite audit')
    rows = [json.loads(s) for s in args.inventory.read_text().splitlines()]
    rows = [r for r in rows if r['source']=='lieder' and (args.all_indexed_scores or r['masked_reasons'].get('ratio_source_duration_disagreement'))]
    manifest = json.loads(args.target_manifest.read_text()) if args.target_manifest else None
    if manifest is not None and manifest.get('durationPolicy') != 'tick-quantized':
        raise ValueError('Requires frozen normalized target policy')
    labels = [json.loads(s) for s in args.labels.read_text().splitlines()]
    alignment = json.loads(args.alignment.read_text())
    if alignment.get('model_predictions_used') is not False:
        raise ValueError('Untrusted alignment provenance')
    tick=time.monotonic();results=[]
    for row in rows:
        result=audit_score(row, [l for l in labels if l['source']=='lieder' and l['score']==row['score']], alignment['scores'][row['score']], args.out, manifest)
        results.append(result)
        write(args.out/(row['score']+'.json'),result)
        print(row['score'],dict(result['counts']),flush=True)
    totals=Counter()
    for r in results:totals.update(r['counts'])
    write(args.out/'report.json',dict(scores=results, totals=totals, seconds=time.monotonic()-tick,
        input_sha256={str(p):sha(p) for p in [args.inventory,args.labels,args.alignment]+([args.target_manifest] if args.target_manifest else [])},
        tool_sha256=sha(Path(__file__)), threads=1,
        limitation='Native-source audit and replay of actual indexed Lieder label boundaries. '
        'No corpus modification, target repair, image re-detection, inference or full crop support census. '
        'Diagnostic audit_source_event metadata exists only in regenerated audit labels. '
        'Token mismatches are reported; they are not silently paired with unchanged published crops.'))


if __name__ == '__main__':
    main()
