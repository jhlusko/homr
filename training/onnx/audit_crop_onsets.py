"""Qualify changed crop onset groups and annotation anchors against source XML.

The published corpus is read-only. Event tags exist only in diagnostic copies.
"""
import argparse
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import json
from pathlib import Path
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

from homr import music_xml_generator as generator
from homr.circle_of_fifths import strip_naturals
from homr.transformer.vocabulary import serialized_chords
from training.omr_datasets import music_xml_parser as parser
from training.omr_datasets.recover_excluded_pairs import slice_voice_measures
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.audit_grace_cursor_ablation import legacy_grace_cursor
from training.onnx.audit_lieder_rest_stretch import clean_crop
from training.onnx.qualify_joint_export import sha, write
from training.transformer.training_vocabulary import read_token_lines, token_lines_to_str


def source_events(root):
    events = {}
    for part_no, part in enumerate(root.findall('part'), 1):
        divisions = Fraction(1)
        meter = {}
        for measure_no, measure in enumerate(part.findall('measure'), 1):
            cursor = onset = Fraction(0)
            note_no = 0
            for child in measure:
                if child.tag == 'attributes' and child.find('divisions') is not None:
                    divisions = Fraction(child.findtext('divisions'))
                if child.tag == 'attributes':
                    for signature in child.findall('time'):
                        meter[signature.get('number','all')] = dict(beats=signature.findtext('beats'),beat_type=signature.findtext('beat-type'))
                if child.tag in ('backup', 'forward'):
                    cursor += (1 if child.tag == 'forward' else -1) * Fraction(child.findtext('duration')) / divisions
                if child.tag != 'note':
                    continue
                note_no += 1
                if child.find('chord') is None:
                    onset = cursor
                event = f'p{part_no}-m{measure_no}-n{note_no}'
                child.set('data-onset-audit-id', event)
                events[event] = dict(part=part_no, physical_measure=measure_no,
                    printed_measure=measure.get('number'), onset_quarters=str(onset),
                    staff=child.findtext('staff', '1'), voice=child.findtext('voice', '1'),
                    time_signature=copy.deepcopy(meter.get(child.findtext('staff','1'),meter.get('all'))),
                    grace=child.find('grace') is not None,
                    slurs=[s.get('type') for s in child.findall('./notations/slur') if s.get('type') in ('start', 'stop')],
                    legacy_slur_events=sorted({s.get('type') for container in child.findall('notations')[:1] if container.get('print-object')!='no' for s in container if s.tag in ('tied','slur') and s.get('type') in ('start','stop')}))
                if child.find('chord') is None and child.find('grace') is None:
                    cursor += Fraction(child.findtext('duration', '0')) / divisions
    return events


def traced_parse(root):
    current = None
    process = parser._process_note
    append_note, append_rest = parser.TokensPart.append_note, parser.TokensPart.append_rest
    def process_note(part, note):
        nonlocal current
        current = note.get('data-onset-audit-id')
        return process(part, note)
    def wrap(original):
        def append(part, staff, is_chord, duration, invisible, symbol):
            symbol._onset_audit_id = current
            return original(part, staff, is_chord, duration, invisible, symbol)
        return append
    with patch.object(parser, '_process_note', process_note), patch.object(parser.TokensPart, 'append_note', wrap(append_note)), patch.object(parser.TokensPart, 'append_rest', wrap(append_rest)):
        return parser.music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))


def group_audit(symbols, events):
    groups = []
    split = defaultdict(list)
    mixed = []
    for index, chord in enumerate(serialized_chords(symbols)):
        ids = [s._onset_audit_id for s in chord if hasattr(s, '_onset_audit_id') and not events[s._onset_audit_id]['grace']]
        if not ids:
            continue
        keys = {(events[e]['part'], events[e]['physical_measure'], events[e]['onset_quarters']) for e in ids}
        if len(keys) != 1:
            mixed.append(dict(group=index, events=ids, source=[events[e] for e in ids]))
        for key in keys:
            split[key].append(index)
        groups.append(dict(group=index, events=ids))
    return dict(groups=groups, mixed_onsets=mixed,
                split_onsets=[dict(source_onset=list(k),groups=v) for k,v in split.items() if len(v)>1])


def serialized_copy(symbols, preserve_annotations=False):
    text = ("\n".join("&".join(str(s) for s in chord) for chord in serialized_chords(symbols))
            if preserve_annotations else token_lines_to_str(symbols))
    result = read_token_lines(text.splitlines())
    originals, copies = serialized_chords(symbols), serialized_chords(result)
    assert len(originals) == len(copies)
    for a,b in zip(originals,copies,strict=True):
        assert len(a)==len(b)
        for original, copied in zip(a,b,strict=True):
            assert (original.rhythm,original.pitch,original.position)==(copied.rhythm,copied.pitch,copied.position)
            if hasattr(original, '_onset_audit_id'):
                copied._onset_audit_id = original._onset_audit_id
    return result


def export_audit(symbols, events, start_measure=0):
    original = generator.build_note_or_rest
    def build(model_note, *args, **kwargs):
        note = original(model_note, *args, **kwargs)
        if hasattr(model_note, '_onset_audit_id'):
            note.set('data-original-source-id', model_note._onset_audit_id)
        return note
    try:
        with patch.object(generator, 'build_note_or_rest', build), patch.object(generator, 'structured_export_enabled', return_value=False):
            root = generator.generate_xml(generator.XmlGeneratorArguments(), copy.deepcopy([symbols]), 'Diagnostic')
    except Exception as error:
        return dict(error=f'{type(error).__name__}: {error}')
    output = source_events(root)
    # source_events adds new IDs; recover original tags from the emitted XML first.
    actual = {}
    for part in root.findall('part'):
        for measure in part.findall('measure'):
            for note in measure.findall('note'):
                generated = note.get('data-onset-audit-id')
                original_id = note.get('data-original-source-id')
                if original_id:
                    actual.setdefault(original_id, []).append(output[generated])
    expected = {s._onset_audit_id for s in symbols if hasattr(s, '_onset_audit_id')}
    onset_errors=[];slur_errors=[];measure_errors=[];legacy_endpoint_errors=[]
    for event, records in actual.items():
        for record in records:
            source=events[event]
            if record['physical_measure'] != source['physical_measure'] - start_measure:
                measure_errors.append(dict(event=event,source_measure=source['physical_measure'],expected_crop_measure=source['physical_measure']-start_measure,exported_crop_measure=record['physical_measure']))
            if not source['grace'] and record['onset_quarters']!=source['onset_quarters']:
                onset_errors.append(dict(event=event, source=source['onset_quarters'], exported=record['onset_quarters']))
            if Counter(source['legacy_slur_events'])!=Counter(record['slurs']):
                legacy_endpoint_errors.append(dict(event=event,source=source['legacy_slur_events'],exported=record['slurs']))
            if Counter(source['slurs'])!=Counter(record['slurs']):
                slur_errors.append(dict(event=event,source=source['slurs'],exported=record['slurs']))
    return dict(onset_errors=onset_errors,measure_errors=measure_errors,slur_errors=slur_errors,
                legacy_endpoint_errors=legacy_endpoint_errors,
                missing_events=sorted(expected-set(actual)),duplicate_events={k:len(v) for k,v in actual.items() if len(v)!=1})


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--replay',type=Path,required=True)
    cli.add_argument('--inventory',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--anchor-prototype',action='store_true',help='Audit-only per-note flat annotations; no default serializer change')
    args=cli.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'report.json').exists():raise ValueError('Refusing overwrite')
    inventory={r['score']:r for r in map(json.loads,args.inventory.read_text().splitlines())}
    replay=json.loads(args.replay.read_text());results=[];start=time.monotonic()
    for score in replay['scores']:
        crops=[c for c in score['crops'] if not c['exact_legacy_tokens']]
        if not crops:continue
        row=inventory[score['score']];path=Path(row['reference'])
        assert sha(path)==row['reference_sha256']
        root=read_mxl(path);events=source_events(root);voices=traced_parse(root)
        with patch.object(parser.TokensMeasure, 'append_note', legacy_grace_cursor(parser.TokensMeasure.append_note)):
            old_voices=traced_parse(root)
        records=[]
        for crop in crops:
            symbols=clean_crop(voices,crop['start_measure'],crop['end_measure'],crop['voice_index'])
            assert token_lines_to_str(symbols)==Path(crop['replayed_label']).read_text()
            groups=group_audit(symbols,events)
            old_symbols=clean_crop(old_voices,crop['start_measure'],crop['end_measure'],crop['voice_index'])
            old_parity=token_lines_to_str(old_symbols)==Path(crop['token']).read_text()
            old_groups=group_audit(old_symbols,events)
            old_export=export_audit(serialized_copy(old_symbols),events,crop['start_measure']) if old_parity else dict(error='Frozen token replay differs; paired export not qualified')
            serial=serialized_copy(symbols)
            annotations=[]
            for before,after in zip(serialized_chords(symbols),serialized_chords(serial),strict=True):
                for a,b in zip(before,after,strict=True):
                    if hasattr(a,'_onset_audit_id') and (a.slur!=b.slur or a.articulation!=b.articulation):
                        annotations.append(dict(event=a._onset_audit_id,source=events[a._onset_audit_id],
                            before_slur=a.slur,after_slur=b.slur,before_articulation=a.articulation,after_articulation=b.articulation))
            pre=strip_naturals(slice_voice_measures(voices[crop['voice_index']],crop['start_measure'],crop['end_measure']))
            modal=generator.modal_measure_duration(generator.add_tuplet_start_stop(generator.group_into_chords(pre)))
            records.append(dict(token=crop['token'],splits=crop['splits'],**groups,
                source_time_signatures=[dict(event=s._onset_audit_id,time_signature=events[s._onset_audit_id]['time_signature']) for s in symbols if hasattr(s,'_onset_audit_id')],
                annotation_moves=annotations,legacy_export=export_audit(serial,events,crop['start_measure']),
                anchor_prototype_export=export_audit(serialized_copy(symbols,preserve_annotations=True),events,crop['start_measure']) if args.anchor_prototype else None,
                old_token_parity=old_parity,old_grouping=old_groups,old_export=old_export,modal_whole_notes=str(modal),
                numerator_removed=generator.stated_numerator_contradicts_bars(pre,modal)))
        result=dict(score=score['score'],reference=str(path),crops=records)
        results.append(result);write(args.out/(score['score']+'.json'),result)
        print(score['score'],len(records),sum(len(c['mixed_onsets']) for c in records),sum(len(c['split_onsets']) for c in records),flush=True)
    counts=Counter()
    for score in results:
        for crop in score['crops']:
            counts['crops']+=1;counts['mixed_onset_crops']+=bool(crop['mixed_onsets']);counts['split_onset_crops']+=bool(crop['split_onsets'])
            counts['annotation_move_crops']+=bool(crop['annotation_moves']);counts['numerator_removed_crops']+=crop['numerator_removed']
            export=crop['legacy_export']
            counts['old_token_parity_crops']+=crop['old_token_parity']
            counts['old_mixed_onset_crops']+=bool(crop['old_grouping']['mixed_onsets'])
            counts['old_split_onset_crops']+=bool(crop['old_grouping']['split_onsets'])
            old=crop['old_export']
            counts['old_export_onset_error_crops']+=bool(old.get('onset_errors'))
            counts['old_export_slur_error_crops']+=bool(old.get('slur_errors'))
            counts['export_failure_crops']+=bool(export.get('error'))
            counts['export_onset_error_crops']+=bool(export.get('onset_errors'))
            counts['export_slur_error_crops']+=bool(export.get('slur_errors'))
            counts['export_measure_error_crops']+=bool(export.get('measure_errors'))
            counts['export_missing_event_crops']+=bool(export.get('missing_events') or export.get('duplicate_events'))
    write(args.out/'report.json',dict(scores=results,counts=counts,seconds=time.monotonic()-start,
        replay_sha256=sha(args.replay),inventory_sha256=sha(args.inventory),tool_sha256=sha(Path(__file__)),
        export_policy='legacy tokens only; no source notation/timing sidecars or predicted heads',
        anchor_prototype=args.anchor_prototype,
        annotation_policies=['literal MusicXML phrase-slur endpoints','legacy per-note deduplicated union of tied/slur start/stop endpoints'],
        limitation='Ordinary visible source onset grouping and legacy annotation relocation, legacy export onset/slur anchoring (no structured heads); not inference or complete export fidelity. No corpus modification.'))


if __name__=='__main__':main()
