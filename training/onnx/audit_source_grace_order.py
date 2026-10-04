"""Check source grace sequences and explicit chords through serialization.

Diagnostic attributes stay in memory. No reference or published crop is changed.
"""
import argparse
from collections import defaultdict
import copy
from fractions import Fraction
import json
from pathlib import Path
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

from homr.transformer.vocabulary import sort_token_chords
from training.omr_datasets import music_xml_parser as parser
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write


def check(root):
    root = copy.deepcopy(root)
    # Document order can visit later onsets in one voice before backing up
    # to an earlier onset in another. Only compare source sequence *within*
    # one onset; ordering across onsets must follow MusicXML's clock.
    for part in root.findall('part'):
        divisions = Fraction(1)
        for measure in part.findall('measure'):
            cursor = onset = Fraction(0)
            for child in measure:
                if child.tag == 'attributes' and child.find('divisions') is not None:
                    divisions = Fraction(child.findtext('divisions'))
                if child.tag in ('backup', 'forward'):
                    cursor += (1 if child.tag == 'forward' else -1) * Fraction(child.findtext('duration')) / divisions
                if child.tag != 'note':
                    continue
                if child.find('chord') is None:
                    onset = cursor
                child.set('data-grace-audit-onset', str(onset))
                if child.find('chord') is None and child.find('grace') is None:
                    cursor += Fraction(child.findtext('duration', '0')) / divisions
    expected = defaultdict(list)
    actual = defaultdict(list)
    source_note = None
    ordinal = 0
    retained_measures = {}
    process = parser._process_note
    append = parser.TokensPart.append_note

    def process_note(part, note):
        nonlocal source_note
        source_note = note
        return process(part, note)

    def append_note(part, staff, is_chord, duration, invisible, symbol):
        nonlocal ordinal
        append(part, staff, is_chord, duration, invisible, symbol)
        if invisible or source_note.find('grace') is None:
            return
        key = (id(part.current_measure), staff, source_note.get('data-grace-audit-onset'))
        retained_measures[key[0]] = part.current_measure
        ordinal += 1
        symbol._audit_grace = (key, ordinal)
        if source_note.find('chord') is None or not expected[key]:
            expected[key].append([])
        expected[key][-1].append(ordinal)

    with patch.object(parser, '_process_note', process_note), patch.object(parser.TokensPart, 'append_note', append_note):
        voices = parser.music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
    for voice in voices:
        for measure in voice:
            # This is the same group partition used by serialized_chords.
            # Avoid its vocabulary lookup on unrelated, unsupported legacy
            # duration aliases: within-chord sorting cannot change group order.
            for chord in sort_token_chords(measure):
                groups = defaultdict(list)
                for symbol in chord:
                    diagnostic = getattr(symbol, '_audit_grace', None)
                    if diagnostic is not None:
                        key, event = diagnostic
                        groups[key].append(event)
                for key, events in groups.items():
                    actual[key].append(sorted(events))
    mismatches = [dict(expected=groups, actual=actual[key]) for key, groups in expected.items() if groups != actual[key]]
    return dict(grace_notes=ordinal, grace_groups=sum(map(len, expected.values())),
                explicit_grace_chords=sum(len(g)>1 for groups in expected.values() for g in groups),
                mismatches=mismatches)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--inventory',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite qualification')
    results=[];start=time.monotonic()
    for row in map(json.loads,args.inventory.read_text().splitlines()):
        path=Path(row['reference'])
        if sha(path)!=row['reference_sha256']:
            raise ValueError('Frozen source changed')
        root=read_mxl(path) if path.suffix=='.mxl' else ET.parse(path).getroot()
        if not any(n.find('grace') is not None for n in root.findall('./part/measure/note')):
            result=dict(grace_notes=0,grace_groups=0,explicit_grace_chords=0,mismatches=[])
        else:
            result=check(root)
        results.append(dict(source=row['source'],score=row['score'],reference=str(path),**result))
        print(row['source'],row['score'],result['grace_notes'],len(result['mismatches']),flush=True)
    write(args.out,dict(scores=results,seconds=time.monotonic()-start,inventory_sha256=sha(args.inventory),
        tool_sha256=sha(Path(__file__)),limitation='Visible pitched grace sequences per source measure/staff; '
        'chord members may sort within an explicit chord. Does not qualify ordinary onset grouping or runtime inference.'))
    if any(r['mismatches'] for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
