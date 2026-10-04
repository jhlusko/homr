"""Audit exact-duration alternatives without changing corpora or token semantics.

Uses hash-verified canonical notes located by the real-source census. Alternative
tokens are diagnostics, never replacements. Retains original XML and a proposed
versioned exact-timing sidecar for each affected event.
"""
import argparse
import copy
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from homr.transformer.vocabulary import build_rhythm, kern_to_symbol_duration
from training.omr_datasets.exact_timing_labels import extract_exact_timing, apply_exact_timing
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write


def duration_candidates(kind, quarters):
    """Existing non-grace, non-multirest tokens with exactly this elapsed time."""
    result = []
    for token in build_rhythm():
        if not token.startswith(kind + '_') or 'G' in token or token.endswith('m'):
            continue
        duration = kern_to_symbol_duration(token.split('_', 1)[1])
        if 4 * duration.fraction == quarters:
            result.append(dict(token=token, written_base_quarters=str(4 * duration.base_duration),
                dots=duration.dots, actual=duration.actual_notes, normal=duration.normal_notes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'report.json').exists() or (args.out / 'events.jsonl').exists():
        raise ValueError('Refusing to overwrite audit')
    issues = defaultdict(list)
    for line in (args.census / 'reference-issues.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['indexed_score']:
            issues[row['source'], row['score']].append(row)
    references = [json.loads(line) for line in (args.census / 'references.jsonl').read_text().splitlines()]
    totals, scores, candidate_cache = defaultdict(Counter), defaultdict(set), {}
    tick = time.monotonic()
    with (args.out / 'events.jsonl').open('w', buffering=1) as output:
        for reference in references:
            key = reference['source'], reference['score']
            if key not in issues:
                continue
            path = Path(reference['reference'])
            if sha(path) != reference['sha256']:
                raise ValueError('Reference hash mismatch: ' + str(path))
            root = read_mxl(path) if path.suffix == '.mxl' else ET.parse(path).getroot()
            for issue in issues[key]:
                loc = issue['location']
                part = root.findall('part')[loc['part'] - 1]
                measures = part.findall('measure')
                measure = measures[loc['physical_measure'] - 1]
                note = measure.findall('note')[loc['note_index'] - 1]
                divisions = Fraction(1)
                for earlier in measures[:loc['physical_measure']]:
                    for child in earlier:
                        if child.tag == 'attributes' and child.find('divisions') is not None:
                            divisions = Fraction(child.findtext('divisions'))
                        if child is note:
                            break
                exact = Fraction(note.findtext('duration')) / divisions
                if exact != Fraction(issue['expected_quarters']):
                    raise ValueError('Located duration does not replay census')
                kind = 'rest' if note.find('rest') is not None else 'note'
                cache_key = kind, exact
                if cache_key not in candidate_cache:
                    candidate_cache[cache_key] = duration_candidates(kind, exact)
                candidates = candidate_cache[cache_key]
                modification = note.find('time-modification')
                full_rest = note.find('rest') is not None and note.find('rest').get('measure') == 'yes'
                payload = extract_exact_timing(note, divisions)
                restored = copy.deepcopy(note)
                apply_exact_timing(restored, json.loads(json.dumps(payload)), divisions * 2)
                if extract_exact_timing(restored, divisions * 2) != payload:
                    raise ValueError('Exact-timing sidecar roundtrip mismatch')
                # A ratio-equivalent duration alone does not preserve a duplet
                # glyph, bracket, normal-type, nesting or a full-measure rest.
                row = dict(**issue, reference_sha256=reference['sha256'],
                    existing_exact_duration_candidates=candidates, proposal=payload,
                    canonical_note_xml=ET.tostring(note, encoding='unicode'))
                output.write(json.dumps(row) + '\n')
                counts = totals[reference['source']]
                prefix = 'full_rest' if full_rest else 'tuplet'
                counts[prefix + '_loss_events'] += 1
                counts['sidecar_roundtrip_verified'] += 1
                counts[prefix + ('_existing_exact_duration' if candidates else '_no_existing_exact_duration')] += 1
                scores[reference['source'], prefix].add(reference['score'])
                if modification is not None:
                    counts['tuplet_normal_type_present'] += modification.find('normal-type') is not None
                    counts['tuplet_normal_dot_present'] += modification.find('normal-dot') is not None
                counts[prefix + '_quarters_' + str(exact)] += 1
            print(f'{reference["source"]} {reference["score"]}: {len(issues[key])} located events', flush=True)
    code_root = Path(__file__).resolve().parents[2]
    files = [str(Path(__file__).resolve().relative_to(code_root)),
        'homr/transformer/vocabulary.py', 'training/omr_datasets/music_xml_parser.py',
        'training/omr_datasets/exact_timing_labels.py',
        'training/onnx/census_real_timing_labels.py']
    write(args.out / 'report.json', dict(sources=totals,
        affected_score_counts={s: {k: len(v) for (source, k), v in scores.items() if source == s}
            for s in totals}, seconds=time.monotonic()-tick, threads=1,
        code_sha256={f: sha(code_root / f) for f in files},
        input_sha256={f: sha(args.census / f) for f in ('references.jsonl', 'reference-issues.jsonl')},
        limitation='Indexed canonical census issues only, including references that fail the full parser. '
            'Exact token alternatives preserve elapsed duration only, not necessarily printed notation. '
            'Proposal sidecars are diagnostic and are not consumed by training or runtime. '
            'No crop labels, corpora, vocabulary, inference, deployment or model pins changed.'))


if __name__ == '__main__':
    main()
