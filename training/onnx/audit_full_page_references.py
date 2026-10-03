"""Audit independent page references and known score exposure before inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image
import yaml

from training.onnx.qualify_joint_export import sha, write


def read_mxl(path):
    with zipfile.ZipFile(path) as archive:
        container = ET.fromstring(archive.read('META-INF/container.xml'))
        files = container.findall('.//{*}rootfile')
        if len(files) != 1:
            raise ValueError('Ambiguous compressed MusicXML root')
        return ET.fromstring(archive.read(files[0].get('full-path')))


def certified_page(rows):
    if not rows or any(r.get('status') != 'aligned' or not r.get('boundary_certified')
                       or not r.get('count_exact') for r in rows):
        return False
    rows = sorted(rows, key=lambda r: r['system_index'])
    return ([r['system_index'] for r in rows] == list(range(len(rows)))
            and all(int(r['end_measure']) > int(r['start_measure']) for r in rows)
            and all(a['end_measure'] == b['start_measure'] for a, b in zip(rows, rows[1:])))


def audit(workspace, out, max_pixels, extra_pages=None):
    base = workspace / 'homr-artifacts/gpu-roadmap-20260919/instance-results/workspace'
    public = base / 'train-20260928/public-core-reproducibility/provenance/indexes'
    paths = list(public.glob('*/index_train.txt')) + list(public.glob('*/index_valid.txt'))
    paths += [workspace / 'homr-artifacts/instance-salvage-20261001/files/runs/public-joint-core-heads-20260928/frozen-train-index.txt']
    paths += [workspace / 'homr-artifacts/instance-salvage-20261001/files/runs/headmix-audit-20260927/train_index.txt']
    paths += list((workspace / 'homr-artifacts/instance-2026-08-30/b0').glob('imslp*index*.txt'))
    paths += list((workspace / 'homr-artifacts/corpora/lieder_scanned_pairs').glob('stage2*.txt'))
    exposure = defaultdict(list)
    identities = {}
    for path in sorted(set(paths)):
        identities[str(path)] = sha(path)
        for score in set(re.findall(r'(?:sq|IMSLP)\d+', path.read_text())):
            exposure[score].append(str(path))
    alignment_path = workspace / 'lieder-omr-data/provenance/alignment_v4_boundary_safe.json'
    alignment = json.loads(alignment_path.read_text())
    if alignment.get('model_predictions_used') is not False:
        raise ValueError('Reference alignment uses model predictions')
    runlog = workspace / 'homr/RUNLOG.md'
    history = runlog.read_text().split('## IV.9 Lieder recount recovered, coverage-gated, and aligned', 1)[1].split('\n## ', 1)[0]
    prior_evaluated = set(re.findall(r'IMSLP\d+', history))
    source = workspace / 'lieder-omr-data/sources/lieder-all-source/openscore-lieder'
    catalogue_path = source / 'data/scores.yaml'
    catalogue = yaml.safe_load(catalogue_path.read_text())
    keys = defaultdict(list)
    for key, entry in catalogue.items():
        digits = re.fullmatch(r'#?(\d+)', str(entry.get('imslp', '')))
        if digits:
            keys['IMSLP' + digits[1]].append(str(key))
    mxls = {p.stem[2:]: p for p in source.rglob('lc*.mxl')}
    rows, candidates = [], []
    for score, record in sorted(alignment['scores'].items()):
        reasons = []
        if exposure.get(score):
            reasons.append('known_index_exposure')
        matches = keys.get(score, [])
        if len(matches) != 1:
            reasons.append('ambiguous_or_missing_openscore_identity')
        gt = mxls.get(matches[0]) if len(matches) == 1 else None
        gt_count = None
        if gt is None:
            reasons.append('missing_local_independent_musicxml')
        else:
            try:
                root = read_mxl(gt)
                if root.tag != 'score-partwise' or not root.findall('part'):
                    raise ValueError('Unsupported MusicXML structure')
                if any('homr' in (s.text or '').lower() for s in root.findall('.//software')):
                    raise ValueError('HOMR output cannot be reference ground truth')
                counts = [len(p.findall('measure')) for p in root.findall('part')]
                gt_count = counts[0]
                if len(set(counts)) != 1 or gt_count != sum(record['source_counts']):
                    reasons.append('physical_reference_measure_count_mismatch')
            except Exception as error:
                reasons.append('invalid_reference:' + str(error))
        groups = defaultdict(list)
        for system in record['systems']:
            groups[system['page_image']].append(system)
        page_rows = []
        for relative, systems in sorted(groups.items()):
            page_reasons = list(reasons)
            if not certified_page(systems):
                page_reasons.append('incomplete_or_uncertified_page_boundaries')
            image_roots = [
                workspace / 'homr-artifacts/datasets/lieder_vocal_text/pages',
                workspace / 'homr-artifacts/lieder-v8/_build/pages',
            ]
            if extra_pages:
                image_roots.append(extra_pages)
            image = next((p / relative for p in image_roots if (p / relative).is_file()), None)
            if image is None:
                page_reasons.append('missing_original_page')
                dimensions = None
            else:
                try:
                    with Image.open(image) as pixels:
                        dimensions = list(pixels.size)
                except Image.DecompressionBombError:
                    dimensions = None
                    page_reasons.append('above_fixed_pixel_budget')
                except OSError as error:
                    dimensions = None
                    page_reasons.append('unreadable_original:' + str(error))
                if dimensions and dimensions[0] * dimensions[1] > max_pixels:
                    page_reasons.append('above_fixed_pixel_budget')
            row = dict(source='lieder', score=score, image=str(image) if image else None,
                       reference=str(gt) if gt else None, dimensions=dimensions,
                       reference_total_measures=gt_count, systems=systems,
                       known_prior_bar_count_evaluation=score in prior_evaluated,
                       known_exposure=exposure.get(score, []), exclusions=page_reasons)
            if not page_reasons:
                row.update(image_sha256=sha(image), reference_sha256=sha(gt),
                           start_measure=min(r['start_measure'] for r in systems),
                           end_measure=max(r['end_measure'] for r in systems),
                           exposure_status='No exposure found in audited indexes; historical accepted-model provenance incomplete')
                candidates.append(row)
            page_rows.append(row)
        rows.extend(page_rows)
    ossq = []
    for gt in sorted((workspace / 'ossq-omr/scores').glob('*/*/sq*.musicxml')):
        if not re.fullmatch(r'sq\d+', gt.stem):
            continue
        images = sorted((gt.parent / 'images/scanned/original').glob('*.png'))
        aligned = sorted((gt.parent / 'metadata/scanned/systemwise').glob('*.yaml'))
        ossq.append(dict(score=gt.stem, reference=str(gt), original_pages=len(images),
                         aligned_system_metadata=len(aligned), known_exposure=exposure.get(gt.stem, []),
                         readiness='Aligned scan metadata absent: do not substitute synthetic/unaligned metadata'
                         if not aligned else 'Requires page/window audit'))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'lieder-page-audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out / 'candidate-pages.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in candidates))
    write(out / 'ossq-score-audit.json', ossq)
    # Fixed selection by identity hash, never by recognition output or model quality.
    per_score = defaultdict(list)
    for row in candidates:
        per_score[row['score']].append(row)
    selected = [min(v, key=lambda r: hashlib.sha256(r['image'].encode()).hexdigest())
                for v in per_score.values()]
    selected.sort(key=lambda r: hashlib.sha256(('20261003:' + r['score']).encode()).hexdigest())
    write(out / 'pilot-pages.json', selected[:2])
    report = dict(script_sha256=sha(__file__), max_pixels=max_pixels,
        identities=dict(identities, **{str(alignment_path): sha(alignment_path),
                                      str(catalogue_path): sha(catalogue_path)}),
        lieder_scores=len(alignment['scores']), lieder_pages=len(rows),
        lieder_no_known_index_exposure_scores=len(set(alignment['scores']) - set(exposure)),
        candidate_pages=len(candidates), candidate_scores=len(per_score), pilot_pages=len(selected[:2]),
        candidate_known_prior_evaluation_pages=sum(r['known_prior_bar_count_evaluation'] for r in candidates),
        verified_untouched_pages=0,
        historical_evaluation_evidence={'source': str(runlog), 'section': 'IV.9',
                                       'section_sha256': hashlib.sha256(history.encode()).hexdigest()},
        page_exclusions=dict(Counter(reason for r in rows for reason in r['exclusions'])),
        ossq_scores=len(ossq), ossq_original_pages=sum(r['original_pages'] for r in ossq),
        ossq_aligned_metadata_scores=sum(bool(r['aligned_system_metadata']) for r in ossq),
        limitation='Conservative known-index exposure audit, not proof of untouched historical training/selection. '
                   'Lieder alignment uses count/boundary evidence; notation-level reference fidelity still needs review. '
                   'No recognizer output used to select pages. OSSQ CSV remapping is not silently inferred.')
    write(out / 'audit.json', report)
    (out / 'historical-evaluation-excerpt.md').write_text(history)
    print(json.dumps({k: v for k, v in report.items() if k not in ('identities', 'page_exclusions')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-pixels', type=int, default=8_000_000)
    parser.add_argument('--extra-pages', type=Path)
    args = parser.parse_args()
    audit(args.workspace.resolve(), args.out.resolve(), args.max_pixels,
          args.extra_pages.resolve() if args.extra_pages else None)
