"""Audit score-disjoint inputs and run a resumable, bounded complete-stack comparison.

The published validation scores are training-held-out, not model-selection-untouched.
This runner evaluates raw crop decoding; full-page runtime qualification is separate.
"""
import argparse
from collections import Counter, defaultdict
import difflib
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
from rapidfuzz.distance import Levenshtein

from training.transformer.audit_cached_joint_decoding import events, fields, predicted_fields, projection, sha

ARMS = ('accepted_oldcore_v7', 'public_joint27')
SEED = 20261001


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def score_id(value):
    match = re.search(r'(sq\d+|IMSLP\d+)', Path(value).name)
    if not match:
        raise ValueError('Unrecognized score identity: ' + value)
    return match[1]


def index_rows(path):
    return [line.split(',') for line in path.read_text().splitlines() if line.strip()]


def prepare(workspace, run):
    archived = workspace / 'homr-artifacts/instance-salvage-20261001/files/runs'
    base = workspace / 'homr-artifacts/gpu-roadmap-20260919/instance-results/workspace'
    provenance = base / 'train-20260928/public-core-reproducibility/provenance/indexes'
    joint = archived / 'public-joint-core-heads-20260928'
    training = [joint / 'frozen-train-index.txt'] + [provenance / f'{s}-omr-data/index_train.txt' for s in ('ossq', 'lieder')]
    trained = {score_id(tokens) for path in training for _, tokens in index_rows(path) if re.search(r'(sq\d+|IMSLP\d+)', Path(tokens).name)}
    manifest = []
    exclusions = Counter()
    for source in ('ossq', 'lieder'):
        for remote_image, remote_tokens in index_rows(provenance / f'{source}-omr-data/index_valid.txt'):
            score = score_id(remote_tokens)
            if score in trained:
                exclusions[source + ':training_score_overlap'] += 1
                continue
            if source == 'ossq':
                root = run / 'corpus/ossq-v2-valid'
                image = root / ('scanned/' + remote_image.split('/scanned/', 1)[1])
                tokens = root / ('scanned/' + remote_tokens.split('/scanned/', 1)[1])
            else:
                root = workspace / 'homr-artifacts/lieder-v8/pairs'
                image, tokens = root / Path(remote_image).name, root / Path(remote_tokens).name
            sidecar = Path(str(tokens) + '.notation.json')
            for path in (image, tokens, sidecar):
                if not path.is_file():
                    raise ValueError('Missing qualification input: ' + str(path))
            manifest.append(dict(id=remote_tokens, source=source, score=score, image=str(image), tokens=str(tokens),
                                 image_sha256=sha(image), reference_sha256=sha(tokens), sidecar_sha256=sha(sidecar)))
    manifest.sort(key=lambda r: (r['source'], r['score'], r['id']))
    groups = defaultdict(list)
    for row in manifest:
        groups[(row['source'], row['score'])].append(row)
    pilot = [min(rows, key=lambda r: hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).hexdigest()) for rows in groups.values()]
    for filename, rows in [('holdout.jsonl', manifest), ('pilot.jsonl', pilot)]:
        (run / filename).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    models = {
        'accepted_oldcore_v7': dict(core=str(archived / 'headmix-audit-20260927/old-core-20260918.pth'),
                                   heads=str(base / 'train-20260924/heads-mixed-v2stems-v7/heads.pth')),
        'public_joint27': dict(joint=str(base / 'train-20260928/public-joint-core-heads-20260928/full/epoch-27.pth')),
    }
    for spec in models.values():
        spec['sha256'] = {key: sha(path) for key, path in list(spec.items())}
    assert models['accepted_oldcore_v7']['sha256']['core'] == '1b2855ce3facea9aa8b0017f6e30143abd9aa43569a2f9ce1ab46bf8bf818fa3'
    assert models['public_joint27']['sha256']['joint'] == '6b146d0d7d2b5a7e4ecfe6d2d004bfd49ef5eef9a5843251b3696b466d1255cc'
    write(run / 'models.json', models)
    write(run / 'split-audit.json', dict(seed=SEED, rows=dict(Counter(r['source'] for r in manifest)),
          scores=dict(Counter(source for source, _ in groups)), pilot_rows=len(pilot), exclusions=dict(exclusions),
          checked_training_indices={str(p): sha(p) for p in training}, training_score_overlap=0,
          limitation='Public-core and joint training indexes checked. Historical old-core/head training membership is not fully proven here. Published validation was used previously for model selection; no untouched-test claim.',
          lieder_symbol_regression_margin_pp=.25, margin_authorized=True))
    print(f'Prepared {len(manifest)} held-out crops / {len(groups)} scores; pilot {len(pilot)} crops.', flush=True)


def pixels(path):
    import cv2
    import torch
    from homr.staff_parsing import add_image_into_tr_omr_canvas
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Image unreadable: ' + str(path))
    image = cv2.cvtColor(add_image_into_tr_omr_canvas(image), cv2.COLOR_BGR2GRAY)
    tensor = torch.tensor(image.astype(np.float32) / 255.)
    return (tensor - torch.tensor([.7931]).view(1, 1, 1)) / torch.tensor([.1738]).view(1, 1, 1)


def serialize(symbol):
    row = {key: getattr(symbol, key) for key in ('rhythm', 'pitch', 'lift', 'articulation', 'slur', 'position')}
    if symbol.notation is not None:
        n = symbol.notation
        row['notation'] = dict(beams=list(n.beam_levels), stem=str(n.stem), tie=str(n.tie),
                               slurs=[list(pair) for pair in n.slurs], advance=str(n.advance))
    return row


def decode(run, arm, scope, device, threads, limit):
    import torch
    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR
    from training.omr_datasets.notation_sidecar import attach_sidecar
    from training.transformer.training_vocabulary import read_tokens
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    rows = [json.loads(line) for line in (run / f'{scope}.jsonl').read_text().splitlines()]
    if limit:
        rows = rows[:limit]
    spec = json.loads((run / 'models.json').read_text())[arm]
    for key, digest in spec['sha256'].items():
        if sha(spec[key]) != digest:
            raise ValueError('Checkpoint changed: ' + key)
    folder = run / scope
    folder.mkdir(exist_ok=True)
    identity = dict(arm=arm, model=spec, index_sha256=sha(run / f'{scope}.jsonl'), scorer_sha256=sha(__file__),
                    device=device, threads=threads, limit=limit, torch=str(torch.__version__),
                    runtime_sources={str(p): sha(p) for p in [Path(__file__).resolve().parents[2] / relative for relative in
                        ('homr/transformer/configs.py', 'homr/transformer/vocabulary.py', 'homr/staff_parsing.py',
                         'training/architecture/transformer/encoder.py', 'training/architecture/transformer/decoder.py',
                         'training/architecture/transformer/custom_x_transformer.py')]},
                    protocol='float32 eager greedy math-SDPA, final-divider enforcement off, raw crop outputs, no runtime policies')
    identity_path = folder / f'{arm}.identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Resume identity changed; use a new run directory')
    write(identity_path, identity)
    output = folder / f'{arm}.jsonl'
    done = set()
    if output.exists():
        recovered = output.with_suffix('.recovering')
        with output.open() as original, recovered.open('w') as clean:
            for line in original:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if line.endswith('\n'):
                        raise
                    continue  # interrupted final write; redo this crop
                if row['status'] == 'ok':
                    assert row['id'] not in done, 'Duplicate completed crop'
                    done.add(row['id'])
                    clean.write(json.dumps(row) + '\n')
        recovered.replace(output)
        assert done <= {r['id'] for r in rows}, 'Unexpected cached crop'
    config = Config()
    config.encoder_pretrained = False
    config.enable_structured_heads = True
    config.structured_beam_levels, config.structured_slur_slots = 4, 2
    config.enforce_final_divider = False
    model = TrOMR(config)
    if arm == 'public_joint27':
        model.load_state_dict(torch.load(spec['joint'], map_location='cpu', weights_only=True), strict=True)
    else:
        result = model.load_state_dict(torch.load(spec['core'], map_location='cpu', weights_only=True), strict=False)
        assert not result.unexpected_keys
        assert all(k.startswith('decoder.structured_heads.') for k in result.missing_keys), result.missing_keys
        model.decoder.structured_heads.load_state_dict(torch.load(spec['heads'], map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    model.decoder.device = torch.device(device)
    print('Loaded complete stack:', arm, flush=True)
    start = time.monotonic()
    newly_completed = 0
    with output.open('a') as handle:
        for row in rows:
            if row['id'] in done:
                continue
            tick = time.monotonic()
            try:
                for key, filename in [('image_sha256', row['image']), ('reference_sha256', row['tokens']), ('sidecar_sha256', row['tokens'] + '.notation.json')]:
                    assert sha(filename) == row[key], 'Input changed: ' + filename
                expected = read_tokens(row['tokens'])
                assert attach_sidecar(row['tokens'], expected)
                tensor = pixels(row['image'])
                with torch.inference_mode():
                    actual = model.generate(tensor.unsqueeze(0).to(device))
                reference, predicted = [serialize(s) for s in expected], [serialize(s) for s in actual]
                metrics = {}
                for metric in ('symbol', 'rhythm', 'pitch', 'accidental'):
                    a, b = projection(reference, metric), projection(predicted, metric)
                    metrics[metric] = dict(errors=Levenshtein.distance(a, b), reference=len(a))
                refnotes = [s for s in events(reference) if s['rhythm'].startswith('note')]
                gotnotes = [s for s in events(predicted) if s['rhythm'].startswith('note')]
                key = lambda s: (s['rhythm'], s['pitch'], s['lift'])
                match = difflib.SequenceMatcher(a=list(map(key, refnotes)), b=list(map(key, gotnotes)), autojunk=False)
                pairs = {b.a+i: b.b+i for b in match.get_matching_blocks() for i in range(b.size)}
                support, correct = Counter(), Counter()
                for i, note in enumerate(refnotes):
                    target = fields(note)
                    support.update(target.keys())
                    got = predicted_fields(gotnotes[pairs[i]], target) if i in pairs else {}
                    correct.update(k for k, value in target.items() if got.get(k) == value)
                record = dict(**row, status='ok', seconds=time.monotonic()-tick, metrics=metrics,
                              pixel_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                              reference=reference, predicted=predicted, notation_support=dict(support), notation_correct=dict(correct))
            except Exception:
                record = dict(**row, status='failed', seconds=time.monotonic()-tick, error=traceback.format_exc())
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            newly_completed += 1
            count = len(done) + newly_completed
            elapsed = time.monotonic()-start
            write(folder / f'{arm}.progress.json', dict(completed=count, total=len(rows), last_status=record['status'],
                  elapsed_seconds=elapsed, eta_seconds=elapsed/newly_completed*(len(rows)-count), updated=time.time()))
            print(f'{arm}: {count}/{len(rows)} {row["source"]}/{row["score"]} {record["status"]} {record["seconds"]:.1f}s', flush=True)
    (folder / f'{arm}.FINISHED').write_text('Loop finished; failures must be checked before qualification.\n')


def report(run, scope):
    folder = run / scope
    results = {}
    for arm in ARMS:
        if not (folder / f'{arm}.FINISHED').exists():
            raise ValueError('Incomplete arm: ' + arm)
        results[arm] = {r['id']: r for r in map(json.loads, (folder / f'{arm}.jsonl').read_text().splitlines())}
    assert set(results[ARMS[0]]) == set(results[ARMS[1]])
    expected = {json.loads(line)['id'] for line in (run / f'{scope}.jsonl').read_text().splitlines()}
    assert set(results[ARMS[0]]) == expected, 'Incomplete input coverage'
    identities = [json.loads((folder / f'{arm}.identity.json').read_text()) for arm in ARMS]
    for key in ('index_sha256', 'scorer_sha256', 'device', 'threads', 'limit', 'torch', 'runtime_sources', 'protocol'):
        assert identities[0][key] == identities[1][key], 'Unpaired protocol: ' + key
    rows = list(results[ARMS[0]].values())
    lines = ['# Training-held-out core/head-pair crop comparison', '',
             'Previously used validation scores; raw eager crop decoding, not untouched-test or full-page qualification. '
             + ('Pilot selects one deterministic crop per score. No promotion decision from this sample.' if scope == 'pilot' else 'All published real-source validation crops included; runtime policies and full-page detection/export remain separate.'), '',
             '| Source | Metric | Old core + v7 | Public joint 27 | Delta (pp) |', '|---|---|---:|---:|---:|']
    summaries = {}
    for source in ('ossq', 'lieder'):
        ids = [r['id'] for r in rows if r['source'] == source]
        failed = [rid for rid in ids if any(results[a][rid]['status'] != 'ok' for a in ARMS)]
        if failed:
            summaries[source] = dict(failed=failed)
            lines.append(f'\n{source}: {len(failed)} paired failures; no pass result.')
            continue
        for rid in ids:
            for key in ('image_sha256', 'pixel_sha256', 'reference_sha256', 'sidecar_sha256', 'reference'):
                assert results[ARMS[0]][rid][key] == results[ARMS[1]][rid][key]
        summary = dict(crops=len(ids), scores=len({results[ARMS[0]][rid]['score'] for rid in ids}), metrics={})
        for metric in ('symbol', 'rhythm', 'pitch', 'accidental'):
            totals = []
            for arm in ARMS:
                records = [results[arm][rid]['metrics'][metric] for rid in ids]
                totals.append(dict(errors=sum(r['errors'] for r in records), reference=sum(r['reference'] for r in records)))
            rates = [100*t['errors']/max(1,t['reference']) for t in totals]
            summary['metrics'][metric] = dict(accepted=totals[0], joint=totals[1], delta_pp=rates[1]-rates[0])
            groups = defaultdict(list)
            for rid in ids:
                groups[results[ARMS[0]][rid]['score']].append(rid)
            score_totals = np.array([[sum(results[arm][rid]['metrics'][metric]['errors'] for rid in group)
                                     for arm in ARMS] +
                                    [sum(results[ARMS[0]][rid]['metrics'][metric]['reference'] for rid in group)]
                                    for group in groups.values()])
            rng = np.random.default_rng(SEED)
            samples = score_totals[rng.integers(0, len(groups), size=(2000, len(groups)))].sum(axis=1)
            delta = 100 * (samples[:, 1]-samples[:, 0]) / np.maximum(1, samples[:, 2])
            summary['metrics'][metric]['score_bootstrap_95ci_pp'] = np.percentile(delta, [2.5, 97.5]).tolist()
            lines.append(f'| {source} | {metric} | {rates[0]:.3f}% | {rates[1]:.3f}% | {rates[1]-rates[0]:+.3f} |')
        summary['notation'] = {}
        for arm in ARMS:
            support, correct = Counter(), Counter()
            for rid in ids:
                support.update(results[arm][rid]['notation_support'])
                correct.update(results[arm][rid]['notation_correct'])
            summary['notation'][arm] = {k: dict(support=n, correct=correct[k], coverage=correct[k]/n) for k,n in support.items() if n}
        summaries[source] = summary
    lines += ['', '## Correct-recognition-plus-notation coverage', '',
              '| Source | Field | Old core + v7 | Public joint 27 | GT support |', '|---|---|---:|---:|---:|']
    for source, summary in summaries.items():
        if 'notation' not in summary:
            continue
        for field in ('stem_direction', 'beam_vector', 'tie_state', 'slur_1_side'):
            a = summary['notation'][ARMS[0]].get(field)
            b = summary['notation'][ARMS[1]].get(field)
            if a and b:
                assert a['support'] == b['support']
                lines.append(f'| {source} | {field} | {100*a["coverage"]:.2f}% | {100*b["coverage"]:.2f}% | {a["support"]} |')
    write(folder / 'comparison.json', dict(scope=scope, sources=summaries, lieder_margin_pp=.25,
          status='training-held-out raw crop comparison; no promotion or full-page qualification'))
    lines += ['', 'The agreed Lieder margin is +0.25 percentage points in symbol error. '
              + ('This pilot is too small to establish a non-inferiority gate; ' if scope == 'pilot' else 'Consult score-bootstrap intervals in comparison.json; ')
              + 'retain all pitch/rhythm/accidental and notation results separately.', '']
    (folder / 'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'decode', 'report'))
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--scope', choices=('pilot', 'holdout'), default='pilot')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare':
        prepare(args.workspace.resolve(), args.run)
    elif args.command == 'decode':
        if not args.arm:
            parser.error('--arm required for decode')
        decode(args.run, args.arm, args.scope, args.device, args.threads, args.limit)
    else:
        report(args.run, args.scope)


if __name__ == '__main__':
    main()
