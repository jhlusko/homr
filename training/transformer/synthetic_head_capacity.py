"""Fit fresh linear heads on fixed, owner-qualified MuseScore decoder features.

This is a diagnostic capacity test, not a real-source comparison or release.
No core weights move; dynamics is excluded. Family splits are preserved.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from training.architecture.transformer.structured_losses import IGNORE_INDEX


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def qualify(dataset: Path, approval: Path) -> list[dict]:
    """Promote only the approved generator's audit-clean subset for this diagnostic."""
    accepted = json.loads(approval.read_text())
    identity = json.loads((dataset / 'identity.json').read_text())
    if not accepted.get('approved') or accepted.get('scope') != 'synthetic-capacity-only':
        raise ValueError('Explicit visual qualification for the capacity probe is required')
    for field in ('schema', 'generator_sha256', 'musescore', 'music_fonts'):
        if accepted.get(field) != identity.get(field):
            raise ValueError(f'Approval does not cover renderer/generator identity: {field}')
    if not (dataset / 'GENERATION_COMPLETE').exists():
        raise ValueError('Generation has not completed')
    # Check all original artifacts before constructing any training index.
    for line in (dataset / 'SHA256SUMS').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        path = dataset / relative
        if not path.resolve().is_relative_to(dataset.resolve()) or sha(path) != digest:
            raise ValueError(f'Artifact checksum mismatch: {relative}')
    rows = [json.loads(line) for line in (dataset / 'samples.jsonl').read_text().splitlines()]
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest['total'] != len(rows) or manifest['identical_pixels_across_variants']:
        raise ValueError('Incomplete or duplicate-contrast dataset')
    if any(r['status'] != 'requires_visual_review' or r['renderer_audit']['differences'] for r in rows):
        raise ValueError('Renderer failures or changed labels must be resolved before this probe')
    families = defaultdict(set)
    for row in rows:
        families[row['family']].add(row['split'])
    if any(len(splits) != 1 for splits in families.values()):
        raise ValueError('Family leakage across splits')
    return rows


def image_controls(rows: list[dict]) -> dict[str, str]:
    """Swap images only within a family/font and identical teacher-forcing tokens."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['family'], row['font'], row['files'][row['id'] + '.tokens'])].append(row['id'])
    controls = {}
    for group in groups.values():
        if len(group) > 1:
            ordered = sorted(group)
            controls.update({name: ordered[(i + 1) % len(ordered)] for i, name in enumerate(ordered)})
    return controls


def core_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith('decoder.structured_heads.'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def clean_sample(dataset: Path, row: dict, image_id: str | None = None) -> dict:
    from homr.staff_parsing import add_image_into_tr_omr_canvas
    from training.transformer.image_utils import (
        read_image_to_ndarray, prepare_for_tensor, ndarray_to_tensor, pad_to_3_dims,
    )
    from training.transformer.training_vocabulary import read_tokens, to_decoder_branches
    from training.omr_datasets.notation_sidecar import attach_sidecar
    from training.architecture.transformer.structured_targets import build_targets, notation_positions
    dataset = Path(row.get('_dataset', dataset))
    symbols = read_tokens(str(dataset / 'samples' / (row['id'] + '.tokens')))
    if not attach_sidecar(dataset / 'samples' / (row['id'] + '.tokens'), symbols):
        raise ValueError('Missing notation sidecar')
    branches = to_decoder_branches(symbols)
    length = len(symbols) + 2
    targets = build_targets([notation_positions(symbols, length)], 4, 2)
    image = read_image_to_ndarray(str(dataset / 'samples' / ((image_id or row['id']) + '.png')))
    image = prepare_for_tensor(add_image_into_tr_omr_canvas(image))
    batch = {name: getattr(branches, name)[:length].unsqueeze(0)
             for name in ('rhythms', 'pitchs', 'lifts', 'positions', 'articulations', 'slurs', 'mask')}
    batch['inputs'] = pad_to_3_dims(ndarray_to_tensor(image)).unsqueeze(0)
    return dict(batch=batch, targets={name: value[:, 1:] for name, value in targets.items()
                                     if name != 'dynamic.mark'})


def extract(model: nn.Module, dataset: Path, rows: list[dict], out: Path, device: str) -> list[dict]:
    model.eval()
    controls = image_controls(rows)
    records = []
    started = time.monotonic()
    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            item = clean_sample(dataset, row)
            batch = {k: v.to(device) for k, v in item['batch'].items()}
            result = model(**batch)
            hidden = result['hidden'].detach().cpu()[0]
            assert all(value.shape[1] == hidden.shape[0] for value in item['targets'].values())
            # Pin the cache to the production head input, not an intermediate layer.
            replay = model.decoder.structured_heads(result['hidden'])
            assert all(torch.equal(replay[name], value) for name, value in result['structured_logits'].items())
            record = dict(id=row['id'], family=row['family'], split=row['split'], kind=row['kind'],
                          hidden=hidden, targets={k: v[0] for k, v in item['targets'].items()})
            if row['id'] in controls:
                swapped = clean_sample(dataset, row, controls[row['id']])
                swap_batch = {k: v.to(device) for k, v in swapped['batch'].items()}
                assert all(torch.equal(batch[k], swap_batch[k]) for k in batch if k != 'inputs')
                record['swapped_hidden'] = model(**swap_batch)['hidden'].detach().cpu()[0]
                record['image_control_id'] = controls[row['id']]
            records.append(record)
            if i == 1 or i % 32 == 0 or i == len(rows):
                elapsed = time.monotonic() - started
                progress = dict(phase='features', completed=i, total=len(rows), elapsed_seconds=elapsed,
                                remaining_seconds=elapsed / i * (len(rows) - i))
                write_json(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    # Controls are measured on the positions whose labels differ between token twins.
    by_id = {r['id']: r for r in records}
    for record in records:
        if 'image_control_id' in record:
            other = by_id[record['image_control_id']]
            record['contrast_masks'] = {name: (target != other['targets'][name]) &
                                       (target != IGNORE_INDEX) & (other['targets'][name] != IGNORE_INDEX)
                                       for name, target in record['targets'].items()}
    return records


def projection(heads: nn.Module, name: str) -> nn.Module:
    if name.startswith('beam.level.'):
        return heads.beam[int(name.split('.')[-1]) - 1]
    if name.startswith('slur.slot.'):
        slot, kind = name.split('.')[2:]
        return (heads.slur_event if kind == 'event' else heads.slur_side)[int(slot) - 1]
    return {'stem.direction': heads.stem, 'tie.state': heads.tie, 'advance.delta': heads.advance}[name]


def pools_for(records: list[dict], names: list[str], device: str) -> dict:
    pools = {}
    for name in names:
        pieces, labels, families = [], [], []
        for record in records:
            if record['split'] != 'train':
                continue
            target = record['targets'][name]
            keep = target != IGNORE_INDEX
            pieces.append(record['hidden'][keep])
            labels.append(target[keep])
            families.extend([record['family']] * int(keep.sum()))
        y = torch.cat(labels).to(device)
        if y.numel():
            pools[name] = dict(x=torch.cat(pieces).to(device), y=y,
                               classes={int(c): (y == c).nonzero().flatten() for c in y.unique()},
                               families=families)
    return pools


def balanced_indices(pool: dict, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    classes = list(pool['classes'].values())
    choices = torch.randint(len(classes), (batch_size,), generator=generator)
    result = torch.empty(batch_size, dtype=torch.long, device=pool['y'].device)
    for i, candidates in enumerate(classes):
        selected = (choices == i).nonzero().flatten()
        draws = torch.randint(len(candidates), (len(selected),), generator=generator)
        result[selected.to(result.device)] = candidates[draws.to(candidates.device)]
    return result


def class_report(predicted: torch.Tensor, actual: torch.Tensor, size: int) -> dict:
    keep = actual != IGNORE_INDEX
    predicted, actual = predicted[keep].cpu(), actual[keep].cpu()
    rows = {}
    for c in range(size):
        tp = int(((predicted == c) & (actual == c)).sum())
        fp = int(((predicted == c) & (actual != c)).sum())
        fn = int(((predicted != c) & (actual == c)).sum())
        rows[str(c)] = dict(support=tp + fn, tp=tp, fp=fp, fn=fn,
                            precision=tp / (tp + fp) if tp + fp else 0,
                            recall=tp / (tp + fn) if tp + fn else 0,
                            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    supported = [r['f1'] for r in rows.values() if r['support']]
    return dict(classes=rows, macro_f1=sum(supported) / len(supported) if supported else None,
                support=len(actual), accuracy=float((predicted == actual).float().mean()) if len(actual) else None)


@torch.no_grad()
def evaluate_cache(heads: nn.Module, records: list[dict], names: list[str], device: str) -> dict:
    from training.architecture.transformer.structured_decoding import decode_predictions, decode_reference
    from training.transformer.structured_metrics import Evaluation
    results = {}
    for split in ('train', 'valid', 'test'):
        raw = defaultdict(lambda: [[], []])
        contrasts = defaultdict(lambda: [[], [], []])
        evaluation = Evaluation(4, 2)
        simple_spans = Evaluation(4, 2)
        sizes = {}
        for record in records:
            if record['split'] != split:
                continue
            hidden = record['hidden'].to(device)
            logits = {n: projection(heads, n)(hidden).unsqueeze(0).cpu() for n in names}
            targets = {n: record['targets'][n].unsqueeze(0) for n in names}
            got, want = decode_predictions(logits, targets, 4, 2)[0], decode_reference(targets, 4, 2)[0]
            evaluation.observe(got, want)
            if record['kind'] == 'slurs': simple_spans.observe(got, want)
            for n in names:
                raw[n][0].append(logits[n].argmax(-1)[0]); raw[n][1].append(targets[n][0])
                sizes[n] = logits[n].shape[-1]
            if 'swapped_hidden' in record:
                swapped = record['swapped_hidden'].to(device)
                for n in names:
                    keep = record['contrast_masks'][n]
                    if keep.any():
                        contrasts[n][0].append(logits[n].argmax(-1)[0][keep])
                        contrasts[n][1].append(projection(heads, n)(swapped).argmax(-1).cpu()[keep])
                        contrasts[n][2].append(targets[n][0][keep])
        classes = {n: class_report(torch.cat(v[0]), torch.cat(v[1]), sizes[n]) for n, v in raw.items()}
        controls = {n: dict(normal=class_report(torch.cat(v[0]), torch.cat(v[2]), sizes[n]),
                            swapped=class_report(torch.cat(v[1]), torch.cat(v[2]), sizes[n]))
                    for n, v in contrasts.items() if v[0]}
        spans = simple_spans.slur_spans
        results[split] = dict(heads=classes, metrics=evaluation.to_dict(),
                              simple_slur_spans=dict(f1=spans.f1, support=spans.support,
                                                    tp=spans.true_positive, fp=spans.false_positive, fn=spans.false_negative),
                              matched_token_image_controls=controls)
    return results


def fit_step(heads: nn.Module, pools: dict, optimizer: torch.optim.Optimizer,
             batch_size: int, generator: torch.Generator) -> float:
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for name, pool in pools.items():
        selected = balanced_indices(pool, batch_size, generator)
        loss = torch.nn.functional.cross_entropy(projection(heads, name)(pool['x'][selected]), pool['y'][selected])
        losses.append(loss)
    total = torch.stack(losses).sum()
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for name in pools for p in projection(heads, name).parameters())
    optimizer.step()
    return float(total.detach())


def save_state(path: Path, heads: nn.Module, optimizer: torch.optim.Optimizer,
               generator: torch.Generator, step: int, identity: dict) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(dict(heads=heads.state_dict(), optimizer=optimizer.state_dict(),
                    generator=generator.get_state(), step=step, identity=identity), temporary)
    temporary.replace(path)


def load_state(path: Path, heads: nn.Module, optimizer: torch.optim.Optimizer,
               generator: torch.Generator, identity: dict) -> int:
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['identity'] != identity:
        raise ValueError('Resume identity does not match data/core/probe settings')
    heads.load_state_dict(state['heads'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    generator.set_state(state['generator'])
    return state['step']


def gate_report(result: dict, pools: dict) -> dict:
    eligible, blocked, scores = [], [], {}
    for name, pool in pools.items():
        supports = {}
        for c, positions in pool['classes'].items():
            supports[str(c)] = dict(notes=len(positions), families=len({pool['families'][i] for i in positions.cpu().tolist()}))
        supported = len(supports) >= 2 and all(v['notes'] >= 20 and v['families'] >= 4 for v in supports.values())
        f1 = result['train']['heads'][name]['macro_f1']
        scores[name] = dict(class_support=supports, eligible=supported, train_macro_f1=f1,
                            passes=supported and f1 >= .95)
        (eligible if supported else blocked).append(name)
    spans = result['train']['simple_slur_spans']
    return dict(heads=scores, insufficient_support=blocked, eligible_heads=eligible,
                simple_slur_spans=spans, capacity_pass=bool(eligible) and not blocked and
                all(scores[n]['passes'] for n in eligible) and spans['support'] >= 20 and spans['f1'] >= .90,
                scope='Diagnostic synthetic fit only; no real-source improvement or release claim')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--approval', required=True, type=Path)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--supplement-approval', type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=128, help='Balanced positions per head per update')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every) < 1 or args.lr <= 0:
        parser.error('Positive steps, batch size, evaluation interval and learning rate required')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()) and args.resume is None:
        raise ValueError('Choose an empty output directory or explicitly resume')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR
    from training.transformer.train_structured_heads import load_pinned
    from training.architecture.transformer.structured_heads import StructuredNotationHeads
    from training.transformer import image_utils, training_vocabulary
    rows = qualify(args.dataset, args.approval)
    for row in rows: row['_dataset'] = str(args.dataset.resolve())
    if bool(args.supplement) != bool(args.supplement_approval):
        parser.error('Supplement requires its explicit approval manifest')
    if args.supplement:
        extra = qualify(args.supplement, args.supplement_approval)
        for row in extra: row['_dataset'] = str(args.supplement.resolve())
        if set(r['id'] for r in rows) & set(r['id'] for r in extra):
            raise ValueError('Duplicate sample identity across datasets')
        original = {r['family']: r for r in rows}
        for row in extra:
            if row['family'] in original and (row['split'], row['seed']) != (original[row['family']]['split'], original[row['family']]['seed']):
                raise ValueError('Supplement changed family split or musical seed')
        rows.extend(extra)
    config = Config()
    config.enable_structured_heads = True
    model = TrOMR(config)
    load_pinned(model, args.checkpoint)
    for p in model.parameters(): p.requires_grad_(False)
    model.to(args.device).eval()
    original_core_hash = core_hash(model)
    identity = dict(schema='synthetic-capacity.v1', checkpoint_sha256=sha(args.checkpoint),
                    dataset_sha256=sha(args.dataset / 'SHA256SUMS'), approval_sha256=sha(args.approval),
                    supplement_sha256=sha(args.supplement / 'SHA256SUMS') if args.supplement else None,
                    supplement_approval_sha256=sha(args.supplement_approval) if args.supplement else None,
                    script_sha256=sha(Path(__file__)), seed=args.seed, lr=args.lr, batch_size=args.batch_size,
                    optimizer='Adam', balanced_class_sampling=True, frozen_core=True, dynamics_excluded=True,
                    image_augmentation=False, teacher_forced=True, training_gate_macro_f1=.95,
                    simple_span_gate_f1=.90, class_min_notes=20, class_min_families=4,
                    torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                    source_hashes={str(p): sha(p) for p in (Path(image_utils.__file__),
                                  Path(training_vocabulary.__file__), Path(__file__).parents[2] / 'homr/transformer/configs.py')})
    write_json(args.out / 'identity.json', identity)
    write_json(args.out / 'qualification.json', dict(approval=json.loads(args.approval.read_text()),
               dataset_sha256=identity['dataset_sha256'], renders=len(rows),
               training_ready=True, scope='synthetic-capacity-only', source_dataset_unchanged=True))
    cache = args.out / 'features.pt'
    if cache.exists():
        cached = torch.load(cache, map_location='cpu', weights_only=True)
        if cached['identity'] != identity: raise ValueError('Cached feature identity mismatch')
        records = cached['records']
    else:
        records = extract(model, args.dataset, rows, args.out, args.device)
        torch.save(dict(identity=identity, records=records), cache)
    fixed = clean_sample(args.dataset, rows[0])
    fixed_batch = {k: v.to(args.device) for k, v in fixed['batch'].items()}
    with torch.no_grad(): baseline_logits = tuple(v.clone() for v in model(**fixed_batch)['logits'])
    torch.manual_seed(args.seed)
    heads = StructuredNotationHeads(config.decoder_dim, 4, 2).to(args.device)
    names = [n for n in heads.head_names() if n != 'dynamic.mark']
    pools = pools_for(records, names, args.device)
    names = list(pools)
    for p in heads.parameters(): p.requires_grad_(False)
    for name in names:
        for p in projection(heads, name).parameters(): p.requires_grad_(True)
    optimizer = torch.optim.Adam([p for p in heads.parameters() if p.requires_grad], lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    initial = {k: v.clone() for k, v in heads.state_dict().items()}
    step = load_state(args.resume, heads, optimizer, generator, identity) if args.resume else 0
    started = time.monotonic()
    start_step = step
    while step < args.steps:
        loss = fit_step(heads, pools, optimizer, args.batch_size, generator)
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            save_state(args.out / 'resume.pt', heads, optimizer, generator, step, identity)
            result = evaluate_cache(heads, records, names, args.device)
            gate = gate_report(result, pools)
            write_json(args.out / f'eval-step-{step}.json', dict(step=step, metrics=result, gate=gate))
            elapsed = time.monotonic() - started
            progress = dict(phase='linear-fit', step=step, steps=args.steps, loss=loss,
                            elapsed_seconds=elapsed, remaining_seconds=elapsed / max(1, step-start_step) * (args.steps-step),
                            capacity_pass=gate['capacity_pass'])
            write_json(args.out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    assert all(torch.isfinite(v).all() for v in heads.state_dict().values())
    assert len(optimizer.state) == 2 * len(names) and all(float(v['step']) > 0 for v in optimizer.state.values())
    assert any(not torch.equal(initial[k], v) for k, v in heads.state_dict().items())
    assert core_hash(model) == original_core_hash
    with torch.no_grad(): final_logits = model(**fixed_batch)['logits']
    assert all(torch.equal(a, b) for a, b in zip(baseline_logits, final_logits, strict=True))
    result = evaluate_cache(heads, records, names, args.device)
    gate = gate_report(result, pools)
    torch.save(heads.state_dict(), args.out / 'heads.pth')
    write_json(args.out / 'report.json', dict(identity=identity, steps=step, metrics=result, gate=gate,
               guards=dict(core_tensor_sha256=original_core_hash, frozen_core_unchanged=True,
                           recognition_logits_bit_identical=True, active_heads=names,
                           optimizer_tensors=len(optimizer.state))))
    lines = ['# Synthetic head capacity probe', '',
             'Diagnostic teacher-forced synthetic fit; no real-source/release claim.', '',
             '| Head | Train macro-F1 | Family holdout macro-F1 | Capacity bar |',
             '|---|---:|---:|---|']
    for name in names:
        train = result['train']['heads'][name]['macro_f1']
        valid = result['valid']['heads'].get(name, {}).get('macro_f1')
        verdict = 'insufficient support' if not gate['heads'][name]['eligible'] else ('pass' if gate['heads'][name]['passes'] else 'fail')
        lines.append(f'| {name} | {train:.3f} | {valid:.3f} | {verdict} |' if valid is not None else f'| {name} | {train:.3f} | n/a | {verdict} |')
    lines += ['', f"Simple training slur-span F1: {gate['simple_slur_spans']['f1']:.3f}.",
              f"Overall capacity gate: {gate['capacity_pass']}.",
              'Image-swap controls, per-class support and gaps are in report.json.',
              'Level-1 hooks and flags are absent; inspect per-class support for remaining coverage gaps.']
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    files = sorted(p for p in args.out.iterdir() if p.is_file() and p.name not in ('SHA256SUMS', 'COMPLETE'))
    (args.out / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files))
    (args.out / 'COMPLETE').write_text('complete\n')


if __name__ == '__main__':
    main()
