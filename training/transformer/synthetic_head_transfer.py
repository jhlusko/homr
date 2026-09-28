"""Matched fixed-pixel pilot transfer: real-only, shuffled mixture, staged mixture.

This uses qualified clean/font/context motifs, not the richer proposed curriculum.
No synthetic degradation or backbone adaptation is introduced in this pilot.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset

from training.architecture.transformer.structured_losses import IGNORE_INDEX, structured_loss
from training.transformer.synthetic_head_capacity import core_hash, projection, qualify, sha, write_json

STAGES = ((1000, .75), (2000, .5), (2000, .25), (3000, .1))
SOURCES = ('ossq', 'pdmx', 'lieder')
ARMS = ('real_only', 'unordered', 'curriculum')


def read_index(path: Path) -> list[dict]:
    root = Path(__file__).resolve().parents[2]
    records = []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        image, tokens = line.split(',')
        image, tokens = root / image, root / tokens
        source = 'ossq' if '/scanned/' in str(tokens) else 'lieder' if '/pairs/' in str(tokens) else 'pdmx'
        if not image.is_file() or not tokens.is_file() or not Path(str(tokens)+'.notation.json').is_file():
            raise ValueError(f'Missing image/token/sidecar: {tokens}')
        records.append(dict(image=str(image), tokens=str(tokens), source=source, id=str(tokens)))
    return records


def quotas(total: int, sizes: dict[str, int]) -> dict[str, int]:
    denominator = sum(sizes.values())
    counts = {k: total*v//denominator for k,v in sizes.items()}
    order = sorted(sizes, key=lambda k: (-(total*sizes[k] % denominator), k))
    for k in order[:total-sum(counts.values())]: counts[k] += 1
    return counts


def draw_real(rows: list[dict], count: int, rng: random.Random) -> list[int]:
    pools = {s: [i for i,r in enumerate(rows) if r['source']==s] for s in SOURCES}
    if any(not pool for pool in pools.values()): raise ValueError('All three real sources are required')
    selected = []
    for source,n in quotas(count,{s:len(p) for s,p in pools.items()}).items():
        while n:
            cycle = pools[source].copy(); rng.shuffle(cycle); cycle = cycle[:n]
            selected.extend(cycle); n -= len(cycle)
    rng.shuffle(selected)
    return selected


def draw_synthetic(rows: list[dict], count: int, stage: int, rng: random.Random) -> list[int]:
    groups = defaultdict(list)
    for i,row in enumerate(rows):
        if row['split']!='train': continue
        if stage==0 and not (row['variant']<2 or row['kind']=='slurs' and row['variant']==2): continue
        if stage==1 and row['variant']<2: continue
        if stage==2 and row['font']!='Bravura': continue
        groups[row['family']].append(i)
    if not groups: raise ValueError('Empty synthetic stage')
    selected = []
    while len(selected)<count:
        families = sorted(groups); rng.shuffle(families)
        selected.extend(rng.choice(groups[f]) for f in families[:count-len(selected)])
    return selected


def make_schedules(real: list[dict], synthetic: list[dict], seed: int,
                   stages=STAGES, batch_size: int=8) -> dict[str,list[dict]]:
    rng = random.Random(seed)
    ordered = []
    for stage,(updates,ratio) in enumerate(stages):
        draws = updates*batch_size
        synthetic_count = round(draws*ratio)
        choices = [('synthetic',i) for i in draw_synthetic(synthetic,synthetic_count,stage,rng)]
        choices += [('real',i) for i in draw_real(real,draws-synthetic_count,rng)]
        rng.shuffle(choices)
        for kind,index in choices:
            ordered.append(dict(draw_id=len(ordered), kind=kind, index=index, stage=stage,
                                pixel_seed=rng.randrange(2**32)))
    unordered = ordered.copy(); random.Random(seed+1).shuffle(unordered)
    control = []
    indices = draw_real(real,len(ordered),random.Random(seed))
    for i,index in enumerate(indices):
        control.append(dict(draw_id=i,kind='real',index=index,stage=ordered[i]['stage'],pixel_seed=ordered[i]['pixel_seed']))
    return dict(real_only=control,unordered=unordered,curriculum=ordered)


def item_for(row: dict) -> dict:
    from homr.staff_parsing import add_image_into_tr_omr_canvas
    from training.transformer.image_utils import read_image_to_ndarray, prepare_for_tensor, ndarray_to_tensor, pad_to_3_dims
    from training.transformer.training_vocabulary import read_tokens, to_decoder_branches
    from training.omr_datasets.notation_sidecar import attach_sidecar
    from training.architecture.transformer.structured_targets import build_targets, notation_positions
    symbols = read_tokens(row['tokens'])
    if not attach_sidecar(row['tokens'],symbols): raise ValueError('Unsupervised pilot row')
    from homr.transformer.configs import default_config
    length = len(symbols)+2
    if length>default_config.max_seq_len: raise ValueError('Overlength training example')
    branches = to_decoder_branches(symbols)
    result = {k:getattr(branches,k)[:length] for k in ('rhythms','pitchs','lifts','positions','articulations','slurs','mask')}
    image = read_image_to_ndarray(row['image'])
    if image.shape[0]<1 or image.shape[1]<1: raise ValueError('Empty image')
    image = prepare_for_tensor(add_image_into_tr_omr_canvas(image))
    result['inputs'] = pad_to_3_dims(ndarray_to_tensor(image))
    result.update({name:value[0] for name,value in build_targets([notation_positions(symbols,length)],4,2).items()
                   if name!='dynamic.mark'})
    return result


class DrawDataset(Dataset):
    def __init__(self,real,synthetic,schedule): self.real,self.synthetic,self.schedule=real,synthetic,schedule
    def __len__(self): return len(self.schedule)
    def __getitem__(self,index):
        draw = self.schedule[index]
        row = (self.real if draw['kind']=='real' else self.synthetic)[draw['index']]
        return item_for(row)


def collate_fixed(items: list[dict]) -> dict:
    length = max(len(item['rhythms']) for item in items)
    result = {}
    for name in items[0]:
        if name=='inputs': result[name]=torch.stack([item[name] for item in items]); continue
        fill = IGNORE_INDEX if '.' in name else 0
        result[name]=torch.stack([torch.nn.functional.pad(item[name],(0,length-len(item[name])),value=fill)
                                 for item in items])
    return result


def loader(real,synthetic,schedule,batch_size,workers):
    from training.transformer.train_structured_heads import _limit_worker_threads
    return DataLoader(DrawDataset(real,synthetic,schedule),batch_size=batch_size,shuffle=False,
                      num_workers=workers,worker_init_fn=_limit_worker_threads if workers else None,
                      pin_memory=True,collate_fn=collate_fixed)


def move(raw: dict,device: str,names: list[str]) -> tuple[dict,dict]:
    targets = {k:v[:,1:].to(device,non_blocking=True) for k,v in raw.items() if k in names}
    batch = {k:v.to(device,non_blocking=True) for k,v in raw.items() if k not in names}
    return batch,targets


def class_details(report) -> dict:
    return {name:dict(support=m.support,tp=m.true_positive,fp=m.false_positive,fn=m.false_negative,
                      precision=m.precision,recall=m.recall,f1=m.f1) for name,m in report.classes.items()}


@torch.no_grad()
def score_source(model,rows,names,path,workers,device):
    from training.architecture.transformer.structured_decoding import decode_predictions,decode_reference
    from training.transformer.structured_metrics import Evaluation
    evaluation = Evaluation(4,2)
    schedule = [dict(kind='real',index=i) for i in range(len(rows))]
    position = 0
    reference_hash = hashlib.sha256()
    pixels_hash = hashlib.sha256()
    with path.with_suffix('.jsonl').open('w') as handle:
        for raw in loader(rows,[],schedule,8,workers):
            batch,targets = move(raw,device,names)
            logits = model(**batch)['structured_logits']
            got = decode_predictions(logits,targets,4,2)
            want = decode_reference(targets,4,2)
            for i,(predicted,actual) in enumerate(zip(got,want,strict=True)):
                evaluation.observe(predicted,actual)
                reference = {k:targets[k][i].cpu().tolist() for k in names}
                predicted_indices = {k:logits[k][i].argmax(-1).cpu().tolist() for k in names}
                # Remove collate-dependent padding from paired identities.
                length = int(raw['mask'][i].sum())-1
                reference = {k:v[:length] for k,v in reference.items()}
                predicted_indices = {k:v[:length] for k,v in predicted_indices.items()}
                encoded = json.dumps(reference,sort_keys=True).encode()
                reference_hash.update(rows[position]['id'].encode()); reference_hash.update(encoded)
                pixel_sha = hashlib.sha256(raw['inputs'][i].contiguous().numpy().tobytes()).hexdigest()
                pixels_hash.update(rows[position]['id'].encode()); pixels_hash.update(pixel_sha.encode())
                handle.write(json.dumps(dict(id=rows[position]['id'], pixel_sha256=pixel_sha, reference_sha256=hashlib.sha256(encoded).hexdigest(),
                                             reference=reference,predicted=predicted_indices))+'\n')
                position += 1
    assert position==len(rows)
    report = evaluation.to_dict()
    report['directional_stem_support']=sum(m.support for n,m in evaluation.stems.classes.items() if n in ('up','down'))
    report['class_details']=dict(beams={str(n):class_details(v) for n,v in evaluation.per_level.items()},
                                 stems=class_details(evaluation.stems),ties=class_details(evaluation.ties),
                                 slur_sides=class_details(evaluation.slur_sides))
    m=evaluation.slur_spans
    report['slur_span_counts']=dict(tp=m.true_positive,fp=m.false_positive,fn=m.false_negative,
                                    precision=m.precision,recall=m.recall,f1=m.f1,support=m.support)
    report['reference_sha256']=reference_hash.hexdigest()
    report['pixels_sha256']=pixels_hash.hexdigest()
    write_json(path,report)
    return report


def cells(report: dict) -> dict:
    result = {f'beam{k}':(v['macro_f1'],v['support']) for k,v in report['beam_levels'].items()}
    for name,key in (('hooks','f1'),('ties','macro_f1'),('slur_spans','f1'),('slur_sides','macro_f1'),('advance_nontrivial','macro_f1')):
        v=report[name]; result[name]=(v[key],v['support'])
    for name,key in (('stem_direction','direction_only'),('stem_macro','macro_f1')):
        result[name]=(report['stems'][key],report['directional_stem_support'])
    v=report['exact_beam_vector'];result['exact_beam_vector']=(v['rate'],v['total'])
    return result


def compare(control: dict,candidate: dict) -> dict:
    deltas,guards={},[]
    for source in SOURCES:
        a,b=control[source],candidate[source]
        if a['reference_sha256']!=b['reference_sha256'] or a['pixels_sha256']!=b['pixels_sha256']:
            raise ValueError('Unmatched evaluation references or pixels')
        ac,bc=cells(a),cells(b)
        deltas[source]={name:dict(delta_pp=100*(v-ac[name][0]),support=n,supported=n>=100 and ac[name][1]>=100)
                        for name,(v,n) in bc.items()}
        for group in ('stems','ties','slur_sides'):
            for label,want in a['class_details'][group].items():
                got=b['class_details'][group].get(label,dict(support=0,recall=0))
                if label!='none' and want['support']>=100 and got['support']>=100:
                    guards.append(dict(source=source,metric=f'{group}.{label}.recall',
                                       delta_pp=100*(got['recall']-want['recall']),passes=got['recall']>=want['recall']-.01))
        for level,rows in a['class_details']['beams'].items():
            for label in ('forward_hook','backward_hook'):
                want=rows.get(label,dict(support=0));got=b['class_details']['beams'][level].get(label,dict(support=0))
                if min(want['support'],got['support'])>=100:
                    guards.append(dict(source=source,metric=f'beam{level}.{label}.recall',
                                       delta_pp=100*(got['recall']-want['recall']),passes=got['recall']>=want['recall']-.01))
        x,y=a['slur_span_counts'],b['slur_span_counts']
        if min(x['support'],y['support'])>=100:
            delta=y['fp']/y['support']-x['fp']/x['support']
            guards.append(dict(source=source,metric='slur_inventions_per_reference',delta_pp=100*delta,passes=delta<=.01))
    supported=[c['delta_pp'] for src in deltas.values() for c in src.values() if c['supported']]
    return dict(by_source=deltas,guardrails=guards,worst_delta_pp=min(supported),best_delta_pp=max(supported),
                eligible=min(supported)>=-1 and max(supported)>=1 and all(g['passes'] for g in guards))


def analyze(run: Path) -> None:
    reports={arm:{s:json.loads((run/arm/'eval'/f'{s}.json').read_text()) for s in SOURCES} for arm in ARMS}
    metadata={arm:json.loads((run/arm/'identity.json').read_text()) for arm in ARMS}
    for key in ('checkpoint_sha256','plan_sha256','initial_heads_sha256','source_sha256','lr','pixel_policy',
                'preprocessing_sha256','vocabulary_sha256','sidecar_reader_sha256','torch_version','cuda_version','runtime_source_sha256'):
        assert len({json.dumps(v[key],sort_keys=True) for v in metadata.values()})==1,key
    contrasts={arm:compare(reports['real_only'],reports[arm]) for arm in ARMS[1:]}
    ordering=compare(reports['unordered'],reports['curriculum'])
    selected='curriculum' if contrasts['curriculum']['eligible'] and ordering['eligible'] else 'unordered' if contrasts['unordered']['eligible'] else None
    summary=dict(contrasts=contrasts,ordering=ordering,selected=selected,
                 scope='One-seed, fixed-pixel pilot internal transfer; no release or robust curriculum claim',reports=reports)
    write_json(run/'report.json',summary)
    lines=['# Fixed-pixel synthetic pilot transfer — 2026-09-28','',
           'Matched 8,000-update fresh-head arms; source-grouped internal validation only.',
           'This pilot covers qualified motifs/fonts, not the full richer/degraded curriculum.','',
           '| Arm | Worst supported change vs real-only | Passes all gates |','|---|---:|---|']
    lines += [f"| {a} | {c['worst_delta_pp']:+.2f} pp | {c['eligible']} |" for a,c in contrasts.items()]
    lines += ['',f"Selected for further confirmation: **{selected or 'none'}**.",
              f"Ordering advantage eligible versus unordered: {ordering['eligible']}.",
              'Per-source metrics, directional recalls and inventions are in report.json.',
              'Published validation, full-data confirmation, seed repeat and deployment are not performed.']
    (run/'report.md').write_text('\n'.join(lines)+'\n')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--analyze',action='store_true')
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--real-index',type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--approval',type=Path)
    parser.add_argument('--supplement',type=Path)
    parser.add_argument('--supplement-approval',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--capacity-report',type=Path)
    parser.add_argument('--validation-dir',type=Path)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int,default=20260928)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--skip-eval',action='store_true')
    args=parser.parse_args()
    if args.workers < 0 or args.max_steps is not None and args.max_steps < 1:
        parser.error('Invalid worker/update count')
    torch.set_num_threads(1)
    if args.analyze: analyze(args.run);return
    if args.prepare:
        if args.run.exists() and any(args.run.iterdir()): raise ValueError('Plan output must be empty')
        args.run.mkdir(parents=True,exist_ok=True)
        capacity=json.loads(args.capacity_report.read_text())
        if not capacity['gate']['capacity_pass']:raise ValueError('Capacity gate has not passed')
        real=read_index(args.real_index)
        if sha(args.real_index)!='f8c7f272b0fddd2397c5db169698a28e1ebaca82f6cd7a8a2baba5e7441202ca':
            raise ValueError('Real training index differs from the frozen internal split')
        if len(real)!=40290 or Counter(r['source'] for r in real)!=Counter(dict(ossq=31496,pdmx=5701,lieder=3093)):
            raise ValueError('Unexpected real-source pool')
        synthetic=[]
        for dataset,approval in ((args.dataset,args.approval),(args.supplement,args.supplement_approval)):
            rows=qualify(dataset,approval)
            synthetic.extend(dict(r,image=str(dataset/'samples'/f"{r['id']}.png"),
                                  tokens=str(dataset/'samples'/f"{r['id']}.tokens")) for r in rows)
        original={}
        for r in synthetic:
            if r['family'] in original and original[r['family']]!=(r['seed'],r['split']): raise ValueError('Family identity mismatch')
            original[r['family']]=(r['seed'],r['split'])
        schedules=make_schedules(real,synthetic,args.seed)
        identity=dict(capacity_report_sha256=sha(args.capacity_report),seed=args.seed,stages=STAGES,batch_size=8,updates=8000,real_index_sha256=sha(args.real_index),
                      real_rows=len(real),real_sources=dict(Counter(r['source'] for r in real)),
                      dataset_sha256=sha(args.dataset/'SHA256SUMS'),supplement_sha256=sha(args.supplement/'SHA256SUMS'),
                      pixel_policy='fixed-original-canvas-normalization-no-augmentation',
                      synthetic_stage_pools='simple; context; Bravura; complete qualified replay')
        for arm,draws in schedules.items():
            identity[arm]=dict(draws=len(draws),source_counts=dict(Counter(real[d['index']]['source'] if d['kind']=='real' else 'synthetic' for d in draws)))
        write_json(args.run/'plan.json',dict(identity=identity,real=real,synthetic=synthetic,schedules=schedules))
        return
    plan=json.loads((args.run/'plan.json').read_text())
    if not args.arm or not args.checkpoint or not args.validation_dir: parser.error('Training requires arm/checkpoint/validation-dir')
    out=args.run/args.arm;out.mkdir(exist_ok=True)
    if any(out.iterdir()) and not args.resume: raise ValueError('Choose fresh arm output or resume explicitly')
    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR
    from training.transformer.train_structured_heads import load_pinned,write_manifest
    from training.architecture.transformer.structured_heads import StructuredNotationHeads
    from training.transformer import image_utils,training_vocabulary
    from training.omr_datasets import notation_sidecar
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(args.seed)
    config=Config();config.enable_structured_heads=True
    model=TrOMR(config);load_pinned(model,args.checkpoint)
    # Core loading may include old projections: discard them to guarantee fresh matched initialization.
    torch.manual_seed(args.seed)
    model.decoder.structured_heads=StructuredNotationHeads(config.decoder_dim,4,2)
    model.freeze_core_for_structured_heads()
    for p in model.decoder.structured_heads.dynamic.parameters():p.requires_grad_(False)
    names=[n for n in model.decoder.structured_heads.head_names() if n!='dynamic.mark']
    model.to('cuda').eval()
    heads=model.decoder.structured_heads
    digest=hashlib.sha256()
    for k,v in sorted(heads.state_dict().items()):digest.update(k.encode());digest.update(v.detach().cpu().numpy().tobytes())
    source_root=Path(__file__).resolve().parents[2]
    runtime_files=sorted((source_root/'homr/transformer').glob('*.py'))+sorted((source_root/'training/architecture/transformer').glob('*.py'))
    runtime_hash=hashlib.sha256()
    for path in runtime_files:
        runtime_hash.update(str(path.relative_to(source_root)).encode());runtime_hash.update(sha(path).encode())
    identity=dict(runtime_source_sha256=runtime_hash.hexdigest(),checkpoint_sha256=sha(args.checkpoint),plan_sha256=sha(args.run/'plan.json'),
                  source_sha256=sha(Path(__file__)),initial_heads_sha256=digest.hexdigest(),arm=args.arm,
                  lr=.001,batch_size=8,pixel_policy=plan['identity']['pixel_policy'],seed=args.seed,
                  torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                  preprocessing_sha256=sha(Path(image_utils.__file__)),vocabulary_sha256=sha(Path(training_vocabulary.__file__)),
                  sidecar_reader_sha256=sha(Path(notation_sidecar.__file__)),
                  tf32=False,deterministic_cudnn=True)
    write_json(out/'identity.json',identity)
    frozen_hash=core_hash(model)
    fixed=collate_fixed([item_for(plan['real'][0])]);fixed_batch,_=move(fixed,'cuda',names)
    with torch.no_grad(): fixed_logits=tuple(v.clone() for v in model(**fixed_batch)['logits'])
    initial_projections={n:{k:v.clone() for k,v in projection(heads,n).state_dict().items()} for n in names}
    initial_dynamic={k:v.clone() for k,v in heads.dynamic.state_dict().items()}
    optimizer=torch.optim.Adam([p for p in heads.parameters() if p.requires_grad],lr=.001)
    state_path=out/'resume.pt'
    completed=0;support=Counter()
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=True)
        if state['identity']!=identity:raise ValueError('Resume identity changed')
        heads.load_state_dict(state['heads']);optimizer.load_state_dict(state['optimizer'])
        completed=state['completed'];support.update(state['support'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    steps=args.max_steps or plan['identity']['updates']
    if completed>steps:raise ValueError('Checkpoint already beyond requested update budget')
    draws=plan['schedules'][args.arm][completed*8:steps*8]
    boundaries={1000,3000,5000,8000}
    started=time.monotonic();loss_total=0
    for offset,raw in enumerate(loader(plan['real'],plan['synthetic'],draws,8,args.workers),1):
        batch,targets=move(raw,'cuda',names)
        outputs=model(**batch)
        result=structured_loss(outputs['structured_logits'],targets)
        if not result.supervised_heads():raise ValueError('Unsupervised draw batch')
        optimizer.zero_grad(set_to_none=True);result.total.backward()
        assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('decoder.structured_heads.'))
        assert heads.dynamic.weight.grad is None and heads.dynamic.bias.grad is None
        assert all(torch.isfinite(p.grad).all() for p in heads.parameters() if p.grad is not None)
        optimizer.step();loss_total+=float(result.total.detach())
        for head in result.heads:support[head.name]+=head.supervised
        step=completed+offset
        if step%200==0 or step in boundaries or step==steps:
            temporary=state_path.with_suffix('.tmp')
            torch.save(dict(identity=identity,heads=heads.state_dict(),optimizer=optimizer.state_dict(),
                            completed=step,support=dict(support),torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all()),temporary);temporary.replace(state_path)
            if step in boundaries:torch.save(heads.state_dict(),out/f'heads-step-{step}.pth')
            elapsed=time.monotonic()-started
            progress=dict(arm=args.arm,step=step,steps=steps,mean_loss=loss_total/offset,elapsed_seconds=elapsed,
                          remaining_seconds=elapsed/offset*(steps-step),support=dict(support))
            write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert all(torch.isfinite(v).all() for v in heads.state_dict().values())
    assert core_hash(model)==frozen_hash
    with torch.no_grad():final_logits=model(**fixed_batch)['logits']
    assert all(torch.equal(a,b) for a,b in zip(fixed_logits,final_logits,strict=True))
    assert all(torch.equal(initial_dynamic[k],v) for k,v in heads.dynamic.state_dict().items())
    if not args.max_steps:
        assert len(optimizer.state)==22 and all(support[n]>0 for n in names)
        assert all(any(not torch.equal(initial_projections[n][k],v) for k,v in projection(heads,n).state_dict().items()) for n in names)
    torch.save(heads.state_dict(),out/'heads.pth')
    write_manifest(out/'manifest.json',config,tuple(n for n in names if support[n]),
                   model_revision='public-rareNum-20260927 sha256:'+identity['checkpoint_sha256'],
                   training_revision='synthetic-fixed-pixel-pilot-20260928 '+identity['source_sha256'],run_id=args.arm)
    write_json(out/'guards.json',dict(frozen_core_unchanged=True,recognition_logits_bit_identical=True,
               core_tensor_sha256=frozen_hash,optimizer_tensors=len(optimizer.state),support=dict(support),
               dynamics_unchanged=True,updates=steps))
    if not args.skip_eval:
        eval_dir=out/'eval';eval_dir.mkdir(exist_ok=True)
        for source in SOURCES:
            rows=read_index(args.validation_dir/f'index_{source}_internal_valid.txt')
            score_source(model,rows,names,eval_dir/f'{source}.json',args.workers,'cuda')
    (out/'COMPLETE').write_text('complete\n')


if __name__=='__main__':main()
