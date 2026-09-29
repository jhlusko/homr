"""Continue the PDMX-v2 joint model with bounded-cost scan augmentation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
import types

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

from training.transformer.synthetic_head_transfer import item_for,collate_fixed,read_index
from training.transformer.train_structured_heads import _limit_worker_threads
from training.transformer.synthetic_head_capacity import sha,write_json
from training.architecture.transformer.structured_losses import structured_loss


def objective(recognition,notation):
    # Normalize the head sum by the fixed 12-head inventory, never batch support.
    return recognition + notation / 12


def deterministic_cross_entropy(self,logits,target,label_smoothing=0.0,weights=None):
    # Same objective; flattening selects the deterministic 1-D NLL kernel.
    return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),
            reduction='mean',weight=weights,ignore_index=self.ignore_index,label_smoothing=label_smoothing)


class JointRows(Dataset):
    def __init__(self,rows,seed,epoch,augment):self.rows,self.seed,self.epoch,self.augment=rows,seed,epoch,augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        from training.transformer.training_vocabulary import read_tokens
        from training.omr_datasets.notation_sidecar import attach_sidecar
        from training.architecture.transformer.structured_targets import build_targets,notation_positions
        r=self.rows[i];result=item_for(r)
        symbols=read_tokens(r['tokens']);assert attach_sidecar(r['tokens'],symbols)
        result['dynamic.mark']=build_targets([notation_positions(symbols,len(symbols)+2)],4,2)['dynamic.mark'][0]
        if self.augment:
            from training.transformer.image_utils import distort_image,read_image_to_ndarray,prepare_for_tensor,ndarray_to_tensor,pad_to_3_dims
            from homr.staff_parsing import add_image_into_tr_omr_canvas,get_tr_omr_canvas_size
            import cv2
            seed=(self.seed+self.epoch*1000003+i)%2**32
            random.seed(seed);np.random.seed(seed)
            image=read_image_to_ndarray(r['image'])
            canvas=get_tr_omr_canvas_size(image.shape)
            cap_width,cap_height=int(canvas[0])*2,int(canvas[1])*2
            if r['source']=='lieder' and (image.shape[1]>cap_width or image.shape[0]>cap_height):
                image=cv2.resize(image,(cap_width,cap_height),interpolation=cv2.INTER_AREA)
            image=distort_image(image,allow_occlusions=True)
            result['inputs']=pad_to_3_dims(ndarray_to_tensor(prepare_for_tensor(add_image_into_tr_omr_canvas(image))))
        return result


def atomic_save(value,path):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


def restore_rng(state):
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])



def evaluate(model,rows,names,path,workers,batch_size):
    from training.architecture.transformer.structured_decoding import decode_predictions,decode_reference
    from training.transformer.structured_metrics import Evaluation
    evaluation=Evaluation(4,2);correct=Counter();counts=Counter();loss_total=0.;n=0
    data=DataLoader(JointRows(rows,0,0,False),batch_size=batch_size,num_workers=workers,worker_init_fn=_limit_worker_threads if workers else None,
                    collate_fn=collate_fixed,generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for raw in data:
            targets={k:v[:,1:].to('cuda') for k,v in raw.items() if k in names}
            batch={k:v.to('cuda') for k,v in raw.items() if k not in names}
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):outputs=model(**batch,sampling_prob=1.)
            size=len(raw['inputs']);loss_total+=float(outputs['loss'])*size;n+=size
            mask=batch['mask'][:,1:]
            for branch,logits in zip(('rhythms','pitchs','lifts','positions','articulations','slurs'),outputs['logits'],strict=True):
                correct[branch]+=int(((logits.argmax(-1)==batch[branch][:,1:]) & mask).sum())
                counts[branch]+=int(mask.sum())
            for got,want in zip(decode_predictions(outputs['structured_logits'],targets,4,2),decode_reference(targets,4,2),strict=True):
                evaluation.observe(got,want)
    write_json(path,dict(staves=n,recognition_loss=loss_total/n,
                        teacher_forced_accuracy={k:correct[k]/counts[k] for k in correct},
                        structured=evaluation.to_dict()))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--train-index',required=True,type=Path)
    p.add_argument('--validation-dir',required=True,type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--core-lr',type=float,default=3e-4);p.add_argument('--heads-lr',type=float,default=3e-4)
    p.add_argument('--warmup-updates',type=int,default=5000)
    p.add_argument('--encoder-pretrained',action='store_true')
    p.add_argument('--name',default='random-core-heads-pdmx-v2-capped-20260929')
    p.add_argument('--source-shares',type=float,nargs=3,default=[.60,.15,.25],metavar=('OSSQ','PDMX','LIEDER'))
    p.add_argument('--epochs',type=int,default=160);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--accumulation',type=int,default=4);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260928);p.add_argument('--max-updates',type=int)
    p.add_argument('--resume',type=Path);p.add_argument('--migrate-from',type=Path)
    p.add_argument('--draws-per-epoch',type=int,default=40290)
    p.add_argument('--no-augmentation',action='store_true')
    p.add_argument('--skip-eval',action='store_true');a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or a.accumulation<1 or a.draws_per_epoch<1:raise ValueError('Invalid budget')
    if not a.migrate_from:raise ValueError('An immutable source checkpoint is required')
    a.run.mkdir(parents=True,exist_ok=True)
    if (a.run/'resume.pt').exists() and not a.resume:raise ValueError('Resume explicitly; do not overwrite')
    torch.set_num_threads(1);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    rows=read_index(a.train_index)
    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR
    from training.architecture.transformer.structured_heads import StructuredNotationHeads
    from training.transformer.train_structured_heads import load_pinned
    from training.transformer import image_utils,training_vocabulary
    config=Config();config.enable_structured_heads=True;config.structured_beam_levels=4;config.structured_slur_slots=2
    if a.checkpoint is not None:raise ValueError('This runner is random initialization only')
    config.encoder_pretrained=a.encoder_pretrained
    model=TrOMR(config)
    torch.manual_seed(a.seed);model.decoder.structured_heads=StructuredNotationHeads(config.decoder_dim,4,2)
    for param in model.parameters():param.requires_grad_(True)
    model.decoder.cross_entropy=types.MethodType(deterministic_cross_entropy,model.decoder)
    model.to('cuda');names=model.decoder.structured_heads.head_names()
    assert len(names)==12
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    core=[p for n,p in model.named_parameters() if not n.startswith('decoder.structured_heads.')]
    heads=list(model.decoder.structured_heads.parameters())
    optimizer=torch.optim.AdamW([dict(params=core,lr=a.core_lr),dict(params=heads,lr=a.heads_lr)],betas=(.9,.999),eps=1e-8,weight_decay=.05)
    batches=math.ceil(a.draws_per_epoch/a.batch_size);per_epoch=math.ceil(batches/a.accumulation);total=per_epoch*a.epochs
    warmup=a.warmup_updates
    if not 0<warmup<total:raise ValueError('Invalid warmup')
    if any(v<=0 for v in a.source_shares) or not math.isclose(sum(a.source_shares),1.):raise ValueError('Source shares must be positive and sum to one')
    pools={source:[i for i,row in enumerate(rows) if row['source']==source] for source in ('ossq','pdmx','lieder')}
    if any(not v for v in pools.values()):raise ValueError('Missing training source')
    def scale(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,scale)
    runtime=hashlib.sha256()
    root=Path(__file__).resolve().parents[2]
    for f in sorted((root/'homr/transformer').glob('*.py'))+sorted((root/'training/architecture/transformer').glob('*.py')):
        runtime.update(str(f.relative_to(root)).encode());runtime.update(sha(f).encode())
    identity=dict(name=a.name,initialization='migrated model, optimizer, scheduler, and RNG from PDMX-v2 run before bounded augmentation',encoder_pretrained=a.encoder_pretrained,checkpoint_sha256=sha(a.migrate_from),source_shares=dict(zip(pools,a.source_shares)),sampler='source-weighted uniform crops with replacement',train_index_sha256=sha(a.train_index),
                  rows=len(rows),draws_per_epoch=a.draws_per_epoch,sources=dict(Counter(r['source'] for r in rows)),epochs=a.epochs,batch_size=a.batch_size,
                  accumulation=a.accumulation,seed=a.seed,core_lr=a.core_lr,heads_lr=a.heads_lr,structured_scale=1/12,
                  optimizer='AdamW',weight_decay=.05,warmup_updates=warmup,total_updates=total,
                  bf16=True,tf32=False,deterministic_algorithms=True,attention_backend="math",augmentation=not a.no_augmentation,augmentation_pre_resize='Lieder-only-2x-model-canvas-for-oversize-inputs',scheduled_sampling=False,
                  source_sha256=sha(Path(__file__)),runtime_sha256=runtime.hexdigest(),
                  preprocessing_sha256=sha(Path(image_utils.__file__)),vocabulary_sha256=sha(Path(training_vocabulary.__file__)),
                  torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                  validation_indices={s:sha(a.validation_dir/f'index_{s}_internal_valid.txt') for s in ('ossq','pdmx','lieder')})
    write_json(a.run/'identity.json',identity)
    epoch=0;batch_offset=0;updates=0;support=Counter();gradient_seen=set()
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        if state['identity']!=identity:raise ValueError('Resume identity differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch,batch_offset,updates=state['epoch'],state['batch_offset'],state['updates']
        support.update(state['support']);gradient_seen.update(state['gradient_seen']);restore_rng(state)
    else:
        state=torch.load(a.migrate_from,map_location='cpu',weights_only=False)
        old=state['identity']
        expected={'name':'random-core-heads-pdmx-v2-20260929',
                  'train_index_sha256':'e94b63898f7175913a759a381675291e88ad0ca016dd9880afa69418e8b525c1',
                  'total_updates':total,'batch_size':a.batch_size,'accumulation':a.accumulation,
                  'source_shares':dict(zip(pools,a.source_shares)),
                  'sampler':'source-weighted uniform crops with replacement'}
        for key in ('encoder_pretrained','epochs','seed','core_lr','heads_lr','structured_scale',
                    'optimizer','weight_decay','warmup_updates','bf16','tf32','deterministic_algorithms',
                    'attention_backend','augmentation','scheduled_sampling','runtime_sha256',
                    'preprocessing_sha256','vocabulary_sha256','torch_version','cuda_version',
                    'validation_indices'):
            expected[key]=identity[key]
        for key,value in expected.items():
            if old[key]!=value:raise ValueError(f'Incompatible source checkpoint: {key}')
        if state['updates'] != state['epoch']*per_epoch + math.ceil(state['batch_offset']/a.accumulation):
            raise ValueError('Checkpoint epoch, batch offset, and updates differ')
        if state['updates']!=state['scheduler']['_step_count']-1:
            raise ValueError('Optimizer and scheduler progress differ')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch,batch_offset,updates=state['epoch'],state['batch_offset'],state['updates']
        # Preserve the update budget exactly. The first v2 epoch begins at the
        # remaining draw slots in this epoch; later epochs use all v2 draws.
        support.update(state['support']);gradient_seen.update(state['gradient_seen']);restore_rng(state)
        write_json(a.run/'migration.json',dict(source_checkpoint=str(a.migrate_from),source_sha256=sha(a.migrate_from),
                   source_epoch=state['epoch'],source_batch_offset=state['batch_offset'],source_updates=updates,
                   target_epoch=epoch,target_batch_offset=batch_offset,target_updates=updates))
    if not a.skip_eval and not a.resume:
        model.eval();out=a.run/'eval-initial';out.mkdir(exist_ok=True)
        for source in ('ossq','pdmx','lieder'):
            evaluate(model,read_index(a.validation_dir/f'index_{source}_internal_valid.txt'),names,out/f'{source}.json',a.workers,a.batch_size)
    started=time.monotonic();start_update=updates
    def save():
        atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                         epoch=epoch,batch_offset=batch_offset,updates=updates,support=dict(support),gradient_seen=sorted(gradient_seen),
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state()),a.run/'resume.pt')
    while epoch<a.epochs:
        sampler=random.Random(a.seed+epoch)
        sources=sampler.choices(list(pools),weights=a.source_shares,k=a.draws_per_epoch)
        indices=[sampler.choice(pools[source]) for source in sources]
        # Dedicated generator prevents iterator startup consuming dropout's RNG on resume.
        data=DataLoader(JointRows(rows,a.seed,epoch,not a.no_augmentation),batch_sampler=[indices[i:i+a.batch_size] for i in range(batch_offset*a.batch_size,len(indices),a.batch_size)],
                        num_workers=a.workers,worker_init_fn=_limit_worker_threads if a.workers else None,collate_fn=collate_fixed,pin_memory=True,generator=torch.Generator().manual_seed(a.seed+epoch))
        iterator=iter(data);model.train()
        while batch_offset<batches:
            optimizer.zero_grad(set_to_none=True);count=min(a.accumulation,batches-batch_offset)
            rec=notation=0.;head_losses=Counter();head_support=Counter();recognition_components=Counter()
            for micro in range(count):
                raw=next(iterator);targets={k:v[:,1:].to('cuda',non_blocking=True) for k,v in raw.items() if k in names}
                batch={k:v.to('cuda',non_blocking=True) for k,v in raw.items() if k not in names}
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                    result=model(**batch,sampling_prob=1.0);losses=structured_loss(result['structured_logits'],targets)
                    loss=objective(result['loss'],losses.total)/count
                if not torch.isfinite(loss):raise ValueError('Nonfinite joint loss')
                for key,value in result.items():
                    if key.startswith('loss_'):recognition_components[key]+=float(value.detach())/count
                loss.backward();rec+=float(result['loss'].detach())/count;notation+=float(losses.total.detach())/count
                for h in losses.heads:
                    support[h.name]+=h.supervised;head_support[h.name]+=h.supervised;head_losses[h.name]+=float(h.loss.detach())/count
                batch_offset+=1
            for n,param in model.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():raise ValueError(f'Nonfinite gradient {n}')
                    if torch.count_nonzero(param.grad):gradient_seen.add(n)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step();scheduler.step();updates+=1
            if updates%10==0 or updates==1:
                elapsed=time.monotonic()-started
                write_json(a.run/'progress.json',dict(epoch=epoch+batch_offset/batches,updates=updates,total_updates=total,
                         recognition_loss=rec,structured_loss=notation,recognition_components=dict(recognition_components),head_losses=dict(head_losses),head_support=dict(head_support),core_lr=optimizer.param_groups[0]['lr'],heads_lr=optimizer.param_groups[1]['lr'],elapsed_seconds=elapsed,
                         remaining_seconds=elapsed/max(1,updates-start_update)*(total-updates),support=dict(support),gpu_peak_bytes=torch.cuda.max_memory_allocated()))
                print(json.dumps(json.loads((a.run/'progress.json').read_text())),flush=True)
            if updates%200==0 or a.max_updates and updates>=a.max_updates:save()
            if a.max_updates and updates>=a.max_updates:
                changed={n:bool(not torch.equal(initial[n],p.detach().cpu())) for n,p in model.named_parameters()}
                write_json(a.run/'smoke-guards.json',dict(gradient_seen=sorted(gradient_seen),changed=changed,support=dict(support)))
                return
        epoch+=1;batch_offset=0;save()
        # Keep every epoch's weights, and latest complete optimizer/RNG state.
        atomic_save(model.state_dict(),a.run/f'epoch-{epoch:02d}.pth')
        if not a.skip_eval:
            model.eval();out=a.run/f'eval-epoch-{epoch:02d}';out.mkdir(exist_ok=True)
            with torch.no_grad():
                for source in ('ossq','pdmx','lieder'):
                    evaluate(model,read_index(a.validation_dir/f'index_{source}_internal_valid.txt'),names,out/f'{source}.json',a.workers,a.batch_size)
        write_json(a.run/'epoch-status.json',dict(completed_epoch=epoch,updates=updates,support=dict(support)))
        if not a.skip_eval and epoch>=2:
            bad=[]
            for source in ('ossq','pdmx','lieder'):
                baseline=json.loads((a.run/'eval-initial'/f'{source}.json').read_text())
                current=json.loads((out/f'{source}.json').read_text())
                previous=json.loads((a.run/f'eval-epoch-{epoch-1:02d}'/f'{source}.json').read_text())
                bar=baseline['teacher_forced_accuracy']['rhythms']-.03
                if current['teacher_forced_accuracy']['rhythms']<bar and previous['teacher_forced_accuracy']['rhythms']<bar:
                    bad.append(source)
            if bad:
                write_json(a.run/'EARLY_STOPPED.json',dict(reason='Rhythm token accuracy lost >3 points for two successive epochs',sources=bad,epoch=epoch))
                return
    (a.run/'COMPLETE').write_text('complete\n')

if __name__=='__main__':main()
