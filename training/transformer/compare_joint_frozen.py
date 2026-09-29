"""Matched teacher-forced recognition/notation scoring of joint versus frozen-core fits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

import torch
from homr.transformer.configs import Config
from training.architecture.transformer.tromr_arch import TrOMR
from training.transformer.train_structured_heads import load_pinned
from training.transformer.train_joint_public import deterministic_cross_entropy
from training.transformer.synthetic_head_transfer import read_index,score_source,compare,SOURCES


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write(path,x):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint',required=True,type=Path);p.add_argument('--core',required=True,type=Path);p.add_argument('--heads',required=True,type=Path);p.add_argument('--validation-dir',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(20260929)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    rows={s:read_index(a.validation_dir/f'index_{s}_internal_valid.txt') for s in SOURCES}
    if a.limit:rows={s:r[:a.limit] for s,r in rows.items()}
    identity=dict(joint_sha256=sha(a.joint),frozen_core_sha256=sha(a.core),frozen_heads_sha256=sha(a.heads),indexes={s:sha(a.validation_dir/f'index_{s}_internal_valid.txt') for s in SOURCES},dtype='float32',attention='math SDPA',tf32=False,limit=a.limit,mode='teacher-forced, internal diagnostic, dynamics excluded')
    write(a.out/'identity.json',identity);reports={};recognition={};completed=0;total=2*sum(map(len,rows.values()));started=time.monotonic()
    for arm in ['frozen','joint']:
        c=Config();c.enable_structured_heads=True;c.structured_beam_levels=4;c.structured_slur_slots=2
        model=TrOMR(c)
        if arm=='joint':model.load_state_dict(torch.load(a.joint,map_location='cpu',weights_only=True))
        else:
            load_pinned(model,a.core);model.decoder.structured_heads.load_state_dict(torch.load(a.heads,map_location='cpu',weights_only=True))
        model.decoder.cross_entropy=MethodType(deterministic_cross_entropy,model.decoder)
        model.to('cuda').eval();names=[n for n in model.decoder.structured_heads.head_names() if n!='dynamic.mark'];reports[arm]={};recognition[arm]={}
        for source in SOURCES:
            original=model.forward;correct=Counter();counts=Counter();loss_sum=0.;n=0
            def observed(self,**batch):
                nonlocal loss_sum,n,completed
                result=original(**batch,sampling_prob=1.0);size=len(batch['inputs']);n+=size;completed+=size;loss_sum+=float(result['loss'].detach())*size;mask=batch['mask'][:,1:]
                for branch,logits in zip(('rhythms','pitchs','lifts','positions','articulations','slurs'),result['logits'],strict=True):
                    correct[branch]+=int(((logits.argmax(-1)==batch[branch][:,1:])&mask).sum());counts[branch]+=int(mask.sum())
                elapsed=time.monotonic()-started;write(a.out/'progress.json',dict(arm=arm,source=source,completed_staves=completed,total_staves=total,elapsed_seconds=elapsed,eta_seconds=elapsed/completed*(total-completed)))
                return result
            model.forward=MethodType(observed,model)
            path=a.out/f'{arm}-{source}.json';reports[arm][source]=score_source(model,rows[source],names,path,a.workers,'cuda');model.forward=original
            recognition[arm][source]=dict(staves=n,recognition_loss=loss_sum/n,teacher_forced_accuracy={k:correct[k]/counts[k] for k in correct},correct=dict(correct),counts=dict(counts));write(a.out/f'{arm}-{source}-recognition.json',recognition[arm][source]);print(arm,source,n,'scored',flush=True)
        del original,model;torch.cuda.empty_cache()
    comparison=compare(reports['frozen'],reports['joint']);recognition_deltas={s:dict(loss_delta=recognition['joint'][s]['recognition_loss']-recognition['frozen'][s]['recognition_loss'],accuracy_delta_pp={k:100*(recognition['joint'][s]['teacher_forced_accuracy'][k]-v) for k,v in recognition['frozen'][s]['teacher_forced_accuracy'].items()}) for s in SOURCES}
    write(a.out/'comparison.json',dict(identity=identity,notation=comparison,recognition=recognition,recognition_deltas=recognition_deltas))
    lines=['# Joint epoch 27 versus frozen public core + real-only heads','','Matched pixels and reference hashes verified for every source. Both rescored in float32, math SDPA, TF32 disabled. Dynamics excluded because frozen dynamic.mark was not trained. Internal teacher-forced diagnostic; autoregressive comparison is separate.','','| Source | Frozen recognition loss | Joint loss | Rhythm delta pp | Pitch delta pp | Accidentals delta pp |','|---|---:|---:|---:|---:|---:|']
    for s in SOURCES:
        d=recognition_deltas[s]['accuracy_delta_pp'];lines.append(f"| {s} | {recognition['frozen'][s]['recognition_loss']:.5f} | {recognition['joint'][s]['recognition_loss']:.5f} | {d['rhythms']:+.4f} | {d['pitchs']:+.4f} | {d['lifts']:+.4f} |")
    lines+=['','| Source | Notation metric | Joint minus frozen pp | Support |','|---|---|---:|---:|']
    for s in SOURCES:
        for metric,d in comparison['by_source'][s].items():lines.append(f"| {s} | {metric} | {d['delta_pp']:+.3f} | {d['support']} |")
    (a.out/'comparison.md').write_text('\n'.join(lines)+'\n');write(a.out/'progress.json',dict(stage='complete',completed_staves=completed,total_staves=total));(a.out/'COMPLETE').write_text('Matched diagnostic comparison complete.\n')

if __name__=='__main__':main()
