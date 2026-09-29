"""Resumable, sharded autoregressive comparison on identical real-source staff crops."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import time
import traceback

try:
    from rapidfuzz.distance.Levenshtein import distance as edit_distance
except ModuleNotFoundError:
    def edit_distance(a,b):
        # Exact unit-cost Levenshtein fallback; no environment installation required.
        start=0
        while start<min(len(a),len(b)) and a[start]==b[start]:start+=1
        a,b=a[start:],b[start:]
        while a and b and a[-1]==b[-1]:a,b=a[:-1],b[:-1]
        if len(a)<len(b):a,b=b,a
        previous=list(range(len(b)+1))
        for i,x in enumerate(a,1):
            current=[i]
            for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
            previous=current
        return previous[-1]
import torch
from homr.transformer.configs import Config
from homr.transformer.vocabulary import sort_token_chords
from training.architecture.transformer.tromr_arch import TrOMR
from training.transformer.train_structured_heads import load_pinned
from training.transformer.synthetic_head_transfer import read_index,item_for,SOURCES
from training.transformer.training_vocabulary import read_tokens
from training.omr_datasets.notation_sidecar import attach_sidecar
from training.transformer.compare_joint_frozen import sha,write


def events(symbols):
    return [s for group in sort_token_chords(symbols) for s in group if s.rhythm.startswith(('note','rest','keySignature'))]

def serialize(s):
    value=dict(rhythm=s.rhythm,pitch=s.pitch,lift=s.lift,articulation=s.articulation,slur=s.slur,position=s.position)
    if s.notation is not None:
        n=s.notation;value['notation']=dict(beams=list(n.beam_levels),stem=str(n.stem),tie=str(n.tie),slurs=[list(p) for p in n.slurs],advance=str(n.advance))
    return value

def metrics(expected,actual):
    ref=events(expected);got=events(actual)
    rnotes=[s for s in ref if s.rhythm.startswith('note')];gnotes=[s for s in got if s.rhythm.startswith('note')]
    projections=dict(symbol=lambda s:(s.rhythm,s.pitch,s.lift),rhythm=lambda s:s.rhythm,pitch=lambda s:s.pitch,accidental=lambda s:s.lift)
    scores={}
    for key,project in projections.items():
        a,b=(rnotes,gnotes) if key in ('pitch','accidental') else (ref,got);distance=edit_distance([project(s) for s in a],[project(s) for s in b]);scores[key]=dict(errors=distance,reference=len(a),rate=distance/max(1,len(a)))
    key=lambda s:(s.rhythm,s.pitch,s.lift)
    match=difflib.SequenceMatcher(a=[key(s) for s in rnotes],b=[key(s) for s in gnotes],autojunk=False)
    pairs=[(block.a+i,block.b+i) for block in match.get_matching_blocks() for i in range(block.size)]
    return scores,dict(reference_notes=len(rnotes),predicted_notes=len(gnotes),matched_notes=len(pairs),pairs=pairs)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['joint','core','heads','validation-dir','out']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);p.add_argument('--limit',type=int);a=p.parse_args()
    if not 0<=a.part<a.parts:raise ValueError('Invalid shard')
    a.out.mkdir(parents=True,exist_ok=True);root=a.out/f'part-{a.part:02d}';root.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(20260929);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    rows={s:read_index(a.validation_dir/f'index_{s}_internal_valid.txt') for s in SOURCES}
    if a.limit:rows={s:r[:a.limit] for s,r in rows.items()}
    assigned=[(s,i,row) for s,items in rows.items() for i,row in enumerate(items) if i%a.parts==a.part]
    identity=dict(joint_sha256=sha(a.joint),frozen_core_sha256=sha(a.core),frozen_heads_sha256=sha(a.heads),index_sha256={s:sha(a.validation_dir/f'index_{s}_internal_valid.txt') for s in SOURCES},part=a.part,parts=a.parts,limit=a.limit,dtype='float32',attention='math SDPA',tf32=False,mode='eager greedy autoregressive; raw core/head outputs, no post-decode policies')
    if (root/'identity.json').exists() and json.loads((root/'identity.json').read_text())!=identity:raise ValueError('Scoring identity changed')
    write(root/'identity.json',identity);completed=0;total=len(assigned)*2;started=time.monotonic()
    for arm in ['frozen','joint']:
        output=root/f'{arm}.jsonl';done={}
        if output.exists():
            for line in output.read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if r.get('status')=='ok':done[r['id']]=r
            output.write_text(''.join(json.dumps(x)+'\n' for x in done.values()))
        completed+=len(done)
        config=Config();config.enable_structured_heads=True;config.structured_beam_levels=4;config.structured_slur_slots=2
        model=TrOMR(config)
        if arm=='joint':model.load_state_dict(torch.load(a.joint,map_location='cpu',weights_only=True))
        else:
            load_pinned(model,a.core);model.decoder.structured_heads.load_state_dict(torch.load(a.heads,map_location='cpu',weights_only=True))
        model.to('cuda').eval()
        with output.open('a') as handle:
            for source,index,row in assigned:
                if row['id'] in done:continue
                t=time.monotonic()
                try:
                    raw=item_for(row);pixels=raw['inputs'];expected=read_tokens(row['tokens']);assert attach_sidecar(row['tokens'],expected)
                    with torch.inference_mode():actual=model.generate(pixels.unsqueeze(0).to('cuda'))
                    scores,alignment=metrics(expected,actual)
                    result=dict(id=row['id'],source=source,index=index,status='ok',seconds=time.monotonic()-t,pixel_sha256=hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest(),reference_sha256=sha(row['tokens']),sidecar_sha256=sha(row['tokens']+'.notation.json'),metrics=scores,alignment=alignment,reference=[serialize(s) for s in expected],predicted=[serialize(s) for s in actual],reference_notes=[serialize(s) for s in events(expected) if s.rhythm.startswith('note')],predicted_notes=[serialize(s) for s in events(actual) if s.rhythm.startswith('note')])
                except Exception:
                    result=dict(id=row['id'],source=source,index=index,status='failed',seconds=time.monotonic()-t,error=traceback.format_exc())
                handle.write(json.dumps(result)+'\n');handle.flush();completed+=1;elapsed=time.monotonic()-started
                write(root/'progress.json',dict(stage='decoding',arm=arm,source=source,completed_staves=completed,total_staves=total,elapsed_seconds=elapsed,eta_seconds=elapsed/max(1,completed)*(total-completed),updated=time.time()))
                if completed%10==0:print(f'{completed}/{total}, {arm}/{source}; elapsed {elapsed:.1f}s',flush=True)
        del model;torch.cuda.empty_cache()
    write(root/'progress.json',dict(stage='complete',completed_staves=completed,total_staves=total,elapsed_seconds=time.monotonic()-started,updated=time.time()));(root/'COMPLETE').write_text('Both arms decoded; aggregator must inspect failures and pairing.\n')

if __name__=='__main__':main()
