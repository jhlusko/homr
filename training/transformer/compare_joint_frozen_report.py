"""Aggregate paired decoded results with score-clustered bootstrap intervals."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re

import numpy as np
from training.transformer.compare_joint_frozen import write
from training.transformer.synthetic_head_transfer import SOURCES


def score_id(path):
    name=Path(path).name
    match=re.search(r'(sq\d+|IMSLP\d+|Qm[1-9A-HJ-NP-Za-km-z]{44})',name,re.I)
    if not match:raise ValueError('Cannot establish score cluster: '+name)
    return match.group(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--parts',type=int,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
    records={arm:{} for arm in ['frozen','joint']};identities=[]
    for n in range(a.parts):
        part=a.root/f'part-{n:02d}'
        if not (part/'COMPLETE').exists():raise ValueError('Incomplete decoding shard')
        identities.append(json.loads((part/'identity.json').read_text()))
        for arm in records:
            for line in (part/f'{arm}.jsonl').read_text().splitlines():
                x=json.loads(line)
                if x['id'] in records[arm]:raise ValueError('Duplicate decoded crop')
                records[arm][x['id']]=x
    for key in ['joint_sha256','frozen_core_sha256','frozen_heads_sha256','index_sha256','parts','limit','dtype','attention','tf32','mode']:
        if any(x[key]!=identities[0][key] for x in identities):raise ValueError('Shard identities differ')
    if set(records['frozen'])!=set(records['joint']):raise ValueError('Arms have different crop identities')
    failures=[];paired=[]
    for rid in sorted(records['frozen']):
        f,j=records['frozen'][rid],records['joint'][rid]
        if f['status']!='ok' or j['status']!='ok':failures.append(dict(id=rid,frozen=f['status'],joint=j['status']));continue
        for key in ['source','index','pixel_sha256','reference_sha256','sidecar_sha256']:
            if f[key]!=j[key]:raise ValueError('Unmatched '+key+': '+rid)
        paired.append((f,j))
    output=dict(identity=identities[0],total_crops=len(records['frozen']),paired_successes=len(paired),failures=failures,scope='Internal crop-level eager greedy decoding; raw outputs before post-decode policies; score-cluster bootstrap; no release claim',sources={})
    lines=['# Decoded joint epoch 27 versus frozen public core + real-only heads','',output['scope'],f"\nPaired successful crops: {len(paired)}/{output['total_crops']}; failed pairs: {len(failures)}.",'','| Source | Metric | Frozen error % | Joint error % | Joint − frozen pp | 95% paired score-bootstrap CI |','|---|---|---:|---:|---:|---|']
    rng=np.random.default_rng(20260929)
    for source in SOURCES:
        subset=[(f,j) for f,j in paired if f['source']==source];report=dict(staves=len(subset),metrics={},recognition_alignment={})
        for metric in ['symbol','rhythm','pitch','accidental']:
            groups=defaultdict(lambda:[0,0,0]);rates={arm:[] for arm in records}
            for f,j in subset:
                fm,jm=f['metrics'][metric],j['metrics'][metric]
                if fm['reference']!=jm['reference']:raise ValueError('Reference denominators differ')
                cluster=groups[score_id(f['id'])];cluster[0]+=fm['errors'];cluster[1]+=jm['errors'];cluster[2]+=fm['reference'];rates['frozen'].append(fm['rate']);rates['joint'].append(jm['rate'])
            v=np.array(list(groups.values()),dtype=float);total=v.sum(axis=0);base=total[0]/max(1,total[2]);joint=total[1]/max(1,total[2]);samples=[]
            for _ in range(a.bootstrap):
                sums=v[rng.integers(0,len(v),size=len(v))].sum(axis=0);samples.append(100*(sums[1]-sums[0])/max(1,sums[2]))
            lo,hi=np.percentile(samples,[2.5,97.5]);report['metrics'][metric]=dict(frozen_error=base,joint_error=joint,delta_pp=100*(joint-base),ci95_pp=[float(lo),float(hi)],score_clusters=len(v),reference_events=int(total[2]),frozen_errors=int(total[0]),joint_errors=int(total[1]),frozen_mean_staff_error=float(np.mean(rates['frozen'])),joint_mean_staff_error=float(np.mean(rates['joint'])))
            lines.append(f'| {source} | {metric} | {base*100:.3f} | {joint*100:.3f} | {100*(joint-base):+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
        refnotes=sum(f['alignment']['reference_notes'] for f,j in subset)
        for arm in records:
            ix=0 if arm=='frozen' else 1;matched=sum(pair[ix]['alignment']['matched_notes'] for pair in subset);report['recognition_alignment'][arm]=dict(reference_notes=refnotes,matched_notes=matched,coverage=matched/max(1,refnotes))
        # Fair head comparison on exactly the same GT notes recognized by both arms.
        common=Counter();correct={arm:Counter() for arm in records};all_reference=Counter();end_to_end_correct={arm:Counter() for arm in records}
        def fields(note):
            n=note.get('notation');out={}
            if not n:return out
            if n['stem'] in ('up','down'):out['stem_direction']=n['stem']
            applicable=[(i,b) for i,b in enumerate(n['beams'][:4]) if b!='not_applicable']
            if applicable:
                out['beam_vector']=tuple(applicable)
                for i,b in applicable:out['beam_level_'+str(i+1)]=b
            if n['tie'] not in ('unknown','unsupervised'):out['tie_state']=n['tie']
            for i,(event,side) in enumerate(n['slurs'],1):
                if event not in ('unknown','unsupervised'):out[f'slur_slot_{i}_event']=event
                if side in ('above','below'):out[f'slur_slot_{i}_side']=side
            return out
        def predictions(note,reference_fields):
            n=note.get('notation');out={}
            if not n:return out
            for key in reference_fields:
                if key=='stem_direction':out[key]=n['stem']
                elif key=='beam_vector':out[key]=tuple((i,n['beams'][i]) for i,_ in reference_fields[key])
                elif key.startswith('beam_level_'):out[key]=n['beams'][int(key.rsplit('_',1)[1])-1]
                elif key=='tie_state':out[key]=n['tie']
                elif key.startswith('slur_slot_'):
                    _,_,slot,field=key.split('_');slot=int(slot)-1
                    out[key]=n['slurs'][slot][0 if field=='event' else 1] if slot<len(n['slurs']) else None
            return out
        for f,j in subset:
            if f['reference_notes']!=j['reference_notes']:raise ValueError('Sorted note references differ')
            maps={arm:dict(x['alignment']['pairs']) for arm,x in [('frozen',f),('joint',j)]}
            for idx,note in enumerate(f['reference_notes']):
                ref=fields(note);all_reference.update(ref.keys())
                for arm,x in [('frozen',f),('joint',j)]:
                    if idx in maps[arm]:
                        pred=predictions(x['predicted_notes'][maps[arm][idx]],ref)
                        for key,value in ref.items():end_to_end_correct[arm][key]+=pred.get(key)==value
                if idx not in maps['frozen'] or idx not in maps['joint']:continue
                common.update(ref.keys())
                for arm,x in [('frozen',f),('joint',j)]:
                    pred=predictions(x['predicted_notes'][maps[arm][idx]],ref)
                    for key,value in ref.items():correct[arm][key]+=pred.get(key)==value
        report['raw_note_head_accuracy_common_recognized_notes']={key:dict(support=n,frozen=correct['frozen'][key]/n,joint=correct['joint'][key]/n,delta_pp=100*(correct['joint'][key]-correct['frozen'][key])/n) for key,n in common.items()}
        report['gt_note_coverage_with_correct_recognition_and_notation']={key:dict(support=n,frozen=end_to_end_correct['frozen'][key]/n,joint=end_to_end_correct['joint'][key]/n) for key,n in all_reference.items()}
        output['sources'][source]=report
    lines+=['','Raw head accuracies are conditional on common correctly recognized notes; tie/slur-event accuracies include NONE and can be dominated by it. Teacher-forced macro-F1 and span reports provide complementary class-balanced diagnostics. GT-note coverage also counts missing/wrongly recognized notes against each arm. No complete span-pairing metric is inferred after dropping unmatched notes.']
    write(a.root/'comparison.json',output);(a.root/'comparison.md').write_text('\n'.join(lines)+'\n');(a.root/'COMPLETE').write_text('Decoded comparison aggregated; consult failures and scope.\n');print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
