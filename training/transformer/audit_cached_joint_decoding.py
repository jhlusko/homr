"""Strict four-model cached decoding comparison and score-clustered error audit.

No model inference. Recompute recognition metrics and match pixels/labels/protocol
before aggregating. Error operations are minimum unit-cost Levenshtein alignments;
ambiguous alignments are diagnostics, not established musical causes.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
from rapidfuzz.distance import Levenshtein

MODELS=('frozen_public','public_joint27','random73','random80')
SOURCES=('ossq','lieder','pdmx')
PROTOCOL=('frozen_core_sha256','frozen_heads_sha256','index_sha256','parts','limit','dtype','attention','tf32','mode')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def cluster(rid):
 m=re.search(r'(sq\d+|IMSLP\d+|Qm[1-9A-HJ-NP-Za-km-z]{44})',Path(rid).name,re.I)
 if not m:raise ValueError('Cannot cluster '+rid)
 return m.group(1)

def events(symbols):
 # Same chord ordering as the original scorer, not raw autoregressive order.
 from homr.transformer.vocabulary import EncodedSymbol,sort_token_chords
 # Reconstruct once and preserve mapping by object identity through sorting.
 objects=[];mapping={}
 for r in symbols:
  s=EncodedSymbol(**{k:r[k] for k in ('rhythm','pitch','lift','articulation','slur','position')});objects.append(s);mapping[id(s)]=r
 return [mapping[id(s)] for group in sort_token_chords(objects) for s in group if s.rhythm.startswith(('note','rest','keySignature'))]

def projection(symbols,metric):
 ev=events(symbols)
 if metric in ('pitch','accidental'):ev=[s for s in ev if s['rhythm'].startswith('note')]
 if metric=='symbol':return [(s['rhythm'],s['pitch'],s['lift']) for s in ev]
 return [s[{'rhythm':'rhythm','pitch':'pitch','accidental':'lift'}[metric]] for s in ev]

def fields(n):
 x=n.get('notation');out={}
 if not x:return out
 if x['stem'] in ('up','down'):out['stem_direction']=x['stem']
 beams=[(i,b) for i,b in enumerate(x['beams'][:4]) if b!='not_applicable']
 if beams:out['beam_vector']=tuple(beams)
 if x['tie'] not in ('unknown','unsupervised'):out['tie_state']=x['tie']
 for i,(event,side) in enumerate(x['slurs'][:2]):
  if event not in ('unknown','unsupervised'):out[f'slur_{i+1}_event']=event
  if side in ('above','below'):out[f'slur_{i+1}_side']=side
 return out

def predicted_fields(n,reference):
 x=n.get('notation');out={}
 if not x:return out
 for key in reference:
  if key=='stem_direction':out[key]=x['stem']
  elif key=='beam_vector':out[key]=tuple((i,x['beams'][i]) for i,_ in reference[key])
  elif key=='tie_state':out[key]=x['tie']
  else:
   _,slot,field=key.split('_');i=int(slot)-1
   out[key]=x['slurs'][i][0 if field=='event' else 1] if i<len(x['slurs']) else None
 return out

def protocol(root):
 ids=[]
 for part in range(4):
  folder=root/f'part-{part:02d}';assert (folder/'COMPLETE').exists()
  identity=json.loads((folder/'identity.json').read_text());assert identity['part']==part;ids.append(identity)
 for key in PROTOCOL+('joint_sha256',):assert all(i[key]==ids[0][key] for i in ids),key
 return ids[0]

def iter_cases(previous,random):
 seen=set()
 for part in range(4):
  roots=[previous,previous,random/'epoch-73',random/'epoch-80',random/'epoch-73',random/'epoch-80']
  arms=['frozen','joint','joint','joint','frozen','frozen']
  handles=[(root/f'part-{part:02d}'/f'{arm}.jsonl').open() for root,arm in zip(roots,arms,strict=True)]
  try:
   for lines in itertools.zip_longest(*handles):
    assert all(line is not None for line in lines),'Unequal shard lengths'
    records=[json.loads(line) for line in lines]
    assert records[4]==records[0] and records[5]==records[0],'Reused frozen predictions differ'
    rid=records[0]['id'];assert rid not in seen;seen.add(rid)
    assert all(r['id']==rid and r['status']=='ok' for r in records),'Unmatched or failed crop'
    yield dict(zip(MODELS,records[:4],strict=True))
  finally:
   for handle in handles:handle.close()
 assert len(seen)==3909,len(seen)

def diagnostics(row):
 a=projection(row['reference'],'symbol');b=projection(row['predicted'],'symbol')
 ops=Levenshtein.editops(a,b);counts=Counter(o.tag for o in ops)
 details=Counter()
 for op in ops:
  if op.tag!='replace':continue
  x,y=a[op.src_pos],b[op.dest_pos]
  if x[0]!=y[0]:details['rhythm_changed']+=1
  if x[1]!=y[1]:
   details['pitch_changed']+=1
   if re.fullmatch(r'[A-G]\d+',x[1]) and re.fullmatch(r'[A-G]\d+',y[1]) and x[1][0]==y[1][0]:details['same_letter_octave_changed']+=1
  if x[1]==y[1] and x[2]!=y[2]:details['accidental_only_changed']+=1
 refkeys=[s['rhythm'] for s in row['reference'] if s['rhythm'].startswith('keySignature')]
 gotkeys=[s['rhythm'] for s in row['predicted'] if s['rhythm'].startswith('keySignature')]
 def grams(seq):return Counter(tuple(seq[i:i+4]) for i in range(max(0,len(seq)-3)))
 aa,bb=grams(a),grams(b)
 excess=sum(max(0,count-max(1,aa[g])) for g,count in bb.items() if count>=2)
 notes=[s for s in row['predicted'] if s['rhythm'].startswith('note')]
 invalid_pitch=sum(re.fullmatch(r'[A-G]\d+',s['pitch']) is None for s in notes)
 invalid_position=sum(s['position'] not in ('upper','lower') for s in notes)
 invalid_lift=sum(s['lift'] not in ('_','N','b','bb','#','##') for s in notes)
 return dict(invalid_pitch_notes=invalid_pitch,invalid_staff_notes=invalid_position,invalid_accidental_notes=invalid_lift,operations=dict(counts),substitution_details=dict(details),reference_events=len(a),predicted_events=len(b),length_ratio=len(b)/max(1,len(a)),event_length_delta=len(b)-len(a),reference_sequence_length=len(row['reference']),predicted_sequence_length=len(row['predicted']),possible_sequence_limit=len(row['predicted'])>=606,key_signature_mismatch=refkeys!=gotkeys,reference_keys=refkeys,predicted_keys=gotkeys,excess_repeated_fourgrams=excess)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--random',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 rows={m:{} for m in MODELS};identities={}
 for name,root in [('frozen_public',args.previous),('public_joint27',args.previous),('random73',args.random/'epoch-73'),('random80',args.random/'epoch-80')]:identities[name]=protocol(root)
 for key in PROTOCOL:assert all(identity[key]==identities[MODELS[0]][key] for identity in identities.values()),key
 print('Shard protocol identities match. Auditing with bounded-memory streaming.',flush=True)
 crop_rows=[];source_reports={};common=defaultdict(Counter);correct=defaultdict(lambda:defaultdict(Counter));total_support=defaultdict(Counter);endcorrect=defaultdict(lambda:defaultdict(Counter));ops=defaultdict(lambda:defaultdict(Counter));stats=defaultdict(lambda:defaultdict(list))
 for n,records in enumerate(iter_cases(args.previous,args.random)):
  rid=records[MODELS[0]]['id'];reference=records[MODELS[0]]
  for model in MODELS:
   r=records[model]
   for key in ('source','index','pixel_sha256','reference_sha256','sidecar_sha256','reference','reference_notes'):assert r[key]==reference[key],(rid,model,key)
   for metric,m in r['metrics'].items():
    ref=projection(r['reference'],metric);got=projection(r['predicted'],metric)
    assert len(ref)==m['reference'] and Levenshtein.distance(ref,got)==m['errors'],(rid,model,metric)
  crophead=defaultdict(Counter)
  source=reference['source'];maps={model:dict(records[model]['alignment']['pairs']) for model in MODELS}
  for index,note in enumerate(reference['reference_notes']):
   ref=fields(note);total_support[source].update(ref.keys());matched=all(index in mapping for mapping in maps.values())
   if matched:common[source].update(ref.keys())
   for model in MODELS:
    if index not in maps[model]:continue
    r=records[model];got=predicted_fields(r['predicted_notes'][maps[model][index]],ref)
    for key,value in ref.items():
     endcorrect[source][model][key]+=got.get(key)==value
     crophead[model][key]+=got.get(key)==value
     if matched:correct[source][model][key]+=got.get(key)==value
  diagnostic={model:diagnostics(records[model]) for model in MODELS}
  for model,d in diagnostic.items():
   ops[source][model].update(d['operations']);stats[source][model].append(d)
  base=reference['metrics']['symbol'];deltas={model:records[model]['metrics']['symbol']['errors']-base['errors'] for model in MODELS}
  crop_rows.append(dict(head_correct={m:dict(crophead[m]) for m in MODELS},id=rid,source=source,score=cluster(rid),reference_events=base['reference'],errors={model:records[model]['metrics']['symbol']['errors'] for model in MODELS},delta_errors_vs_frozen=deltas,diagnostics=diagnostic))
  if n%500==0:print('Audited',n,'crops',flush=True)
  for model in MODELS:rows[model][rid]=dict(metrics=records[model]['metrics'])
 with (args.out/'crop-audit.jsonl').open('w') as f:
  for r in crop_rows:f.write(json.dumps(r)+'\n')
 pairs=[('frozen_public',m) for m in MODELS[1:]]+[('public_joint27','random80'),('random73','random80')]
 lines=['# Four-model cached autoregressive comparison','','Matched 3,909 crops; all saved recognition edit distances recomputed exactly. Pixel/label/protocol matching passes. No GPU or new inference. Internal diagnostic crops, some core-seen score identities; raw outputs without runtime repairs.','','Frozen reference: public-rareNum-20260927 + 8,400-update real-only frozen-core heads. Public joint 27 started from that public core; random 73/80 started from an ImageNet encoder with a random music decoder/heads on 2026-09-29.','','| Source | Metric | Frozen | Public joint 27 | Random 73 | Random 80 |','|---|---|---:|---:|---:|---:|']
 rng=np.random.default_rng(20261001);score_rows=[]
 for source in SOURCES:
  subset=[r for r in crop_rows if r['source']==source];report=dict(crops=len(subset),models={},paired_comparisons={},heads_common_all_four={},heads_end_to_end={})
  for model in MODELS:
   modelrows=[rows[model][r['id']] for r in subset];metric_report={}
   for metric in ('symbol','rhythm','pitch','accidental'):
    errors=sum(r['metrics'][metric]['errors'] for r in modelrows);den=sum(r['metrics'][metric]['reference'] for r in modelrows)
    metric_report[metric]=dict(errors=errors,reference=den,error_rate=errors/max(1,den))
   ds=stats[source][model];report['models'][model]=dict(invalid_pitch_notes=sum(d['invalid_pitch_notes'] for d in ds),invalid_pitch_crops=sum(d['invalid_pitch_notes']>0 for d in ds),invalid_staff_notes=sum(d['invalid_staff_notes'] for d in ds),invalid_accidental_notes=sum(d['invalid_accidental_notes'] for d in ds),metrics=metric_report,operations=dict(ops[source][model]),short_output_crops=sum(d['length_ratio']<.75 for d in ds),long_output_crops=sum(d['length_ratio']>1.25 for d in ds),possible_sequence_limit_crops=sum(d['possible_sequence_limit'] for d in ds),key_signature_mismatch_crops=sum(d['key_signature_mismatch'] for d in ds),excess_repeat_crops=sum(d['excess_repeated_fourgrams']>0 for d in ds),length_ratio_percentiles=np.percentile([d['length_ratio'] for d in ds],[0,10,50,90,100]).tolist(),exact_crops=sum(r['metrics']['symbol']['errors']==0 for r in modelrows))
  for metric in ('symbol','rhythm','pitch','accidental'):lines.append('| '+source+' | '+metric+' | '+' | '.join(f"{report['models'][m]['metrics'][metric]['error_rate']*100:.3f}%" for m in MODELS)+' |')
  for base,model in pairs:
   pair_report={}
   for metric in ('symbol','rhythm','pitch','accidental'):
    groups=defaultdict(lambda:np.zeros(3))
    for r in subset:
     a,b=rows[base][r['id']]['metrics'][metric],rows[model][r['id']]['metrics'][metric];assert a['reference']==b['reference'];groups[r['score']]+=np.array([a['errors'],b['errors'],a['reference']])
    v=np.array(list(groups.values()));sums=v.sum(0);boot=v[rng.integers(0,len(v),(args.bootstrap,len(v)))].sum(1);d=100*(boot[:,1]-boot[:,0])/np.maximum(1,boot[:,2]);pair_report[metric]=dict(delta_pp=float(100*(sums[1]-sums[0])/max(1,sums[2])),ci95_pp=np.percentile(d,[2.5,97.5]).tolist(),score_clusters=len(v))
   report['paired_comparisons'][base+'__'+model]=pair_report
  for key,n in common[source].items():report['heads_common_all_four'][key]=dict(support=n,accuracy={m:correct[source][m][key]/n for m in MODELS})
  for key,n in total_support[source].items():report['heads_end_to_end'][key]=dict(support=n,coverage={m:endcorrect[source][m][key]/n for m in MODELS})
  for model in MODELS[1:]:
   ranked=sorted(subset,key=lambda r:-r['delta_errors_vs_frozen'][model]);positive=sum(max(0,r['delta_errors_vs_frozen'][model]) for r in subset)
   scores=defaultdict(lambda:dict(delta=0,positive=0,reference=0,crops=0))
   for r in subset:
    d=scores[r['score']];delta=r['delta_errors_vs_frozen'][model];d['delta']+=delta;d['positive']+=max(0,delta);d['reference']+=r['reference_events'];d['crops']+=1
   report['models'][model]['regression_distribution']=dict(worse_crops=sum(r['delta_errors_vs_frozen'][model]>0 for r in subset),better_crops=sum(r['delta_errors_vs_frozen'][model]<0 for r in subset),same_crops=sum(r['delta_errors_vs_frozen'][model]==0 for r in subset),positive_excess_errors=positive,top10_positive_share=sum(max(0,r['delta_errors_vs_frozen'][model]) for r in ranked[:10])/max(1,positive),top5percent_positive_share=sum(max(0,r['delta_errors_vs_frozen'][model]) for r in ranked[:max(1,int(len(subset)*.05))])/max(1,positive),top5percent_removed_delta_errors=sum(r['delta_errors_vs_frozen'][model] for r in ranked[max(1,int(len(subset)*.05)):]))
   for score,data in scores.items():score_rows.append(dict(source=source,model=model,score=score,**data))
  source_reports[source]=report
 lines+=['','## Paired score-clustered comparisons','','Candidate minus reference in percentage points; positive means worse. Fixed seed 20261001; '+str(args.bootstrap)+' bootstrap resamples.','','| Source | Reference → candidate | Metric | Delta pp | 95% interval |','|---|---|---|---:|---|']
 for source,r in source_reports.items():
  for pair,metrics in r['paired_comparisons'].items():
   for metric,v in metrics.items():lines.append(f"| {source} | {pair.replace('__',' → ')} | {metric} | {v['delta_pp']:+.3f} | [{v['ci95_pp'][0]:+.3f}, {v['ci95_pp'][1]:+.3f}] |")
 lines+=['','## Four-way common-note and end-to-end notation','','Common-note accuracy uses exactly the same notes recognized correctly by all four models. GT-note coverage counts recognition mistakes against each model. NONE-dominated tie/slur-event accuracy is not span F1.','','| Source | Field | Common-note support | Frozen / public27 / random73 / random80 accuracy | GT-note coverage in the same order |','|---|---|---:|---|---|']
 for source,r in source_reports.items():
  for key,v in r['heads_common_all_four'].items():
   lines.append(f"| {source} | {key} | {v['support']} | "+' / '.join(f"{100*v['accuracy'][m]:.2f}%" for m in MODELS)+' | '+' / '.join(f"{100*r['heads_end_to_end'][key]['coverage'][m]:.2f}%" for m in MODELS)+' |')
 (args.out/'comparison.md').write_text('\n'.join(lines)+'\n')
 (args.out/'comparison.json').write_text(json.dumps(dict(models=MODELS,identities=identities,paired_crops=len(crop_rows),saved_metrics_recomputed=True,sources=source_reports),indent=2)+'\n')
 with (args.out/'score-audit.jsonl').open('w') as f:
  for r in sorted(score_rows,key=lambda r:(r['source'],r['model'],-r['delta'])):f.write(json.dumps(r)+'\n')
 # Deterministic selected examples; cap one source/model/category entry per score.
 chosen=[];used=set()
 for source in SOURCES:
  sub=[r for r in crop_rows if r['source']==source]
  for category,ranking in [('largest_random80_regression',lambda r:r['delta_errors_vs_frozen']['random80']),('largest_random80_improvement',lambda r:-r['delta_errors_vs_frozen']['random80']),('accidental_regression',lambda r:rows['random80'][r['id']]['metrics']['accidental']['errors']-rows['frozen_public'][r['id']]['metrics']['accidental']['errors']),('short_output',lambda r:1-r['diagnostics']['random80']['length_ratio']),('excess_repetition',lambda r:r['diagnostics']['random80']['excess_repeated_fourgrams'])]:
   ranked=sorted(sub,key=lambda r:(-ranking(r),r['id']))
   for r in ranked:
    if r['id'] not in used and ranking(r)>0:
     chosen.append(dict(category=category,**r));used.add(r['id']);break
 for source in ('ossq','lieder'):
  pool=[r for r in crop_rows if r['source']==source and r['errors']['frozen_public']==0 and r['errors']['random80']==0 and r['id'] not in used]
  if pool:
   rank=lambda r:sum(r['head_correct']['random80'].get(k,0)-r['head_correct']['frozen_public'].get(k,0) for k in ('stem_direction','beam_vector'))
   best=max(pool,key=rank)
   if rank(best)>0:chosen.append(dict(category='notation_gain_exact_recognition',**best));used.add(best['id'])
 (args.out/'selected-cases.json').write_text(json.dumps(chosen,indent=2)+'\n')
 print('Wrote comparison, crop/score audit and',len(chosen),'selected cases to',args.out,flush=True)

if __name__=='__main__':main()
