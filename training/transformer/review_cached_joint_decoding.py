"""Render selected cached decoding cases with original pixels and audited MuseScore exports."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from training.transformer.audit_cached_joint_decoding import MODELS,events,projection,sha
from training.transformer.synthetic_transfer_review import locations
from homr.transformer.vocabulary import EncodedSymbol
from homr.transformer.structured_notation import NoteNotation,BeamLevelState,StemDirection,SlurEvent,SlurSide,TieState,AdvanceClass
from homr.music_xml_generator import generate_xml,XmlGeneratorArguments
from rapidfuzz.distance import Levenshtein

LABELS={'reference':'Reference labels','frozen_public':'Public core + frozen real-only heads','public_joint27':'Public-initialized joint · epoch 27','random73':'ImageNet/random music · epoch 73','random80':'ImageNet/random music · epoch 80'}
CATEGORIES={'largest_random80_regression':'Largest recognition regression','largest_random80_improvement':'Largest recognition improvement','accidental_regression':'Accidental regression','short_output':'Short-output diagnostic','excess_repetition':'Repeated-pattern diagnostic','notation_gain_exact_recognition':'Notation gain with exact recognition'}

def restore(rows):
 out=[]
 for r in rows:
  kw={k:r[k] for k in ('rhythm','pitch','lift','articulation','slur','position')}
  n=r.get('notation')
  if n:kw['notation']=NoteNotation(beam_levels=tuple(BeamLevelState(x) for x in n['beams']),stem=StemDirection(n['stem']),slurs=tuple((SlurEvent(e),SlurSide(s)) for e,s in n['slurs']),tie=TieState(n['tie']),advance=AdvanceClass(n['advance']))
  out.append(EncodedSymbol(**kw))
 return out

def pixel_hash(path):
 # Exact audited unaugmented item_for operations; no training augmentation imports.
 import cv2,numpy as np,torch
 from homr.staff_parsing import add_image_into_tr_omr_canvas
 a=add_image_into_tr_omr_canvas(cv2.imread(str(path),cv2.IMREAD_COLOR));a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
 x=torch.tensor(a.astype(np.float32)/255.0);x=(x-torch.tensor([.7931]).view(1,1,1))/torch.tensor([.1738]).view(1,1,1)
 return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()

def image_for(workspace,r):
 rid=r['id'];name=Path(rid).stem
 if r['source']=='ossq':candidates=[workspace/'homr-artifacts/gpu-roadmap-20260919/ossq-scanned-v1/dataset/scanned'/rid.split('/scanned/',1)[1]];candidates=[p.with_suffix('.png') for p in candidates]
 elif r['source']=='lieder':candidates=[workspace/'homr-artifacts/lieder-v8/pairs'/f'{name}.png']
 else:candidates=[workspace/'homr/datasets/pdmx/out'/rid.split('/out/',1)[1]];candidates=[p.with_suffix('.jpg') for p in candidates]
 for p in candidates:
  if p.exists() and pixel_hash(p)==r['pixel_sha256']:return p
 raise ValueError('No matching original pixels: '+rid)

def xml_events(text):
 root=ET.fromstring(text);notes={};divisions=Fraction(1)
 for part in root.findall('part'):
  for mi,measure in enumerate(part.findall('measure'),1):
   cursor=Fraction(0);last=Fraction(0)
   for child in measure:
    if child.tag=='attributes':
     value=child.findtext('divisions')
     if value:divisions=Fraction(value)
    elif child.tag in ('backup','forward'):
     cursor+=(1 if child.tag=='forward' else -1)*Fraction(child.findtext('duration','0'))/divisions
    elif child.tag=='note':
     duration=Fraction(child.findtext('duration','0'))/divisions;onset=last if child.find('chord') is not None else cursor
     if child.find('chord') is None:last=onset;cursor+=duration
     pitch=child.find('pitch')
     if pitch is None:continue
     key=(part.attrib.get('id'),mi,str(onset),pitch.findtext('step'),str(Fraction(pitch.findtext('alter','0'))),pitch.findtext('octave'),str(duration),child.findtext('staff','1'))
     value=dict(stem=child.findtext('stem'),beams=tuple((b.attrib.get('number'),b.text) for b in child.findall('beam')))
     notes.setdefault(key,[]).append(value)
 return notes

def audit_xml(original,exported):
 a,b=xml_events(original),xml_events(exported)
 recognition=Counter({k:len(v) for k,v in a.items()})!=Counter({k:len(v) for k,v in b.items()})
 changes=Counter()
 for key,values in a.items():
  for left,right in zip(values,b.get(key,[])):
   if left['stem'] is not None and left['stem']!=right['stem']:changes['stem']+=1
   if left['beams']!=right['beams']:changes['beams']+=1
 return dict(pitch_duration_staff_onset_changed=recognition,engraving_changes=dict(changes),original_pitched_notes=sum(map(len,a.values())),exported_pitched_notes=sum(map(len,b.values())))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['audit','previous','random','workspace','musescore','out']:p.add_argument('--'+k,type=Path,required=True)
 args=p.parse_args();out=args.out.resolve();assets=out/'assets';assets.mkdir(parents=True,exist_ok=True)
 cases=json.loads((args.audit/'selected-cases.json').read_text());wanted={r['id'] for r in cases};records={m:{} for m in MODELS}
 for model,root,arm in [('frozen_public',args.previous,'frozen'),('public_joint27',args.previous,'joint'),('random73',args.random/'epoch-73','joint'),('random80',args.random/'epoch-80','joint')]:
  for file in root.glob(f'part-*/{arm}.jsonl'):
   with file.open() as f:
    for line in f:
     rid=line.split('"id": "',1)[1].split('"',1)[0]
     if rid in wanted:records[model][rid]=json.loads(line)
  assert set(records[model])==wanted
 os.environ['HOMR_STRUCTURED_EXPORT']='beam,stem,tie,slur'
 jobs=[];items=[]
 for ci,c in enumerate(cases,1):
  r=records['frozen_public'][c['id']];source=image_for(args.workspace.resolve(),r);key=f'case-{ci:02d}';source_target=assets/(key+'-original'+source.suffix);shutil.copy2(source,source_target)
  reference=r['reference'];symbols=restore(reference);loc=locations(symbols);event_rows=events(reference);index={id(row):i for i,row in enumerate(reference)}
  refproj=projection(reference,'symbol');candidateproj=projection(records['random80'][c['id']]['predicted'],'symbol');edits=Levenshtein.editops(refproj,candidateproj)
  labels=[]
  def human(value):
   if value is None:return 'end of crop'
   rhythm,pitch,lift=value
   if rhythm.startswith(('note_','rest_')):
    kind,duration=rhythm.split('_',1);dots=duration.count('.');plain=duration.replace('.','')
    names={'1':'whole','2':'half','4':'quarter','8':'eighth','16':'sixteenth','32':'thirty-second','64':'sixty-fourth','3':'half-note triplet','6':'quarter-note triplet','12':'eighth-note triplet','24':'sixteenth-note triplet','48':'thirty-second-note triplet'}
    name=('dotted ' if dots==1 else 'double-dotted ' if dots==2 else '')+names.get(plain,'duration '+duration)
    accidental={'b':'♭','#':'♯','bb':'𝄫','##':'𝄪','N':'♮'}.get(lift,'')
    rendered_pitch=pitch[:-1]+accidental+pitch[-1] if pitch not in ('_','.') else ''
    return name+' '+kind+(' '+rendered_pitch if rendered_pitch else '')
   return rhythm.replace('keySignature_','key signature ')
  for edit in edits[:6]:
   i=min(edit.src_pos,max(0,len(event_rows)-1));where=loc.get(index[id(event_rows[i])],'Score attribute at crop start') if event_rows else 'Crop start'
   a=refproj[edit.src_pos] if edit.src_pos<len(refproj) else None;b=candidateproj[edit.dest_pos] if edit.dest_pos<len(candidateproj) else None
   message={'replace':f'reference {human(a)} → epoch 80 {human(b)}','delete':f'epoch 80 omits reference {human(a)}','insert':f'epoch 80 adds {human(b)} before this position'}[edit.tag]
   labels.append('Crop-local '+where+' — '+message)
  if c['category']=='notation_gain_exact_recognition':
   candidate=records['random80'][c['id']];maps={m:dict(records[m][c['id']]['alignment']['pairs']) for m in ['frozen_public','random80']}
   notes=[(i,s) for i,s in enumerate(reference) if s['rhythm'].startswith('note')]
   # Match the original scorer's sorted reference-note order back to raw symbols.
   byvalue={};available=list(notes)
   for ni,note in enumerate(r['reference_notes']):
    idx,symbol=next((i,s) for i,s in available if s==note);available.remove((idx,symbol));byvalue[ni]=idx
    if ni not in maps['frozen_public'] or ni not in maps['random80']:continue
    a=r['predicted_notes'][maps['frozen_public'][ni]].get('notation',{});b=candidate['predicted_notes'][maps['random80'][ni]].get('notation',{});gt=note.get('notation',{})
    if gt.get('stem') in ('up','down') and a.get('stem')!=gt['stem'] and b.get('stem')==gt['stem']:
     labels.append('Crop-local '+loc.get(idx,'Unmapped note')+f" — stem: reference {gt['stem']}; frozen {a.get('stem')}; epoch 80 {b.get('stem')}")
     if len(labels)>=6:break
  c.update(key=key,original_image='assets/'+source_target.name,original_pixel_hash_verified=True,musical_labels=labels,panels={})
  for model in ['reference',*MODELS]:
   values=reference if model=='reference' else records[model][c['id']]['predicted']
   try:
    (assets/f'{key}-{model}.symbols.json').write_text(json.dumps(values,indent=2)+'\n')
    xml=generate_xml(XmlGeneratorArguments(),[restore(values)],LABELS[model]);path=assets/f'{key}-{model}.musicxml';path.write_text(ET.tostring(xml,encoding='unicode'))
    png=assets/f'{key}-{model}.png';exported=assets/f'{key}-{model}.exported.musicxml'
    if not (exported.exists() and (png.exists() or list(assets.glob(png.stem+'-*.png')))):
     jobs.extend([{'in':str(path),'out':str(png)},{'in':str(path),'out':str(exported)}])
    items.append((c,model,path,png,exported,values))
   except Exception as e:c['panels'][model]=dict(error='MusicXML reconstruction failed: '+str(e))
  print('Prepared',key,c['source'],c['category'],flush=True)
 (out/'render-jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
 cmd=['xvfb-run','-a',str(args.musescore.resolve()),'--appimage-extract-and-run','--force','-j',str(out/'render-jobs.json')]
 if jobs:
  with (out/'musescore.log').open('w') as f:
   result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 else:
  result=subprocess.CompletedProcess(cmd,0)
 (out/'renderer-provenance.json').write_text(json.dumps(dict(version='4.6.5',binary_sha256=sha(args.musescore),returncode=result.returncode,command=cmd,render_jobs=len(jobs),cached_exports_reused=not bool(jobs),structured_export='beam,stem,tie,slur; advance disabled equally to retain rhythm-token timing'),indent=2)+'\n')
 from PIL import Image,ImageChops
 for c,model,path,png,exported,values in items:
  pages=sorted(assets.glob(png.stem+'-*.png'))
  if png.exists():pages=[png]
  if not pages or not exported.exists():c['panels'][model]=dict(error='MuseScore did not export this panel');continue
  cropped=[]
  for page in pages:
   im=Image.open(page).convert('RGBA');white=Image.new('RGBA',im.size,'white');im=Image.alpha_composite(white,im).convert('RGB');box=ImageChops.difference(im,Image.new('RGB',im.size,'white')).getbbox()
   if box:im=im.crop((max(0,box[0]-12),max(0,box[1]-12),min(im.width,box[2]+12),min(im.height,box[3]+12)))
   dest=page.with_name(page.stem+'-crop.jpg');im.save(dest,quality=90);cropped.append('assets/'+dest.name)
  audit=audit_xml(path.read_text(),exported.read_text());audit['raw_note_symbols']=sum(v['rhythm'].startswith('note') for v in values);audit['raw_to_xml_note_count_changed']=audit['raw_note_symbols']!=audit['original_pitched_notes'];c['panels'][model]=dict(images=cropped,audit=audit,source_xml='assets/'+path.name,exported_xml='assets/'+exported.name)
 (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
 cards=[]
 for c in cases:
  panels=[]
  for model in ['reference',*MODELS]:
   panel=c['panels'].get(model,{})
   if 'error' in panel:content='<p class="warning">'+html.escape(panel['error'])+'</p>'
   else:
    audit=panel['audit'];warning=''
    if audit['raw_to_xml_note_count_changed']:warning=f'<p class="warning">MusicXML reconstruction retained {audit["original_pitched_notes"]} of {audit["raw_note_symbols"]} raw note symbols. Invalid predictions cannot all be rendered; this panel is not faithful raw output.</p>'
    if audit['pitch_duration_staff_onset_changed']:warning+='<p class="warning">MuseScore changed pitched-note timing/content. This panel is diagnostic reconstruction, not faithful raw output; consult encoded labels and MusicXML.</p>'
    elif audit['engraving_changes']:warning+='<p class="warning">MuseScore re-engraved '+html.escape(', '.join(f'{n} {field}' for field,n in audit['engraving_changes'].items()))+'. Read the encoded differences above when judging notation.</p>'
    content=warning+f'<p><a href="assets/{c["key"]}-{model}.symbols.json">Raw encoded symbols</a></p>'+''.join(f'<a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a>' for url in panel['images'])+f'<p><a href="{panel["source_xml"]}">Input MusicXML</a> · <a href="{panel["exported_xml"]}">MuseScore export</a></p>'
   errors='' if model=='reference' else f'<p>Symbol edits: {c["errors"][model]} / {c["reference_events"]} reference events.</p>'
   panels.append('<section><h3>'+LABELS[model]+'</h3>'+errors+content+'</section>')
  labels=''.join('<li>'+html.escape(label)+'</li>' for label in c['musical_labels'])
  cards.append(f'<article id="{c["key"]}"><h2>{c["source"].upper()} · {CATEGORIES[c["category"]]}</h2><p class="id">{html.escape(Path(c["id"]).name)}</p><p>Selected example; not a representative sample.</p><a href="{c["original_image"]}" target="_blank"><img class="original" src="{c["original_image"]}"></a><ul>{labels}</ul><div class="panels">'+''.join(panels)+'</div></article>')
 (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Four-model decoding review</title><style>
body{font:16px/1.5 system-ui;background:#f3f5f9;color:#17243b;margin:24px}article{background:white;border:1px solid #d3dbe8;border-radius:12px;padding:20px;margin:28px 0}.original{width:100%;max-height:480px;object-fit:contain}.panels{display:grid;grid-template-columns:repeat(5,minmax(240px,1fr));gap:16px;overflow:auto}.panels img{width:100%}.panels section{min-width:240px}h3{font-size:15px}.warning{background:#fff0c5;padding:8px;font-size:13px}.id{overflow-wrap:anywhere;font-size:13px}a{color:#1658a5}li{margin:6px 0}</style>
<h1>Four-model decoding review</h1><p>Compare the original crop with reference labels, the frozen public reference, public-initialized joint epoch 27, and ImageNet/random-music epochs 73/80. No individual decisions are required: these examples explain where recognition deteriorates and where notation improves.</p>
<p>The original image matches the benchmark pixel hash. Reconstructed panels use MuseScore 4.6.5; export checks flag changes to note timing/content, stems or beams. Those warnings are part of the result, not a hidden renderer substitution. Rhythmic-advance export is disabled equally in every panel, retaining rhythm-token timing.</p>
<p>Measure and beat labels are crop-local. Printed score measure numbers may differ. For short-output and repeated-pattern examples, the labels describe what differs, not proof of an early-EOS or repetition mechanism. The full render may span multiple images; click to enlarge.</p><p><a href="../comparison.md">Four-model numerical report</a> · <a href="../error-audit.md">Error audit and interpretation</a></p>'''+''.join(cards)+'</html>')
 print('Gallery complete:',out/'index.html',flush=True)

if __name__=='__main__':main()
