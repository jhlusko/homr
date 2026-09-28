"""Select deterministic gain/loss cases and render isolated field comparisons."""
import argparse
import copy
from dataclasses import replace
from fractions import Fraction
import html
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def select(run):
    cases=[]
    for source in ('ossq','pdmx','lieder'):
        def rows(arm):
            return {r['id']:r for r in map(json.loads,(run/arm/'eval'/f'{source}.jsonl').read_text().splitlines())}
        control=rows('real_only')
        for arm in ('unordered','curriculum'):
            candidate=rows(arm)
            for family,heads in [('stem',['stem.direction']),('slur',['slur.slot.1.side','slur.slot.2.side'])]:
                pool=[]
                for key,a in control.items():
                    b=candidate[key]
                    assert a['reference_sha256']==b['reference_sha256'] and a['pixel_sha256']==b['pixel_sha256']
                    gains=[];losses=[]
                    for head in heads:
                        for i,(ref,x,y) in enumerate(zip(a['reference'][head],a['predicted'][head],b['predicted'][head],strict=True)):
                            if ref<0 or x==y:continue
                            record=dict(position=i,head=head,reference=ref,control=x,candidate=y)
                            if x==ref and y!=ref:losses.append(record)
                            if y==ref and x!=ref:gains.append(record)
                    pool.append(dict(id=key,source=source,arm=arm,family=family,gains=gains,losses=losses,control=a,candidate=b))
                for direction in ('gains','losses'):
                    ranked=sorted((r for r in pool if r[direction]),key=lambda r:(-len(r[direction]),r['id']))
                    if ranked:
                        chosen=copy.deepcopy(ranked[0]);chosen['direction']=direction;chosen['case_id']=f'{source}-{arm}-{family}-{direction}'
                        cases.append(chosen)
    return cases


def locations(symbols):
    from homr.music_xml_generator import group_into_chords
    indices={id(s):i for i,s in enumerate(symbols)}
    result={};measure=1;onset=Fraction(0);unit=4
    for group in group_into_chords(symbols):
        rhythm=group.symbols[0].rhythm
        if rhythm.startswith('timeSignature/'):unit=int(rhythm.split('/',1)[1])
        elif rhythm.startswith(('note','rest')):
            for symbol in group.symbols:
                pitch=symbol.pitch
                if symbol.lift in ('b','bb','#','##') and pitch and pitch[-1].isdigit():pitch=pitch[:-1]+symbol.lift+pitch[-1]
                result[indices[id(symbol)]]=f'measure {measure}, beat {1+onset*unit} ({pitch})'
            onset+=group.get_duration()
        elif 'barline' in rhythm or rhythm in ('repeatEnd','repeatEndStart'):
            measure+=1;onset=Fraction(0)
    return result


def render(out,cases):
    import verovio
    from homr.music_xml_generator import XmlGeneratorArguments,generate_xml,xml_to_string
    from homr.transformer.structured_notation import STEM_CLASSES,SLUR_SIDE_CLASSES
    from training.transformer.training_vocabulary import read_tokens
    from training.omr_datasets.notation_sidecar import attach_sidecar
    cards=[]
    for c in cases:
        key=c['case_id'];token=out/'assets'/f'{key}.tokens'
        symbols=read_tokens(token);assert attach_sidecar(token,symbols)
        loc=locations(symbols)
        changes=c[c['direction']]
        labels=[]
        for d in changes:
            i=d['position'];assert i<len(symbols)
            cls=STEM_CLASSES if c['family']=='stem' else SLUR_SIDE_CLASSES
            label='Stem' if c['family']=='stem' else 'First slur' if '.1.' in d['head'] else 'Overlapping slur'
            labels.append(f"{loc.get(i,'Unmapped score position')} — {label}: reference {cls[d['reference']]}; real-only {cls[d['control']]}; {c['arm']} {cls[d['candidate']]}")
        images=[]
        for variant in ('reference','control','candidate'):
            output=copy.deepcopy(symbols)
            row=c['control'] if variant!='candidate' else c['candidate']
            values=row['reference'] if variant=='reference' else row['predicted']
            targets=row['reference']
            for i,symbol in enumerate(output):
                if symbol.notation is None:continue
                if c['family']=='stem' and targets['stem.direction'][i]>=0:
                    symbol.notation=replace(symbol.notation,stem=STEM_CLASSES[values['stem.direction'][i]])
                if c['family']=='slur':
                    pairs=list(symbol.notation.slurs)
                    for slot in range(min(2,len(pairs))):
                        head=f'slur.slot.{slot+1}.side'
                        if targets[head][i]>=0:pairs[slot]=(pairs[slot][0],SLUR_SIDE_CLASSES[values[head][i]])
                    symbol.notation=replace(symbol.notation,slurs=tuple(pairs))
            xml=xml_to_string(generate_xml(XmlGeneratorArguments(),[output],key))
            if c['family']=='stem':
                root=ET.fromstring(xml)
                # A beam group can override explicit, conflicting per-note stems.
                # Remove beam connections equally from all diagnostic panels.
                for note in root.findall('.//note'):
                    for beam in list(note.findall('beam')):note.remove(beam)
                xml=ET.tostring(root,encoding='unicode')
            (out/'assets'/f'{key}-{variant}.musicxml').write_text(xml)
            toolkit=verovio.toolkit();toolkit.setOptions(dict(pageWidth=1600,pageHeight=2500,scale=35,adjustPageHeight=True))
            if not toolkit.loadData(xml):raise ValueError(f'Cannot render {key}')
            for page in range(1,toolkit.getPageCount()+1):
                filename=f'{key}-{variant}-{page}.svg';(out/'assets'/filename).write_text(toolkit.renderToSVG(page))
                images.append((variant,page,filename+'?v='+hashlib.sha256((out/'assets'/filename).read_bytes()).hexdigest()[:12]))
        panels=''.join('<div><h3>'+html.escape(name)+'</h3>'+''.join(f'<img src="assets/{f}" alt="{name} page {p}">' for v,p,f in images if v==variant)+'</div>' for variant,name in [('reference','Reference'),('control','Real-only'),('candidate',c['arm'])])
        explanation='<p><strong>Stem diagnostic:</strong> beam connections are removed from all three rendered panels so each predicted stem direction remains visible. The original crop above retains its beams.</p>' if c['family']=='stem' else ''
        cards.append(f'<article><h2>{html.escape(key)}</h2>{explanation}<p>{html.escape(Path(c["id"]).name)}</p><img class="source" src="assets/{key}.png"><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in labels)+f'</ul><section>{panels}</section></article>')
        c['musical_labels']=labels
    (out/'cases.json').write_text(json.dumps(cases,indent=2))
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Synthetic transfer divergences</title><style>body{font:16px system-ui;margin:24px;background:#f3f4f6}article{background:white;padding:20px;margin:24px 0}section{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{width:100%}.source{max-width:1100px}li{margin:8px 0}</style><h1>Synthetic transfer divergences</h1><p>Deterministic largest gain/loss cases per corpus and field. These are selected examples, not a representative sample. Recognition tokens and other notation are held to the reference, except that beam connections are suppressed consistently in stem diagnostics to prevent engraving from overriding individual predictions. Slur events remain reference events, so these panels isolate side predictions. Measure numbers start at this crop; beats use its time signature, or quarter-note units when unavailable. These are teacher-forced head comparisons, not full pipeline output.</p>'''+''.join(cards))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--render',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.render:render(a.out,json.loads((a.out/'cases.json').read_text()))
    else:(a.out/'cases.json').write_text(json.dumps(select(a.run),indent=2))

if __name__=='__main__':main()
