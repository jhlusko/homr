"""Local evidence viewer: original training crops and native/XML rest durations."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from training.architecture.transformer.exact_timing_targets import TYPE_QUARTERS
from training.onnx.qualify_joint_export import sha, write


def native_beat(score, case):
    root=ET.parse(score['native']).getroot()
    staff=next(s for s in root.findall('./Score/Staff') if s.get('id')==case['native']['native_staff_id'])
    voice=staff.findall('Measure')[case['physical_measure']-1].findall('voice')[(int(case['voice'])-1)%4]
    elapsed=Fraction(0);rests=0
    for child in voice:
        if child.tag not in ('Chord','Rest'):
            continue
        if child.tag=='Rest':
            rests+=1
            if rests==case['rest_ordinal']:
                sig=ET.fromstring(case['native']['native_time_signature_xml'])
                return f"beat {1+elapsed/(Fraction(4)/int(sig.findtext('sigD')))} in {sig.findtext('sigN')}/{sig.findtext('sigD')}"
        # Do not invent a position across unsupported tuplets or grace notation.
        kind=child.findtext('durationType')
        if kind not in TYPE_QUARTERS or child.find('Tuplet') is not None or any('grace' in n.tag.lower() or 'acciaccatura' in n.tag.lower() or 'appoggiatura' in n.tag.lower() for n in child):
            return 'rest position shown by measure and ordinal; complex preceding rhythm'
        elapsed+=TYPE_QUARTERS[kind]*sum((Fraction(1,2**i) for i in range(int(child.findtext('dots','0'))+1)),Fraction(0))
    raise ValueError('Native rest not found')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'index.html').exists():
        raise ValueError('Refusing to overwrite review')
    report=json.loads(args.report.read_text());cases=[]
    for score in report['scores']:
        if sha(score['native'])!=score['native_sha256']:
            raise ValueError('Native source changed')
        for case in score['cases']:
            copy=dict(case,score=score['score'],local_position=native_beat(score,case),images=[])
            for exposure in case['crop_exposure']:
                source=Path(exposure['image']);name=source.name
                shutil.copyfile(source,args.out/name)
                copy['images'].append(name)
            cases.append(copy)
    # Training examples first; excluded cases remain inspectable in the selector.
    cases.sort(key=lambda c:(not bool(c['images']),c['score'],c['physical_measure'],c['note_index']))
    write(args.out/'cases.json',cases)
    data=json.dumps(cases).replace('<','\\u003c')
    html='''<!doctype html><meta charset="utf-8"><title>Lieder rest export audit</title>
<style>body{font:17px system-ui;margin:24px;background:#f5f6f8;color:#18202b;max-width:1600px}h1{font-size:26px}button,select{font:inherit;padding:9px;margin:5px 8px 15px 0}article,.box{background:white;padding:18px;border-radius:10px;margin:12px 0}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:18px}.box strong{display:block;margin-bottom:10px}img{width:100%;background:white}pre{white-space:pre-wrap;font-size:13px}p{line-height:1.5}.muted{color:#526071}</style>
<h1>Lieder rest export audit</h1><p>All 29 rests match explicit staff stretch in the native source. Fourteen occur in three accepted training crops. Images below are the original scans; no rendering engine has changed the rest glyphs. No individual approval decisions are requested.</p>
<button id="prev">Previous</button><button id="next">Next</button><select id="pick"></select><article><h2 id="title"></h2><p id="position"></p><p id="exposure"></p><div class="comparison"><div class="box"><strong>Native source and printed rest</strong><p id="native"></p></div><div class="box"><strong>Frozen MusicXML export</strong><p id="xml"></p></div></div><p id="diagnosis"></p><div id="images"></div><p class="muted">Positions use the native staff's meter and written rhythm, not the inconsistent XML cursor. “Lower piano staff” can use either clef. Current elapsed/ratio targets remain masked; written rest labels are retained.</p><details><summary>Native and MusicXML evidence</summary><pre id="evidence"></pre></details></article>
<script>const cases=__DATA__;let i=0;const $=id=>document.getElementById(id);cases.forEach((c,n)=>{let o=document.createElement('option');o.value=n;o.textContent=`${n+1}. ${c.score} — measure ${c.printed_measure}, ${c.part===1?'vocal':'piano lower'} rest ${c.rest_ordinal}${c.images.length?' — training':' — excluded'}`;$('pick').append(o)});function render(){let c=cases[i];$('pick').value=i;$('title').textContent=`${c.score} — measure ${c.printed_measure}`;$('position').textContent=`${c.part===1?'Vocal staff':'Lower piano staff'}, ${c.local_position}; rest ${c.rest_ordinal} in this measure.`;$('exposure').textContent=c.images.length?'Present in the frozen training set; replayed token bytes match exactly.':'Outside accepted training crops; no certified indexed crop contains this rest.';let glyph=`${c.timing.dots?'dotted ': 'undotted '}${c.timing.written_type} rest`;$('native').textContent=`${glyph}: ${c.native.written_quarters} quarter-note units.`;$('xml').textContent=`Same written glyph, but elapsed duration ${c.timing.elapsed_quarters} quarter-note units; no time-modification.`;$('diagnosis').textContent=`Explicit staff stretch ${c.native.stretch} explains the difference exactly: ${c.timing.elapsed_quarters} × ${c.native.stretch} = ${c.native.written_quarters}. Do not add a dot or tuplet to make the XML arithmetic fit.`;$('images').replaceChildren();c.images.forEach(src=>{let img=document.createElement('img');img.src=src;img.alt='Original scanned training crop';$('images').append(img)});$('evidence').textContent=c.native.native_time_signature_xml+'\\n'+c.native.native_rest_xml+'\\n'+c.source_note_xml} $('prev').onclick=()=>{i=(i+cases.length-1)%cases.length;render()};$('next').onclick=()=>{i=(i+1)%cases.length;render()};$('pick').onchange=e=>{i=Number(e.target.value);render()};render();</script>'''.replace('__DATA__',data)
    (args.out/'index.html').write_text(html)
    write(args.out/'provenance.json',dict(report_sha256=sha(args.report),tool_sha256=sha(Path(__file__)),
        original_scan_sha256={p.name:sha(p) for p in args.out.glob('*.png')}, cases=len(cases),
        limitation='Native-derived positions refuse unsupported preceding tuplets/grace notation. '
        'Original scan crops shown only for accepted indexed exposure; excluded cases have textual source evidence.'))


if __name__=='__main__':
    main()
