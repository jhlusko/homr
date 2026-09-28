"""Generate an original-MusicXML head pilot, quarantined pending visual qualification.

Run with ``python -m training.omr_datasets.synthetic_head_curriculum --out DIR``.
MuseScore renders the original MusicXML; the parser uses the invoking environment.
Nothing here trains a model or declares XML intent to be verified rendered truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import xml.etree.ElementTree as ET

VERSION = "synthetic-head-pilot.v5"
# Supplements share the v4 musical families and splits; schema changes must not
# create a second identity for the same melody.
FAMILY_SEED_VERSION = "synthetic-head-pilot.v4"
KINDS = ("stems", "ties", "slurs", "beams")
DIVISIONS = 240
MUSIC_FONTS = ("Leland", "Bravura", "Petaluma", "MuseJazz", "Gonville")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_seed(seed: int, family: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{FAMILY_SEED_VERSION}:{seed}:{family}".encode()).digest()[:4], "big")


def split_for(seed: int, family: int) -> str:
    value = family_seed(seed, family) % 10
    return "valid" if value == 0 else "test" if value == 1 else "train"


def child(node, tag, text=None, **attrs):
    value = ET.SubElement(node, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})
    if text is not None: value.text = str(text)
    return value


def build_score(seed: int, family: int, variant: int) -> tuple[str, dict]:
    """Four contrasts per family; all derived images share a family split."""
    if variant not in range(5) or (variant == 4 and KINDS[family % len(KINDS)] != "slurs"):
        raise ValueError("variant 4 is the reversed-overlap slur contrast; other motifs use 0..3")
    rng = random.Random(family_seed(seed, family))
    kind = KINDS[family % len(KINDS)]
    pitches = [("CDEFGAB"[p % 7], 4 + p // 7) for p in [rng.randrange(0, 12) for _ in range(8)]]
    if kind == "ties" or (kind == "slurs" and family // 4 % 2 == 0):
        pitches = [pitches[0]] * 8
    score = ET.Element("score-partwise", version="4.0")
    partlist = child(score, "part-list")
    named = child(partlist, "score-part", id="P1")
    child(named, "part-name", "")
    part = child(score, "part", id="P1")
    locations, counter = [], 0
    depth = 1 + family // 4 % 4

    def note(measure, duration, note_type, pitch, beat, *, chord=False, rest=False, stem="up", beams=None, ties=(), slurs=()):
        nonlocal counter
        counter += 1
        n = child(measure, "note", id=f"n{counter:04d}")
        if chord: child(n, "chord")
        if rest: child(n, "rest")
        else:
            p = child(n, "pitch")
            child(p, "step", pitch[0]); child(p, "octave", pitch[1])
        child(n, "duration", duration)
        for event in ties: child(n, "tie", type=event)
        child(n, "voice", 1); child(n, "type", note_type)
        if not rest: child(n, "stem", stem)
        for level, state in (beams or {}).items(): child(n, "beam", state, number=level)
        if ties or slurs:
            notations = child(n, "notations")
            for event in ties: child(notations, "tied", type=event)
            for number, event, side in slurs: child(notations, "slur", type=event, number=number, placement=side)
        locations.append(dict(id=n.get("id"), measure=int(measure.get("number")), beat=str(Fraction(beat, DIVISIONS)+1),
                              pitch=None if rest else f"{pitch[0]}{pitch[1]}",note_type=note_type,
                              chord=chord, rest=rest, stem=None if rest else stem, ties=list(ties), slurs=list(slurs), beams=beams or {}))

    for m in range(2):
        measure = child(part, "measure", number=m+1)
        if m == 0:
            attrs = child(measure, "attributes")
            child(attrs, "divisions", DIVISIONS)
            key = child(attrs, "key"); child(key, "fifths", 0)
            meter = child(attrs, "time"); child(meter, "beats", 4); child(meter, "beat-type", 4)
            clef = child(attrs, "clef"); child(clef, "sign", "G"); child(clef, "line", 2)
        if kind != "beams":
            for i in range(4):
                index = m*4+i
                ties, slurs = [], []
                if kind == "ties":
                    pairs = [(0,1)] if variant == 1 else [(0,2)] if variant == 2 else [(3,4)] if variant == 3 else []
                    for start, stop in pairs:
                        if index == start: ties.append("start")
                        elif index == stop: ties.append("stop")
                        elif start < index < stop: ties.extend(("stop", "start"))
                    if variant == 3:
                        if index == 0: slurs.append((1,"start","above"))
                        if index == 2: slurs.append((1,"stop","above"))
                if kind == "slurs":
                    spans = [(1,0,3,"above")] if variant == 1 else [(1,0,3,"below")] if variant == 2 else [(1,0,5,"above"),(2,2,7,"below")] if variant == 3 else [(1,0,5,"below"),(2,2,7,"above")] if variant == 4 else []
                    for number,start,stop,side in spans:
                        if index == start: slurs.append((number,"start",side))
                        if index == stop: slurs.append((number,"stop",side))
                stem = "down" if kind == "stems" and variant % 2 else "up"
                note(measure,DIVISIONS,"quarter",pitches[index],i*DIVISIONS,stem=stem,ties=ties,slurs=slurs)
                if kind == "stems" and variant >= 2:
                    step, octave = pitches[index]
                    p = "CDEFGAB".index(step) + (octave-4)*7 + 2
                    note(measure,DIVISIONS,"quarter",("CDEFGAB"[p%7],4+p//7),i*DIVISIONS,chord=True,stem=stem)
        else:
            duration = DIVISIONS // (2**depth)
            note_type = ("eighth","16th","32nd","64th")[depth-1]
            # A hook must be isolated at its level. MuseScore ignores direction
            # overrides among equal-duration notes, so use mixed durations within
            # one beat for the forward/backward examples at levels 2--4.
            pattern = [(note_type,duration)]*4
            if depth >= 2 and variant >= 2:
                shallow = ("eighth","16th","32nd")[depth-2]
                pattern = [(note_type,duration),(shallow,duration*2)]
                if variant == 3: pattern.reverse()
            onset = 0
            for i,(written_type,written_duration) in enumerate(pattern):
                beams = {}
                for level in range(1,depth+1):
                    state = "begin" if i == 0 else "end" if i == len(pattern)-1 else "continue"
                    if depth == 1 and variant >= 2:
                        state = "begin" if i % 2 == 0 else "end"
                    elif depth >= 2 and variant >= 2 and level == depth:
                        if written_type != note_type: continue
                        state = "forward hook" if variant == 2 else "backward hook"
                    beams[level] = state
                stem = "down" if variant == 1 or (depth == 1 and variant == 3) else "up"
                note(measure,written_duration,written_type,pitches[(m*4+i)%8],onset,
                     stem=stem,beams=beams)
                onset += written_duration
            for rest_type,rest_duration in (("quarter",240),("eighth",120),("16th",60),("32nd",30),("64th",15)):
                while onset + rest_duration <= DIVISIONS*4:
                    note(measure,rest_duration,rest_type,pitches[0],onset,rest=True)
                    onset += rest_duration
            if onset != DIVISIONS*4: raise ValueError("Beam measure does not fill 4/4")
        child(measure, "barline", location="right").append(ET.Element("bar-style"))
        measure.find("barline/bar-style").text = "light-light" if m == 1 else "regular"
    metadata = dict(family=f"family-{family:04d}",kind=kind,variant=variant,seed=family_seed(seed,family),
                    split=split_for(seed,family),beam_depth=depth if kind=="beams" else None,source_notes=locations)
    return ET.tostring(score,encoding="unicode",xml_declaration=True), metadata


def parse_symbols(xml):
    from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
    from training.transformer.training_vocabulary import check_token_lines
    parts=music_xml_string_to_tokens(xml)
    if len(parts)!=1: raise ValueError("Expected exactly one staff")
    symbols=[symbol for measure in parts[0] for symbol in measure]
    check_token_lines(symbols)
    return symbols


def audit_renderer_labels(source_xml, exported_xml):
    """Detect importer changes; agreement still requires a visible-label review."""
    from training.omr_datasets.structured_notation_parser import parse_score
    from homr.transformer.structured_notation import SlurSide
    source,_ = parse_score(ET.fromstring(source_xml))
    exported,findings = parse_score(ET.fromstring(exported_xml))
    source_notes = [n for part in source.values() for n in part]
    exported_notes = [n for part in exported.values() for n in part]
    differences = []
    unstated_sides = 0
    if len(source_notes) != len(exported_notes):
        differences.append(dict(field="note_count",source=len(source_notes),exported=len(exported_notes)))
    if not findings.clean or findings.ambiguous_beaming or findings.voices_beyond_cap:
        differences.append(dict(field="exported_parser_findings",detail=str(findings)))
    for index,(original,rendered) in enumerate(zip(source_notes,exported_notes)):
        for field,left,right in (("beams",original.beam_levels[:4],rendered.beam_levels[:4]),
                                 ("stem",original.stem,rendered.stem),("tie",original.tie,rendered.tie)):
            if left != right: differences.append(dict(note=index,field=field,source=left,exported=right))
        for slot,(left,right) in enumerate(zip(original.slurs[:2],rendered.slurs[:2])):
            if left[0] != right[0]:
                differences.append(dict(note=index,field=f"slur{slot+1}",source=left[0],exported=right[0]))
            if right[1] == SlurSide.UNSPECIFIED and left[1] != right[1]:
                unstated_sides += 1
            elif left[1] != right[1]:
                differences.append(dict(note=index,field=f"slur_side{slot+1}",source=left[1],exported=right[1]))
    return dict(differences=differences,unstated_exported_slur_sides=unstated_sides,
                source_notes=len(source_notes),exported_notes=len(exported_notes),
                visually_verified=False)


def generate_case(task):
    out, seed, family, variant, font, musescore_bin, style_file, revision = task
    started=time.monotonic()
    xml, info=build_score(seed,family,variant)
    name=f"{info['family']}-{info['kind']}-v{variant}-{font}"
    directory=Path(out)/"samples"; directory.mkdir(exist_ok=True)
    stem=directory/name
    try:
        from PIL import Image, ImageChops
        from training.omr_datasets.notation_sidecar import write_sidecar, attach_sidecar
        from training.omr_datasets.structured_notation_parser import parse_score
        from training.transformer.training_vocabulary import token_lines_to_str, read_tokens
        from homr.transformer.vocabulary import serialized_chords
        _, findings = parse_score(ET.fromstring(xml))
        if not findings.clean or findings.ambiguous_beaming:
            raise ValueError(f"Structured labels failed audit: {findings}")
        symbols=parse_symbols(xml)
        token=stem.with_suffix(".tokens")
        token.write_text(token_lines_to_str(symbols))
        sidecar=write_sidecar(token,symbols)
        if sidecar is None: raise ValueError("Missing structured labels")
        loaded=read_tokens(str(token)); attach_sidecar(token,loaded)
        ordered=[s for chord in serialized_chords(symbols) for s in chord if s.notation is not None]
        assert [s.notation for s in loaded if s.notation is not None]==[s.notation for s in ordered]
        original=stem.with_suffix(".musicxml"); original.write_text(xml)
        exported=stem.with_name(stem.name+"-export.png")
        exported_xml=stem.with_suffix(".rendered.musicxml")
        job=stem.with_suffix(".render-job.json")
        job.write_text(json.dumps([
            dict(**{"in":str(original)},out=str(exported),style=style_file),
            dict(**{"in":str(original)},out=str(exported_xml),style=style_file)
        ],indent=2)+"\n")
        command=["xvfb-run","-a",musescore_bin]
        if musescore_bin.lower().endswith(".appimage"):
            command.append("--appimage-extract-and-run")
        command.extend(["--force","-j",str(job)])
        subprocess.run(command,capture_output=True,timeout=90,check=True)
        renderer_audit=audit_renderer_labels(xml,exported_xml.read_text())
        rendered_png=exported.with_name(exported.stem+"-1.png")
        if not rendered_png.exists() and exported.exists():
            rendered_png=exported
        if not rendered_png.exists():
            raise ValueError("MuseScore completed without producing a PNG")
        rendered_image=Image.open(rendered_png).convert("RGBA")
        white=Image.new("RGBA",rendered_image.size,"white")
        image=Image.alpha_composite(white,rendered_image).convert("RGB")
        box=ImageChops.difference(image,Image.new("RGB",image.size,"white")).getbbox()
        if box is None: raise ValueError("Blank rendering")
        x0,y0,x1,y1=box
        box=(max(0,x0-12),max(0,y0-12),min(image.width,x1+12),min(image.height,y1+12))
        image=image.crop(box)
        path=stem.with_suffix(".png"); image.save(path)
        rendered_png.unlink()
        payload=json.loads(sidecar.read_text())
        support=Counter()
        for item in payload["notation"]:
            support[f"stem:{item['stem']}"]+=1; support[f"tie:{item['tie']}"]+=1
            for level,state in enumerate(item["beams"][:4],1): support[f"beam{level}:{state}"]+=1
            for slot,(event,side) in enumerate(item["slurs"][:2],1):
                support[f"slur{slot}:{event}"]+=1
                if event!="none" and side!="unspecified": support[f"side{slot}:{side}"]+=1
        status="rejected_renderer_labels" if renderer_audit["differences"] else "requires_visual_review"
        info.update(id=name,font=font,source_revision=revision,status=status,renderer_audit=renderer_audit,
                    training_ready=False,seconds=time.monotonic()-started,shape=[image.width,image.height],
                    crop_box=box,renderer="MuseScore",support=dict(support),
                    annotated_symbols=payload["annotatedSymbols"],
                    files={p.name:sha(p) for p in (original,token,sidecar,path,exported_xml,job)})
        stem.with_suffix(".json").write_text(json.dumps(info,indent=2)+"\n")
        return info
    except Exception as exc:
        info.update(id=name,font=font,status="failed",training_ready=False,error=str(exc),seconds=time.monotonic()-started)
        return info


def review_caption(row):
    kind, variant = row['kind'], row['variant']
    if kind == 'stems':
        return f"{'Two-note chords' if variant >= 2 else 'Single notes'}, stems {'down' if variant % 2 else 'up'}; both measures, beats 1–4."
    if kind == 'ties':
        return ["No ties.", "Tie: measure 1, beat 1 to beat 2.",
                "Tie chain: measure 1, beats 1 → 2 → 3.",
                "Tie across barline: measure 1, beat 4 → measure 2, beat 1; slur above measure 1, beats 1–3."][variant]
    if kind == 'slurs':
        return ["No slurs.", "Slur above: measure 1, beats 1–4.",
                "Slur below: measure 1, beats 1–4.",
                "Overlapping slurs: above from measure 1, beat 1 to measure 2, beat 2; below from measure 1, beat 3 to measure 2, beat 4.",
                "Overlapping slurs: below from measure 1, beat 1 to measure 2, beat 2; above from measure 1, beat 3 to measure 2, beat 4."][variant]
    depth = row['beam_depth']
    if depth == 1:
        return f"Eighth notes in {'two pairs' if variant >= 2 else 'one four-note beam group'}; stems {'down' if variant % 2 else 'up'}; both measures, beats 1–2."
    if variant < 2:
        return f"Four notes joined at beam levels 1–{depth}; stems {'down' if variant else 'up'}; starts at beat 1 in each measure."
    hook = next(n for n in row['source_notes'] if not n['rest'] and depth in n['beams'])
    beat = Fraction(hook['beat'])
    whole, remainder = divmod(beat.numerator, beat.denominator)
    label = str(whole) + (f" + {remainder}/{beat.denominator}" if remainder else "")
    return f"Level {depth} {'forward' if variant == 2 else 'backward'} hook on the {'first' if variant == 2 else 'second'} note, beat {label} in each measure. Mixed note lengths create the hook; rhythm differs from the four-note examples."


def review_html(rows):
    import html
    chunks=['<!doctype html><meta charset="utf-8"><title>Synthetic head pilot</title>',
            '<style>body{font:16px system-ui;margin:2rem;background:#f6f7f8}section{background:white;padding:1rem;margin:1rem 0;border:1px solid #ccc}.cases{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}img{width:100%}summary{cursor:pointer}@media(max-width:800px){.cases{grid-template-columns:1fr}}</style>',
            '<h1>Synthetic head pilot: renderer qualification</h1><p>All examples await visual qualification. Compare visible notation with the musical descriptions. Click an image to enlarge. No individual preference votes are needed.</p>']
    families={}
    for row in rows: families.setdefault((row['family'],row['font']),[]).append(row)
    for (family,font),cases in sorted(families.items()):
        chunks.append(f'<section><h2>{family} — {html.escape(font)} — {cases[0]["kind"]}</h2><div class="cases">')
        for row in sorted(cases,key=lambda r:r['variant']):
            name=row['id']
            chunks.append(f'<article><h3>{html.escape(review_caption(row))}</h3><p>{html.escape(row["status"])}</p>')
            if row['status']=='failed':
                chunks.append(f'<p>Failed: {html.escape(row["error"])}</p></article>'); continue
            chunks.append(f'<a href="samples/{name}.png" target="_blank"><img loading="lazy" src="samples/{name}.png" alt="{html.escape(review_caption(row))}"></a><details><summary>Labels and provenance</summary><a href="samples/{name}.musicxml">Original MusicXML</a> · <a href="samples/{name}.rendered.musicxml">MuseScore export</a> · <a href="samples/{name}.json">Metadata</a></details></article>')
        chunks.append('</div></section>')
    return '\n'.join(chunks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--families',type=int,default=256)
    p.add_argument('--seed',type=int,default=20260928)
    p.add_argument('--family-ids',nargs='+',type=int,help='Render an explicit subset, retaining canonical family identities')
    p.add_argument('--variant-ids',nargs='+',type=int,default=[0,1,2,3])
    p.add_argument('--music-fonts',nargs='+',default=['Leland','Bravura'])
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--musescore-bin',default=str(Path('datasets/MuseScore')))
    p.add_argument('--source-revision',default='uncommitted')
    a=p.parse_args()
    if not 1<=a.families<=100000 or not 1<=a.workers<=16: p.error('Invalid families/workers')
    families=sorted(set(a.family_ids)) if a.family_ids is not None else list(range(a.families))
    variants=sorted(set(a.variant_ids))
    if not families or min(families)<0 or max(families)>=100000 or any(v not in range(5) for v in variants):
        p.error('Invalid selected family or variant IDs')
    if 4 in variants and any(KINDS[f % len(KINDS)]!='slurs' for f in families):
        p.error('Variant 4 requires slur families only')
    if any(font not in MUSIC_FONTS for font in a.music_fonts):
        p.error(f'Music fonts must be selected from {", ".join(MUSIC_FONTS)}')
    a.out.mkdir(parents=True,exist_ok=True)
    if any(a.out.iterdir()):
        raise SystemExit('Output directory must be empty; choose a new directory (no overwrites)')
    musescore=Path(a.musescore_bin)
    if not musescore.is_file() or not os.access(musescore,os.X_OK):
        p.error(f'MuseScore executable is missing or not executable: {musescore}')
    version_command=["xvfb-run","-a",str(musescore)]
    if musescore.suffix.lower()=='.appimage': version_command.append('--appimage-extract-and-run')
    version_command.append('--version')
    version_result=subprocess.run(version_command,text=True,capture_output=True,timeout=60,check=True)
    version_lines=(version_result.stdout+'\n'+version_result.stderr).splitlines()
    version=next((line.strip() for line in version_lines if 'MuseScore' in line),version_lines[0].strip())
    style_dir=a.out/'styles'; style_dir.mkdir()
    styles={}
    for font in a.music_fonts:
        style=style_dir/f'{font}.mss'
        style.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<museScore version="4.00"><Style>'
            f'<musicalSymbolFont>{font}</musicalSymbolFont>'
            f'<musicalTextFont>{font} Text</musicalTextFont>'
            '</Style></museScore>\n'
        )
        styles[font]=str(style)
    identity=dict(schema=VERSION,generator_sha256=sha(Path(__file__)),seed=a.seed,families=len(families),family_ids=families,variant_ids=variants,
                  music_fonts=a.music_fonts,source_revision=a.source_revision,musescore=version,
                  variants=len(variants),training_ready=False)
    previous=a.out/'identity.json'
    if previous.exists(): raise SystemExit('Output already initialized; choose a new directory (no overwrites)')
    previous.write_text(json.dumps(identity,indent=2)+'\n')
    os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
    tasks=[(str(a.out),a.seed,family,variant,font,str(musescore),styles[font],a.source_revision)
           for family in families for variant in variants for font in a.music_fonts]
    rows=[]; started=time.monotonic()
    with (a.out/'samples.jsonl').open('w') as handle, ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,row in enumerate(pool.map(generate_case,tasks),1):
            rows.append(row); handle.write(json.dumps(row)+'\n'); handle.flush()
            if i==1 or i%32==0 or i==len(tasks):
                elapsed=time.monotonic()-started
                progress=dict(completed=i,total=len(tasks),elapsed_seconds=elapsed,
                              remaining_seconds=elapsed/i*(len(tasks)-i),failures=sum(r['status']=='failed' for r in rows))
                (a.out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(progress),flush=True)
    summary=Counter(); support=Counter(); duplicate_contrasts=[]
    grouped={}
    for row in rows:
        grouped.setdefault((row['family'],row['font']),[]).append(row)
    for group in grouped.values():
        for i,left in enumerate(group):
            for right in group[i+1:]:
                if left['status']!='failed' and right['status']!='failed' and left['files'][left['id']+'.png']==right['files'][right['id']+'.png']:
                    duplicate_contrasts.append([left['id'],right['id']])
    duplicate_ids={name for pair in duplicate_contrasts for name in pair}
    for row in rows:
        if row['id'] in duplicate_ids:
            row['status']='rejected_identical_pixels'
        if row['status']!='failed':
            (a.out/'samples'/f"{row['id']}.json").write_text(json.dumps(row,indent=2)+'\n')
        summary[row['status']]+=1
        if row['status']=='requires_visual_review': support.update(row.get('support',{}))
    (a.out/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    for split in ('train','valid','test'):
        records=[r for r in rows if r['split']==split and r['status']=='requires_visual_review']
        (a.out/f'index_candidates_{split}.txt').write_text(''.join(f"samples/{r['id']}.png,samples/{r['id']}.tokens\n" for r in records))
        summary[f'{split}_renders']=len(records); summary[f'{split}_families']=len({r['family'] for r in records})
    report=dict(identity=identity,total=len(rows),counts=dict(summary),label_support=dict(support),
                identical_pixels_across_variants=duplicate_contrasts,training_ready=False,
                reason='Renderer intent has not been visually qualified. Candidate indexes are not training releases.',
                elapsed_seconds=time.monotonic()-started)
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'review.html').write_text(review_html(rows))
    files=sorted(f for f in a.out.rglob('*') if f.is_file())
    (a.out/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(a.out)}\n' for f in files))
    (a.out/'GENERATION_COMPLETE').write_text(json.dumps(dict(total=len(rows),failed=summary['failed']))+'\n')
    if summary['failed']: raise SystemExit(1)


if __name__=='__main__':
    main()
