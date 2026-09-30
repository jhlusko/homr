"""Plot the ImageNet-backed, fresh music-model run from local collected metadata."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SOURCES=('ossq','lieder','pdmx')
COLORS={'ossq':'#1769aa','lieder':'#088f73','pdmx':'#c46b16'}
MAIN_HEADS=('beam.level.1','stem.direction','tie.state','slur.slot.1.event','advance.delta','dynamic.mark')
RARE_HEADS=('beam.level.2','beam.level.3','beam.level.4','slur.slot.1.side','slur.slot.2.event','slur.slot.2.side')
BRANCHES=('loss_rhythm','loss_pitch','loss_lift','loss_position','loss_articulations','loss_slurs')
HEAD_LABELS={'beam.level.1':'Primary beams','beam.level.2':'Secondary beams','beam.level.3':'Third-level beams',
             'beam.level.4':'Fourth-level beams','stem.direction':'Stem direction','tie.state':'Ties',
             'slur.slot.1.event':'First slur: start / stop','slur.slot.1.side':'First slur: above / below',
             'slur.slot.2.event':'Second overlapping slur: start / stop','slur.slot.2.side':'Second overlapping slur: above / below',
             'advance.delta':'Rhythmic advance','dynamic.mark':'Dynamics (experimental)'}
NOTATION_METRICS=(('Primary beams',('beam_levels','1','macro_f1')),
                  ('Secondary beams',('beam_levels','2','macro_f1')),
                  ('Third-level beams',('beam_levels','3','macro_f1')),
                  ('Fourth-level beams',('beam_levels','4','macro_f1')),
                  ('Stem direction',('stems','macro_f1')),('Ties',('ties','macro_f1')),
                  ('Slur spans',('slur_spans','f1')),('Slur side',('slur_sides','macro_f1')),
                  ('Rhythmic advance',('advance_nontrivial','macro_f1')),
                  ('Dynamics (experimental)',('dynamics','macro_f1')))


def notation_value(report,path):
    value=report.get('structured',{})
    for key in path[:-1]:value=value.get(key,{})
    if value.get('support',1)==0:return np.nan
    return value.get(path[-1],np.nan)


def save(fig, out, name):
    for fmt in ('png','svg'):
        temp=out/f'{name}.tmp.{fmt}'
        fig.savefig(temp,dpi=155,bbox_inches='tight',format=fmt)
        temp.replace(out/f'{name}.{fmt}')
    plt.close(fig)


def plot_series(ax, x, y, *, color, label=None, smooth=40, supervised=None):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if supervised is not None:
        keep=np.asarray(supervised,dtype=int)>0;x=x[keep];y=y[keep]
    if len(x)==0:
        ax.text(.5,.5,'No supervised samples',ha='center',va='center',transform=ax.transAxes)
        return
    ax.plot(x,y,color=color,alpha=.13,lw=.55)
    window=min(smooth,max(1,len(x)//5))
    if window>1:
        ys=np.convolve(y,np.ones(window)/window,mode='valid')
        ax.plot(x[window-1:],ys,color=color,lw=1.7,label=label)
    else:ax.plot(x,y,color=color,lw=1.4,label=label)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,action='append',help='Training phases, in chronological order')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();runs=args.run;run=runs[-1];out=args.out;out.mkdir(parents=True,exist_ok=True)
    rows={};phase_starts=[];phase_names=[]
    for phase,folder in enumerate(runs):
        phase_names.append(('Original PDMX-v1','PDMX-v2','PDMX-v2 + capped Lieder','PDMX-v2: 6 unsupported crops excluded')[phase] if phase<4 else folder.name)
        for line in (folder/'train.log').read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(r,dict) and 'updates' in r and 'recognition_loss' in r:
                r['phase']=phase_names[-1];rows[r['updates']]=r
        if phase:
            migration=folder/'full/migration.json'
            phase_starts.append(json.loads(migration.read_text())['source_updates'] if migration.exists() else min(r['updates'] for r in rows.values() if r['phase']==phase_names[-1]))
    if not rows:raise ValueError('No logged optimizer-step samples')
    train=[rows[i] for i in sorted(rows)]
    val={};val_phase={}
    for phase,phase_run in enumerate(runs):
        migration=phase_run/'full/migration.json'
        start=json.loads(migration.read_text())['source_updates']/1260 if migration.exists() else 0
        for folder in (phase_run/'full').glob('eval-*'):
            epoch=start if folder.name=='eval-initial' else int(folder.name.rsplit('-',1)[1])
            found={}
            for source in SOURCES:
                file=folder/f'{source}.json'
                if file.exists():found[source]=json.loads(file.read_text())
            if len(found)==3:val[(phase,epoch)]=found;val_phase[(phase,epoch)]=phase_names[phase]
    if not val:raise ValueError('No complete source-wise validation epoch')
    epochs=sorted(val,key=lambda item:(item[1],item[0]));latest=epochs[-1];progress=json.loads((run/'full/progress.json').read_text())
    def phase_lines(ax,values,*,marker='o'):
        for source in SOURCES:
            for phase,name in enumerate(phase_names):
                points=[e for e in epochs if e[0]==phase]
                if points:ax.plot([e[1] for e in points],[values(e,source) for e in points],marker=marker,ms=3,color=COLORS[source],linestyle=('-','--',':','-.')[phase%4],label=source.upper() if phase==0 else None)
    def boundaries(ax):
        low,high=ax.get_xlim()
        for step,name in zip(phase_starts,phase_names[1:]):
            if low<=step/1260<=high:ax.axvline(step/1260,color='#64748b',ls=':',lw=1)
    x=np.array([r['epoch'] for r in train]);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17})
    fig,axes=plt.subplots(1,3,figsize=(17,4.4))
    plot_series(axes[0],x,[r['recognition_loss'] for r in train],color=COLORS['ossq']);axes[0].set_title('Recognition: training');axes[0].set_yscale('log')
    plot_series(axes[1],x,[r['structured_loss']/12 for r in train],color=COLORS['lieder']);axes[1].set_title('Notation contribution: sum of 12 heads / 12');axes[1].set_yscale('log')
    phase_lines(axes[2],lambda e,s:val[e][s]['recognition_loss'])
    axes[2].set_title('Recognition: internal validation');axes[2].set_yscale('log');axes[2].legend(frameon=False,fontsize=7,ncol=2)
    for ax in axes:boundaries(ax)
    for ax in axes:ax.set_xlabel('Epoch');ax.set_ylabel('Loss (log scale)')
    fig.tight_layout();save(fig,out,'overview')
    fig,axes=plt.subplots(1,3,figsize=(17,4.4))
    selected=[e for e in epochs if e[1]>=max(2,latest[1]-9)]
    for source,ax in zip(SOURCES,axes):
        for phase,name in enumerate(phase_names):
            points=[e for e in selected if e[0]==phase]
            if points:ax.plot([e[1] for e in points],[val[e][source]['recognition_loss'] for e in points],color=COLORS[source],linestyle=('-','--',':','-.')[phase%4],marker='o',ms=4,label=name)
        ax.set_title(source.upper());ax.set_xlabel('Epoch');ax.set_ylabel('Validation recognition loss')
        boundaries(ax)
    axes[0].legend(frameon=False,fontsize=7)
    fig.suptitle('Recent validation trend; source-specific vertical scales');fig.tight_layout();save(fig,out,'validation-zoom')
    fig,axes=plt.subplots(2,3,figsize=(15.5,8))
    for ax,key in zip(axes.flat,BRANCHES):
        plot_series(ax,x,[r.get('recognition_components',{}).get(key,np.nan) for r in train],color='#1769aa')
        ax.set_title(key.removeprefix('loss_').replace('_',' ').title());ax.set_xlabel('Epoch');ax.set_ylabel('Raw branch loss')
    fig.suptitle('Training: six recognition branches');fig.tight_layout();save(fig,out,'recognition-branches')
    for names,slug,title in ((MAIN_HEADS,'common-heads','Training: common notation heads'),(RARE_HEADS,'rare-heads','Training: rare notation heads')):
        fig,axes=plt.subplots(2,3,figsize=(15.5,8))
        for ax,key in zip(axes.flat,names):
            support=[r.get('head_support',{}).get(key,0) for r in train]
            plot_series(ax,x,[r.get('head_losses',{}).get(key,np.nan) for r in train],color='#088f73',supervised=support)
            ax.set_title(HEAD_LABELS[key]+'\n'+f'{sum(int(v>0) for v in support):,} logged batches with targets');ax.set_xlabel('Epoch');ax.set_ylabel('Raw head loss')
        fig.suptitle(title+' · zero-support batches omitted');fig.tight_layout();save(fig,out,slug)
    fig,axes=plt.subplots(1,3,figsize=(17,4.4))
    for ax,key in zip(axes,('rhythms','pitchs','lifts')):
        phase_lines(ax,lambda e,s:100*val[e][s]['teacher_forced_accuracy'][key])
        ax.set_title({'rhythms':'Rhythm','pitchs':'Pitch','lifts':'Accidentals'}[key]);ax.set_xlabel('Epoch');ax.set_ylabel('Teacher-forced token accuracy %')
    axes[0].legend(frameon=False,fontsize=7,ncol=2);fig.tight_layout();save(fig,out,'recognition-accuracy')
    fig,axes=plt.subplots(4,3,figsize=(15.5,13))
    for ax,(label,path) in zip(axes.flat,NOTATION_METRICS):
        phase_lines(ax,lambda e,s:100*notation_value(val[e][s],path))
        ax.set_title(label);ax.set_xlabel('Epoch');ax.set_ylabel('Validation F1 %');boundaries(ax)
    for ax in list(axes.flat)[len(NOTATION_METRICS):]:ax.set_visible(False)
    axes.flat[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Notation validation with reference recognition sequence · macro F1, except slur-span F1')
    fig.tight_layout();save(fig,out,'notation-validation')
    with (out/'training-samples.csv').open('w',newline='') as f:
        names=['phase','updates','epoch','recognition_loss','structured_loss','core_lr','heads_lr']+list(BRANCHES)+list(MAIN_HEADS)+list(RARE_HEADS)+[v+'.support' for v in MAIN_HEADS+RARE_HEADS]
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for row in train:
            item={k:row.get(k) for k in names}
            for k in BRANCHES:item[k]=row.get('recognition_components',{}).get(k)
            for k in MAIN_HEADS+RARE_HEADS:item[k]=row.get('head_losses',{}).get(k);item[k+'.support']=row.get('head_support',{}).get(k,0)
            writer.writerow(item)
    with (out/'validation.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['phase','epoch','source','recognition_loss','rhythm_accuracy','pitch_accuracy','accidental_accuracy'])
        for e in epochs:
            for source in SOURCES:
                v=val[e][source];acc=v['teacher_forced_accuracy'];writer.writerow([val_phase[e],e[1],source,v['recognition_loss'],acc['rhythms'],acc['pitchs'],acc['lifts']])
    with (out/'notation-validation.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['phase','epoch','source','metric','f1'])
        for e in epochs:
            for source in SOURCES:
                for label,path in NOTATION_METRICS:
                    value=notation_value(val[e][source],path)
                    writer.writerow([val_phase[e],e[1],source,label,'' if np.isnan(value) else value])
    summary={'generated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'training_epoch':progress['epoch'],'updates':progress['updates'],'total_updates':progress['total_updates'],'last_complete_validation_epoch':latest[1],'validation_phase':val_phase[latest],'validation':{source:{'recognition_loss':val[latest][source]['recognition_loss'],'rhythm_accuracy':val[latest][source]['teacher_forced_accuracy']['rhythms'],'pitch_accuracy':val[latest][source]['teacher_forced_accuracy']['pitchs'],'accidental_accuracy':val[latest][source]['teacher_forced_accuracy']['lifts']} for source in SOURCES}}
    watch_file=run/'WATCH_STATUS.json'
    if watch_file.exists():
        watch=json.loads(watch_file.read_text())
        summary['watch_state']=watch.get('state','unknown')
        summary['watch_checked_utc']=watch.get('checked_utc')
    summary['training_only_remaining_hours']=progress.get('remaining_seconds',0)/3600
    summary['estimated_remaining_hours']=summary['training_only_remaining_hours']
    summary['eta_basis']='elapsed time per update in the active phase; future validation overhead is not yet measured'
    # Startup update speed omits the expensive per-epoch evaluations. Prefer
    # intervals between completed epochs, including the evaluations between them.
    for phase in reversed(range(len(runs))):
        complete=[r for r in train if r['phase']==phase_names[phase] and r['updates']%1260==0
                  and (phase,r['updates']//1260) in val and 'elapsed_seconds' in r]
        if len(complete)<2:continue
        window=complete[-6:];first,last=window[0],window[-1]
        elapsed=last['elapsed_seconds']-first['elapsed_seconds']
        if elapsed<=0:continue
        seconds_per_update=elapsed/(last['updates']-first['updates'])
        summary['estimated_remaining_hours']=seconds_per_update*(progress['total_updates']-progress['updates'])/3600
        summary['eta_basis']=f'{len(window)-1} completed epoch intervals in {phase_names[phase]}, including validation overhead'
        break
    summary['progress_updated_utc']=dt.datetime.fromtimestamp((run/'full/progress.json').stat().st_mtime,dt.timezone.utc).isoformat()
    summary['latest_validation_epoch']=latest[1]
    summary['last_complete_validation_epoch']=max(e[1] for e in epochs if isinstance(e[1],int))
    recent=[e for e in epochs if e[0]>=min(2,len(runs)-1) and isinstance(e[1],int)]
    summary['best_validation_loss']={source:{'epoch':min(recent,key=lambda e:val[e][source]['recognition_loss'])[1],
                                           'loss':min(val[e][source]['recognition_loss'] for e in recent)} for source in SOURCES} if recent else {}
    if summary.get('watch_state')!='running':
        summary['projected_remaining_hours_if_resumed']=summary['estimated_remaining_hours']
        summary['estimated_remaining_hours']=None
        failures=[line for line in (run/'train.log').read_text().splitlines() if line.startswith(('KeyError:','RuntimeError:','ValueError:','FileNotFoundError:'))]
        if failures:summary['stop_reason']=failures[-1]
    audits=sorted(run.glob('token-audit-*/report.json'))
    if audits:summary['token_audit']=json.loads(audits[-1].read_text())
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    style='body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px;background:#edf1f6;color:#162b3e}section{background:white;padding:20px;margin:20px 0;border-radius:10px}img{width:100%;height:auto}a{color:#1769aa}nav a{margin-right:18px}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.muted{color:#526274}'
    html=f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="300"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fresh music-model loss curves</title><style>{style}</style><h1>Fresh music-model training curves</h1><p>ImageNet encoder; music decoder and notation heads initialized randomly. Snapshot: epoch {progress["epoch"]:.2f}/160, {progress["updates"]:,}/{progress["total_updates"]:,} updates. Last fully evaluated epoch: {summary["last_complete_validation_epoch"]:g}. Latest validation snapshot: epoch {latest[1]:g} ({val_phase[latest]}).</p>'
    html+=f'<p class="muted">Page generated: {summary["generated_utc"]}. Latest training update: {summary["progress_updated_utc"]}. Watcher: {summary.get("watch_state","unknown")} at {summary.get("watch_checked_utc","unknown")}.</p>'
    if summary['estimated_remaining_hours'] is not None:
        html+=f'<p>Completion ETA: about {summary["estimated_remaining_hours"]:.1f} hours, based on {summary["eta_basis"]}. Interruptions add time.</p>'
    else:
        from html import escape
        html+=f'<p><strong>Training is stopped.</strong> {escape(summary.get("stop_reason","No active training session at the last check."))} No completion ETA while stopped.</p>'
    if 'token_audit' in summary:
        audit=summary['token_audit'];total=sum(audit['rows_by_source'].values())
        html+=f'<p>Training-label audit: {audit["invalid_rows"]:,} of {total:,} crops contain labels outside the current decoder vocabulary. The exclusion inventory and filtered index are recorded with the run.</p>'
    html+='<p class="muted">Training phases: original PDMX-v1 to update 21,420 (solid lines); PDMX-v2 to update 22,000 (dashed); then PDMX-v2 with oversized Lieder crops capped before augmentation (dotted). Vertical lines mark switches. Validation lines remain separate by phase; PDMX validation is the same v1 holdout throughout.</p>'
    if len(runs)>=4:html+='<p class="muted">From update 49,800, six PDMX crops with unsupported articulation combinations are excluded (dash-dot lines). This continuation preserves the existing optimizer, schedule, and 60% OSSQ / 25% Lieder / 15% PDMX sampling.</p>'
    html+='<p class="muted">Training lines average up to 40 logged batches (each logged every 10 updates); faint lines are raw batch losses. A head is plotted only when its batch has targets. Validation is teacher-forced on fixed internal crops; these curves do not measure decoded-score quality or justify promotion.</p>'
    html+='<nav><a href="#overview">Overview</a><a href="#branches">Recognition losses</a><a href="#heads">Notation head losses</a><a href="#accuracy">Recognition accuracy</a><a href="#notation">Notation validation</a><a href="training-samples.csv">Training CSV</a><a href="validation.csv">Recognition validation CSV</a><a href="notation-validation.csv">Notation validation CSV</a></nav>'
    html+='<section><h2>Latest validation</h2><table><tr><th>Source</th><th>Recognition loss</th><th>Rhythm accuracy</th><th>Pitch accuracy</th><th>Accidental accuracy</th></tr>'
    for source in SOURCES:
        v=summary['validation'][source];html+=f'<tr><td>{source.upper()}</td><td>{v["recognition_loss"]:.4f}</td><td>{100*v["rhythm_accuracy"]:.2f}%</td><td>{100*v["pitch_accuracy"]:.2f}%</td><td>{100*v["accidental_accuracy"]:.2f}%</td></tr>'
    html+='</table></section>'
    if summary['best_validation_loss']:
        html+='<section><h2>Lowest validation loss since the Lieder cap</h2><p>Each source favors a different epoch. These are diagnostics for selecting checkpoints to evaluate with full decoding.</p><table><tr><th>Source</th><th>Epoch</th><th>Recognition loss</th></tr>'
        for source,best in summary['best_validation_loss'].items():html+=f'<tr><td>{source.upper()}</td><td>{best["epoch"]}</td><td>{best["loss"]:.4f}</td></tr>'
        html+='</table></section>'
    for anchor,title,slug in (('overview','Training and validation loss','overview'),('recent','Recent validation trend','validation-zoom'),('branches','Recognition branch losses','recognition-branches'),('heads','Common notation heads','common-heads'),('rare','Rare notation heads','rare-heads'),('accuracy','Recognition accuracy','recognition-accuracy'),('notation','Notation validation quality','notation-validation')):
        html+=f'<section id="{anchor}"><h2>{title}</h2><a href="{slug}.png"><img src="{slug}.png?v={summary["generated_utc"]}" alt="{title}"></a><p><a href="{slug}.svg">Download SVG</a> · <a href="{slug}.png">Download PNG</a></p></section>'
    temporary=out/'index.tmp.html';temporary.write_text(html);temporary.replace(out/'index.html')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
