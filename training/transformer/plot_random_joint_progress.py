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
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run=args.run;out=args.out;out.mkdir(parents=True,exist_ok=True)
    rows={}
    for line in (run/'train.log').read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(r,dict) and 'updates' in r and 'recognition_loss' in r:
            rows[r['updates']]=r
    if not rows:raise ValueError('No logged optimizer-step samples')
    train=[rows[i] for i in sorted(rows)]
    val={}
    for folder in (run/'full').glob('eval-*'):
        epoch=0 if folder.name=='eval-initial' else int(folder.name.rsplit('-',1)[1])
        found={}
        for source in SOURCES:
            file=folder/f'{source}.json'
            if file.exists():found[source]=json.loads(file.read_text())
        if len(found)==3:val[epoch]=found
    if not val:raise ValueError('No complete source-wise validation epoch')
    epochs=sorted(val);latest=epochs[-1];progress=json.loads((run/'full/progress.json').read_text())
    x=np.array([r['epoch'] for r in train]);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17})
    fig,axes=plt.subplots(1,3,figsize=(17,4.4))
    plot_series(axes[0],x,[r['recognition_loss'] for r in train],color=COLORS['ossq']);axes[0].set_title('Recognition: training');axes[0].set_yscale('log')
    plot_series(axes[1],x,[r['structured_loss']/12 for r in train],color=COLORS['lieder']);axes[1].set_title('Notation contribution: sum of 12 heads / 12');axes[1].set_yscale('log')
    for source in SOURCES:axes[2].plot(epochs,[val[e][source]['recognition_loss'] for e in epochs],marker='o',ms=3,color=COLORS[source],label=source.upper())
    axes[2].set_title('Recognition: internal validation');axes[2].set_yscale('log');axes[2].legend(frameon=False)
    for ax in axes:ax.set_xlabel('Epoch');ax.set_ylabel('Loss (log scale)')
    fig.tight_layout();save(fig,out,'overview')
    fig,axes=plt.subplots(1,3,figsize=(17,4.4))
    selected=[e for e in epochs if e>=max(2,latest-9)]
    for source,ax in zip(SOURCES,axes):
        ax.plot(selected,[val[e][source]['recognition_loss'] for e in selected],color=COLORS[source],marker='o',ms=4)
        ax.set_title(source.upper());ax.set_xlabel('Epoch');ax.set_ylabel('Validation recognition loss')
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
            ax.set_title(key+'  ·  '+f'{sum(int(v>0) for v in support):,} logged batches with targets');ax.set_xlabel('Epoch');ax.set_ylabel('Raw head loss')
        fig.suptitle(title+' · zero-support batches omitted');fig.tight_layout();save(fig,out,slug)
    fig,axes=plt.subplots(1,3,figsize=(17,4.4))
    for ax,key in zip(axes,('rhythms','pitchs','lifts')):
        for source in SOURCES:
            ax.plot(epochs,[100*val[e][source]['teacher_forced_accuracy'][key] for e in epochs],marker='o',ms=3,label=source.upper(),color=COLORS[source])
        ax.set_title({'rhythms':'Rhythm','pitchs':'Pitch','lifts':'Accidentals'}[key]);ax.set_xlabel('Epoch');ax.set_ylabel('Teacher-forced token accuracy %')
    axes[0].legend(frameon=False);fig.tight_layout();save(fig,out,'recognition-accuracy')
    with (out/'training-samples.csv').open('w',newline='') as f:
        names=['updates','epoch','recognition_loss','structured_loss','core_lr','heads_lr']+list(BRANCHES)+list(MAIN_HEADS)+list(RARE_HEADS)+[v+'.support' for v in MAIN_HEADS+RARE_HEADS]
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for row in train:
            item={k:row.get(k) for k in names}
            for k in BRANCHES:item[k]=row.get('recognition_components',{}).get(k)
            for k in MAIN_HEADS+RARE_HEADS:item[k]=row.get('head_losses',{}).get(k);item[k+'.support']=row.get('head_support',{}).get(k,0)
            writer.writerow(item)
    with (out/'validation.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['epoch','source','recognition_loss','rhythm_accuracy','pitch_accuracy','accidental_accuracy'])
        for e in epochs:
            for source in SOURCES:
                v=val[e][source];acc=v['teacher_forced_accuracy'];writer.writerow([e,source,v['recognition_loss'],acc['rhythms'],acc['pitchs'],acc['lifts']])
    summary={'generated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'training_epoch':progress['epoch'],'updates':progress['updates'],'total_updates':progress['total_updates'],'last_complete_validation_epoch':latest,'validation':{source:{'recognition_loss':val[latest][source]['recognition_loss'],'rhythm_accuracy':val[latest][source]['teacher_forced_accuracy']['rhythms'],'pitch_accuracy':val[latest][source]['teacher_forced_accuracy']['pitchs'],'accidental_accuracy':val[latest][source]['teacher_forced_accuracy']['lifts']} for source in SOURCES}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    style='body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px;background:#edf1f6;color:#162b3e}section{background:white;padding:20px;margin:20px 0;border-radius:10px}img{width:100%;height:auto}a{color:#1769aa}nav a{margin-right:18px}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.muted{color:#526274}'
    html=f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="300"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fresh music-model loss curves</title><style>{style}</style><h1>Fresh music-model training curves</h1><p>ImageNet encoder; music decoder and notation heads initialized randomly. Snapshot: epoch {progress["epoch"]:.2f}/160, {progress["updates"]:,}/{progress["total_updates"]:,} updates. Last complete validation: epoch {latest}.</p>'
    html+='<p class="muted">Training lines average up to 40 logged batches (each logged every 10 updates); faint lines are raw batch losses. A head is plotted only when its batch has targets. Validation is teacher-forced on fixed internal crops; these curves do not measure decoded-score quality or justify promotion.</p>'
    html+='<nav><a href="#overview">Overview</a><a href="#branches">Recognition losses</a><a href="#heads">Notation heads</a><a href="#accuracy">Accuracy</a><a href="training-samples.csv">Training CSV</a><a href="validation.csv">Validation CSV</a></nav>'
    html+='<section><h2>Latest validation</h2><table><tr><th>Source</th><th>Recognition loss</th><th>Rhythm accuracy</th><th>Pitch accuracy</th><th>Accidental accuracy</th></tr>'
    for source in SOURCES:
        v=summary['validation'][source];html+=f'<tr><td>{source.upper()}</td><td>{v["recognition_loss"]:.4f}</td><td>{100*v["rhythm_accuracy"]:.2f}%</td><td>{100*v["pitch_accuracy"]:.2f}%</td><td>{100*v["accidental_accuracy"]:.2f}%</td></tr>'
    html+='</table></section>'
    for anchor,title,slug in (('overview','Training and validation loss','overview'),('overview','Recent validation trend','validation-zoom'),('branches','Recognition branch losses','recognition-branches'),('heads','Common notation heads','common-heads'),('heads','Rare notation heads','rare-heads'),('accuracy','Recognition accuracy','recognition-accuracy')):
        html+=f'<section id="{anchor}"><h2>{title}</h2><a href="{slug}.png"><img src="{slug}.png?v={progress["updates"]}" alt="{title}"></a><p><a href="{slug}.svg">Download SVG</a> · <a href="{slug}.png">Download PNG</a></p></section>'
    (out/'index.html').write_text(html)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
