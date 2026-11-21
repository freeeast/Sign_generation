"""Score the fixed paired pilot and make the report's figures."""
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sacrebleu.metrics import BLEU, CHRF

from view_aitviewer import BODY_EDGES, hand_edges

CONDITIONS = {'real':'Real RTMPose','generated_proxy':'Generated + proxy','static_proxy':'Static proxy'}
COLORS = ['#2b665a','#2e64a1','#9c704c']


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def norm(text):
    return ' '.join(re.findall('[a-z0-9]+',text.lower()))


def draw_pose(ax,xy):
    for edges,color,width in [(BODY_EDGES,'#344555',1.75),
                              (hand_edges(91),'#117f86',1.45),
                              (hand_edges(112),'#cc5439',1.45)]:
        for start,end in edges:
            ax.plot(*xy[[start,end]].T,color=color,lw=width,solid_capstyle='round')
        selected=np.unique(edges)
        ax.scatter(*xy[selected].T,color=color,s=5,lw=0,zorder=3)
    ax.set(xlim=(110,530),ylim=(375,35),aspect='equal')
    ax.axis('off')


def figures(package, metrics, margins, difference, stats):
    directory=package/'figures'
    directory.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'text.color':'#172e3b','axes.labelcolor':'#172e3b',
                         'xtick.color':'#172e3b','ytick.color':'#172e3b','axes.titleweight':'bold'})
    fig, axs=plt.subplots(1,3,figsize=(9.6,2.8),layout='constrained')
    for ax, key, title in zip(axs,['bleu4','chrf','wins'],
                              ['Corpus BLEU-4','Corpus chrF','Reference preferred']):
        vals=[row[key] for row in metrics]
        y=np.arange(3)
        ax.barh(y,vals,color=COLORS,height=.58)
        ax.set_yticks(y, list(CONDITIONS.values()) if ax==axs[0] else ['']*3)
        ax.invert_yaxis()
        ax.set_title(title,loc='left',pad=9)
        ax.set_xlim(0,max(vals)*1.25 if key!='wins' else 28)
        ax.xaxis.grid(True,alpha=.22)
        ax.set_axisbelow(True)
        for i,v in enumerate(vals):
            ax.annotate(f'{v}/24' if key=='wins' else f'{v:.2f}',(v,i),
                        xytext=(5,0),textcoords='offset points',va='center',fontsize=9)
    fig.savefig(directory/'01_translation.png',dpi=185,facecolor='white')
    plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(9.6,3.1),layout='constrained',width_ratios=(1,1))
    for i,(label, color) in enumerate(zip(CONDITIONS.values(),COLORS)):
        jit=np.linspace(-.14,.14,len(margins[i]))
        axs[0].scatter(margins[i],i+jit,color=color,s=19,alpha=.63,linewidth=0)
        axs[0].scatter(np.mean(margins[i]),i,marker='D',color=color,edgecolor='white',s=63,zorder=4)
    axs[0].set_yticks(range(3),list(CONDITIONS.values()))
    axs[0].invert_yaxis()
    axs[0].set_title('Preference margin by clip',loc='left')
    axs[0].set_xlabel('NLL(decoy) - NLL(reference)')
    axs[0].axvline(0,color='#53616b',ls='--',lw=1)
    inds=np.arange(len(difference))
    axs[1].bar(inds,difference,color=np.where(difference>=0,'#2e64a1','#9c704c'))
    axs[1].axhline(0,color='#53616b',lw=.8)
    axs[1].set(title='Generated minus static margin',ylabel='Paired change in margin',xlabel='Sample in locked order')
    axs[1].set_xticks([0,5,11,17,23],['S01','S06','S12','S18','S24'])
    fig.savefig(directory/'02_preference.png',dpi=185,facecolor='white')
    plt.close(fig)

    fig,axs=plt.subplots(1,6,figsize=(9.6,2.12),layout='constrained')
    with np.load(package/'proxy'/'S20.npz',allow_pickle=False) as pose:
        xy=pose['keypoints']
        fps=float(pose['fps'])
    for ax,i in zip(axs,[4,10,16,22,28,36]):
        draw_pose(ax,xy[i])
        ax.set_title(f'{i/fps:.2f} s',fontsize=10,pad=3)
    fig.savefig(directory/'03_s20_frames.png',dpi=185,facecolor='white')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.6,2.0),layout='constrained')
    durations=np.array([float(r['duration_s']) for r in stats])
    ax.bar(range(len(durations)),durations,color='#2e64a1',width=.7)
    ax.axhline(np.median(durations),color='#172e3b',ls='--',lw=1)
    ax.set(ylabel='Duration (s)',xticks=[0,5,11,17,23],xticklabels=['S01','S06','S12','S18','S24'])
    ax.grid(axis='y',alpha=.22)
    ax.set_axisbelow(True)
    fig.savefig(directory/'04_durations.png',dpi=185,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample',type=Path)
    parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    package=args.package
    records=json.loads(args.sample.read_text())
    refs=[r['reference'] for r in records]
    ids=[r['id'] for r in records]
    metrics=[]
    margins=[]
    prediction_rows={}
    for condition,label in CONDITIONS.items():
        raw=read_csv(package/'results'/f'{condition}.csv')
        data={x['id']:x for x in raw}
        assert len(raw)==len(data)==len(ids) and set(data)==set(ids),(condition,len(data))
        hyps=[data[i]['prediction'] for i in ids]
        ref=np.array([float(data[i]['reference_nll']) for i in ids])
        decoy=np.array([float(data[i]['decoy_nll']) for i in ids])
        margin=decoy-ref
        margins.append(margin)
        prediction_rows[condition]=data
        bleu=BLEU(tokenize='13a',lowercase=True).corpus_score(hyps,[refs]).score
        chrf=CHRF(char_order=6,word_order=0,beta=2).corpus_score(
            [h.lower() for h in hyps],[[r.lower() for r in refs]]).score
        metrics.append(dict(condition=condition,n=len(ids),bleu4=round(bleu,4),chrf=round(chrf,4),
                            exact=sum(norm(x)==norm(y) for x,y in zip(refs,hyps)),
                            distinct=len(set(map(norm,hyps))),wins=int((margin>0).sum()),
                            mean_nll=round(float(ref.mean()),6),mean_margin=round(float(margin.mean()),6)))
    difference=margins[1]-margins[2]
    result=dict(conditions=metrics,generated_minus_static_margin=float(difference.mean()),
                positive_paired_change=int((difference>0).sum()))
    p=package/'results'/'summary.json'
    p.write_text(json.dumps(result,indent=2)+'\n')
    with (package/'results'/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)
    stats=read_csv(package/'results'/'pose_stats.csv')
    assert [r['id'] for r in stats]==ids
    with (package/'results'/'paired_predictions.csv').open('w',encoding='utf-8',newline='') as f:
        columns=['id','clip_id',*CONDITIONS,'dynamic_minus_static_margin']
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        for idx,r in enumerate(records):
            predicted={cond:by_id[r['id']]['prediction'] for cond,by_id in prediction_rows.items()}
            w.writerow(dict(id=r['id'],clip_id=r['clip_id'],**predicted,
                            dynamic_minus_static_margin=round(float(difference[idx]),6)))
    figures(package,metrics,margins,difference,stats)
    print(p.read_text())


if __name__=='__main__':
    main()
