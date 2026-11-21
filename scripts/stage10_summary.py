"""Aggregate completed closed-loop metrics and make audit plots."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.stage7_optimize import read
from scripts.stage10_audit import hand_shape
from src.evaluate_feedback import write_csv

COLORS={"A_geometry":"#6c757d","B_joint":"#9e423e","C_local_expert":"#7d65a4",
        "D_guarded":"#cc783f","E_dense_geometry":"#527d87","F_dense_guarded":"#157b90",
        "G_reanchored_geometry":"#547852","H_reanchored_guarded":"#068364"}


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    work=a.output
    cfg=read(Path("configs/stage10.json"))
    out=work/"summary"; out.mkdir(exist_ok=True)
    sources=cfg["development"]+cfg["locked"]
    mets=[]; digit=[]; rgb=[]; decisions=[]
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for source,ax in zip(sources,axes.ravel()):
        folder=work/source
        if not (folder/"results.json").exists(): continue
        values=read(folder/"results.json")
        for name,v in values.items():
            mets.append(v)
            digit.extend(table(folder/name/"held_per_digit.csv"))
            trace=folder/name/"trace.csv"
            result=folder/name/"refinement.json"
            if result.exists():
                z=read(result)
                decisions.append(dict(source=source,variant=name,**{k:v for k,v in z.items() if k!="rejection"},
                                      rejection=str(z["rejection"])))
                rows=table(trace)
                t=[int(r["step"]) for r in rows]
                ax.plot(t,[float(r["expert"]) for r in rows],label=name.split("_")[0],color=COLORS[name])
            elif name in ("A_geometry","B_joint","G_reanchored_geometry"):
                ax.axhline(float(v["expert"]),color=COLORS[name],linestyle="--",label=name.split("_")[0])
        r=folder/"rgb/same_detector.csv"
        if r.exists():
            rgb.extend(table(r))
        ax.set_title(source[:7]+(" · locked" if source in cfg["locked"] else ""))
        ax.set_xlabel("Expert proposal"); ax.set_ylabel("Same direct expert distance")
        ax.grid(alpha=.24)
    axes[0,1].legend(ncol=4,fontsize=8,frameon=False)
    fig.tight_layout(); fig.savefig(out/"optimization_trajectories.png",dpi=170); plt.close(fig)
    for f,v in (("closed_loop_comparison",mets),("heldout_each_digit",digit),("rgb_same_detector",rgb),("update_decisions",decisions)):
        if v:
            names=list(dict.fromkeys(k for row in v for k in row))
            write_csv(out/f"{f}.csv",[{key:row.get(key) for key in names} for row in v])

    aud=work/"audit"
    tr=table(aud/"joint_frame90_replay.csv")
    path=Path("experiments/Stage9")/cfg["focus"]/"B/art_joint_factor/trace.csv"
    arch=table(path)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    t=[int(r["proposal"]) for r in tr]
    axes[0].plot(t,[float(r["response"]) for r in arch],color="#14627b")
    axes[0].set_ylabel("Stage 9 whole-clip expert score")
    axes[1].plot(t,[float(r["opening_px_ratio"]) for r in tr],color="#925039",label="Stage 9 avatar")
    human=next(float(r["opening_2d"]) for r in table(aud/"handshape_frame90.csv") if r["variant"]=="real_RTMW" and r["frame"]=="90" and r["side"]=="left")
    axes[1].axhline(human,color="#19735b",label="Real RTMW frame 90")
    axes[1].set_ylabel("Left aperture / palm length")
    axes[1].set_xlabel("Stage 9 proposal"); axes[1].legend(frameon=False)
    for x in axes:x.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/"frame90_stage9_causal_trace.png",dpi=170);plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    with np.load(aud/"world_trajectories.npz") as f:
        for k,col in (("legacy","#9e423e"),("fixed_gauge","#7d65a4"),("stabilized","#068364")):
            for label,ax,y in (("pelvis",axes[0],"Pelvis"),("knee",axes[1],"Knees")):
                z=f[k+"_"+label]
                magn=np.linalg.norm(np.diff(z,n=3,axis=0),axis=-1)
                if magn.ndim==2: magn=magn.max(1)
                ax.plot((np.arange(len(magn))+3)/24,magn*1000,label=k,c=col,linewidth=1.2)
                ax.set_ylabel(y+" (mm/frame³)")
    axes[0].legend(frameon=False);axes[1].set_xlabel("Playback time (s)")
    fig.tight_layout();fig.savefig(out/"pelvis_and_knees.png",dpi=170);plt.close(fig)

    sid=cfg["focus"]
    obs=np.load(Path("experiments/Stage7/data/real_pose")/f"{sid}.npz")
    other=np.load(Path("experiments/Stage7/data/paired")/f"{sid}.npz")
    times=np.arange(86,97)
    rows=[]
    for name,seq in (("RTMW",obs["keypoints"]),("MediaPipe",other["mp"])):
        for side,ids in (("left",slice(91,112)),("right",slice(112,133))):
            _,bend,ap=hand_shape(seq[times,ids]*[1280,720])
            for t,value,angle in zip(times,ap,bend):
                rows.append(dict(detector=name,side=side,frame=t,aperture=value,projected_bend_deg=angle[1:].mean()))
    write_csv(out/"frame90_detector_crosscheck.csv",rows)


if __name__=="__main__":
    main()
