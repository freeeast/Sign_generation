"""Summarize source-paired outcomes and render diagnostic motion comparisons."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.stage4_geometry import per_frame_metrics
from scripts.stage4_review_media import pose_on, render_pair
from scripts.stage7_optimize import read
from src.evaluate_feedback import write_csv

METHODS=("geometry","stage6","bilateral","inframe","confidence","robust_geometry","factor","hierarchical","reliable","safe")
METRICS=("mp_held_combined","mp_held_wrists","mp_held_tips","mp_held_handshape","hand_accel_p95_mm")


def table(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def paired(root,dirs,baseline,seed):
    rows=[]
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            r=read(f)
            if "method" in r and "mp_held_combined" in r:
                rows.append(r)
    if not rows:
        return [],[],[]
    g={}
    for r in rows:
        g.setdefault((r["clip"],r["subset"]),{})[r["method"]]=r
    delta=[]
    for (clip,subset),by in g.items():
        for name,row in by.items():
            for b in baseline:
                if b not in by or name==b:
                    continue
                delta.append(dict(clip=clip,subset=subset,method=name,baseline=b,**{
                    m:row[m]-by[b][m] if row[m] is not None and by[b][m] is not None else None for m in METRICS}))
    stats=[]
    rng=np.random.default_rng(seed)
    for key in sorted({(r["method"],r["baseline"],r["subset"]) for r in delta}):
        s=[r for r in delta if (r["method"],r["baseline"],r["subset"])==key]
        for metric in METRICS:
            v=np.array([x[metric] for x in s if x[metric] is not None])
            if not len(v):
                continue
            lo,hi=np.quantile(rng.choice(v,size=(10000,len(v)),replace=True).mean(axis=1),[.025,.975])
            stats.append(dict(method=key[0],baseline=key[1],subset=key[2],metric=metric,n_sources=len(v),
                    mean=v.mean(),lower95=lo,upper95=hi,better=int((v<-.0001).sum()),worse=int((v>.0001).sum()),
                    min=v.min(),max=v.max()))
    return rows,delta,stats


def report_direct(root):
    seed=read(Path("configs/stage7.json"))["bootstrap_seed"]
    s,d,stats=paired(root,sorted((root/"direct").glob("*")),("geometry","robust_geometry"),seed)
    for name,values in (("direct_results",s),("direct_contrasts",d),("direct_summary",stats)):
        if values:
            write_csv(root/f"{name}.csv",values)
    if not s:
        return
    clips=sorted({r["clip"] for r in s})
    prep=[]
    for location in [*(root/"direct").glob("*"),*(root/"visibility_calibration").glob("*"),*(root/"visibility_injected").glob("*")]:
        for a,b in (("stage6","bilateral"),("bilateral","inframe"),("inframe","confidence")):
            x,y=[location/f"{m}_trace.csv" for m in (a,b)]
            if not (x.is_file() and y.is_file()):
                continue
            first,second=table(x),table(y)
            changed=[int(i["step"]) for i,j in zip(first,second) if i["coefficients"]!=j["coefficients"]]
            prep.append(dict(clip=location.name,partition=location.parent.name,comparison=f"{b}_minus_{a}",
                changed_steps=len(changed),first_divergence=changed[0] if changed else None,
                endpoint_changed=first[-1]["coefficients"]!=second[-1]["coefficients"],
                support_a=float(first[-1]["support_retained"]),support_b=float(second[-1]["support_retained"]),
                heldout_delta=float(second[-1]["mp_held_combined"])-float(first[-1]["mp_held_combined"])))
    if prep:
        write_csv(root/"preprocessing_decisions.csv",prep)
    df={(r["clip"],r["method"]):r["mp_held_combined"] for r in s}
    fig,ax=plt.subplots(figsize=(9.6,4.8),layout="constrained")
    for c in clips:
        if (c,"geometry") not in df:
            continue
        values=[df.get((c,m),float("nan"))-df[c,"geometry"] for m in METHODS]
        ax.plot(range(len(METHODS)),values,"o-",label=c.split("_")[0])
    ax.axhline(0,color="#555555",lw=.8)
    ax.set_xticks(range(len(METHODS)),[x.replace("_"," ") for x in METHODS],rotation=25,ha="right")
    ax.set_ylabel("Held-out MediaPipe composite change (px)")
    ax.grid(axis="y",alpha=.25)
    ax.legend(frameon=False,ncols=max(1,len(clips)),loc="upper left",title="Recording")
    fig.savefig(root/"new_source_comparison.png",dpi=180)
    plt.close(fig)
    anatomy=[]
    for r in s:
        c,m=r["clip"],r["method"]
        with np.load(root/"direct"/c/f"{m}_fit.npz") as f, np.load(root/"data/inputs"/f"{c}.npz") as original:
            orig_equal=bool(np.array_equal(f["betas"],original["betas"]))
        with np.load(root/"direct"/c/f"{m}_tracks.npz") as f:
            xyz=f["points"]
        neck=xyz[:,[5,6]].mean(axis=1)
        hip=xyz[:,[11,12]].mean(axis=1)
        vector=hip-neck
        wrist=xyz[:,[91,112]]
        scale=((wrist-neck[:,None])*vector[:,None]).sum(axis=-1)/np.square(vector).sum(axis=-1,keepdims=True)
        nearest=neck[:,None]+np.clip(scale,0,1)[...,None]*vector[:,None]
        distance=np.linalg.norm(wrist-nearest,axis=-1)*1000
        anatomy.append(dict(clip=c,method=m,shape_parameters_unchanged=orig_equal,
            min_wrist_torso_centerline_mm=float(distance.min()),wrist_torso_centerline_50mm_fraction=float((distance<50).mean()),
            correction_joint_deg=r["correction_joint_deg"],correction_velocity_deg_per_frame=r["correction_velocity_deg_per_frame"],
            correction_accel_deg_per_frame2=r["correction_accel_deg_per_frame2"],constraint_violations=r["constraint_violations"]))
    write_csv(root/"plausibility.csv",anatomy)


def report_rgb(root):
    seed=read(Path("configs/stage7.json"))["bootstrap_seed"]
    dirs=sorted((root/"rgb").glob("*/*"))
    rows,deltas,_=paired(root,dirs,("geometry","robust_geometry"),seed)
    for name,data in (("rgb_results",rows),("rgb_contrasts",deltas)):
        if data:
            write_csv(root/f"{name}.csv",data)
    if not dirs:
        return
    fig,axes=plt.subplots(len(dirs),2,figsize=(10,2.7*len(dirs)),squeeze=False,layout="constrained")
    for (left,right),d in zip(axes,dirs):
        for method in ("geometry","direct","rgb","robust_rgb","robust_geometry"):
            f=d/f"{method}_trace.csv"
            if not f.is_file():
                continue
            rows=table(f)
            left.plot([x["step"] for x in rows],[float(x["mp_held_combined"]) for x in rows],"o-",label=method)
            right.plot([x["step"] for x in rows],[float(x["legacy"]) for x in rows],"o-",label=method)
        left.set_ylabel(d.parent.name.split("_")[0]+" "+d.name+" (px)")
        left.legend(frameon=False,ncols=2,fontsize=8)
    axes[0,0].set_title("Independent held-out geometry")
    axes[0,1].set_title("Legacy temporal signal by path")
    for a in axes[-1]:
        a.set_xlabel("Coordinate decisions")
    fig.savefig(root/"rgb_trajectories.png",dpi=160)
    plt.close(fig)


def control(root):
    rows=[]
    for dest in sorted((root/"controller").glob("*/*")):
        src=[dest/x/"geometry.json" for x in ("six","nine")]
        if all(f.is_file() for f in src):
            six,nine=[read(f) for f in src]
            rows.append(dict(clip=dest.parent.name,condition=dest.name,
                six_px=six["mp_held_combined"],nine_px=nine["mp_held_combined"],
                mp_delta=nine["mp_held_combined"]-six["mp_held_combined"],
                train_delta=nine["robust_geometry"]-six["robust_geometry"],
                wrist_delta=nine["mp_held_wrists"]-six["mp_held_wrists"],
                tip_delta=nine["mp_held_tips"]-six["mp_held_tips"],
                shape_delta=nine["mp_held_handshape"]-six["mp_held_handshape"],
                smoothness_delta=nine["hand_accel_p95_mm"]-six["hand_accel_p95_mm"],
                six_accepted=six["accepted"],nine_accepted=nine["accepted"]))
    if rows:
        write_csv(root/"motion_control.csv",rows)


def failure_media(root):
    if not (root/"direct_contrasts.csv").exists():
        return
    delta=table(root/"direct_contrasts.csv")
    pool=[r for r in delta if r["method"]=="safe" and r["baseline"]=="robust_geometry" and r["mp_held_wrists"]]
    if not pool:
        return
    worst=max(pool,key=lambda r:float(r["mp_held_wrists"]))
    d=root/"direct"/worst["clip"]
    with np.load(d/"safe_tracks.npz") as x, np.load(d/"robust_geometry_tracks.npz") as y:
        f=[y["xy"],x["xy"]]
        p=[y["points"],x["points"]]
    out=root/"geometry_vs_safe_3d.mp4"
    if not out.exists():
        render_pair(out,f,p)
    with np.load(root/"data/paired"/f"{d.name}.npz") as source:
        mp=source["mp"]
    a,b=[per_frame_metrics(x,mp,(1280,720))["wrists"] for x in f]
    idx=np.unravel_index(np.nanargmax((b-a)[1::2]),a[1::2].shape)
    frame=2*int(idx[0])+1
    cap=cv2.VideoCapture(str(root/"data/inputs"/f"{d.name}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
    ret,im=cap.read()
    cap.release()
    if not ret:
        return
    tiles=[]
    for title,pose in (("Held-out MediaPipe",mp),("Robust geometry",f[0]),("Geometry-constrained expert",f[1])):
        copy=im.copy()
        pose_on(copy,pose[frame]*[1280,720],(13,240,251))
        copy=cv2.resize(copy,(640,360))
        cv2.rectangle(copy,(0,0),(640,32),(28,28,28),-1)
        cv2.putText(copy,title,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        tiles.append(copy)
    cv2.imwrite(str(root/"heldout_wrist_failure.png"),np.hstack(tiles))
    (root/"heldout_wrist_failure.json").write_text(json.dumps(dict(clip=d.name,frame_zero_based=frame,
        hand=("left","right")[idx[1]],geometry_px=float(a[frame,idx[1]]),safe_px=float(b[frame,idx[1]])),indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    report_direct(a.root)
    report_rgb(a.root)
    control(a.root)
    failure_media(a.root)


if __name__=="__main__":
    main()
