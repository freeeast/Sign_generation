"""Aggregate proposal traces and compare held-frame finger transitions and side evidence."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.stage7_evaluate import project_side
from scripts.stage7_optimize import read
from scripts.stage8_imitate import geometry_holdout
from scripts.stage9_audit import calculate,finger_rows
from src.evaluate_feedback import write_csv
from src.stage7_feedback import in_frame
from src.stage9_hand import DIGITS,SIDES
from src.stage9_timing import track_stats,wrist_local
from src.virtual_pose import load_fit

SIZE=np.array([1280,720])
OLD=("A_geometry","C_sequential","C3_local_wide")
MAIN=("art_geo_factor","art_hier_factor","art_joint_factor")


def dynamic(source,level,variant,xy,mp,pts,fit,asset):
    result=[]
    local=wrist_local(pts,fit,asset)[1::2]*1000
    xy,mp=xy[1::2]*SIZE,mp[1::2]*SIZE
    for h,start in enumerate((91,112)):
        for j,finger in enumerate(DIGITS):
            k=(j+1)*4
            y=mp[:,start+k]-mp[:,start]
            valid=np.isfinite(y).all(1)
            pred=(xy[:,start+k]-xy[:,start])[valid]
            y=y[valid]
            if len(y)<5:
                continue
            centered=lambda m:m-np.median(m,axis=0)
            length=lambda m:np.linalg.norm(m,axis=-1)
            y_spread=length(centered(y))
            pred_spread=length(centered(pred))
            wrist_local_delta=local[:,h,k]-np.median(local[:,h,k],axis=0)
            result.append(dict(source=source,level=level,variant=variant,side=SIDES[h],finger=finger,
                               frames=len(y),centered_motion_error_px=float(length(centered(y)-centered(pred)).mean()),
                               raw_wrist_relative_error_px=float(length(y-pred).mean()),
                               real_motion_p95_px=float(np.percentile(y_spread,95)),
                               avatar_motion_p95_px=float(np.percentile(pred_spread,95)),
                               avatar_wrist_local_3d_p95_mm=float(np.percentile(length(wrist_local_delta),95))))
    return result


def trace_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def curves(cfg,root,summary):
    fig,axes=plt.subplots(3,3,figsize=(11,7.6),sharex=True)
    for i,source in enumerate(cfg["sources"]):
        for j,level in enumerate("ABC"):
            ax=axes[i,j]
            for name,c in zip(MAIN,("#62686c","#db7840","#136b99")):
                p=root/source/level/name/"trace.csv"
                if not p.exists():
                    continue
                rows=trace_rows(p)
                ax.plot([int(r["step"]) for r in rows],[float(r["response"]) for r in rows],color=c,
                        linewidth=1.8,label=name.replace("art_","").replace("_factor",""))
            ax.axvline(1,color="0.75",linestyle=":",linewidth=1)
            if i==0:
                ax.set_title({"A":"A NSA","B":"B coarse IK","C":"C challenging"}[level])
            if j==0:
                ax.set_ylabel(source[:5]+"\nexpert discrepancy")
            ax.grid(alpha=.2)
    axes[0,2].legend(frameon=False,ncol=1)
    for ax in axes[-1]:
        ax.set_xlabel("Proposal count")
    fig.tight_layout()
    fig.savefig(summary/"optimization_curves.png",dpi=160)
    plt.close(fig)


def timing(root,summary):
    panels=("raw","corrected")
    frames=[]
    for variant in panels:
        with np.load(root/"rgb/timing"/f"{variant}.npz") as f:
            frames.append({k:f[k] for k in ("direct","detected","points")})
    fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True)
    indices=[91,112,95,99,103,107,111,116,120,124,128,132]
    for group,variant,c in zip(frames,panels,("#cd624b","#157a98")):
        for a,field,scale,title in zip(ax,("points","direct","detected"),(1000,SIZE,SIZE),
                                       ("3D mesh-linked joints (mm/frame³)","Direct 2D projection (px/frame³)",
                                        "RGB → RTMW keypoints (px/frame³)")):
            pts=group[field][:,indices]*scale
            v=np.linalg.norm(np.diff(pts,n=3,axis=0),axis=-1)
            a.plot(np.arange(3,len(pts))/24,np.median(v,axis=1),c,label=variant,linewidth=1.2)
            a.set_ylabel(title,fontsize=9)
    ax[0].legend(frameon=False)
    ax[-1].set_xlabel("Signing video time (s)")
    fig.tight_layout()
    fig.savefig(summary/"jitter_layers.png",dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    root=a.output; cfg=read(Path("configs/stage9.json"))
    old=Path(cfg["upstream"]); st7=Path(cfg["observations"]); data=st7/"data"
    out=root/"summary"; out.mkdir(exist_ok=True)
    camera_status={r["clip"]:r for r in read(st7/"multiview/status.json")}
    aggregate=[]; digits=[]; joints=[]; sides=[]
    with TemporaryDirectory() as td,np.load(a.asset) as mesh:
        temp=Path(td)/"pose.npz"
        mean=[mesh[k].reshape(15,3) for k in ("hands_meanl","hands_meanr")]
        for source in cfg["sources"]:
            with np.load(data/"paired"/f"{source}.npz") as f:
                cam=f["camera"]; mp=np.where(in_frame(f["mp"])[...,None],f["mp"],np.nan)
            for level in "ABC":
                before=old/source/level; current=root/source/level
                paths={"initial":before/"initial_fit.npz"}
                paths.update({m:before/m/"step_18_fit.npz" for m in OLD})
                paths.update({m:current/m/"final_fit.npz" for m in (*MAIN,"art_joint_geo_factor","art_guarded_factor","art_guarded_geo_factor")})
                for variant,path in paths.items():
                    if not path.exists():
                        continue
                    fit=load_fit(path)
                    pts,xy,_=calculate(fit,a.asset,cam,temp)
                    geo=geometry_holdout(xy[1::2],mp[1::2],SIZE)
                    d=dynamic(source,level,variant,xy,mp,pts,fit,a.asset)
                    digits.extend(d)
                    motion=track_stats(pts,fit,a.asset)
                    meta=read(path.parent/"result.json") if variant!="initial" else {}
                    baseline=read(before/"C_sequential/result.json")
                    aggregate.append(dict(source=source,level=level,variant=variant,
                       response_start=meta.get("feedback_start",baseline["initial_response"]),
                       response_final=meta.get("feedback_final",meta.get("final_response",baseline["initial_response"])),
                       wall_time_s=meta.get("optim_time_s",meta.get("elapsed_s",0)),
                       proposals=meta.get("proposal_count",meta.get("proposals",0)),
                       accepted=meta.get("accepted",0),expert_accepted=meta.get("accepted_expert",meta.get("expert_phase_updates",0)),
                       coefficients=meta.get("parameters",12 if variant=="C3_local_wide" else (6 if variant in OLD else 0)),
                       centered_finger_motion_error_px=float(np.mean([r["centered_motion_error_px"] for r in d])),
                       joint_guard_violations=meta.get("validity_violations",meta.get("new_angle_excess",0)),
                       **geo,**motion))
                    if level=="B":
                        joints.extend(finger_rows(source,variant,fit,a.asset,pts,mean))
                    if source==cfg["focus"] and level=="B":
                        r=camera_status[source]
                        c=(r["calibration_error_px"],r["camera_yaw_deg"],r["camera_scale"],r["camera_offset_px"])
                        with np.load(st7/"multiview/pose"/f"{source}_mp.npz") as f:
                            held=f["mp"][1::2]
                        scores=geometry_holdout(project_side(pts,c)[1::2],held,SIZE)
                        sides.append(dict(source=source,level=level,variant=variant,**scores,
                            held_hand_frames=r["associated_odd_hands"],proxy_camera_anchor_error_px=c[0]))
    for name,rows in (("comparison",aggregate),("per_finger_held_dynamics",digits),("per_joint_coarse",joints),("side_proxy",sides)):
        write_csv(out/f"{name}.csv",rows)
    all_rgb=[r for source in cfg["sources"] for r in trace_rows(root/"rgb"/source/"results.csv")]
    write_csv(out/"rgb_comparison.csv",all_rgb)
    extras=[read(p) for p in (root/cfg["focus"]/"B").glob("*/result.json") if p.parent.name not in MAIN]
    write_csv(out/"feedback_and_stabilization_ablation.csv",extras)
    curves(cfg,root,out); timing(root,out)


if __name__=="__main__":
    main()
