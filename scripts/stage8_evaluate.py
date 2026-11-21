"""Aggregate the actual imitation trajectories and run held-out pose checks."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.stage7_evaluate import project_side
from scripts.stage7_optimize import read, subset
from scripts.stage8_imitate import METHODS, REFINEMENT, WIDE, geometry_holdout
from src.evaluate_feedback import write_csv
from src.stage7_feedback import expert_input, in_frame, valid_pose
from src.unisign import UniSign, cosine

METHOD_LABELS = {METHODS[0]:"Geometry",METHODS[1]:"Expert-only",METHODS[2]:"Sequential",
                 METHODS[3]:"Joint",REFINEMENT:"Local fingers",WIDE:"Local + larger steps"}
COLORS = ["#42637a","#ba582e","#086b61","#6e59a0","#b68212","#c24272"]
CHAINS = [(5,7),(7,9),(6,8),(8,10)]


def csv_rows(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def audit(root, cfg, data):
    valid = []
    side = []
    statuses = {r["clip"]:r for r in read(data/"multiview/status.json")}
    for name in cfg["sources"]:
        d=root/name
        for level in ("A","B","C"):
            for method in METHOD_LABELS:
                arm=d/level/method
                if not (arm/"result.json").exists():
                    continue
                r = read(arm/"result.json")
                with np.load(arm/"step_00_tracks.npz") as p0,np.load(arm/"step_18_tracks.npz") as p1:
                    p0,p1=p0["points"],p1["points"]
                with np.load(arm/"step_18_fit.npz") as npz:
                    same_betas=bool(np.all(npz["betas"]==npz["betas"][0]))
                a,b=np.array(CHAINS).T
                length0=np.linalg.norm(p0[:,a]-p0[:,b],axis=-1)
                length1=np.linalg.norm(p1[:,a]-p1[:,b],axis=-1)
                bone_error=float(abs(length1-length0).max()*1000)
                if same_betas is False or not np.isfinite(p1).all():
                    raise ValueError("Invalid shape or kinematics")
                valid.append({**r,"static_betas":same_betas,"bone_length_max_delta_mm":bone_error})
                info=statuses.get(name,{})
                if info.get("associated_odd_hands",0)<5:
                    continue
                with np.load(data/"multiview/pose"/f"{name}_mp.npz") as p:
                    mp=p["mp"][1::2]
                mp=np.where(in_frame(mp)[...,None],mp,np.nan)
                camera=(info["calibration_error_px"],info["camera_yaw_deg"],info["camera_scale"],info["camera_offset_px"])
                for label,pts in (("initial",p0),("final",p1)):
                    h=geometry_holdout(project_side(pts,camera)[1::2],mp,(1280,720))
                    side.append(dict(source=name,level=level,method=method,time=label,
                                     side_hands=info["associated_odd_hands"],**h))
    write_csv(root/"main_results.csv",valid)
    if side:
        write_csv(root/"side_heldout.csv",side)
    return valid


def trajectories(root, cfg):
    fig, ax=plt.subplots(3,len(cfg["sources"]),figsize=(12,8),sharex=True,layout="constrained")
    for r,level in enumerate(("A","B","C")):
        for j,name in enumerate(cfg["sources"]):
            axes=ax[r,j]
            for (method,label),color in zip(METHOD_LABELS.items(),COLORS):
                file=root/name/level/method/"trace.csv"
                if not file.is_file():
                    continue
                trace=csv_rows(file)
                axes.plot([int(x["step"]) for x in trace],[float(x["response"]) for x in trace],
                          color=color, label=label, ls="--" if method in (REFINEMENT,WIDE) else "-")
            axes.axvline(6,color="#888888",lw=.65,ls=":")
            axes.grid(alpha=.2)
            if r==0:
                axes.set_title(name.split("_")[0])
            if j==0:
                axes.set_ylabel({"A":"NSA fit","B":"Coarse IK","C":"Synthetic challenge"}[level]+"\nResponse discrepancy")
            if r==2:
                axes.set_xlabel("Coordinate iterations")
    h,l=ax[1,-1].get_legend_handles_labels()
    fig.legend(h,l,loc="outside lower center",ncols=3,frameon=False)
    fig.savefig(root/"STAGE8_VISUAL_DEMONSTRATION"/"closed_loop_trajectories.png",dpi=180)
    plt.close(fig)


def successful_trajectories(root, rows):
    c=[]
    for r in rows:
        if r["level"]!="B":
            continue
        folder=root/r["source"]/r["level"]/r["method"]
        for t in csv_rows(folder/"proposals.csv"):
            if t["accepted"]!="True":
                continue
            c.append(dict(source=r["source"],level=r["level"],method=r["method"],step=int(t["step"]),
                          phase=t["phase"],proposed_coefficients=t["proposed_coefficients"],
                          response_gain=float(t["candidate_gain"]),held_px=float(t["held_geometry"]),
                          whole_run_response_gain=r["response_drop"],
                          whole_run_held_gain=r["initial_full_held"]-r["final_full_held"],
                          motion_guard_violations=int(t["constraint_violations"])+int(t["new_angle_excess"])))
    if c:
        write_csv(root/"accepted_coarse_updates.csv",c)


def secondary(a,cfg,data):
    sign=UniSign(a.repo,a.checkpoint,a.mt5,"cpu",a.threads)
    rows=[]
    for name in cfg["sources"]:
        txt=next(x["reference"] for x in read(data/"data/manifest.json") if x["clip_id"]==name)
        with np.load(data/"data/real_pose"/f"{name}.npz") as poses:
            real,conf=poses["keypoints"],poses["scores"]
        idx=subset(2,len(real),44,start=1)
        r=sign.infer(*expert_input(real[idx],conf[idx],valid_pose(real[idx],conf[idx],.3)),[txt],max_frames=256)
        paths={"initial":a.output/name/"B/A_geometry/step_00_tracks.npz"}
        for method in METHOD_LABELS:
            paths[method]=a.output/name/"B"/method/"step_18_tracks.npz"
        for variant, path in paths.items():
            if not path.exists():
                continue
            with np.load(path) as k:
                xy=k["xy"][idx]
            score=np.ones(xy.shape[:2],np.float32)
            c=sign.infer(*expert_input(xy,score,valid_pose(xy,score,.3)),[txt],max_frames=256)
            rows.append(dict(source=name,level="B",variant=variant,n_frames=len(idx),
                             real_nll=r["target_nll"][0],virtual_nll=c["target_nll"][0],
                             global_distance=float(1-cosine(r["embedding"],c["embedding"]))))
        print(name,"off-objective scoring complete",flush=True)
    write_csv(a.output/"secondary_expert.csv",rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--secondary",action="store_true")
    for k in ("repo","checkpoint","mt5"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--threads",type=int,default=3)
    a=p.parse_args()
    cfg=read(Path("configs/stage8.json"))
    stage7=Path(cfg["stage7_root"])
    if a.secondary:
        secondary(a,cfg,stage7)
    else:
        (a.output/"STAGE8_VISUAL_DEMONSTRATION").mkdir(exist_ok=True)
        rows=audit(a.output,cfg,stage7)
        trajectories(a.output,cfg)
        successful_trajectories(a.output,rows)


if __name__=="__main__":
    main()
