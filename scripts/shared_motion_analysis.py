"""Quantify source-level uncertainty, motion artifacts, and exact FK round-trip."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.shared_motion_train import GROUPS,populate,write_csv
from scripts.stage9_audit import calculate,sample_fit
from src.shared_motion import baseline,decode,exp,hand_mean,load_predictor,log,rotations
from src.stage9_timing import wrist_local
from src.virtual_pose import load_fit


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def plots(a):
    metrics=rows(a.model/"test_forecast.csv")
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    colors={"hold":"#697381","velocity":"#a65545","damped":"#d99a4e","gru":"#137d74"}
    for m in colors:
        here=[x for x in metrics if x["method"]==m]
        t=[float(x["seconds"]) for x in here]
        ax[0].plot(t,[float(x["finger"]) for x in here],"o-",c=colors[m],label=m)
        ax[1].plot(t,[float(x["wrist"]) for x in here],"o-",c=colors[m],label=m)
    ax[0].legend(frameon=False)
    for a0,label in zip(ax,("Finger geodesic error","Wrist geodesic error")):
        a0.set_xlabel("Free-running horizon (seconds)");a0.set_ylabel(label+" (degrees)");a0.grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.output/"rollout_error.png",dpi=180);plt.close(fig)
    table=rows(a.model/"test_per_clip.csv")
    rng=np.random.default_rng(1301); uncertainty=[]
    for h in (1,3,6,12,24,48):
        r=defaultdict(dict)
        for x in table:
            if int(x["horizon"])==h:
                r[x["source"]][x["method"]]=x
        for key in ("finger","wrist","upper","left","right"):
            d=np.array([float(y["gru"][key])-float(y["hold"][key]) for y in r.values()])
            q=np.percentile(d[rng.integers(len(d),size=(4000,len(d)))].mean(1),[2.5,97.5])
            uncertainty.append(dict(horizon=h,group=key,sources=len(d),mean_deg=float(d.mean()),lo_deg=q[0],hi_deg=q[1]))
    write_csv(a.output/"paired_source_bootstrap.csv",uncertainty)


def checks(a):
    table=rows(a.dataset/"sample.csv")
    stats={}
    for split in ("train","val","test"):
        use=[r for r in table if r["split"]==split]
        vals={}
        cols=("beta_max_span","camera_z_span","cam_step_jerk_p95","root_step_p99_deg","legs_step_p99_deg",
              "body_step_p99_deg","left_step_p99_deg","right_step_p99_deg","local_rotation_gt150_fraction")
        for k in cols:vals[k]=np.percentile([float(r[k]) for r in use],[50,95]).tolist()
        vals.update(clips=len(use),source_count=len({x["source"] for x in use}),frames=sum(int(x["frames"]) for x in use),
                    no_rotation_step_over90=sum(int(r["step_gt90_count"])==0 for r in use),
                    rotation_step_gt90_count=sum(int(r["step_gt90_count"]) for r in use))
        stats[split]=vals
    model,ck=load_predictor(a.model/json.loads((a.model/"selection.json").read_text())["checkpoint"])
    bank=populate(a.dataset,a.asset)
    means=hand_mean(a.asset); ctx=ck["context"]
    the=a.output/"motion_checks.json"
    fks=[]; physics=[]
    with TemporaryDirectory() as t:
        tmp=Path(t)/"fit.npz"; camera=np.array([[1,0,0],[0,1,0]])
        for cap in ("train","val","test"):
            row=next(q for q in table if q["split"]==cap)
            f=sample_fit(load_fit(a.dataset/row["path"]),np.arange(0,min(int(row["frames"]),31),2))
            z=decode(f,rotations(f,means),means)
            p,y,_=calculate(f,a.asset,camera,tmp)
            q,w,_=calculate(z,a.asset,camera,tmp)
            fks.append(dict(split=cap,clip=row["clip"],landmark_parity_max_m=float(abs(p-q).max()),
                            projection_parity_max_normalized=float(abs(y-w).max())))
        coeff=torch.tensor([[.03,-.09,.3],[1.7,-1.1,.8]],dtype=torch.float32)
        so3=float((log(exp(coeff))-coeff).abs().max())
        for h in (6,48):
            eligible=[r for r in bank["test"] if len(r[2])>=ctx+h]
            step=max(1,len(eligible)//12)
            mapping={r["clip"]:r["path"] for r in table if r["split"]=="test"}
            for sid,group,r,d in eligible[::step][:12]:
                ids=np.arange(0,2*(ctx+h),2)
                f=sample_fit(load_fit(a.dataset/mapping[sid]),ids)
                # Ground-truth shape/translation do not enter the model.
                f["betas"][:]=np.median(f["betas"][:ctx],axis=0)
                for key in ("cam_trans","expression"):
                    f[key][ctx:]=f[key][ctx-1]
                obs,_,_=calculate(f,a.asset,camera,tmp)
                local=wrist_local(obs,f,a.asset)
                x=torch.tensor(r[:ctx])[None];y=torch.tensor(d[:ctx])[None]
                with torch.no_grad():
                    roll,_=model.rollout(x,y,h)
                predicted=roll[0].numpy()
                candidates={"gru":predicted,"hold":r[ctx-1:ctx].repeat(h,0)}
                for name,z in candidates.items():
                    ob=decode(f,np.concatenate((r[:ctx],z)),means)
                    xyz,_,_=calculate(ob,a.asset,camera,tmp)
                    lk=wrist_local(xyz,ob,a.asset)
                    tip=np.linalg.norm((lk-local)[:,:,4::4],axis=-1)
                    ar=np.concatenate((ob["left_hand_pose"][ctx:],ob["right_hand_pose"][ctx:]),axis=1).reshape(h,30,3)
                    physics.append(dict(clip=sid,source=group,horizon=h,method=name,
                                      left_tip_final_mm=np.mean(tip[-1,0])*1000,right_tip_final_mm=np.mean(tip[-1,1])*1000,
                                      raw_local_axis_over_150_proxy=float((np.linalg.norm(ar,axis=-1)>np.deg2rad(150)).mean()),
                                      wrist_local_tip_speed_mm_s=float(np.linalg.norm(np.diff(lk[ctx-1:,:,4::4],axis=0),axis=-1).mean()*12000)))
    write_csv(a.output/"sample_fk_held.csv",physics)
    (the).write_text(json.dumps(dict(sample_quality=stats,Fk_roundtrip=fks,exp_log_max_rad=so3,
                                     hand_model_order="left/right: index middle little ring thumb",
                                     limitations=["SMPL-X raw angle norm proxy, not anatomical limit","FK sample scaffold uses reference inactive channels for diagnostics only"]),indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("dataset","model","asset","output"):
        p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    plots(a);checks(a)


if __name__=="__main__":
    main()
