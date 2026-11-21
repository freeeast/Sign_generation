"""Attach the frozen shared prior to the existing guarded 3D expert optimizer."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from scripts.shared_motion_train import write_csv
from scripts.stage7_optimize import read
from scripts.stage8_imitate import Response
from scripts.stage10_inputs import clip_inputs
from scripts.stage10_run import LocalSearch,evaluate,save_fit
from src.shared_motion import ACTIVE,load_predictor,log
from src.stage7_feedback import HandGraphAdapter,HandGraphs,in_frame
from src.stage9_hand import tensor
from src.virtual_pose import load_fit

ARMS={"A_geometry":"I_matched_geometry","B_stage9_joint":"B_stage9_joint","B_matched_expert":"H_reanchored_guarded",
      "anchor":"G_reanchored_geometry"}


class SharedPrior:
    def __init__(self,model,context,length):
        self.model=model
        self.ctx=context
        self.ends=torch.arange(context+1,(length+1)//2,3)
        if not len(self.ends):
            raise ValueError("Motion too short for the prior")
        self.hist=self.ends[:,None]-torch.arange(context,0,-1)[None]

    def __call__(self,rotation):
        r=rotation[::2][:,ACTIVE]
        d=log(r[:-1].transpose(-1,-2)@r[1:])
        d=torch.cat((torch.zeros_like(d[:1]),d))
        predicted,_=self.model(r[self.hist],d[self.hist])
        # Prefer a predictor-consistent transition while retaining the observed history.
        # Stop-gradient in the target forecast makes the local step less eager to rewrite history.
        delta=(d[self.ends]-predicted.detach())/self.model.scale
        return torch.nn.functional.smooth_l1_loss(delta[:,11:],torch.zeros_like(delta[:,11:]))


def refine(search,prior,path,variant):
    path.mkdir(parents=True,exist_ok=True)
    cfg=search.cfg
    x=search.zero.clone()
    out=search.start["response"]
    with torch.no_grad():
        p=float(prior(search.ctrl.rotations(x)))
    extra=variant=="D_shared_expert"
    h=x.clone().requires_grad_()
    e,_,r=search.forward(h)
    g=torch.autograd.grad(search.diff.value(e,"factor"),h,retain_graph=True)[0]
    q=torch.autograd.grad(prior(r),h)[0]
    dofs=sorted({c for cols in search.groups.values() for c in cols})
    weight=float((.35*torch.linalg.vector_norm(g[:,dofs])/torch.linalg.vector_norm(q[:,dofs]).clamp_min(1e-8)).clamp(.0001,.08))
    trust=np.deg2rad(cfg["initial_block_step_deg"])
    attempts=[];coefs=[x.numpy().copy()]; trials=[]
    rows=[dict(step=0,accepted=False,expert=out,prior=p,objective=(out+weight*p) if extra else p,wall_s=0.)]
    trials_at=Counter(); directions=None
    weights=search.ctrl.bounds
    start=perf_counter()
    for i in range(1,cfg["proposals"]+1):
        if directions is None:
            seed=x.clone().requires_grad_(True)
            xy,_,rot=search.forward(seed)
            loss=prior(rot)
            if extra:
                loss=search.diff.value(xy,"factor")+weight*loss
            grad=torch.autograd.grad(loss,seed)[0].detach()/weights
            directions=[]
            for name,cols in search.groups.items():
                for k,z in enumerate(grad[:,cols]):
                    length=float(torch.linalg.vector_norm(z))
                    if length>1e-10:
                        directions.append((length,name,k,cols,z/length))
        if not directions:
            break
        _,name,k,cols,z=max(directions,key=lambda v:v[0]*cfg["step_shrink"]**(3*trials_at[(v[1],v[2])]))
        radius=trust*cfg["step_shrink"]**trials_at[(name,k)]
        y=x.clone()
        y[k,cols]-=radius*z/weights[cols]
        with torch.no_grad():
            xy,xyz,rot=search.forward(y)
            obs=search.score(xy)
            pr=float(prior(rot))
            reason=search.guard.screen(rot.numpy(),xy.numpy(),xyz.numpy(),True)
            if not reason:
                px=search.fk.points(rot[search.even],search.even)
                py=search.fk.project(px,search.even)
                reason=search.dense_guard.screen(rot.numpy(),py.numpy(),px.numpy(),True)
                if reason:reason="dense_"+reason
        e=obs["response"]
        val=e+weight*pr if extra else pr
        current=out+weight*p if extra else p
        if obs["support"]<search.start["support"]-.025:reason="visibility"
        if not reason and extra and e>out+1e-7:reason="expert_increase"
        if not reason and (not np.isfinite(val) or val>=current-1e-7):reason="no_objective_gain"
        moved=not reason
        if moved:
            x=y;out=e;p=pr
            trials_at.clear();directions=None
        else:
            trials_at[(name,k)]+=1
        attempts.append(dict(step=i,block=name,knot=int(search.knots[k]),reason=reason,accepted=moved,
                             trial_prior=pr,trial_expert=e,step_degrees=np.rad2deg(radius)))
        trials.append(y.numpy().copy());coefs.append(x.numpy().copy())
        rows.append(dict(step=i,accepted=moved,expert=out,prior=p,objective=out+weight*p if extra else p,
                         wall_s=perf_counter()-start))
    seconds=perf_counter()-start
    fit=search.ctrl.export(x)
    for i in sorted({0,len(rows)//2,len(rows)-1}):
        save_fit(path/f"step_{i:02d}_fit.npz",search.ctrl.export(tensor(coefs[i])))
    save_fit(path/"final_fit.npz",fit)
    np.savez_compressed(path/"trajectory.npz",trial=trials,kept=coefs,names=search.ctrl.names,knots=search.knots)
    write_csv(path/"trace.csv",rows)
    write_csv(path/"proposals.csv",attempts)
    return fit,dict(seconds=seconds,proposals=len(rows)-1,accepted=sum(x["accepted"] for x in attempts),
                    gain=rows[0]["expert"]-out,prior_start=rows[0]["prior"],prior_final=p,
                    prior_weight=weight if extra else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("repo","checkpoint","shared-checkpoint","asset","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--sources",nargs="+")
    a=p.parse_args()
    config=read(Path("configs/stage10.json"))
    old=read(Path("configs/stage9.json"))
    net,ck=load_predictor(a.shared_checkpoint)
    g=HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",2))
    rows=[]
    target=a.output/"integration.csv"
    if target.exists():
        with target.open() as f:
            existing=list(csv.DictReader(f))
        completed={r["source"] for r in existing if r["variant"]=="D_shared_expert"}
        rows.extend(r for r in existing if r["source"] in completed)
    else:
        completed=set()
    with TemporaryDirectory() as tmp:
        temp=Path(tmp)/"fit.npz"
        for source in a.sources or config["development"]:
            if source in completed:
                continue
            root=a.output/source
            root.mkdir(parents=True,exist_ok=True)
            archived=Path("experiments/Stage10")/source
            _,_,idx,ref,conf,affine,mp,_,_=clip_inputs(config,source,a.asset,root)
            expert=Response(g,ref[idx],conf[idx],{"stage7_factor_weights":old["factor_weights"]})
            mp=np.where(in_frame(mp)[...,None],mp,np.nan)
            fit=load_fit(archived/"G_reanchored_geometry/final_fit.npz")
            search=LocalSearch(config,old,fit,a.asset,affine,idx,ref[idx],conf[idx],expert,temp,
                               dense_observations=(ref,conf))
            prior=SharedPrior(net,ck["context"],len(fit["body_pose"]))
            for variant,original in ARMS.items():
                path=root/variant
                path.mkdir(exist_ok=True)
                f=load_fit(archived/original/"final_fit.npz")
                save_fit(path/"final_fit.npz",f)
                m=read(archived/"results.json")[original]
                row,_,_=evaluate(a,config,source,root,old,temp,f,variant,affine,expert,idx,mp)
                r=search.fk.base_rot if variant=="anchor" else None
                arc=read(archived/original/"refinement.json") if (archived/original/"refinement.json").exists() else m
                rows.append(dict(**row,proposals=arc.get("proposals",arc.get("archived_proposals",0)),accepted=arc.get("accepted"),
                                 seconds=arc.get("wall_s",arc.get("archived_wall_s",0)),prior_final=float(prior(r)) if r is not None else None))
            for variant in ("C_shared_only","D_shared_expert"):
                fit,meta=refine(search,prior,root/variant,variant)
                row,_,_=evaluate(a,config,source,root,old,temp,fit,variant,affine,expert,idx,mp)
                rows.append(dict(**row,**meta))
                print(source,variant,round(row["expert"],6),round(row["held_geometry"],4),meta,flush=True)
            write_csv(a.output/"integration.csv",rows)


if __name__=="__main__":
    main()
