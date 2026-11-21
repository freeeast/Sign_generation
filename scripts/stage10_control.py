"""Match the last 36 expert trials with a 36-proposal geometry-only control."""
import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from scripts.stage7_optimize import read,write
from scripts.stage8_imitate import Response
from scripts.stage10_inputs import clip_inputs
from scripts.stage10_run import LocalSearch,evaluate,save_fit
from src.evaluate_feedback import write_csv
from src.stage10_guard import FINGERS,SIZE
from src.stage7_feedback import HandGraphAdapter,HandGraphs,in_frame
from src.stage9_hand import tensor
from src.virtual_pose import load_fit


def geometry_trials(search,root,dense_obs,dense_scores):
    cfg=search.cfg
    start=perf_counter()
    even=search.even
    fk=search.fk
    ref=tensor(dense_obs[even][:,FINGERS]*SIZE)
    valid=tensor(search.dense_guard.valid)
    w=tensor(np.clip(dense_scores[even][:,FINGERS],0,1))*valid*valid[:,:,:1]
    w[:,:,0]=0
    def objective(rot):
        p=fk.points(rot[even],even)
        d=fk.project(p,even)[:,FINGERS]*tensor(SIZE)-ref
        loss=lambda x:(torch.sqrt((x*x).sum(-1)+16)-4)
        value=(w*(.28*loss(d)+.72*loss(d-d[:,:,:1]))).sum()/w.sum().clamp_min(1)
        return value,p
    x=search.zero.clone()
    before=float(objective(search.ctrl.rotations(x))[0])
    weights=search.ctrl.bounds
    trace=[]; kept=[x.numpy().copy()]; trials=[]
    directions=None; visited=Counter()
    for i in range(1,cfg["proposals"]+1):
        if directions is None:
            q=x.detach().clone().requires_grad_()
            grad,=torch.autograd.grad(objective(search.ctrl.rotations(q))[0],q)
            grad=grad.detach()/weights[None]
            directions=[]
            for name,cols in search.groups.items():
                for k,g in enumerate(grad[:,cols]):
                    s=float(torch.linalg.vector_norm(g))
                    if s>1e-9:
                        directions.append((s,name,k,cols,g/s))
        strength,name,k,cols,g=max(directions,key=lambda r:r[0]*cfg["step_shrink"]**(3*visited[(r[1],r[2])]))
        step=cfg["initial_block_step_deg"]*cfg["step_shrink"]**visited[(name,k)]
        y=x.clone()
        y[k,cols]-=np.deg2rad(step)*g/weights[cols]
        with torch.no_grad():
            xy,p,rot=search.forward(y)
            score,pe=objective(rot)
            reason=search.guard.screen(rot.numpy(),xy.numpy(),p.numpy(),True)
            if not reason:
                reason=search.dense_guard.screen(rot.numpy(),fk.project(pe,even).numpy(),pe.numpy(),True)
        if not reason and before-float(score)<1e-4:
            reason="geometry_no_gain"
        if not reason:
            x=y;before=float(score)
            visited.clear();directions=None
        else:
            visited[(name,k)]+=1
        trials.append(y.numpy().copy()); kept.append(x.numpy().copy())
        trace.append(dict(step=i,accepted=not reason,reason=reason,loss=before,trial_loss=float(score),group=name,
                          knot=int(search.knots[k]),radius_deg=step,wall_s=perf_counter()-start))
    fit=search.ctrl.export(x)
    root.mkdir(exist_ok=True)
    save_fit(root/"final_fit.npz",fit)
    write_csv(root/"trace.csv",trace)
    np.savez_compressed(root/"control_trajectory.npz",kept=kept,trial=trials,names=search.ctrl.names,knots=search.knots)
    meta=dict(proposals=len(trace),accepted=sum(row["accepted"] for row in trace),wall_s=perf_counter()-start)
    write(root/"geometry_update.json",meta)
    return fit,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","checkpoint","asset","output"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--sources",nargs="+")
    a=p.parse_args()
    cfg=read(Path("configs/stage10.json")); prior=read(Path("configs/stage9.json"))
    graphs=HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",2))
    with TemporaryDirectory() as td:
        for source in a.sources or cfg["development"]+cfg["locked"]:
            root=a.output/source
            name="I_matched_geometry"
            if (root/name/"geometry_update.json").exists():
                continue
            _,_,idx,obs,conf,camera,mp,_,_=clip_inputs(cfg,source,a.asset,root)
            mp=np.where(in_frame(mp)[...,None],mp,np.nan)
            ref=Response(graphs,obs[idx],conf[idx],{"stage7_factor_weights":prior["factor_weights"]})
            search=LocalSearch(cfg,prior,load_fit(root/"G_reanchored_geometry/final_fit.npz"),a.asset,
                               camera,idx,obs[idx],conf[idx],ref,Path(td)/"fit.npz",dense_observations=(obs,conf))
            fit,meta=geometry_trials(search,root/name,obs,conf)
            result,_,_=evaluate(a,cfg,source,root,prior,Path(td)/"fit.npz",fit,name,camera,ref,idx,mp)
            all_scores=read(root/"results.json")
            all_scores[name]={**result,**meta}
            write(root/"results.json",all_scores)
            print(source,name,result["expert"],result["held_geometry"],meta,flush=True)


if __name__=="__main__":
    main()
