"""Real video → frozen Uni-Sign → local 3D proposals with guarded rollback."""
import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from scripts.stage7_optimize import read,write
from scripts.stage8_imitate import Response,geometry_holdout
from scripts.stage9_audit import calculate
from scripts.stage9_run import Loop
from scripts.stage9_summary import dynamic
from scripts.stage10_audit import snapshot
from scripts.stage10_inputs import clip_inputs
from scripts.stage7_evaluate import project_side
from src.evaluate_feedback import write_csv
from src.stage7_feedback import HandGraphAdapter,HandGraphs,in_frame
from src.stage9_expert import DifferentiableResponse
from src.stage9_hand import Controller,SparseFK,tensor
from src.stage9_timing import track_stats,stabilize
from src.stage10_guard import FINGERS, Guard,blocks,compact_basis
from src.stage10_reanchor import neutral_reanchor
from src.stage10_world import body_joints,body_metrics,fixed_world,stabilized_body
from src.virtual_pose import load_fit

SIZE=np.array([1280,720])


def save_fit(path,fit):
    np.savez_compressed(path,**fit)


class LocalSearch:
    def __init__(self, cfg, prior, fit, asset, affine, idx, reference, scores, expert, tmp, dense_observations=None):
        self.cfg=cfg
        self.idx=idx
        self.expert=expert
        self.all=np.arange(len(fit["body_pose"]))
        self.fit=fit
        xyz,xy,_=calculate(fit,asset,affine,tmp)
        self.fk=SparseFK(fit,asset,affine,xy)
        self.ctrl=Controller(self.fk,prior["knot_spacing"],prior["max_knots"])
        self.knots,self.ctrl.interp=compact_basis(len(xyz),prior["knot_spacing"])
        self.ctrl.knots=len(self.knots)
        self.groups=blocks(self.ctrl.names)
        self.diff=DifferentiableResponse(expert,prior)
        self.zero=self.ctrl.zero()
        xy,points,rot=self.forward(self.zero)
        self.start=self.score(xy)
        self.guard=Guard(cfg,reference,scores,idx,xy.numpy(),points.numpy(),rot.numpy())
        self.dense_guard=None
        if dense_observations is not None:
            obs,confidence=dense_observations
            self.even=np.arange(0,len(xyz),2)
            p=self.fk.points(rot[self.even],self.even)
            y=self.fk.project(p,self.even)
            self.dense_guard=Guard(cfg,obs[self.even],confidence[self.even],self.even,y.numpy(),p.numpy(),rot.numpy())
        self.parity=float(abs(xy.numpy()-self.fk.base_xy.numpy()[idx]).max()*1280)
        diffscore=float(self.diff.value(xy,"factor"))
        if self.parity>.02 or abs(diffscore-self.start["response"])>1e-5:
            raise ValueError(f"Projection/expert replay failed: {self.parity}, {diffscore}, {self.start}")

    def forward(self, coef):
        rot=self.ctrl.rotations(coef)
        points=self.fk.points(rot[self.idx],self.idx)
        return self.fk.project(points,self.idx),points,rot

    def score(self,xy):
        xy=xy.detach().numpy()
        return self.expert.score(xy,np.ones(xy.shape[:2],np.float32))[0]

    def refine(self,out,strict):
        start=perf_counter()
        out.mkdir(exist_ok=True,parents=True)
        current=self.zero.clone()
        before=self.start.copy()
        trust=np.deg2rad(self.cfg["initial_block_step_deg"])
        weights=self.ctrl.bounds
        trace=[dict(step=0,accepted=False,expert=before["response"],candidate=before["response"],wall_s=0.)]
        attempts=[]; kept=[current.numpy().copy()]; trials=[]
        visited=Counter()
        directions=None
        passes=0
        for step in range(1,self.cfg["proposals"]+1):
            if directions is None:
                seed=current.detach().clone().requires_grad_()
                grad,=torch.autograd.grad(self.diff.value(self.forward(seed)[0],"factor"),seed)
                grad=(grad/weights[None]).detach()
                directions=[]
                for name,cols in self.groups.items():
                    for i,value in enumerate(grad[:,cols]):
                        size=float(torch.linalg.vector_norm(value))
                        if size>1e-9:
                            directions.append((size,name,i,cols,value/size))
                passes+=1
            power,group,k,cols,unit=max(directions,key=lambda x:x[0]*self.cfg["step_shrink"]**(3*visited[(x[1],x[2])]))
            name=(group,k)
            radius=trust*self.cfg["step_shrink"]**visited[name]
            trial=current.clone()
            trial[k,cols]-=radius*unit/weights[cols]
            with torch.no_grad():
                xy,xyz,rot=self.forward(trial)
                cand=self.score(xy)
                passes+=1
                reason=self.guard.screen(rot.numpy(),xy.numpy(),xyz.numpy(),strict)
                if strict and not reason and self.dense_guard is not None:
                    xyze=self.fk.points(rot[self.even],self.even)
                    xye=self.fk.project(xyze,self.even)
                    reason=self.dense_guard.screen(rot.numpy(),xye.numpy(),xyze.numpy(),True)
                    if reason:
                        reason="dense_"+reason
            if cand["support"]<self.start["support"]-.025:
                reason="visibility"
            if not reason and before["response"]-cand["response"]<self.cfg["minimum_gain"]:
                reason="expert_no_gain"
            accepted=not reason
            if accepted:
                current=trial; before=cand
                directions=None
                visited.clear()
            else:
                visited[name]+=1
            attempts.append(dict(step=step,group=group,knot=int(self.knots[k]),radius_deg=np.rad2deg(radius),
                                 accepted=accepted,reason=reason,**cand))
            trials.append(trial.numpy().copy())
            kept.append(current.numpy().copy())
            trace.append(dict(step=step,accepted=accepted,expert=before["response"],candidate=cand["response"],
                              wall_s=perf_counter()-start))
        elapsed=perf_counter()-start
        save_fit(out/"final_fit.npz",self.ctrl.export(current))
        for step in (0,9,18,self.cfg["proposals"]):
            save_fit(out/f"step_{step:02}_fit.npz",self.ctrl.export(tensor(kept[step])))
        np.savez_compressed(out/"control_trajectory.npz",kept=kept,trial=trials,names=self.ctrl.names,knots=self.knots)
        write_csv(out/"trace.csv",trace)
        write_csv(out/"proposals.csv",attempts)
        result=dict(accepted=sum(row["accepted"] for row in attempts),proposals=len(attempts),
                    expert_start=self.start["response"],expert_final=before["response"],wall_s=elapsed,
                    expert_passes=passes,train_fk_parity_px=self.parity,
                    rejection=dict(Counter(row["reason"] for row in attempts if row["reason"])))
        write(out/"refinement.json",result)
        return self.ctrl.export(current),result


def dense_geometry(cfg,prior,search,obs,confidence,out):
    """Repair the original geometry anchor from densely sampled even frames."""
    start=perf_counter()
    out.mkdir(exist_ok=True,parents=True)
    ids=np.arange(0,len(obs),2)
    fk=search.fk
    ctrl=Controller(fk,prior["knot_spacing"],prior["max_knots"])
    knots,ctrl.interp=compact_basis(len(obs),cfg["dense_spacing"])
    ctrl.knots=len(knots)
    with torch.no_grad():
        bx=fk.points(fk.base_rot[ids],ids)
        by=fk.project(bx,ids)
    residual=Guard(cfg,obs[ids],confidence[ids],ids,by.numpy(),bx.numpy(),fk.base_rot.numpy())
    valid=tensor(residual.valid&(residual.valid[:,:,:1]))
    real=tensor(obs[ids][:,FINGERS]*SIZE)
    weight=valid*tensor(np.clip(confidence[ids][:,FINGERS],0,1))
    weight[:,:,0]=0
    opt=tensor(np.isin(np.arange(len(ctrl.names)),np.concatenate(list(search.groups.values()))))
    x=torch.nn.Parameter(ctrl.zero())
    def state(q):
        rot=ctrl.rotations(q)
        p=fk.points(rot[ids],ids)
        xy=fk.project(p,ids)
        h=xy[:,FINGERS]*tensor(SIZE)
        d=h-real
        l=d-d[:,:,:1]
        robust=lambda v:torch.sqrt((v*v).sum(-1)+16)-4
        match=(.72*robust(l)+.28*robust(d))*weight
        overall=match.sum()/weight.sum().clamp_min(1.)
        tim=(ctrl.angles(q).diff(n=2,dim=0)**2).mean()
        return overall/20+2*tim+.0004*(q*q).mean(),xy,p,rot
    adam=torch.optim.Adam([x],lr=cfg["dense_lr"])
    trace=[]; coefficients=[x.detach().numpy().copy()]
    with torch.no_grad():
        best=float(state(x)[0])
    for step in range(1,cfg["dense_proposals"]+1):
        old=x.detach().clone()
        adam.zero_grad()
        state(x)[0].backward()
        x.grad*=opt
        adam.step()
        with torch.no_grad():
            value,xy,p,rot=state(x)
            reason=residual.screen(rot.numpy(),xy.numpy(),p.numpy(),False)
            if not reason and float(value)>=best:
                reason="geometry_no_gain"
            if reason:
                x.copy_(old)
                adam.param_groups[0]["lr"]=max(adam.param_groups[0]["lr"]*.85,.001)
            else:
                best=float(value)
        trace.append(dict(step=step,accepted=not reason,reason=reason,candidate_objective=float(value),retained_objective=best))
        coefficients.append(x.detach().numpy().copy())
    result=ctrl.export(x)
    save_fit(out/"final_fit.npz",result)
    write_csv(out/"trace.csv",trace)
    np.savez_compressed(out/"control_trajectory.npz",coefs=coefficients,knots=knots,names=ctrl.names)
    write(out/"geometry_update.json",dict(proposals=len(trace),accepted=sum(r["accepted"] for r in trace),
              optimization_frames=ids.tolist(),reliable_per_hand=residual.valid.sum(axis=(0,2)).tolist(),
              wall_s=perf_counter()-start))
    return result


def get_stage9(args, prior, source, s9, data, root, expert, tmp):
    if source not in args.config_data["locked"]:
        a=s9/"B/art_joint_geo_factor"
        b=s9/"B/art_joint_factor"
    else:
        raw,nsa,idx,obs,scores,affine,mp,_,_=data
        # Same Stage 9 methods from the identical coarse IK; alternate input root so no archive is overwritten.
        a=root/"upstream"/"art_joint_geo"
        b=root/"upstream"/"art_joint"
        staged=root/"upstream_input"/source
        staged.mkdir(exist_ok=True,parents=True)
        (staged/"B").mkdir(exist_ok=True)
        save_fit(staged/"B/initial_fit.npz",raw)
        write(staged/"frames.json",dict(optimization_frames=idx.tolist(),held_frames=list(range(1,len(obs),2))))
        p=dict(prior,upstream=str(staged.parent),observations="experiments/Stage6")
        if not (a/"result.json").exists() or not (b/"result.json").exists():
            loop=Loop(args,p,source,"B",expert.graphs,tmp)
            for name,loc in (("art_joint_geo",a),("art_joint",b)):
                if not (loc/"result.json").exists():
                    loop.solve(name,loc,"factor",prior["budget"])
    return load_fit(a/"final_fit.npz"), load_fit(b/"final_fit.npz"),read(a/"result.json"),read(b/"result.json")


def evaluate(args,cfg,source,root,prior,tmp,fit,variant,affine,expert,idx,mp):
    pts,xy,_=calculate(fit,args.asset,affine,tmp)
    held=geometry_holdout(xy[1::2],mp[1::2],SIZE)
    per_digit=dynamic(source,"B",variant,xy,mp,pts,fit,args.asset)
    if per_digit:
        write_csv(root/variant/"held_per_digit.csv",per_digit)
    else:
        (root/variant/"held_per_digit.csv").write_text("source,variant,side,finger,frames,centered_motion_error_px\n")
    def mean_or_missing(values):
        return float(np.mean(values)) if values else None
    m,_=expert.score(xy[idx],np.ones(xy[idx].shape[:2],np.float32))
    diagnostics=track_stats(pts,fit,args.asset)
    joints=body_joints(fit,args.asset,affine,xy)
    timing=body_metrics(fit,joints,pts,xy,affine)
    score=dict(source=source,variant=variant,frames=len(xy),opt_frames=len(idx),expert=m["response"],
               expert_support=m["support"],
               dynamic_px=mean_or_missing([r["centered_motion_error_px"] for r in per_digit]),
               **held,**{**diagnostics,**timing})
    for side in ("left","right"):
        score[side+"_dynamic_px"]=mean_or_missing([r["centered_motion_error_px"] for r in per_digit if r["side"]==side])
    status={x["clip"]:x for x in read(Path("experiments/Stage7/multiview/status.json"))}.get(source,{})
    if status.get("status")=="proxy_side_assessment" and status["associated_odd_hands"]>=20:
        with np.load(Path("experiments/Stage7/multiview/pose")/f"{source}_mp.npz") as p:
            sidehold=p["mp"]
        cam=(status["calibration_error_px"],status["camera_yaw_deg"],status["camera_scale"],status["camera_offset_px"])
        side=geometry_holdout(project_side(pts,cam)[1::2],sidehold[1::2],SIZE)
        score.update({"side_"+k:v for k,v in side.items()})
        score["proxy_side_camera_anchor_px"]=status["calibration_error_px"]
    write(root/variant/"metrics.json",score)
    return score,pts,xy


def run_source(args,cfg,source,prior,graphs,tmp):
    root=args.output/source
    root.mkdir(parents=True,exist_ok=True)
    data=clip_inputs(cfg,source,args.asset,root)
    coarse,nsa,idx,obs,scores,affine,mp,s9,video=data
    mp=np.where(in_frame(mp)[...,None],mp,np.nan)
    expert=Response(graphs,obs[idx],scores[idx],{"stage7_factor_weights":prior["factor_weights"]})
    expert.save(root/"real_expert_reference.npz")
    g,j,gmeta,jmeta=get_stage9(args,prior,source,s9,data,root,expert,tmp)
    offset=np.median(g["cam_trans"],axis=0)
    def prepare(fit):
        f=fixed_world(fit,offset)
        return stabilized_body(f,idx,scores[idx],cfg) if source in cfg["seated"] else f
    old={"coarse":fixed_world(coarse,offset),"A_stage9_geometry":fixed_world(g,offset),
         "B_stage9_joint":fixed_world(j,offset),"A_geometry":prepare(g),"B_joint":prepare(j)}
    compare={}; fits={}
    hand=[]; per_digit=[]
    for name,fit in old.items():
        loc=root/name
        loc.mkdir(exist_ok=True)
        save_fit(loc/"final_fit.npz",fit)
        fits[name]=fit
    search=LocalSearch(cfg,prior,old["A_geometry"],args.asset,affine,idx,obs[idx],scores[idx],expert,tmp)
    for variant,strict in (("C_local_expert",False),("D_guarded",True)):
        if (root/variant/"refinement.json").exists():
            fits[variant]=load_fit(root/variant/"final_fit.npz")
        else:
            fits[variant],_=search.refine(root/variant,strict)
    dense_dir=root/"E_dense_geometry"
    if (dense_dir/"geometry_update.json").exists():
        fits["E_dense_geometry"]=load_fit(dense_dir/"final_fit.npz")
    else:
        fits["E_dense_geometry"]=dense_geometry(cfg,prior,search,obs,scores,dense_dir)
    dense=LocalSearch(cfg,prior,fits["E_dense_geometry"],args.asset,affine,idx,obs[idx],scores[idx],expert,tmp,
                      dense_observations=(obs,scores))
    if (root/"F_dense_guarded"/"refinement.json").exists():
        fits["F_dense_guarded"]=load_fit(root/"F_dense_guarded/final_fit.npz")
    else:
        fits["F_dense_guarded"],_=dense.refine(root/"F_dense_guarded",True)
    re_dir=root/"G_reanchored_geometry"
    if (re_dir/"geometry_update.json").exists():
        fits["G_reanchored_geometry"]=load_fit(re_dir/"final_fit.npz")
    else:
        reanchored,ledger,used=neutral_reanchor(search,old["A_geometry"],coarse,obs,scores,cfg)
        re_dir.mkdir(exist_ok=True)
        np.savez_compressed(re_dir/"neutral_reanchor.npz",**ledger,accepted=used)
        save_fit(re_dir/"neutral_base_fit.npz",reanchored)
        step=LocalSearch(cfg,prior,reanchored,args.asset,affine,idx,obs[idx],scores[idx],expert,tmp)
        fits["G_reanchored_geometry"]=dense_geometry(cfg,prior,step,obs,scores,re_dir)
    re_search=LocalSearch(cfg,prior,fits["G_reanchored_geometry"],args.asset,affine,idx,obs[idx],scores[idx],expert,tmp,
                          dense_observations=(obs,scores))
    if (root/"H_reanchored_guarded"/"refinement.json").exists():
        fits["H_reanchored_guarded"]=load_fit(root/"H_reanchored_guarded/final_fit.npz")
    else:
        fits["H_reanchored_guarded"],_=re_search.refine(root/"H_reanchored_guarded",True)
    w=search.guard.valid
    coverage=dict(train_valid_frames=np.sum(w[:,:,0],axis=0).tolist(),
                  train_reliable_landmarks_per_hand=np.sum(w,axis=(0,2)).tolist())
    write(root/"protocol.json",dict(source=source,optimization_indices=idx.tolist(),heldout_odd_frames=np.arange(1,len(obs),2).tolist(),
          model_partition="locked" if source in cfg["locked"] else "development",
          fixed_extrinsic_t=offset.tolist(),fixed_extrinsic_R=np.eye(3).tolist(),affine=affine.tolist(),
          seated=source in cfg["seated"],**coverage))
    for name,fit in fits.items():
        m,pts,xy=evaluate(args,cfg,source,root,prior,tmp,fit,name,affine,expert,idx,mp)
        for path in (root/name/"refinement.json",):
            if path.exists():
                detail=read(path)
                m.update({k:detail[k] for k in ("proposals","accepted","wall_s","expert_passes","expert_start","expert_final")})
        if name.startswith(("A_","B_")):
            meta=gmeta if name.startswith("A_") else jmeta
            m.update(archived_proposals=meta["proposal_count"],archived_wall_s=meta["optim_time_s"])
        compare[name]=m
        if source==cfg["focus"]:
            hand.extend(snapshot(name,pts,xy,args.asset,obs,scores,np.r_[86:97],fit))
    write(root/"results.json",compare)
    if hand:
        write_csv(root/"focus_hands.csv",hand)
    for name,m in compare.items():
        print(source,name,round(m["expert"],5),m["held_geometry"],m["dynamic_px"],m.get("accepted"),flush=True)
    return compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("repo","checkpoint","asset","output"):
        p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--sources",nargs="+")
    p.add_argument("--partition",choices=("development","locked"),default="development")
    p.add_argument("--config",type=Path,default=Path("configs/stage10.json"))
    a=p.parse_args()
    cfg=read(a.config); a.config_data=cfg
    prior=read(Path("configs/stage9.json"))
    torch.manual_seed(0)
    graphs=HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",2))
    a.output.mkdir(exist_ok=True,parents=True)
    with TemporaryDirectory() as td:
        for source in a.sources or cfg[a.partition]:
            run_source(a,cfg,source,prior,graphs,Path(td)/"fit.npz")


if __name__=="__main__":
    main()
