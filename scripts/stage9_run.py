"""Run Stage 9 articulated imitation with a 36-proposal ledger and replayable checkpoints."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from scipy.ndimage import minimum_filter1d

from scripts.stage7_optimize import read,write
from scripts.stage8_imitate import Response,geometry_holdout
from scripts.stage9_audit import calculate
from src.evaluate_feedback import write_csv
from src.stage7_feedback import HandGraphAdapter,HandGraphs,in_frame
from src.stage9_expert import DifferentiableResponse
from src.stage9_hand import Controller,SparseFK,WRIST,tensor
from src.stage9_timing import track_stats,stabilize
from src.virtual_pose import load_fit

MAIN=("art_geo","art_hier","art_joint")
PIX=np.array([1280,720])


def scoring_term(terms,which):
    return terms["response" if which=="factor" else which]


class Loop:
    def __init__(self, a,cfg,source,level,graphs,temp,raw=False,guarded=False):
        self.cfg=cfg; self.a=a; self.temp=temp; self.source=source; self.level=level
        root=Path(cfg["upstream"])/source
        data=Path(cfg["observations"])/"data"
        self.indices=np.asarray(read(root/"frames.json")["optimization_frames"])
        self.raw=load_fit(root/level/"initial_fit.npz")
        self.all=np.arange(len(self.raw["body_pose"]))
        with np.load(data/"paired"/f"{source}.npz") as p:
            self.camera=p["camera"]; self.mp=np.where(in_frame(p["mp"])[...,None],p["mp"],np.nan)
        with np.load(data/"real_pose"/f"{source}.npz") as p:
            xy=p["keypoints"][self.indices]; scores=p["scores"][self.indices]
        self.response=Response(graphs,xy,scores,{"stage7_factor_weights":cfg["factor_weights"]})
        self.differentiable=DifferentiableResponse(self.response,cfg)
        self.reference=tensor(xy)
        self.weights=tensor(self.response.valid*np.clip(scores,0,1))
        self.pixel=tensor(PIX)
        self.raw_pts,self.raw_xy,_=calculate(self.raw,a.asset,self.camera,temp)
        self.start_fit=self.raw if raw else stabilize(self.raw)
        self.stable_pts,self.stable_xy,_=calculate(self.start_fit,a.asset,self.camera,temp)
        self.fk=SparseFK(self.start_fit,a.asset,self.camera,self.stable_xy)
        self.ctrl=Controller(self.fk,cfg["knot_spacing"],cfg["max_knots"])
        b=self.fk.base_rot.detach()
        self.store_limits=torch.maximum(
            self.increment(b),tensor([cfg["max_adjacent_deg"]["arm"]]*6+[cfg["max_adjacent_deg"]["finger"]]*30))
        self.abs_limits=torch.maximum(self.angles(b[:,np.r_[16:22,25:55]]),
                            tensor([cfg["max_local_deg"]["arm"]]*4+[cfg["max_local_deg"]["wrist"]]*2+[cfg["max_local_deg"]["finger"]]*30))
        if guarded:
            # Keep inherited critical rotations fixed over a short neighborhood.
            # Use physical joint headroom, before seeing any optimizer results.
            indices=np.r_[16:22,25:55]
            absolute=(self.abs_limits-self.angles(b[:,indices])).numpy()
            adjacent=(self.store_limits-self.increment(b)).numpy()
            cap=np.minimum(absolute,np.minimum(np.vstack([adjacent[:1],adjacent]),
                                                np.vstack([adjacent,adjacent[-1:]])))
            mask=np.clip((cap-0.25)/10,0,1)
            available=np.ones((len(b),55),dtype=np.float32)
            available[:,indices]=minimum_filter1d(mask,size=5,axis=0,mode="nearest")
            self.ctrl.mobility=tensor(available)
        self.original_obs=self.measure_xy(tensor(self.raw_xy[self.indices]))
        self.original_held=geometry_holdout(self.raw_xy[1::2],self.mp[1::2],PIX)

    @staticmethod
    def angles(rot):
        tr=rot.diagonal(dim1=-1,dim2=-2).sum(-1)
        return torch.rad2deg(torch.acos(((tr-1)/2).clamp(-1,1)))

    def increment(self,rot):
        r=rot[:,np.r_[16:22,25:55]]
        v=r[:-1].transpose(-1,-2)@r[1:]
        return self.angles(v)

    def forward(self,x,train=True):
        rot=self.ctrl.rotations(x)
        keep=self.indices if train else self.all
        return self.fk.project(self.fk.points(rot[keep],keep),keep),rot

    def terms(self,xy):
        delta=(xy-self.reference)*self.pixel
        wrist=delta[:,WRIST]
        fingers=delta[:,91:133].reshape(-1,2,21,2)[:,:,1:]
        weights=self.weights[:,91:133].reshape(-1,2,21)[:,:,1:]
        ww=self.weights[:,WRIST]
        sk=weights*ww[:,:,None]
        def average(points,w,epsilon=6.):
            norm=torch.sqrt((points*points).sum(-1)+1e-6)
            robust=torch.sqrt(norm*norm+epsilon*epsilon)-epsilon
            return (robust*w).sum()/w.sum().clamp_min(1)
        step=tensor(np.diff(self.indices))[:,None,None]
        return dict(wrist=average(wrist,ww),arm=average(delta[:,5:9],self.weights[:,5:9]),
                    hand=average(fingers,weights),shape=average(fingers-wrist[:,:,None],sk),
                    motion=average((wrist[1:]-wrist[:-1])/step,ww[:-1]*ww[1:],2.))

    def measure_xy(self,xy):
        geometry=self.terms(xy)
        scores,_=self.response.score(xy.numpy(),np.ones(xy.shape[:2],np.float32))
        return {**scores,**{k:float(v) for k,v in geometry.items()}}

    def constraints(self,rot):
        current=self.increment(rot)
        angles=self.angles(rot[:,np.r_[16:22,25:55]])
        n=int((current>self.store_limits+.05).sum())+int((angles>self.abs_limits+.05).sum())
        return n,float(current[:,:6].max()),float(current[:,6:].max())

    def acceptance(self,metrics,phase,anchor,threshold):
        if metrics["validity_violations"]:
            return "rotation_or_speed"
        if metrics["support"]<self.original_obs["support"]-self.cfg["missing_support_tolerance"]:
            return "visibility"
        if phase.startswith("expert"):
            for field,margin in (("wrist","wrist_drift_px"),("arm","arm_drift_px"),("shape","shape_drift_px")):
                if metrics[field]>anchor[field]+self.cfg[margin]:
                    return "anchor_"+field
        if threshold is not None and not np.isfinite(threshold):
            return "nonfinite"
        return ""

    def objective(self,phase,terms,expert,smooth,coeff):
        reg=.0002*(coeff*coeff).mean()+smooth*(.015 if phase=="temporal" else .003)
        if phase=="arm":
            return (.7*terms["wrist"]+.3*terms["arm"])/20+reg
        if phase=="expert":
            return expert/self.cfg["expert_scale"]+reg
        geo=(.23*terms["wrist"]+.12*terms["arm"]+.19*terms["hand"]+
             .42*terms["shape"]+.04*terms["motion"])/20+reg
        if phase=="temporal":
            return geo+.04*smooth+.09*terms["motion"]/10
        if phase=="joint":
            return geo+self.cfg["joint_weight"]*expert/self.cfg["expert_scale"]
        return geo

    def smooth(self,x):
        a=self.ctrl.angles(x)
        return a.diff(n=2,dim=0).abs().mean()*24**2

    def solve(self,method,path,which,budget,wall_limit=None):
        start=perf_counter()
        restored=method!="art_raw"
        x=torch.nn.Parameter(self.ctrl.zero())
        path.mkdir(parents=True,exist_ok=True)
        trace=[]; proposals=[]; buffers=[]; trial_coefficients=[]
        expert_calls=0
        stages=[(phase,n) for phase,n in self.cfg["hierarchy"]]
        if method=="art_geo":
            stages=[(phase if phase!="expert" else "handshape",n) for phase,n in stages]
        if method in ("art_joint","art_guarded","art_joint_geo","art_guarded_geo"):
            phase="joint_geo" if method.endswith("_geo") else "joint"
            stages=[(phase,sum(n for _,n in stages))]
        def measure_state(x):
            with torch.no_grad():
                xy,rot=self.forward(x)
                m=self.measure_xy(xy)
                n,arm,finger=self.constraints(rot)
            return dict(**m,validity_violations=n,max_arm_adjacent_deg=arm,max_finger_adjacent_deg=finger)
        start_m=measure_state(x)
        with torch.no_grad():
            reference_xy,_=self.forward(x)
            parity=float((self.differentiable.terms(reference_xy)-tensor([start_m[k] for k in ("spatial","temporal","multiscale")])).abs().max())
            sparse_error=float(np.abs(reference_xy.numpy()-self.stable_xy[self.indices]).max()*1280)
        if sparse_error>.02 or parity>1e-5:
            raise ValueError(f"FK/expert parity failure: {sparse_error:g}px, {parity:g}")
        expert_calls+=2
        row=dict(step=0,phase="raw_start",accepted=False,wall_seconds=perf_counter()-start,**self.original_obs,
                 validity_violations=0,max_arm_adjacent_deg=0.,max_finger_adjacent_deg=0.)
        trace.append(row)
        buffers.append(("raw",x.detach().numpy().copy()))
        current=start_m
        if restored:
            trace.append(dict(step=1,phase="stabilize_inherited_tracks",accepted=True,wall_seconds=perf_counter()-start,**current))
            proposals.append(dict(step=1,phase="stabilize_inherited_tracks",accepted=True,rejection="",candidate_score=current["response"],**current))
            buffers.append(("stabilized",x.detach().numpy().copy()))
            trial_coefficients.append(x.detach().numpy().copy())
        anchor=current
        for phase,count in stages:
            if phase=="expert":
                anchor=current
            optimizer=torch.optim.Adam([x],lr=self.cfg["expert_lr"] if phase=="expert" else self.cfg["lr"])
            for _ in range(count):
                if len(proposals)>=budget or (wall_limit is not None and perf_counter()-start>=wall_limit):
                    break
                before=x.detach().clone()
                xy,_=self.forward(x)
                e=self.differentiable.value(xy,which) if phase in ("expert","joint") else 0.
                expert_calls+=int(phase in ("expert","joint"))
                loss=self.objective(phase,self.terms(xy),e,self.smooth(x),x)
                optimizer.zero_grad(); loss.backward()
                if phase=="arm":
                    x.grad[:,18:]=0
                if phase=="handshape":
                    x.grad[:,:18]=0
                torch.nn.utils.clip_grad_norm_([x],1.)
                optimizer.step()
                trial_coefficients.append(x.detach().numpy().copy())
                m=measure_state(x)
                expert_calls+=1
                with torch.no_grad():
                    value=float(self.objective(phase,m,scoring_term(m,which),self.smooth(x),x))
                fail=self.acceptance(m,phase,anchor,value)
                if not fail and value>=float(loss.detach())-1e-7:
                    fail="no_objective_gain"
                if not fail and phase=="expert" and scoring_term(m,which)>=scoring_term(current,which)-1e-6:
                    fail="no_response_gain"
                moved=not fail
                if moved:
                    current=m
                else:
                    with torch.no_grad():
                        x.copy_(before)
                    group=optimizer.param_groups[0]
                    group["lr"]=max(group["lr"]*.83,.003)
                s=len(proposals)+1
                proposals.append(dict(step=s,phase=phase,accepted=moved,rejection=fail,candidate_score=m["response"],**m))
                trace.append(dict(step=s,phase=phase,accepted=moved,wall_seconds=perf_counter()-start,**current))
                buffers.append(("stabilized",x.detach().numpy().copy()))
        runtime=perf_counter()-start
        steps=list(dict.fromkeys([0, min(7,len(trace)-1),min(20,len(trace)-1),len(trace)-1]))
        np.savez_compressed(path/"control_trajectory.npz",coefs=np.stack([b[1] for b in buffers]),kind=[b[0] for b in buffers],
                            names=self.ctrl.names,knots=self.ctrl.knots,
                            trial_coefs=np.asarray(trial_coefficients),
                            mobility=np.empty(0) if self.ctrl.mobility is None else self.ctrl.mobility.numpy())
        checkpoints=[]; arrays={}
        for step in steps:
            kind,coeff=buffers[step]
            fit=self.raw if kind=="raw" else self.ctrl.export(tensor(coeff))
            pts,xy,_=calculate(fit,self.a.asset,self.camera,self.temp)
            valid=geometry_holdout(xy[1::2],self.mp[1::2],PIX)
            motion=track_stats(pts,fit,self.a.asset)
            checkpoints.append(dict(step=step,**valid,**motion))
            label=f"step_{step:02}"
            np.savez_compressed(path/f"{label}_fit.npz",**fit)
            arrays[label+"_xy"]=xy; arrays[label+"_points"]=pts
            if step==steps[-1]:
                np.savez_compressed(path/"final_fit.npz",**fit)
        np.savez_compressed(path/"checkpoint_tracks.npz",**arrays)
        if proposals:
            write_csv(path/"proposals.csv",proposals)
        write_csv(path/"trace.csv",trace); write_csv(path/"checkpoints.csv",checkpoints)
        result=dict(source=self.source,level=self.level,method=method,feedback=which,proposal_count=len(proposals),
                    optim_time_s=runtime,expert_passes_accounted=expert_calls,
                    accepted=sum(r["accepted"] for r in proposals),
                    accepted_expert=sum(r["accepted"] for r in proposals if r["phase"]=="expert"),
                    feedback_start=trace[0]["response"], feedback_final=trace[-1]["response"],
                    spatial_final=trace[-1]["spatial"],temporal_final=trace[-1]["temporal"],
                    multiscale_final=trace[-1]["multiscale"],
                    knots=self.ctrl.knots,parameters=x.numel(),last_step=steps[-1],
                    maximum_fk_pixel_parity=sparse_error,maximum_expert_component_parity=parity,
                    geometry_start=self.original_held["held_geometry"], **{k:v for k,v in checkpoints[-1].items() if k!="step"},
                    max_arm_adjacent_deg=trace[-1]["max_arm_adjacent_deg"],max_finger_adjacent_deg=trace[-1]["max_finger_adjacent_deg"],
                    validity_violations=trace[-1]["validity_violations"])
        write(path/"result.json",result)
        print(self.source,self.level,method,which,round(result["feedback_start"],5),
              round(result["feedback_final"],5),result["accepted_expert"],round(result["held_geometry"],2),
              round(result["optim_time_s"],1),flush=True)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("repo","checkpoint","asset","output"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--sources",nargs="+")
    p.add_argument("--config",default=Path("configs/stage9.json"),type=Path)
    p.add_argument("--levels",nargs="+",default=["B"])
    p.add_argument("--methods",nargs="+",default=list(MAIN),choices=(*MAIN,"art_raw","art_guarded","art_joint_geo","art_guarded_geo"))
    p.add_argument("--feedback",nargs="+",default=["factor"],choices=["factor","spatial","temporal","multiscale"])
    p.add_argument("--threads",default=2,type=int)
    p.add_argument("--wall-limit",type=float)
    p.add_argument("--label",default="")
    a=p.parse_args()
    cfg=read(a.config)
    torch.manual_seed(cfg["seed"])
    graphs=HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",a.threads))
    with TemporaryDirectory() as td:
        temp=Path(td)/"candidate.npz"
        for source in a.sources or cfg["sources"]:
            for level in a.levels:
                runner=Loop(a,cfg,source,level,graphs,temp)
                for method in a.methods:
                    use=runner if method!="art_raw" and not method.startswith("art_guarded") else Loop(
                        a,cfg,source,level,graphs,temp,raw=method=="art_raw",guarded=method.startswith("art_guarded"))
                    for which in a.feedback:
                        d=a.output/source/level/(method+"_"+which+a.label)
                        if not (d/"result.json").exists():
                            use.solve(method,d,which,cfg["budget"],wall_limit=a.wall_limit)


if __name__=="__main__":
    main()
