"""Run resumable visibility, factorization and geometry-constrained closed loops."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from scripts.stage5_closed_loop import assay, geometrical_average
from scripts.stage6_rgb import pose_observations, save_avatar
from src.evaluate_feedback import write_csv
from src.extract_real_pose import load_detector
from src.pose import FINGERS
from src.smplx_to_pose import forward
from src.stage5_motion import MotionSpace
from src.stage6_render import AvatarRenderer
from src.stage7_feedback import (HandGraphAdapter, HandGraphs, cosine_error, expert_input,
    feature_scores, hand_quality, in_frame, landmark_losses, longest_visible_region,
    reliability_weights, valid_pose)
from src.virtual_pose import apply_affine, load_fit, pinhole

DIRECT=("geometry","stage6","bilateral","inframe","confidence","robust_geometry","factor","hierarchical","reliable","safe")
PREV=Path("experiments/Stage6")
NAMES=("rtmw_even_combined","legacy","robust_geometry","factor","adapted","mask","confidence","bilateral")
GEOM_KEYS=("wrist","tips","shape","velocity","robust_geometry")


def read(path):
    return json.loads(path.read_text())


def write(path,obj):
    path.with_suffix(".tmp").write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    path.with_suffix(".tmp").replace(path)


def subset(stride,n,max_frames=256,start=0):
    indices=np.arange(start,n,stride)
    if len(indices)>max_frames:
        indices=indices[np.linspace(0,len(indices)-1,max_frames).astype(int)]
    return indices


class Sequence:
    def __init__(self,a,cfg,prior,row,source,adapter,temp, *, indices=None, evaluation=None,
                 render=None,detector=None,control="six",natural=False,corruption=0,dropout=False):
        self.a,self.cfg,self.prior,self.row=a,cfg,prior.copy(),row
        self.temp,self.render,self.detector=temp,render,detector
        self.graphs=HandGraphs(adapter)
        self.cache,self.pose_cache={},{}
        clip=row["clip_id"]
        d=source/"data"
        fit_dir=d/("inputs" if source==a.output else "source_fits")
        self.fit=load_fit(fit_dir/f"{clip}.npz")
        with np.load(d/"real_pose"/f"{clip}.npz") as data:
            self.real,self.scores=data["keypoints"],data["scores"]
            self.size=(int(data["width"]),int(data["height"]))
        with np.load(d/"paired"/f"{clip}.npz") as data:
            self.original,self.camera,self.mp=data["original3d"],data["camera"],data["mp"]
        self.mp=np.where(in_frame(self.mp)[...,None],self.mp,np.nan)
        n=len(self.real)
        self.indices=subset(2,n) if indices is None else indices
        self.evaluation=np.arange(1,n,2) if evaluation is None else evaluation
        if np.intersect1d(self.indices,self.evaluation).size:
            raise ValueError("Geometry evaluator overlaps search frames")
        if dropout:
            # Known synthetic corruption on optimization frames only. Ground reference is untouched.
            stripe=self.indices[(self.indices>=n//5)&(self.indices<n*3//5)]
            self.real[stripe,91:112,1]+=1.1
        self.real_obs=np.where(self.scores[...,None]>cfg["joint_confidence"],self.real,np.nan)
        self.ref_xy,self.ref_score=self.real[self.indices],self.scores[self.indices]
        self.valid=valid_pose(self.ref_xy,self.ref_score,cfg["joint_confidence"])
        self.strong=valid_pose(self.ref_xy,self.ref_score,cfg["strong_confidence"])
        self.good=hand_quality(self.valid,cfg["min_hand_points"])
        self.strong_hands=hand_quality(self.strong,cfg["min_hand_points"])
        self.ones=np.ones(self.good.shape)
        self.weights=self.valid*np.maximum(np.clip(self.ref_score,0,1),.25)
        self.ref_raw=self.graphs.extract(self.ref_xy,self.ref_score)
        self.ref_mask=self.graphs.extract(*expert_input(self.ref_xy,self.ref_score,self.valid))
        self.ref_strong=self.graphs.extract(*expert_input(self.ref_xy,self.ref_score,self.strong))
        self.reliability=reliability_weights(self.ref_xy,self.ref_score,self.valid,self.mp[self.indices],
                                             self.ref_raw,self.ref_mask,self.size)
        self.reliability_mass=float(self.reliability[self.good].mean()) if self.good.any() else 0.
        if natural:
            self.prior.update(corruption_coeff_deg=[0]*6,off_basis_deg=0)
        elif corruption:
            coefficients=np.asarray(self.prior["corruption_coeff_deg"])
            self.prior["corruption_coeff_deg"]=(np.roll(coefficients,corruption)*
                np.where((np.arange(6)+corruption)%3==0,-1,1)).tolist()
        self.space=MotionSpace(self.fit,a.asset,self.prior)
        self.control=control
        if control=="nine":
            from src.stage7_motion import NestedSpace
            self.space=NestedSpace(self.fit,a.asset,self.prior)
        self.zero=np.zeros(6 if control=="six" else 9)
        self.support_count=int(self.good.sum())

    def reconstruct(self,x,rendered=False):
        fit,kin=self.space.candidate(x)
        kin["constraint_violations"]+=int(np.abs(x).max()>self.prior["max_coeff_deg"])
        if kin["constraint_violations"]:
            return fit,kin,None,None,None
        np.savez(self.temp,**fit)
        if rendered:
            points,_,verts,_=forward(self.temp,self.a.asset,"neutral","model",64,"cpu",False,return_mesh=True)
        else:
            points,_=forward(self.temp,self.a.asset,"neutral","model",64,"cpu",False)
        xy=apply_affine(pinhole(points,fit["cam_trans"]),self.camera)
        if rendered:
            frames=self.render.frames(verts[self.indices],fit["cam_trans"][self.indices],self.camera)
            v,vscore,counts=pose_observations(frames,self.detector)
            return fit,kin,xy,points,(v,vscore,float((counts>0).mean()))
        return fit,kin,xy,points,None

    def at(self,x,rendered=False):
        key=tuple(x)
        common=self.cache.get((False,key))
        if (rendered,key) in self.cache:
            return self.cache[rendered,key]
        if rendered and common is None:
            common=self.at(x)
        if common and common["constraint_violations"]:
            return common
        start=perf_counter()
        _,kin,xy,points,pose=self.reconstruct(x,rendered)
        if kin["constraint_violations"]:
            return kin
        if not rendered:
            g=assay(xy,self.real_obs,self.size,self.indices,"rtmw_even")
            g["rtmw_even_combined"]=geometrical_average(g,"rtmw_even")
            rr=landmark_losses(xy[self.indices],self.ref_xy,self.weights,self.indices,self.size)
            rr["robust_geometry"]=sum(rr[k]*v for k,v in self.cfg["loss_weights"].items())
            g.update(rr)
            g.update(assay(xy,self.mp,self.size,self.evaluation,"mp_held"))
            g["mp_held_combined"]=float(np.mean([v for k,v in g.items() if k.startswith("mp_held") and v is not None])) if any(
                v is not None for k,v in g.items() if k.startswith("mp_held")) else None
            ev=self.mp[self.evaluation]
            g["evaluation_hands"]=int(np.isfinite(ev[:,[91,112],0]).sum())
            g["hand_accel_p95_mm"]=float(np.percentile(np.linalg.norm(np.diff(points[:,FINGERS],n=2,axis=0),axis=-1),95)*1000)
            g.update(**kin)
            v=xy[self.indices]
            vscore=np.ones(v.shape[:2],np.float32)
            person=1.
        else:
            excluded=set(("legacy","bilateral","mask","confidence","factor","adapted","support_retained","virtual_both_hands","virtual_median_confidence","virtual_coverage","candidate_runtime_s","person"))
            g={k:r for k,r in common.items() if k not in excluded}
            v,vscore,person=pose
            self.pose_cache[True,key]=(v,vscore)
        self.pose_cache.setdefault((False,key),(xy[self.indices],np.ones(xy[self.indices].shape[:2],np.float32)))
        raw=self.graphs.extract(v,vscore)
        unmasked=cosine_error(raw[1].reshape(len(v),2,-1),self.ref_raw[1].reshape(len(v),2,-1))
        g["legacy"]=float(unmasked[:,1].mean())
        g["bilateral"]=float(unmasked.mean())
        valid=valid_pose(v,vscore,self.cfg["joint_confidence"])
        present=hand_quality(valid,self.cfg["min_hand_points"])
        same=self.valid&valid
        # Candidate may not remove any real support from the denominator.
        masked=self.graphs.extract(*expert_input(v,vscore,same))
        masked_ref=self.ref_mask
        strong=valid_pose(v,vscore,self.cfg["strong_confidence"])
        confidence=self.graphs.extract(*expert_input(v,vscore,strong&self.strong))
        penalty=self.cfg["missing_feature_penalty"]
        standard=feature_scores(masked,masked_ref,self.good,present,self.ones,penalty)
        weighted=feature_scores(masked,masked_ref,self.good,present,self.reliability,penalty)
        g["mask"]=standard[1]
        g["confidence"]=feature_scores(confidence,self.ref_strong,self.strong_hands,
                                        hand_quality(strong,self.cfg["min_hand_points"]),self.ones,penalty)[1]
        for key,v_scores in (("factor",standard),("adapted",weighted)):
            g[key]=sum(self.cfg["expert_weights"][k]*l for k,l in zip(("spatial","temporal","multiscale"),v_scores[:3]))
        g.update(support_retained=standard[3],virtual_both_hands=float(present.all(axis=1).mean()),
            virtual_median_confidence=float(np.median(vscore[:,91:])),virtual_coverage=float(valid[:,91:].mean()),
            candidate_runtime_s=perf_counter()-start,person=person)
        self.cache[rendered,key]=g
        return g

    def solve(self,method,scales, *, rendered=False,rgb=False):
        raw_method=method
        if method=="stage6":
            signal="legacy"
        elif method in {"bilateral","inframe","confidence"}:
            signal={"bilateral":"bilateral","inframe":"mask","confidence":"confidence"}[method]
        else:
            signal={"factor":"factor","reliable":"adapted","hierarchical":"factor","safe":"adapted"}.get(method)
        robust=method in {"robust_geometry","factor","hierarchical","reliable","safe"}
        geo_key="robust_geometry" if robust else "rtmw_even_combined"
        coeff=self.zero.copy()
        base=self.at(coeff,rendered)
        current=base
        anchor=None
        first=[4,5,2,3,0,1]
        second=first if self.control=="six" else [8,7,6,4,2,0]
        coarse_radius,fine_radius=self.cfg["search_radii_deg"]
        schedule=[(coarse_radius,k) for k in first]+[(fine_radius,k) for k in second]
        if rgb:
            schedule=[(r,k) for r in (coarse_radius,fine_radius) for k in (4,2,0)]
        steps=[]
        probes=[]
        accepted=0
        for step,(radius,axis) in enumerate(schedule,1):
            fine=radius<=self.cfg["step_trust_deg"]
            if fine and anchor is None:
                anchor=current
            def objective(m,x):
                g=(m[geo_key]-base[geo_key])/scales[geo_key]
                v=m["correction_accel_deg_per_frame2"]/self.prior["max_correction_accel_deg_per_frame2"]
                regular=self.prior["coefficient_penalty"]*np.sum((x/self.prior["max_coeff_deg"])**2)/6 + self.prior["smoothness_penalty"]*v*v
                if signal and not (method in ("hierarchical","safe") and not fine):
                    expert_delta=(m[signal]-base[signal])/scales[signal]
                    if method=="hierarchical" and axis not in (0,1,6):
                        return g+regular
                    if method=="safe" and self.support_count>=6:
                        return expert_delta+0.05*g+regular
                    weight=self.cfg["expert_weight"]*(self.reliability_mass if method in ("reliable","safe") else 1.)
                    return g+weight*expert_delta+regular
                return g+regular
            old=objective(current,coeff)
            options=[]
            for sign in (-1,1):
                x=coeff.copy()
                x[axis]+=sign*radius
                m=self.at(x,rendered)
                rejection=""
                if m["constraint_violations"]:
                    rejection="kinematic_bound"
                elif rendered and m["person"]<.90:
                    rejection="person_dropout"
                elif method in ("inframe","confidence","factor","reliable","safe","hierarchical") and m["support_retained"]<base["support_retained"]-.05:
                    rejection="lost_visible_support"
                elif method=="hierarchical" and fine and axis in (0,1,6) and m["wrist"]>anchor["wrist"]+.5:
                    rejection="wrist_guard"
                elif method=="safe" and fine and any(m[k]>anchor[k]+self.cfg["geometry_slack_px"][k] for k in GEOM_KEYS):
                    rejection="geometry_guard"
                value=float(objective(m,x)) if not rejection else None
                record=dict(step=step,axis=axis,sign=sign,radius=radius,objective=value,rejection=rejection,
                            selected=False,coefficients=x.tolist(),**{k:m.get(k) for k in base})
                probes.append(record)
                if value is not None:
                    options.append((value,x,m,record))
            if options:
                value,x,m,record=min(options,key=lambda r:r[0])
                if value < old-1e-7:
                    coeff,current=x,m
                    record["selected"]=True
                    accepted+=1
            steps.append(dict(step=step,axis=axis,accepted=any(x["selected"] for x in probes[-2:]),
                              objective=float(objective(current,coeff)),coefficients=coeff.tolist(),**current))
        final=dict(method=raw_method,rendered=rendered,accepted=accepted,budget=len(probes),**current)
        return coeff,final,[dict(step=0,axis=-1,accepted=False,objective=0.,coefficients=self.zero.tolist(),**base)]+steps,probes


def run_one(seq,scales,output,arms, *, rgb=False,subset_name=None):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,method,rendered in arms:
        marker=output/f"{name}.json"
        if marker.exists():
            cached=read(marker)
            rows.append(cached)
            continue
        t=perf_counter()
        coef,final,path,probes=seq.solve(method,scales,rendered=rendered,rgb=rgb)
        fit,kin,xy,points,_=seq.reconstruct(coef)
        np.savez_compressed(output/f"{name}_fit.npz",**fit)
        np.savez_compressed(output/f"{name}_tracks.npz",xy=xy,points=points)
        np.savez_compressed(output/f"{name}_expert_input.npz",xy=seq.pose_cache[rendered,tuple(coef)][0],
                            scores=seq.pose_cache[rendered,tuple(coef)][1],frames=seq.indices)
        write_csv(output/f"{name}_trace.csv",path)
        write_csv(output/f"{name}_proposals.csv",probes)
        if rgb:
            save_avatar(seq.a,seq.render,output/f"{name}_fit.npz",output/f"{name}.mp4",fit["cam_trans"],seq.camera)
        final.update(method=name,runtime_s=perf_counter()-t,clip=seq.row["clip_id"],subset=subset_name or "full",
                     source_support=seq.support_count,initial_mp=path[0]["mp_held_combined"],initial_expert=path[0]["legacy"])
        write(marker,final)
        rows.append(final)
        print(seq.row["clip_id"],subset_name,name,accepted_text(final),flush=True)
    write_csv(output/"results.csv",rows)
    if rgb and not (output/"initial.mp4").exists():
        fit,_=seq.space.candidate(seq.zero)
        np.savez_compressed(output/"initial_fit.npz",**fit)
        save_avatar(seq.a,seq.render,output/"initial_fit.npz",output/"initial.mp4",fit["cam_trans"],seq.camera)


def accepted_text(final):
    return f'{final["accepted"]} accepted, held {final["mp_held_combined"]}, {final["runtime_s"]:.1f}s'


def calibrate(args,cfg,prior,adapter,temp):
    dest=args.output/"calibration"
    dest.mkdir(parents=True,exist_ok=True)
    marker=dest/"scales.json"
    if marker.exists() and (dest/"selection.json").exists():
        return read(marker),read(dest/"selection.json")
    rows=[r for r in read(PREV/"data/manifest.json") if r["clip_id"] in cfg["calibration"]]
    seqs=[Sequence(args,cfg,prior,r,PREV,adapter,temp) for r in rows]
    probes={k:[] for k in NAMES}
    for s in seqs:
        start=s.at(s.zero)
        for x in np.eye(6):
            for sign in (-1,1):
                r=s.at(sign*6*x)
                if not r["constraint_violations"]:
                    for k in NAMES:
                        probes[k].append(abs(r[k]-start[k]))
    scales={k:float(max(np.median(v),1e-4 if k not in ("rtmw_even_combined","robust_geometry") else .05)) for k,v in probes.items()}
    old=read(Path("experiments/Stage5/calibration_scales.json"))
    scales["legacy"]=old["hand_temporal"]
    scales["rtmw_even_combined"]=old["rtmw_even_combined"]
    write(marker,scales)
    candidates=[("factor","factor",False),("reliable","reliable",False),("safe","safe",False)]
    for s in seqs:
        run_one(s,scales,dest/s.row["clip_id"],[("robust_geometry","robust_geometry",False),*candidates])
    means={}
    for method,_,_ in candidates:
        means[method]=float(np.mean([read(dest/s.row["clip_id"]/f"{method}.json")["mp_held_combined"]-
                          read(dest/s.row["clip_id"]/"robust_geometry.json")["mp_held_combined"] for s in seqs]))
    chosen=min(means,key=means.get)
    selection=dict(rgb_robust_method=chosen,mean_heldout_delta_old_sources_px=means,
                   calibration_recordings=[r["clip_id"] for r in rows])
    write(dest/"selection.json",selection)
    return scales,selection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("repo","checkpoint","mt5","asset","output","detector","pose-model"):
        p.add_argument("--"+key,type=Path,required=key not in ("detector","pose-model","mt5"))
    p.add_argument("--threads",type=int,default=5)
    p.add_argument("--device",default="cpu")
    p.add_argument("--mode",choices=("direct","rgb","controller","calibrate","diagnose","rgb_diagnostic"),required=True)
    p.add_argument("--clips",nargs="+")
    a=p.parse_args()
    cfg=read(Path("configs/stage7.json"))
    prior=read(Path("configs/stage5.json"))
    adapter=HandGraphAdapter(a.repo,a.checkpoint,a.device,a.threads)
    with tempfile.TemporaryDirectory() as d:
        tmp=Path(d)/"candidate.npz"
        scales,selection=calibrate(a,cfg,prior,adapter,tmp)
        if a.mode=="calibrate":
            return
        if a.mode=="diagnose":
            row=next(r for r in read(PREV/"data/manifest.json") if r["clip_id"]==cfg["calibration"][0])
            s=Sequence(a,cfg,prior,row,PREV,adapter,tmp)
            arms=("geometry","stage6","bilateral","inframe","confidence","robust_geometry")
            run_one(s,scales,a.output/"visibility_calibration"/row["clip_id"],[(x,x,False) for x in arms])
            synthetic=read(a.output/"data/manifest.json")[0]
            s=Sequence(a,cfg,prior,synthetic,a.output,adapter,tmp,dropout=True)
            run_one(s,scales,a.output/"visibility_injected"/synthetic["clip_id"],[(x,x,False) for x in arms])
            return
        rows=read(a.output/"data/manifest.json")
        source=a.output
        if a.mode in ("rgb","rgb_diagnostic"):
            if a.mode=="rgb_diagnostic":
                source=PREV
                rows=[r for r in read(source/"data/manifest.json") if r["clip_id"]==cfg["calibration"][0]]
            else:
                rows=rows[:cfg["rgb_sources"]]
            detector=load_detector(a.repo,a.detector,a.pose_model,2)
            with np.load(a.asset) as asset:
                renderer=AvatarRenderer((1280,720),asset["f"],asset["v_template"])
        for i,row in enumerate(rows):
            clip=row["clip_id"]
            if a.clips and clip not in a.clips or not (source/"data/paired"/f"{clip}.npz").exists():
                continue
            if a.mode=="direct":
                s=Sequence(a,cfg,prior,row,a.output,adapter,tmp,corruption=i)
                run_one(s,scales,a.output/"direct"/clip,[(x,x,False) for x in DIRECT])
            elif a.mode=="controller":
                for natural in (False,True):
                    for control in ("six","nine"):
                        s=Sequence(a,cfg,prior,row,a.output,adapter,tmp,corruption=i,control=control,natural=natural)
                        run_one(s,scales,a.output/"controller"/clip/("natural" if natural else "synthetic")/control,
                                [("geometry","robust_geometry",False)])
            else:
                with np.load(source/"data/real_pose"/f"{clip}.npz") as data:
                    interval=longest_visible_region(data["keypoints"],data["scores"],cfg)
                windows=[("full",(0,row["fit_frames"]))]
                if interval is not None:
                    windows.append(("visible",interval))
                arms=[("geometry","geometry",False),("direct","stage6",False),("rgb","stage6",True),
                      ("robust_rgb",selection["rgb_robust_method"],True),("robust_geometry","robust_geometry",False)]
                if a.mode=="rgb_diagnostic":
                    arms=[("robust_geometry","robust_geometry",False),
                          ("rgb","stage6",True),("robust_rgb",selection["rgb_robust_method"],True)]
                for name,(start,stop) in windows:
                    indices=subset(cfg["rgb_stride"],stop,cfg["rgb_max_frames"],start)
                    odd=np.arange(1,stop,2)
                    evaluation=np.setdiff1d(odd[odd>=start],indices)
                    dest=a.output/("rgb" if a.mode=="rgb" else "rgb_diagnostic")/clip/name
                    dest.mkdir(parents=True,exist_ok=True)
                    write(dest/"frames.json",dict(bounds=[start,stop],optimization=indices.tolist(),held_out=evaluation.tolist()))
                    s=Sequence(a,cfg,prior,row,source,adapter,tmp,indices=indices,evaluation=evaluation,
                               render=renderer,detector=detector,corruption=i)
                    run_one(s,scales,dest,arms,rgb=True,subset_name=name)


if __name__=="__main__":
    main()
