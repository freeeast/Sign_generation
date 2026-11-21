"""Independent held-out pose, side-view, and off-objective Uni-Sign evaluation."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from scripts.stage4_geometry import evaluate, reference_hands
from scripts.stage7_optimize import read,write
from src.evaluate_feedback import write_csv
from src.pose import BODY
from src.stage4_features import FrozenFeedback
from src.stage7_feedback import (HANDS, TIPS, WRISTS, HandGraphAdapter, HandGraphs, confidence_diagnostics, cosine_error,
                                 expert_input, in_frame, valid_pose)
from src.unisign import UniSign,cosine

SELECTED=("geometry","stage6","bilateral","robust_geometry","factor","reliable","safe")


def mean_norm_scale(xy,score):
    x=xy[:,BODY,:][score[:,BODY]>.3]
    return float(np.ptp(x,axis=0).max()) if len(x)>3 else None


def wrist_relative(xy,hand):
    ids=HANDS[hand]
    return xy[:,ids]-xy[:,ids[:1]]


def toolkit(output,repo,rows):
    sys.path.insert(0,str(repo.resolve()))
    from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
    metric=DTWDTAIImplementationDistanceMeasure(use_fast=False)
    out=[]
    for row in rows:
        clip=row["clip_id"]
        with np.load(output/"data/paired"/f"{clip}.npz") as d:
            mp=d["mp"][1::2]
        for method in SELECTED:
            path=output/"direct"/clip/f"{method}_tracks.npz"
            if not path.exists():
                continue
            with np.load(path) as d:
                xy=d["xy"][1::2]
            for h,i in enumerate(WRISTS):
                valid=in_frame(mp)[:,HANDS[h]].all(axis=1)
                if valid.sum()<5:
                    continue
                a=wrist_relative(xy,h)[valid,None,1:]*[1280,720]
                b=wrist_relative(mp,h)[valid,None,1:]*[1280,720]
                distance=metric(np.ma.asarray(a),np.ma.asarray(b))
                out.append(dict(clip=clip,method=method,side=("left","right")[h],
                                frames=int(valid.sum()),raw_dtw=float(distance),
                                sqrt_length_scaled=float(distance/np.sqrt(valid.sum()))))
    if out:
        write_csv(output/"pose_evaluation_dtw.csv",out)


def side_camera(points,xy,valid):
    ids=np.array([0,3,4,5,6,7,8])
    even=np.arange(0,len(xy),2)
    use=valid[even][:,ids]
    x=points[even][:,ids][use]
    y=xy[even][:,ids][use]*[1280,720]
    if len(x)<20:
        return None
    best=None
    for theta in np.r_[np.arange(-115,-44,2),np.arange(45,116,2)]:
        c,s=np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))
        u=np.column_stack([x[:,0]*c+x[:,2]*s,x[:,1]])
        t=u-u.mean(axis=0)
        v=y-y.mean(axis=0)
        scale=float(np.sum(t*v)/np.square(t).sum())
        if scale<=0:
            continue
        shift=(y-u*scale).mean(axis=0)
        err=np.mean(np.linalg.norm(u*scale+shift-y,axis=1))
        if best is None or err<best[0]:
            best=(float(err),int(theta),scale,shift.tolist())
    return best


def project_side(points,camera):
    _,theta,scale,shift=camera
    c,s=np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))
    u=np.stack([points[...,0]*c+points[...,2]*s,points[...,1]],axis=-1)
    return (u*scale+shift)/[1280,720]


def multiview(output,rows,videos,model):
    root=output/"multiview"
    out=[]
    status=[]
    for row in rows:
        clip=row["clip_id"]
        path=root/"pose"/f"{clip}.npz"
        if not path.is_file():
            status.append(dict(clip=clip,status="no_side_detection"))
            continue
        with np.load(path) as f:
            xy,score,fps=f["keypoints"],f["scores"],float(f["fps"])
        if fps!=24 or len(xy)!=row["fit_frames"]:
            status.append(dict(clip=clip,status="length_or_rate_mismatch",frames_side=len(xy),
                               frames_front=row["fit_frames"],fps_side=fps))
            continue
        with np.load(output/"data/paired"/f"{clip}.npz") as f:
            points=f["original3d"]
        valid=valid_pose(xy,score,.3)
        obs=np.where(valid[...,None],xy,np.nan)
        camera=side_camera(points,xy,valid)
        if not camera:
            status.append(dict(clip=clip,status="cannot_fit_side_camera"))
            continue
        p=root/"pose"/f"{clip}_mp.npz"
        if p.is_file():
            with np.load(p) as archive:
                mp=archive["mp"]
        else:
            mp,_=reference_hands(videos/f"{clip}.mp4",model,xy,score,(1280,720))
            mp=np.where(in_frame(mp)[...,None],mp,np.nan)
            np.savez_compressed(p,mp=mp)
        coverage=int(np.isfinite(mp[1::2,WRISTS,0]).sum())
        status.append(dict(clip=clip,status="proxy_side_assessment",calibration_error_px=camera[0],
                           camera_yaw_deg=camera[1],camera_scale=camera[2],camera_offset_px=camera[3],
                           associated_odd_hands=coverage))
        for method in ("original",*SELECTED):
            tracks=output/"direct"/clip/f"{method}_tracks.npz"
            if method=="original":
                trajectory=points
            elif tracks.is_file():
                with np.load(tracks) as f:
                    trajectory=f["points"]
            else:
                continue
            p2=project_side(trajectory,camera)
            rt=evaluate(p2,obs,(1280,720),slice(1,None,2))
            m=evaluate(p2,mp,(1280,720),slice(1,None,2))
            metrics=("wrists","tips","handshape")
            out.append(dict(clip=clip,method=method,rt_anchors=rt["anchors"],
                            **{f"side_rtmw_{k}":rt[k] for k in metrics},
                            **{f"side_mp_{k}":m[k] for k in metrics},
                            side_mp_composite=float(np.mean([m[k] for k in metrics])) if coverage>=5 else None))
    write(root/"status.json",status)
    if out:
        write_csv(root/"heldout.csv",out)


def semantic(output,rows,a,repo,checkpoint,mt5):
    sign=UniSign(repo,checkpoint,mt5,"cpu",a.threads)
    graph=HandGraphs(sign)
    evaluation={}
    captions=[x["reference"] for x in rows]
    methods=("original",*SELECTED)
    for r in rows:
        clip=r["clip_id"]
        poses=output/"data/real_pose"/f"{clip}.npz"
        with np.load(poses) as f:
            xy,score=f["keypoints"][1::2],f["scores"][1::2]
        evaluation[clip]={"real":(xy,score)}
        with np.load(output/"data/paired"/f"{clip}.npz") as d:
            original=d["original2d"][1::2]
        evaluation[clip]["original"]=(original,np.ones(original.shape[:2],np.float32))
        for method in SELECTED:
            tracks=output/"direct"/clip/f"{method}_tracks.npz"
            if tracks.is_file():
                with np.load(tracks) as f:
                    x=f["xy"][1::2]
                evaluation[clip][method]=(x,np.ones(x.shape[:2],np.float32))
    first=rows[0]["clip_id"]
    p=evaluation[first]["real"]
    official=FrozenFeedback(sign).extract(*p)
    bilateral=graph.extract(*p)
    fast=HandGraphs(HandGraphAdapter(repo,checkpoint,"cpu",a.threads)).extract(*p)
    captures={k:[] for k in ("hand_spatial","hand_temporal")}
    handles=[]
    for k,module in (("hand_spatial",sign.model.gcn_modules["right"]),
                     ("hand_temporal",sign.model.fusion_gcn_modules["right"])):
        def capture(_module,_arg,out,k=k):
            captures[k].append(out.detach().clone()[0].permute(1,2,0).flatten(1).cpu().numpy())
        handles.append(module.register_forward_hook(capture))
    try:
        with sign.torch.inference_mode():
            sign._encode(*p,256)
    finally:
        for handle in handles:
            handle.remove()
    proof=dict(shared_spatial_module=sign.model.gcn_modules["left"] is sign.model.gcn_modules["right"],
               shared_temporal_module=sign.model.fusion_gcn_modules["left"] is sign.model.fusion_gcn_modules["right"])
    for k,feat,fast_feat in zip(("hand_spatial","hand_temporal"),bilateral,fast):
        feat=feat.reshape(*feat.shape[:2],-1)
        raw=official[k]
        proof[k]=dict(cached_l_r_max_abs=float(np.max(np.abs(raw[:,0]-raw[:,1]))),
                      explicit_l_r_max_abs=float(np.max(np.abs(feat[:,0]-feat[:,1]))),
                      old_vs_explicit_right_max_abs=float(np.max(np.abs(raw[:,0]-feat[:,1]))),
                      original_call_order_vs_explicit_max_abs=float(np.max(np.abs(np.stack(captures[k],axis=1)-feat))),
                      graph_only_loader_parity_max_abs=float(np.max(np.abs(fast_feat.reshape(feat.shape)-feat))))
    write(output/"hook_parity.json",proof)
    embeds={}
    score_rows=[]
    for j,(clip,entries) in enumerate(evaluation.items()):
        for method,(x,c) in entries.items():
            pose=expert_input(x,c,valid_pose(x,c,.3))
            with sign.torch.inference_mode():
                enc,attention,global_feat,_=sign._encode(*pose,256)
                nll=[]
                for start in range(0,len(captions),2):
                    nll.extend(sign._score_targets(enc,attention,captions[start:start+2]))
            embeds[clip,method]=global_feat
            order=np.argsort(nll)
            rank=int(np.flatnonzero(order==j)[0])+1
            score_rows.append(dict(clip=clip,method=method,caption_rank=rank,correct_nll=nll[j],
                                   gap_to_best_distractor=min(v for k,v in enumerate(nll) if j!=k)-nll[j],
                                   nlls=json.dumps(nll)))
        print(clip,"held-out Uni-Sign retrieval evaluated",flush=True)
    clips=list(evaluation)
    for r in score_rows:
        clip=r["clip"]
        data=embeds[clip,r["method"]]
        distances=[1-cosine(data,embeds[c,"real"]) for c in clips]
        ranking=np.argsort(distances)
        r["embedding_rank"]=int(np.flatnonzero(ranking==clips.index(clip))[0])+1
        r["real_embedding_distance"]=distances[clips.index(clip)]
        r["embedding_distances"]=json.dumps(distances)
    if score_rows:
        write_csv(output/"heldout_semantic.csv",score_rows)


def domains(output,rows):
    out=[]
    for row in rows:
        clip=row["clip_id"]
        origin=output/"data/real_pose"/f"{clip}.npz"
        with np.load(origin) as f:
            x,c=f["keypoints"],f["scores"]
        source=output/"data/paired"/f"{clip}.npz"
        with np.load(source) as p:
            v=p["original2d"]
        candidates=[("real",x,c),("original_projection",v,np.ones(v.shape[:2]))]
        d=output/"rgb"/clip/"full"
        if (d/"rgb_expert_input.npz").is_file():
            with np.load(d/"rgb_expert_input.npz") as p:
                f=p["frames"]
                candidates.extend([("real_at_rgb_samples",x[f],c[f]),("rgb_endpoint",p["xy"],p["scores"])])
        for mode,xy,score in candidates:
            valid=valid_pose(xy,score,.3)
            hand=np.linalg.norm((xy[:,[100,121]]-xy[:,WRISTS])*[1280,720],axis=-1)
            include=valid[:,[100,121]]&valid[:,WRISTS]
            covered=score[:,91:][valid[:,91:]]
            out.append(dict(clip=clip,mode=mode,frames=len(xy),
                **confidence_diagnostics(xy,score,.3),inside_median_confidence=float(np.median(covered)) if covered.size else None,
                raw_model_bbox_scale=mean_norm_scale(xy,score),
                visibility_model_bbox_scale=mean_norm_scale(xy,np.where(valid,score,0)),
                median_mcp_to_wrist_px=float(np.median(hand[include])) if include.any() else None))
    write_csv(output/"domain_coverage.csv",out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("output","repo","checkpoint","mt5","toolkit","side-videos","mp-model"):
        p.add_argument("--"+name,type=Path,required=name in ("output",))
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--mode",choices=("semantic","toolkit","side","domains"),required=True)
    a=p.parse_args()
    rows=[r for r in read(a.output/"data/manifest.json") if (a.output/"direct"/r["clip_id"]/"geometry.json").exists()]
    if a.mode=="semantic":
        semantic(a.output,rows,a,a.repo,a.checkpoint,a.mt5)
    elif a.mode=="toolkit":
        toolkit(a.output,a.toolkit,rows)
    elif a.mode=="side":
        multiview(a.output,rows,a.side_videos,a.mp_model)
    else:
        domains(a.output,rows)


if __name__=="__main__":
    main()
