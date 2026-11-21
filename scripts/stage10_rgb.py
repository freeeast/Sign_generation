"""Re-evaluate all fixed-world baselines with identical sampled RTMW and Uni-Sign."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.stage6_rgb import pose_observations
from scripts.stage7_optimize import read,write
from scripts.stage8_imitate import Response
from scripts.stage8_rgb import sampled_video
from scripts.stage9_audit import calculate,sample_fit
from scripts.stage9_rgb import detect_rgb
from scripts.stage10_inputs import clip_inputs
from src.evaluate_feedback import write_csv
from src.extract_real_pose import load_detector
from src.stage6_render import AvatarRenderer
from src.stage7_feedback import HandGraphAdapter,HandGraphs,valid_pose
from src.stage9_timing import p95
from src.virtual_pose import load_fit

WH=(1280,720)
HANDS=[91,112,95,99,103,107,111,116,120,124,128,132]
ARMS=("coarse","A_stage9_geometry","B_stage9_joint","A_geometry","B_joint","C_local_expert","D_guarded","E_dense_geometry","F_dense_guarded","G_reanchored_geometry","H_reanchored_guarded","I_matched_geometry")


def detect_path(detector,renderer,fit,asset,camera,temp):
    _,xy,r=calculate(fit,asset,camera,temp,True)
    pix,confidence,person,_=detect_rgb(renderer,detector,r[0],fit["cam_trans"],camera)
    return pix,confidence,person,xy


def score_source(args,cfg,graphs,detector,renderer,source,temp):
    root=args.output/source
    out=root/"rgb"
    out.mkdir(exist_ok=True)
    coarse,_,idx,_,_,camera,_,_,video=clip_inputs(cfg,source,args.asset,root)
    take=np.linspace(0,len(idx)-1,min(len(idx),cfg["rgb_max_frames"])).astype(int)
    idx=idx[take]
    det,conf,person=pose_observations(sampled_video(video,idx),detector)
    ref=Response(graphs,det,conf,{"stage7_factor_weights":[.35,.40,.25]})
    ref.save(out/"real_response.npz")
    rows=[]
    for name in ARMS:
        if not (root/name/"final_fit.npz").exists():
            continue
        f=sample_fit(load_fit(root/name/"final_fit.npz"),idx)
        xy,cl,n,direct=detect_path(detector,renderer,f,args.asset,camera,temp)
        values=ref.score(xy,cl)[0]
        other=ref.score(direct,np.ones(direct.shape[:2],np.float32))[0]
        rows.append(dict(source=source,variant=name,sampled_frames=len(idx),real_detected=int((person>0).sum()),
                         virtual_detected=int((n>0).sum()),**{"rgb_"+k:v for k,v in values.items()},
                         matched_direct=other["response"]))
        np.savez_compressed(out/f"{name}.npz",indices=idx,xy=xy,confidence=cl,person=n,direct=direct)
    write_csv(out/"same_detector.csv",rows)
    print(source,[(r["variant"],round(r["rgb_response"],4)) for r in rows],flush=True)


def jitter(args,cfg,detector,renderer,temp):
    audit=args.output/"audit"
    sets=("legacy","fixed_gauge","stabilized")
    streams=[]
    with np.load(audit/"focus_world_variants.npz") as f:
        camera=f["affine"]
    for k in sets:
        fit=load_fit(audit/f"{k}_fit.npz")
        xy,conf,n,direct=detect_path(detector,renderer,fit,args.asset,camera,temp)
        streams.append((xy,conf,n,direct))
        np.savez_compressed(audit/f"rgb_{k}.npz",xy=xy,conf=conf,n=n,direct=direct)
        print(k,len(n),flush=True)
    common=np.logical_and.reduce([valid_pose(xy,conf,.3) for xy,conf,n,d in streams])
    cub=common[3:]&common[2:-1]&common[1:-2]&common[:-3]
    rows=[]
    for k,(xy,conf,n,d) in zip(sets,streams):
        score=dict(method=k,frames=len(n),persons=int((n>0).sum()))
        for label,ids in (("wrist",HANDS[:2]),("finger",HANDS[2:]),("hip",[11,12]),("knee",[13,14])):
            mask=cub[:,ids]
            score[label+"_coverage"]=float(mask.mean())
            for name,v in (("rtmw",xy),("direct",d)):
                jerk=np.linalg.norm(np.diff(v[:,ids]*WH,n=3,axis=0),axis=-1)
                score[label+"_"+name+"_p95_px_frame3"]=p95(jerk[mask]) if mask.any() else None
        rows.append(score)
    write_csv(audit/"rgb_jitter.csv",rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("repo","checkpoint","asset","detector","pose-model","output"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--jitter",action="store_true")
    p.add_argument("--partition",choices=("development","locked"),default="development")
    p.add_argument("--sources",nargs="+")
    a=p.parse_args()
    cfg=read(Path("configs/stage10.json"))
    torch.set_num_threads(2)
    detector=load_detector(a.repo,a.detector,a.pose_model,2)
    graphs=None if a.jitter else HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",2))
    with np.load(a.asset) as asset:
        renderer=AvatarRenderer(WH,asset["f"],asset["v_template"])
    try:
        with TemporaryDirectory() as td:
            temp=Path(td)/"sample.npz"
            if a.jitter:
                jitter(a,cfg,detector,renderer,temp)
            else:
                for source in a.sources or cfg[a.partition]:
                    score_source(a,cfg,graphs,detector,renderer,source,temp)
    finally:
        renderer.close()


if __name__=="__main__":
    main()
