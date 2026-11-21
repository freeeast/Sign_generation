"""Validate rendered motions with identical real/virtual detectors and localize temporal jitter."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from scripts.stage6_rgb import pose_observations
from scripts.stage7_optimize import read,write
from scripts.stage8_imitate import Response
from scripts.stage8_rgb import sampled_video
from scripts.stage9_audit import calculate,sample_fit
from src.evaluate_feedback import write_csv
from src.extract_real_pose import load_detector
from src.stage6_render import AvatarRenderer,video_writer
from src.stage7_feedback import HandGraphAdapter,HandGraphs,valid_pose
from src.stage9_timing import track_stats,stabilize,p95
from src.virtual_pose import load_fit

W,H=1280,720
ENDPOINTS=("art_geo_factor","art_hier_factor","art_joint_factor","art_guarded_factor","art_joint_geo_factor","art_guarded_geo_factor")
SELECT=np.array([91,112,95,99,103,107,111,116,120,124,128,132])


def detect_rgb(renderer,detector,vertices,translation,camera):
    centers=[]
    def stream():
        for f in renderer.frames(vertices,translation,camera):
            # A rendering-only signal, free of the pose detector.
            section=f[::5,::5]
            bg=section[0,0].astype(np.int16)
            fg=np.abs(section.astype(np.int16)-bg).max(axis=-1)>12
            rr,cc=np.where(fg)
            centers.append([cc.mean()*5,rr.mean()*5])
            yield f
    xy,conf,persons=pose_observations(stream(),detector)
    return xy,conf,persons,np.array(centers)


def endpoint(args,cfg,graphs,renderer,detector,temp,source):
    prior=Path(cfg["observations"])/"data"
    s8=Path(cfg["upstream"])/source
    root=args.output/source
    with np.load(prior/"paired"/f"{source}.npz") as p:
        camera=p["camera"]
    indices=np.array(read(s8/"frames.json")["optimization_frames"])
    indices=indices[np.linspace(0,len(indices)-1,min(len(indices),cfg["rgb_max_frames"])).astype(int)]
    realf=prior/"inputs"/f"{source}.mp4"
    xy,conf,n=pose_observations(sampled_video(realf,indices),detector)
    response=Response(graphs,xy,conf,{"stage7_factor_weights":cfg["factor_weights"]})
    out=args.output/"rgb"/source
    out.mkdir(exist_ok=True,parents=True)
    response.save(out/"real_response.npz")
    rows=[]
    for level in cfg.get("levels",("B","A","C")):
        variants={"initial":s8/level/"initial_fit.npz"}
        if level=="B":
            variants["stage8_c3"]=s8/level/"C3_local_wide"/"step_18_fit.npz"
        variants.update({m:root/level/m/"final_fit.npz" for m in ENDPOINTS})
        for name,path in variants.items():
            result=out/f"{level}_{name}.json"
            if result.exists():
                rows.append(read(result)); continue
            if not path.exists():
                continue
            time=perf_counter()
            fit=sample_fit(load_fit(path),indices)
            _,direct,r=calculate(fit,args.asset,camera,temp,True)
            pix,confidence,person,_=detect_rgb(renderer,detector,r[0],fit["cam_trans"],camera)
            sr,_=response.score(pix,confidence)
            sd,_=response.score(direct,np.ones(direct.shape[:2],np.float32))
            value=dict(source=source,level=level,variant=name,frames=len(indices),elapsed_s=perf_counter()-time,
                       visible_person=int((person>0).sum()),**{"rgb_"+k:v for k,v in sr.items()},
                       **{"direct_"+k:v for k,v in sd.items()})
            write(result,value); rows.append(value)
            np.savez_compressed(out/f"{level}_{name}_detections.npz",xy=pix,confidence=confidence,direct=direct,indices=indices)
            print(source,level,name,round(sr["response"],5),round(sr["support"],3),flush=True)
    if rows:
        write_csv(out/"results.csv",rows)


def jitter(args,cfg,renderer,detector,temp):
    source=cfg["focus"]; prior=Path(cfg["observations"])/"data"; old=Path(cfg["upstream"])/source
    with np.load(prior/"paired"/f"{source}.npz") as p: camera=p["camera"]
    fit=load_fit(old/"B/initial_fit.npz")
    out=args.output/"rgb"/"timing"
    out.mkdir(exist_ok=True,parents=True)
    values=[]
    for name,f in (("raw",fit),("corrected",stabilize(fit))):
        p=out/f"{name}_metrics.json"
        if p.exists():
            values.append(read(p)); continue
        pts,xy,r=calculate(f,args.asset,camera,temp,True)
        uv,conf,count,centers=detect_rgb(renderer,detector,r[0],f["cam_trans"],camera,None)
        valid=valid_pose(uv,conf,.3)[:,SELECT]
        cub=valid[3:]&valid[2:-1]&valid[1:-2]&valid[:-3]
        v1=np.linalg.norm(np.diff(xy[:,SELECT]*[W,H],n=3,axis=0),axis=-1)
        v2=np.linalg.norm(np.diff(uv[:,SELECT]*[W,H],n=3,axis=0),axis=-1)
        metric=dict(variant=name,frames=len(xy),rtmw_4frame_coverage=float(cub.mean()),
                    direct_2d_jerk_p95_px=p95(v1),direct_2d_matched_jerk_p95_px=p95(v1[cub]),
                    redetected_jerk_p95_px=p95(v2[cub]),
                    silhouette_centroid_jerk_p95_px=p95(np.linalg.norm(np.diff(centers,n=3,axis=0),axis=-1)),
                    **track_stats(pts,f,args.asset))
        write(p,metric); values.append(metric)
        np.savez_compressed(out/f"{name}.npz",direct=xy,detected=uv,confidence=conf,mesh_centers_px=centers,points=pts)
        print(name,"full-frame rendering/detection timing audit",metric["redetected_jerk_p95_px"],flush=True)
    write_csv(out/"metrics.csv",values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("repo","checkpoint","asset","detector","pose-model","output"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--jitter",action="store_true")
    p.add_argument("--sources",nargs="+")
    p.add_argument("--config",default=Path("configs/stage9.json"),type=Path)
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args(); cfg=read(a.config)
    detector=load_detector(a.repo,a.detector,a.pose_model,a.threads)
    graphs=None if a.jitter else HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",a.threads))
    if a.jitter:
        import torch
        torch.set_num_threads(a.threads)
    with np.load(a.asset) as f:
        renderer=AvatarRenderer((W,H),f["f"],f["v_template"])
    try:
        with TemporaryDirectory() as td:
            temp=Path(td)/"sample.npz"
            if a.jitter:
                jitter(a,cfg,renderer,detector,temp)
            else:
                for source in a.sources or cfg["sources"]:
                    endpoint(a,cfg,graphs,renderer,detector,temp,source)
    finally:
        renderer.close()


if __name__=="__main__":
    main()
