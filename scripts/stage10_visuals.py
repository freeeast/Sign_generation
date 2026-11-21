"""Produce fixed-view Stage 10 videos at native 24 fps."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from scripts.stage6_rgb import real_frames
from scripts.stage7_optimize import read
from scripts.stage9_audit import sample_fit
from scripts.stage9_visuals import avatar,assemble,panels,real_hand
from scripts.stage10_audit import hand_shape,HANDS
from scripts.stage10_inputs import clip_inputs
from src.stage6_render import AvatarRenderer
from src.stage10_world import fixed_world
from src.virtual_pose import apply_affine,load_fit,pinhole


def camera_crop(camera, center, extent):
    factor=np.array([1280,720])/extent
    p=camera*factor[:,None]
    p[:,2]+=.5-center*factor
    return p


def center_view(camera,xyz,translation):
    bounds=apply_affine(pinhole(xyz[:,::12],translation),camera)
    lo=bounds.min((0,1)); hi=bounds.max((0,1))
    center=(lo+hi)/2
    aspect=np.array([1280/370,720/380])
    scale=np.min(np.array([.88,.82])/(np.maximum(hi-lo,1e-6)*aspect))*aspect
    mat=camera*scale[:,None]
    mat[:,2]+=.5-scale*center
    return mat


def make(args,cfg,source,temp,faces):
    root=args.output/source
    out=args.output/"visuals"; out.mkdir(exist_ok=True)
    coarse,nsa,idx,obs,conf,camera,mp,s9,video=clip_inputs(cfg,source,args.asset,root)
    video=list(real_frames(video))
    n=len(obs)
    if len(video)!=n:
        raise ValueError(f"{source}: real RGB/pose frames mismatch")
    labels=dict(coarse="Coarse IK",A_stage9_geometry="S9 Geometry",A_geometry="Geometry + stable body",
                B_stage9_joint="S9 Joint",B_joint="S9 Joint + stable body",E_dense_geometry="Dense geometry",
                G_reanchored_geometry="Reanchored geometry",H_reanchored_guarded="Constrained expert refinement")
    focal=source==cfg["focus"]
    def state(name,ids=None):
        f=load_fit(root/name/"final_fit.npz")
        return avatar(f if ids is None else sample_fit(f,ids),args.asset,camera,temp)
    def renderer(size):
        return AvatarRenderer(size,faces["f"],faces["v_template"])

    if focal:
        audit=args.output/"audit"
        methods=("legacy","fixed_gauge","stabilized")
        fits=[load_fit(audit/f"{m}_fit.npz") for m in methods]
        states=[avatar(f,args.asset,camera,temp) for f in fits]
        fixed=states[1]
        common=fixed[0]["cam_trans"][0]
        world_legacy=fixed[3]
        front_cam=center_view(camera,np.concatenate([world_legacy,states[2][3]]),
                              np.repeat(common[None],2*n,axis=0))
        pivot=np.median(fixed[3],axis=(0,1))
        turn=Rotation.from_euler("y",87,degrees=True).as_matrix()
        side_mesh=[(s-pivot)@turn.T+pivot for s in (world_legacy,fixed[3],states[2][3])]
        side_cam=center_view(camera,np.concatenate(side_mesh[1:]),np.repeat(common[None],2*n,axis=0))
        with (audit/"body_decomposition.csv").open() as f:
            metrics={r["method"]:r for r in csv.DictReader(f)}
        r=renderer((370,380))
        try:
            names=("Stage 9 legacy translation","Fixed camera | same motion","Fixed camera | stabilized body")
            rows=[[],[]]
            for label,m,s,side in zip(names,methods,states,side_mesh):
                me=metrics[m]
                foot=f'Pelvis jerk {float(me["pelvis_jerk_p95_mm_frame3"]):.1f} | knee {float(me["knee_jerk_p95_mm_frame3"]):.1f} mm/f^3'
                rows[0].append(panels(r.frames(s[3],s[0]["cam_trans"],front_cam),label,sub=foot))
                rows[1].append(panels(r.frames(side,fixed[0]["cam_trans"],side_cam),"Fixed side view",sub=foot))
            assemble(out/"A_stable_camera_lower_body.mp4",rows,n)
        finally:
            r.close()

        if args.only_body:
            return
        ids=np.arange(74,111)
        real=[video[i] for i in ids]
        variants=("A_stage9_geometry","B_stage9_joint","G_reanchored_geometry","H_reanchored_guarded")
        states=[state(m,ids) for m in variants]
        r=renderer((270,270))
        try:
            rows=[]
            for h,side in enumerate(("Left","Right")):
                k=HANDS[h]
                center=obs[82:99,k].mean((0,1))
                center=np.broadcast_to(center,(len(ids),2)).copy()
                extent=260
                view=camera_crop(camera,center[0],extent)
                def sub(x):
                    _,_,ap=hand_shape(x)
                    return [f'Frame {j}{" *" if j==90 else ""}  |  aperture {q:.2f}' for j,q in zip(ids,ap)]
                first=panels(real_hand(real,center,extent,(270,270)),side+" | real, fixed crop",
                             sub=sub(obs[ids][:,k]*[1280,720]))
                row=[first]
                for m,s in zip(variants,states):
                    row.append(panels(r.frames(s[3],s[0]["cam_trans"],view),side+" | "+labels[m][:28],
                                      sub=sub(s[2][:,k]*[1280,720])))
                rows.append(row)
            assemble(out/"B_handshape_frame90.mp4",rows,len(ids))
        finally:
            r.close()

        refin=root/"H_reanchored_guarded"
        with (refin/"trace.csv").open() as f:
            trace={int(r["step"]):r for r in csv.DictReader(f)}
        r=renderer((365,216))
        try:
            steps=(0,9,18,36)
            pictures=[]
            for step in steps:
                s=avatar(load_fit(refin/f"step_{step:02}_fit.npz"),args.asset,camera,temp)
                score=float(trace[step]["expert"])
                subs=[f"Expert {score:.5f} | playback frame {i:03}" for i in range(n)]
                title="Articulated geometry anchor" if step==0 else f"Fixed state after proposal {step}"
                pictures.append(panels(r.frames(s[3],s[0]["cam_trans"],camera),title,sub=subs))
            assemble(out/"C_guarded_learning_progress.mp4",[pictures],n)
        finally:
            r.close()

    r=renderer((320,192))
    try:
        cols=("coarse","A_stage9_geometry","B_stage9_joint","H_reanchored_guarded")
        front=[panels(video,"Synchronized real front",(320,192))]
        side_rows=[]
        sidepath=Path("experiments/Stage7/multiview/videos")/f"{source}.mp4"
        if focal and sidepath.exists():
            real_side=list(real_frames(sidepath))
            if len(real_side)==n:
                side_rows=[panels(real_side,"Synchronized real side",(320,192))]
        states=[state(m) for m in cols]
        common=states[0][0]["cam_trans"][0]
        if side_rows:
            center=np.median(states[0][3],axis=(0,1))
            turn=Rotation.from_euler("y",87,degrees=True).as_matrix()
        for (m,s) in zip(cols,states):
            front.append(panels(r.frames(s[3],s[0]["cam_trans"],camera),labels[m]))
            if side_rows:
                side=(s[3]-center)@turn.T+center
                side_rows.append(panels(r.frames(side,s[0]["cam_trans"],camera),labels[m]+" | side"))
        name="D_full_signing_comparison.mp4" if focal else f"D_locked_{source.split('_')[0]}.mp4"
        assemble(out/name,[front,side_rows] if side_rows else [front],n)
    finally:
        r.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--sources",nargs="+")
    p.add_argument("--only-body",action="store_true")
    a=p.parse_args()
    cv2.setNumThreads(1); torch.set_num_threads(2)
    cfg=read(Path("configs/stage10.json"))
    with TemporaryDirectory() as td,np.load(a.asset) as mesh:
        for source in a.sources or [cfg["focus"]]:
            make(a,cfg,source,Path(td)/"vis.npz",mesh)


if __name__=="__main__":
    main()
