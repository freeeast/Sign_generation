"""Decompose focal handshape by joint and time without using it in candidate selection."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from scripts.stage7_optimize import read
from scripts.stage9_audit import calculate
from scripts.stage10_audit import HANDS
from scripts.stage10_inputs import clip_inputs
from src.evaluate_feedback import write_csv
from src.stage9_hand import DIGITS
from src.stage9_timing import wrist_frames
from src.virtual_pose import load_fit


def norm(v):
    return np.linalg.norm(v,axis=-1)


def normalize(v):
    return v/np.maximum(norm(v)[...,None],1e-7)


def shape(p,digit):
    base=1+4*digit
    phalanges=normalize(np.diff(p[:,base:base+4],axis=1))
    bend=np.rad2deg(np.arccos(np.clip((phalanges[:,1:]*phalanges[:,:-1]).sum(-1),-1,1)))
    palm=normalize(p[:,9]-p[:,0])
    across=p[:,17]-p[:,5]
    side=normalize(across-(across*palm).sum(-1)[:,None]*palm)
    first=phalanges[:,0]
    spread=np.rad2deg(np.arctan2((first*side).sum(-1),(first*palm).sum(-1)))
    scale=np.maximum(norm(p[:,9]-p[:,0]),1e-7)
    return bend,spread,norm(p[:,4]-p[:,8])/scale


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    cfg=read(Path("configs/stage10.json")); sid=cfg["focus"]
    root=a.output/sid
    _,nsa,_,real,score,camera,other,_,_=clip_inputs(cfg,sid,a.asset,root)
    t=np.arange(86,97); rows=[]
    torch.set_num_threads(2)
    poses={"RTMW":(None,real* [1280,720],None),"MediaPipe":(None,other*[1280,720],None)}
    with TemporaryDirectory() as td:
        for name in ("coarse","A_stage9_geometry","B_stage9_joint","G_reanchored_geometry","H_reanchored_guarded","NSA"):
            fit=nsa if name=="NSA" else load_fit(root/name/"final_fit.npz")
            xyz,xy,_=calculate(fit,a.asset,camera,Path(td)/"fit.npz")
            wr=wrist_frames(fit,a.asset)
            rot=Rotation.from_matrix(wr.reshape(-1,3,3)).as_rotvec(degrees=True).reshape(-1,2,3)
            poses[name]=(xyz,xy*[1280,720],rot)
    for name,(xyz,xy,rot) in poses.items():
        for h,side in enumerate(("left","right")):
            ids=HANDS[h]
            for j,digit in enumerate(DIGITS):
                b2,s2,o2=shape(xy[:,ids],j)
                if xyz is not None:
                    b3,s3,o3=shape(xyz[:,ids],j)
                    tip=xyz[:,ids[4+4*j]]-xyz[:,ids[0]]
                    speed=np.r_[0,24*1000*norm(np.diff(tip,axis=0))]
                for i in t:
                    if not np.isfinite(b2[i]).all():continue
                    w=xy[i,ids[0]]
                    row=dict(variant=name,frame=i,side=side,digit=digit,wrist_x_px=w[0],wrist_y_px=w[1],
                             bend_proximal_projected_deg=b2[i,0],bend_distal_projected_deg=b2[i,1],
                             spread_projected_deg=s2[i],thumb_index_projected_ratio=o2[i],
                             own_finger_min_rtmw_score=float(np.min(score[i,ids[1+4*j:5+4*j]])),
                             **{k:None for k in ("wrist_rotation_x","wrist_rotation_y","wrist_rotation_z",
                                                 "bend_proximal_3d_deg","bend_distal_3d_deg","spread_3d_deg",
                                                 "thumb_index_3d_ratio","wrist_relative_tip_speed_mm_s")})
                    if rot is not None:
                        row.update(wrist_rotation_x=rot[i,h,0],wrist_rotation_y=rot[i,h,1],wrist_rotation_z=rot[i,h,2],
                                   bend_proximal_3d_deg=b3[i,0],bend_distal_3d_deg=b3[i,1],spread_3d_deg=s3[i],
                                   thumb_index_3d_ratio=o3[i],wrist_relative_tip_speed_mm_s=speed[i])
                    rows.append(row)
    write_csv(a.output/"audit/focus_each_joint_and_finger.csv",rows)


if __name__=="__main__":
    main()
