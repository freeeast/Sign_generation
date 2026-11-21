"""Render each primary method at all four checkpoints from synchronized 3D states."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.stage4_review_media import LINKS, pose_on
from scripts.stage7_optimize import read
from scripts.stage8_imitate import METHODS
from scripts.stage8_visualize import restore_checkpoint
from src.smplx_to_pose import forward
from src.stage6_render import video_writer
from src.virtual_pose import apply_affine,load_fit,pinhole

PANEL=(300,180)
LABELS=("A Geometry","B Expert-only","C Sequential","D Joint")
TRACK_STEP=(0,6,12,18)


def tracks(root,stage7,source,level,method,step,asset):
    d=root/source/level/method
    cache=d/f"step_{step:02}_tracks.npz"
    if cache.is_file():
        with np.load(cache) as f:
            return f["xy"],f["points"]
    path=restore_checkpoint(d,step)
    xyz,_=forward(path,asset,"neutral","model",32,"cpu",False)
    with np.load(stage7/"data/paired"/f"{source}.npz") as f:
        xy=apply_affine(pinhole(xyz,load_fit(path)["cam_trans"]),f["camera"])
    return xy,xyz


def put(image,text,pos,scale=.43):
    cv2.putText(image,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,(236,244,250),1,cv2.LINE_AA)


def render(a,stage7,source,level):
    root=a.root
    dest=root/"STAGE8_VISUAL_DEMONSTRATION/all_methods"
    dest.mkdir(exist_ok=True,parents=True)
    path=dest/f"{source}_{level}_all_methods.mp4"
    if path.exists():
        return
    poses=[]
    summary=[]
    for method in METHODS:
        d=root/source/level/method
        with (d/"trace.csv").open() as f:
            line=list(csv.DictReader(f))
        with (d/"full_held_checkpoints.csv").open() as f:
            check={int(r["step"]):r for r in csv.DictReader(f)}
        summary.append([(float(line[s]["response"]),float(check[s]["held_geometry"]),
                         int(line[s]["constraint_violations"])+int(line[s]["new_angle_excess"])) for s in TRACK_STEP])
        poses.append([tracks(root,stage7,source,level,method,s,a.asset) for s in TRACK_STEP])
    viewside=[[x[1][...,[2,1]]*[-1,1] for x in group] for group in poses]
    ids=np.array(sorted({i for edge in LINKS for i in edge}))
    joined=np.concatenate([x[:,ids].reshape(-1,2) for group in viewside for x in group])
    lo,hi=np.quantile(joined,[.01,.99],axis=0)
    scale=min((PANEL[0]-42)/(hi[0]-lo[0]),(PANEL[1]-62)/(hi[1]-lo[1]))
    offset=np.array([PANEL[0]/2,PANEL[1]/2])-(hi+lo)/2*scale
    w,h=PANEL
    blank=np.full((h,w,3),24,np.uint8)
    n=len(poses[0][0][0])
    with video_writer(path,(4*w,2*h),24) as output:
        for j,step in enumerate(TRACK_STEP):
            for i in range(n):
                panels=[[],[]]
                for k,label in enumerate(LABELS):
                    e,held,guard=summary[k][j]
                    a0,b0=blank.copy(),blank.copy()
                    pose_on(a0,poses[k][j][0][i]*[w,h],(84,211,245))
                    pose_on(b0,viewside[k][j][i]*scale+offset,(216,193,105))
                    for im,title in ((a0,"Front"),(b0,"3D side")):
                        cv2.rectangle(im,(0,0),(w,24),(34,29,24),-1)
                        cv2.rectangle(im,(0,h-22),(w,h),(34,29,24),-1)
                        put(im,f"{label} | {title} | Step {step}",(8,17))
                    put(a0,f"E {e:.5f} | held {held:.2f}px",(8,h-7))
                    put(b0,f"Level {level} | guard violations {guard} | {i/24:.2f}s",(8,h-7),.37)
                    panels[0].append(a0)
                    panels[1].append(b0)
                output.write(np.vstack([np.hstack(x) for x in panels]).tobytes())
    print(path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--asset",type=Path,required=True)
    a=p.parse_args()
    cfg=read(Path("configs/stage8.json"))
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    for source in cfg["sources"]:
        for level in "ABC":
            render(a,Path(cfg["stage7_root"]),source,level)


if __name__=="__main__":
    main()
