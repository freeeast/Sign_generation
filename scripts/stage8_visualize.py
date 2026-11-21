"""Export synchronized reference, initial, geometry and expert-refined demonstrations."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.stage4_review_media import LINKS, pose_on
from scripts.stage6_rgb import real_frames
from scripts.stage7_optimize import read
from src.smplx_to_pose import forward
from src.stage6_render import AvatarRenderer, video_writer
from src.virtual_pose import load_fit


PANEL=(432,244)
CONTACT_COLOR=(14,211,250)


def caption(img, title, details, i, fps=24):
    frame=img.copy()
    cv2.rectangle(frame,(0,0),(PANEL[0],34),(28,24,20),-1)
    cv2.putText(frame,title,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.54,(245,249,255),1,cv2.LINE_AA)
    cv2.rectangle(frame,(0,PANEL[1]-25),(PANEL[0],PANEL[1]),(28,24,20),-1)
    cv2.putText(frame,details,(8,PANEL[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.39,(227,235,244),1,cv2.LINE_AA)
    # Common progress indicator for all panels.
    cv2.putText(frame,f"{i/fps:05.2f}s",(PANEL[0]-57,22),cv2.FONT_HERSHEY_SIMPLEX,.37,(216,222,231),1,cv2.LINE_AA)
    return frame


def labels(root,method,step):
    d=root/method
    with (d/"trace.csv").open() as f:
        row=list(csv.DictReader(f))[step]
    with (d/"full_held_checkpoints.csv").open() as f:
        held=next(r for r in csv.DictReader(f) if int(r["step"])==step)
    excess=int(row["constraint_violations"])+int(row["new_angle_excess"])
    return f"Expert {float(row['response']):.4f} | held {float(held['held_geometry']):.1f}px | edit {float(row['correction_joint_deg']):.1f}deg | guards {excess}"


def video(fronts, out, details):
    with video_writer(out,(PANEL[0]*3,PANEL[1]*2),24) as dest:
        for i, row in enumerate(zip(*fronts,strict=True)):
            blocks = [caption(img,title,sub,i) for img,(title,sub) in zip(row,details)]
            dest.write(np.vstack([np.hstack(blocks[:3]),np.hstack(blocks[3:])]).tobytes())


def skeleton_views(out,root):
    specs=(("A_geometry",0,"Initial"),
           ("A_geometry",18,"Geometry-only"),
           ("C_sequential",18,"Sequential expert"))
    xy=[]; points=[]
    for method,step,_ in specs:
        f=root/method/f"step_{step:02}_tracks.npz"
        with np.load(f) as r:
            xy.append(r["xy"])
            points.append(r["points"])
    side=[x[...,[2,1]]*[-1,1] for x in points]
    ids=np.array(sorted({i for edge in LINKS for i in edge}))
    visible=np.concatenate([x[:,ids].reshape(-1,2) for x in side])
    lo,hi=np.quantile(visible,[.01,.99],axis=0)
    scale=min((PANEL[0]-70)/(hi[0]-lo[0]),(PANEL[1]-110)/(hi[1]-lo[1]))
    offset=np.asarray([PANEL[0]/2,PANEL[1]/2+5])-(hi+lo)*scale/2
    side=[x*scale+offset for x in side]
    fronts=[x*np.asarray([PANEL[0],PANEL[1]]) for x in xy]

    def draw(frames):
        for p in frames:
            im=np.full((PANEL[1],PANEL[0],3),24,np.uint8)
            pose_on(im,p,CONTACT_COLOR)
            yield im

    names=[(f"{title} - front",labels(root,m,s)) for m,s,title in specs]
    names += [(f"{title} - side (avatar)",labels(root,m,s)) for m,s,title in specs]
    video([draw(p) for p in [*fronts,*side]],out,names)


def restore_checkpoint(arm,step):
    path=arm/f"step_{step:02}_fit.npz"
    if path.exists():
        return path
    original=load_fit(arm.parent/"initial_fit.npz")
    with np.load(arm/"motion_trajectory.npz") as saved:
        for key in ("body_pose","left_hand_pose","right_hand_pose"):
            original[key]=saved[key][step]
    np.savez_compressed(path,**original)
    return path


def make(a,source):
    cfg=read(Path("configs/stage8.json"))
    d=Path(cfg["stage7_root"])/"data"
    s=a.root/source/"B"
    o=a.root/"STAGE8_VISUAL_DEMONSTRATION"
    o.mkdir(parents=True,exist_ok=True)
    with np.load(d/"paired"/f"{source}.npz") as p:
        camera=p["camera"]
    ref=d/"inputs"/f"{source}.mp4"
    frames=[cv2.resize(f,PANEL,interpolation=cv2.INTER_AREA) for f in real_frames(ref)]
    specs=(("A_geometry",0,"1  Initial - coarse IK"),
           ("A_geometry",18,"2  Geometry-only - step 18"),
           ("C_sequential",6,"3  Sequential - geometry anchor"),
           ("C_sequential",12,"4  Sequential - expert pass 1"),
           ("C_sequential",18,"5  Sequential - expert pass 2"))
    with np.load(a.asset) as skin:
        render=AvatarRenderer(PANEL,skin["f"],skin["v_template"])
    names=[("Real reference","How2Sign / original front view")]
    try:
        out=o/f"{source}_coarse_closed_loop.mp4"
        if not out.exists():
            streams=[frames]
            for method,step,label in specs:
                path=restore_checkpoint(s/method,step)
                fit=load_fit(path)
                _,_,mesh,_=forward(path,a.asset,"neutral","model",32,"cpu",False,return_mesh=True)
                # Materialize one stream at a time because they share the OpenGL context.
                streams.append(list(render.frames(mesh,fit["cam_trans"],camera)))
                names.append((label,labels(s,method,step)))
                del mesh
            video(streams,out,names)
        side=o/f"{source}_coarse_front_side.mp4"
        if not side.exists():
            skeleton_views(side,s)
        if a.local:
            local=(("A_geometry",0,"Initial coarse"),("A_geometry",18,"Geometry-only"),
                   ("C_sequential",18,"Sequential"),("C2_local_hands",18,"Local fingers"),
                   ("C3_local_wide",18,"Local fingers - wider radius"))
            streams=[frames]; names=[("Real reference","How2Sign / original front view")]
            for method,step,label in local:
                path=restore_checkpoint(s/method,step)
                fit=load_fit(path)
                _,_,mesh,_=forward(path,a.asset,"neutral","model",32,"cpu",False,return_mesh=True)
                streams.append(list(render.frames(mesh,fit["cam_trans"],camera)))
                names.append((label,labels(s,method,step)))
                del mesh
            video(streams,o/f"{source}_coarse_algorithm_comparison.mp4",names)
    finally:
        render.close()
    print(source,"demonstrations saved",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",required=True,type=Path)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--sources",nargs="+")
    p.add_argument("--local",action="store_true")
    a=p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cfg=read(Path("configs/stage8.json"))
    for source in a.sources or cfg["rgb_sources"]:
        make(a,source)


if __name__=="__main__":
    main()
