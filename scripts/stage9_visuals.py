"""Render synchronized 24 Hz Stage 9 learning, hand, and timing demonstrations."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from scripts.stage6_rgb import real_frames
from scripts.stage7_optimize import read
from scripts.stage9_audit import calculate
from src.stage6_render import AvatarRenderer,video_writer
from src.stage9_hand import DIGITS
from src.stage9_timing import stabilize,wrist_local
from src.virtual_pose import load_fit

FPS=24
RES=np.array([1280,720])
HAND=(slice(91,112),slice(112,133))
SHORT=("Th","Ix","Mi","Ri","Li")


def chart(img,text,at=(8,22),size=.49,color=(255,255,255)):
    cv2.putText(img,text,at,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)


def tile(frame,title,sub=None):
    img=frame.copy()
    img[:28]=[30,29,28]
    chart(img,title)
    if sub:
        img[-26:]=[30,29,28]; chart(img,sub,(8,len(img)-8),.42)
    return img


def assemble(path,grid,n):
    rows=len(grid); cols=len(grid[0])
    img0=grid[0][0][0]
    wh=(img0.shape[1]*cols,img0.shape[0]*rows)
    with video_writer(path,wh,FPS) as out:
        for i in range(n):
            out.write(np.vstack([np.hstack([g[i] for g in row]) for row in grid]).tobytes())


def panels(stream,title,size=None,sub=None):
    return [tile(cv2.resize(f,size) if size else f,title,sub[i] if isinstance(sub,list) else sub)
            for i,f in enumerate(stream)]


def avatar(fit,asset,cam,temp):
    points,xy,r=calculate(fit,asset,cam,temp,True)
    return fit,points,xy,r[0]


def render(r,state,cam,side=None):
    fit,_,_,mesh=state
    if side is not None:
        center,turn=side
        mesh=(mesh-center)@turn.T+center
    return r.frames(mesh,fit["cam_trans"],cam)


def crop_center(seq,idx):
    h=seq[:,idx,:]
    integer=np.isfinite(h).all(2) & (h>-.05).all(2) & (h<1.05).all(2)
    mark=np.copy(h[:,0])
    for i,row in enumerate(h):
        if integer[i,0]:
            others=integer[i,1:]
            if others.any():
                mark[i]=.42*row[0]+.58*row[1:][others].mean(0)
        else:
            mark[i]=np.nan
    for axis in range(2):
        good=np.isfinite(mark[:,axis])
        mark[:,axis]=np.interp(np.arange(len(mark)),np.flatnonzero(good),mark[good,axis]) if good.any() else .5
    return gaussian_filter1d(mark,1.2,axis=0)


def crop_camera(cam,center,extent):
    scale=RES/extent
    crop=np.broadcast_to(cam,(len(center),2,3)).copy()*scale[None,:,None]
    crop[...,2]+=.5-center*scale
    return crop


def real_hand(video,center,extent,size):
    for frame,loc in zip(video,center):
        pos=tuple((loc*RES).astype(float))
        yield cv2.resize(cv2.getRectSubPix(frame,(extent,extent),pos),size)


def excursions(v):
    return np.percentile(np.linalg.norm(v-np.median(v,axis=0),axis=-1),95,axis=0)


def controls(args,fit,cam,temp,f,faces,out):
    keys=("body_pose","left_hand_pose","right_hand_pose")
    frames=24; labels=list(dict.fromkeys(k.split("__")[0] for k in f.files if "__" in k))
    motion={k:np.repeat(v[:1],frames*len(labels),axis=0) if v.ndim and len(v)==2 else v.copy()
            for k,v in fit.items()}
    alpha=np.sin(np.linspace(0,np.pi,frames))**2
    for n,label in enumerate(labels):
        for k in keys:
            mean=0
            if k.endswith("hand_pose"):
                mean=faces["hands_meanl" if k.startswith("left") else "hands_meanr"].reshape(-1,3)
            start=Rotation.from_rotvec(fit[k][0].reshape(-1,3)+mean)
            end=Rotation.from_rotvec(f[label+"__"+k].reshape(-1,3)+mean)
            delta=(start.inv()*end).as_rotvec()
            for j,u in enumerate(alpha):
                motion[k][n*frames+j]=((start*Rotation.from_rotvec(u*delta)).as_rotvec()-mean).ravel()
    state=avatar(motion,args.asset,cam,temp)
    size=(288,288); r=AvatarRenderer(size,faces["f"],faces["v_template"])
    rows=[]
    try:
        base=state[2].copy()
        for num in range(len(labels)):
            base[num*frames:(num+1)*frames]=base[num*frames]
        for h,idx in enumerate(HAND):
            center=crop_center(base,idx)
            names=("Left" if h==0 else "Right")+" hand | isolated SMPL-X modes"
            subs=[f"{labels[i//frames]}  |  frame {i%frames+1}/24" for i in range(frames*len(labels))]
            rows.append(panels(render(r,state,crop_camera(cam,center,235)),names,sub=subs))
        assemble(out/"independent_controls.mp4",[rows],len(alpha)*len(labels))
    finally:
        r.close()


def closeups(args,cam,faces,out,video,reference,coarse,old,new):
    size=(300,300); extent=220
    r=AvatarRenderer(size,faces["f"],faces["v_template"])
    try:
        rows=[]
        for h,idx in enumerate(HAND):
            side="L" if h==0 else "R"
            loc=crop_center(reference,idx)
            row=[panels(real_hand(video,loc,extent,size),side+" | real (hand-following crop)")]
            for name,state in (("Coarse IK",coarse),("Stage 8 C3",old),("Stage 9 Joint",new)):
                fit,pts,xy,_=state
                local=wrist_local(pts,fit,args.asset)[:,h][:,[4,8,12,16,20]]*1000
                vals=excursions(local)
                stats=" ".join(f"{name}{v:.0f}" for name,v in zip(SHORT,vals))+" mm"
                center=crop_center(xy,idx)
                row.append(panels(render(r,state,crop_camera(cam,center,extent)),side+" | "+name,sub=stats))
            rows.append(row)
        assemble(out/"hand_closeups.mp4",rows,len(video))
    finally:
        r.close()


def scene(args,cam,faces,out,video,sidevideo,coarse,geo,new,stages,trace,stable):
    size=(384,216); n=len(video)
    r=AvatarRenderer(size,faces["f"],faces["v_template"])
    try:
        center=np.median(coarse[1][:,[5,6,11,12]],axis=(0,1))
        turn=Rotation.from_euler("y",87,degrees=True).as_matrix()
        front=[panels(video,"Real front | synchronized",size)]
        side=[panels(sidevideo,"Real side | matched 24 Hz",size)] if sidevideo else [
            panels([np.zeros((size[1],size[0],3),np.uint8)]*n,"No matched side camera clip")]
        for name,c in (("Coarse IK",coarse),("Stage 9 Geometry",geo),("Stage 9 Expert Joint",new)):
            front.append(panels(render(r,c,cam),name+" | front"))
            side.append(panels(render(r,c,cam,(center,turn)),name+" | virtual side"))
        assemble(out/"full_signing.mp4",[front,side],n)
        progress=[]
        for step,st in stages.items():
            m=trace[step]
            sub=[f'D={float(m["response"]):.4f}  |  bound excess {m["validity_violations"]}  |  {i/FPS:.2f}s'
                 if step else f'D={float(m["response"]):.4f}  |  input  |  {i/FPS:.2f}s' for i in range(n)]
            progress.append(panels(render(r,st,cam),f"Proposal {step:02} | frozen motion",sub=sub))
        assemble(out/"learning_progress.mp4",[progress],n)
        raw=panels(render(r,coarse,cam),"Raw IK | fixed view")
        clean=panels(render(r,stable,cam),"Stabilized IK | same view")
        plot=jitter_panel(coarse[2],stable[2],size)
        assemble(out/"jitter_diagnosis.mp4",[[raw,clean,plot]],n)
    finally:
        r.close()


def jitter_panel(raw,stable,size):
    width,height=size; n=len(raw); result=[]
    affected=[91,95,99,103,107,111,112,116,120,124,128,132]
    raw,stable=[np.linalg.norm(np.diff(z[:,affected]*RES,n=3,axis=0),axis=-1).max(axis=1) for z in (raw,stable)]
    canvas=np.full((height,width,3),[37,35,33],np.uint8)
    ny=np.arange(3,n)*((width-44)/(n-1))+30
    scale=max(raw.max(),stable.max())
    for vals,col in ((raw,(104,157,255)),(stable,(238,220,88))):
        pts=np.column_stack((ny,(height-36)-vals/scale*(height-78)))
        cv2.polylines(canvas,[pts.astype(np.int32)],False,col,1,cv2.LINE_AA)
    chart(canvas,"Wrist/fingertip peak 2D jerk (px/frame^3)",(8,21),.43)
    chart(canvas,f"{scale:.0f}",(8,43),.38); chart(canvas,"0",(10,height-34),.38)
    chart(canvas,"raw",(10,height-10),.45,(104,157,255))
    chart(canvas,"stabilized",(63,height-10),.45,(238,220,88))
    for i in range(n):
        img=canvas.copy(); x=int(30+i*(width-44)/(n-1))
        cv2.line(img,(x,32),(x,height-33),(240,240,240),1)
        chart(img,f"{i/FPS:.2f} s",(width-61,height-10),.42)
        result.append(img)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("asset","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--side-video",type=Path)
    p.add_argument("--only",choices=("controls","closeups","scene","front"))
    p.add_argument("--config",type=Path,default=Path("configs/stage9.json"))
    a=p.parse_args()
    import csv,torch
    torch.set_num_threads(2); cv2.setNumThreads(1)
    c=read(a.config); s=c["focus"]; root=a.output
    s8=Path(c["upstream"])/s/"B"; d=Path(c["observations"])/"data"
    work=root/s/"B"; out=root/"STAGE9_VISUAL_DEMONSTRATION"; out.mkdir(exist_ok=True,parents=True)
    with TemporaryDirectory() as td,np.load(a.asset) as meshes,np.load(d/"paired"/f"{s}.npz") as pair:
        tmp=Path(td)/"visual.npz"; cam=pair["camera"]
        if a.only in (None,"controls"):
            with np.load(root/"audit/controls.npz") as f:
                fit={k:f[k] for k in f.files if "__" not in k}
                controls(a,fit,cam,tmp,f,meshes,out)
        if a.only=="controls":
            return
        coarse=avatar(load_fit(s8/"initial_fit.npz"),a.asset,cam,tmp)
        new=avatar(load_fit(work/"art_joint_factor/final_fit.npz"),a.asset,cam,tmp)
        video=list(real_frames(d/"inputs"/f"{s}.mp4"))
        if len(video)!=len(coarse[1]):
            raise RuntimeError("Real and virtual timeline lengths differ")
        if a.only in (None,"closeups"):
            old=avatar(load_fit(s8/"C3_local_wide/step_18_fit.npz"),a.asset,cam,tmp)
            with np.load(d/"real_pose"/f"{s}.npz") as f:
                ref=np.where((f["scores"]>.30)[...,None],f["keypoints"],np.nan)
            closeups(a,cam,meshes,out,video,ref,coarse,old,new)
        if a.only=="front":
            size=(384,216)
            geo=avatar(load_fit(work/"art_joint_geo_factor/final_fit.npz"),a.asset,cam,tmp)
            renderer=AvatarRenderer(size,meshes["f"],meshes["v_template"])
            try:
                row=[panels(video,"Real front | synchronized",size)]
                for name,state in (("Coarse IK",coarse),("Matched Geometry",geo),("Expert Joint",new)):
                    row.append(panels(render(renderer,state,cam),name))
                assemble(out/"full_signing_front.mp4",[row],len(video))
            finally:
                renderer.close()
        if a.only in (None,"scene"):
            sidevideo=list(real_frames(a.side_video)) if a.side_video else []
            if sidevideo and len(sidevideo)!=len(video):
                raise RuntimeError("Frontal/side timelines differ")
            geo=avatar(load_fit(work/"art_joint_geo_factor/final_fit.npz"),a.asset,cam,tmp)
            with (work/"art_joint_factor/trace.csv").open() as f:
                trace={int(row["step"]):row for row in csv.DictReader(f)}
            stages={0:coarse,7:avatar(load_fit(work/"art_joint_factor/step_07_fit.npz"),a.asset,cam,tmp),
                    20:avatar(load_fit(work/"art_joint_factor/step_20_fit.npz"),a.asset,cam,tmp),36:new}
            stable=avatar(stabilize(coarse[0]),a.asset,cam,tmp)
            scene(a,cam,meshes,out,video,sidevideo,coarse,geo,new,stages,trace,stable)


if __name__=="__main__":
    main()
