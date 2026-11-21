"""Render the shared model and its frozen-expert optimizer at a fixed viewpoint."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from scripts.shared_motion_train import write_csv
from scripts.stage6_rgb import real_frames
from scripts.stage9_audit import calculate,sample_fit
from scripts.stage9_visuals import assemble,panels,real_hand
from scripts.stage10_visuals import camera_crop
from src.shared_motion import decode,hand_mean,increments,load_predictor,rotations
from src.stage6_render import AvatarRenderer
from src.virtual_pose import load_fit,pinhole

IDENTITY=np.array([[1.,0,0],[0,1.,0]])


def mesh(fit,asset,temp,camera=IDENTITY):
    return calculate(fit,asset,camera,temp,True)


def footage(panels12):
    return [x for x in panels12 for _ in range(2)]


def backdrop(fit,known):
    output={k:v.copy() for k,v in fit.items()}
    for k,v in output.items():
        if v.ndim and len(v)==len(fit["body_pose"]):
            v[known:]=v[known-1]
    output["betas"][:]=np.median(fit["betas"][:known],axis=0)
    output["cam_trans"][:]=np.median(fit["cam_trans"][:known],axis=0)
    return output


def horizon_fits(fit,means,model,context,mode):
    r=rotations(fit,means)
    d=increments(r)
    start=context
    upper=backdrop(fit,start)
    x=torch.from_numpy(r)
    y=torch.from_numpy(d)
    with torch.no_grad():
        if mode=="one":
            starts=np.arange(len(r)-start)
            ids=starts[:,None]+np.arange(start)
            pred,_=model(x[ids],y[ids])
            from src.shared_motion import exp
            answer=(x[start-1:-1]@exp(pred)).numpy()
            b=r[start-1:-1]
        else:
            seed=(x[:start][None],y[:start][None])
            pred,_=model.rollout(*seed,len(r)-start)
            answer=pred[0].numpy()
            b=r[start-1:start].repeat(len(r)-start,axis=0)
    return {label:decode(upper,np.concatenate([r[:start],q]),means) for label,q in
            (("reference",r[start:]),("hold",b),("gru",answer))}


def display_camera(xyz,fit,seed):
    ids=[0,5,6,7,8,9,10,11,12]
    xy=pinhole(xyz[:seed,ids],fit["cam_trans"][:seed])
    lo=xy.min((0,1)); hi=xy.max((0,1))
    mid=(lo+hi)*.5
    h=hi-lo
    ss=min(.77/h[0],.72/(1.4*h[1]))
    s=np.array([ss,1.4*ss])
    return np.array([[s[0],0,.5-s[0]*mid[0]],[0,s[1],.53-s[1]*mid[1]]])


def avatar_stream(render,state,fit,camera,side=False):
    _,_,vals=state
    v=vals[0]
    if side:
        turn=Rotation.from_euler("y",72,degrees=True).as_matrix()
        fixed=np.median(v[0],axis=0)
        v=(v-fixed)@turn.T+fixed
    return render.frames(v,fit["cam_trans"],camera)


def show_forecast(args,out,means,model,ctx,a,temp):
    with (args.dataset/"sample.csv").open() as f:
        rows=list(csv.DictReader(f))
    eligible=sorted((r for r in rows if r["split"]=="test" and int(r["frames"])>=2*(ctx+48)),
                    key=lambda r:(int(r["frames"]),r["clip"]))
    row=eligible[len(eligible)//2]
    original=load_fit(args.dataset/row["path"])
    pick=len(original["body_pose"])
    raw=sample_fit(original,np.arange(0,min(pick,2*(ctx+48)),2))
    report={"test_visual_source":row["clip"],"seed_12hz_frames":ctx,"length_12hz_frames":len(raw["body_pose"])}
    base,_,_=mesh(raw,args.asset,temp)
    camera=display_camera(base,raw,ctx)
    labels={"reference":"NSA | common fixed scaffold","hold":"Hold last pose","gru":"Shared GRU"}
    sizes=(350,240)
    mdir=out/"motion_exports"
    mdir.mkdir(exist_ok=True)
    for mode,name in (("one","Video1_teacher_forced.mp4"),("open","Video2_free_running.mp4")):
        fits=horizon_fits(raw,means,model,ctx,mode)
        states={m:mesh(f,args.asset,temp,camera) for m,f in fits.items()}
        r=AvatarRenderer(sizes,a["f"],a["v_template"])
        try:
            grid=[]
            for side in (False,True):
                line=[]
                for m,fit in fits.items():
                    mark=[f'{"Seed" if i<ctx else "Teacher-forced step" if mode=="one" else "Free-running future"} | t={i/12:.2f}s'
                          for i in range(len(raw["body_pose"]))]
                    line.append(footage(panels(avatar_stream(r,states[m],fit,camera,side),labels[m]+(" | side" if side else ""),sub=mark)))
                grid.append(line)
            if mode=="one":
                # The same square fixed crop is chosen from the seed wrist landmarks.
                center=states["reference"][1][:ctx,91:112].mean((0,1))
                crop=camera_crop(camera,center,435)
                grid.append([footage(panels(avatar_stream(r,states[m],f,crop),labels[m]+" | left hand")) for m,f in fits.items()])
            assemble(out/name,grid,2*len(raw["body_pose"]))
        finally:
            r.close()
        for m,f in fits.items():
            np.savez_compressed(mdir/f"{mode}_{m}_{row['clip']}.npz",**f,fps=12)
    (out/"video_manifest.json").write_text(json.dumps(report,indent=2)+"\n")


def show_hands(args,out,means,model,ctx,a,temp):
    source="3sZjSNcqXpI_2-1-rgb_front"
    data=Path("experiments/Stage7/data")
    p=data/"inputs"/f"{source}.npz"
    original=load_fit(p)
    ids=np.arange(0,2*(ctx+48),2)
    raw=sample_fit(original,ids)
    fits=horizon_fits(raw,means,model,ctx,"open")
    with np.load(data/"paired"/f"{source}.npz") as b:
        camera=b["camera"]
    with np.load(data/"real_pose"/f"{source}.npz") as b:
        xy=b["keypoints"][ids]
    video=list(real_frames(p.with_suffix(".mp4")))[0:2*(ctx+48):2]
    r=AvatarRenderer((300,300),a["f"],a["v_template"])
    try:
        grid=[]
        states={m:mesh(f,args.asset,temp,camera) for m,f in fits.items() if m!="hold"}
        n=len(video)
        for title,hand in (("Left",slice(91,112)),("Right",slice(112,133))):
            center=np.median(xy[:,hand],axis=(0,1))
            crop=camera_crop(camera,center,460)
            mark=[f'{"Seed" if i<ctx else "Future"} | t={i/12:.2f}s' for i in range(n)]
            real=panels(real_hand(video,np.repeat(center[None],n,axis=0),460,(300,300)),title+" hand | real",sub=mark)
            grid.append([footage(real)]+[footage(panels(avatar_stream(r,q,fits[m],crop),
                                title+(" hand | NSA" if m=="reference" else " hand | shared GRU"),sub=mark)) for m,q in states.items()])
        assemble(out/"Video3_both_hands.mp4",grid,n*2)
    finally:
        r.close()
    for name,f in fits.items():
        np.savez_compressed(out/"motion_exports"/f"paired_{name}_{source}.npz",**f,fps=12)


def show_feedback(args,out,a,temp):
    source="3sZjSNcqXpI_2-1-rgb_front"
    d=Path("experiments/Stage7/data")
    root=args.feedback/source
    with np.load(d/"paired"/f"{source}.npz") as p: camera=p["camera"]
    with (root/"D_shared_expert/trace.csv").open() as f: trace={int(x["step"]):x for x in csv.DictReader(f)}
    r=AvatarRenderer((330,208),a["f"],a["v_template"])
    try:
        video=list(real_frames(d/"inputs"/f"{source}.mp4"))
        n=len(video)
        tile=[panels(video,"Real | 3sZj",size=(330,208))]
        stack=(("B_stage9_joint/final_fit.npz","Stage 9 Joint",None),
               ("C_shared_only/final_fit.npz","Shared prior only",None),
               ("D_shared_expert/step_00_fit.npz","Common geometry seed",0),
               ("D_shared_expert/step_18_fit.npz","Shared + expert | Proposal 18",18),
               ("D_shared_expert/final_fit.npz","Shared + expert | final",max(trace)))
        for path,label,i in stack:
            fit=load_fit(root/path)
            s=mesh(fit,args.asset,temp,camera)
            subtitle=None if i is None else [f'Expert {float(trace[i]["expert"]):.5f} | video frame {j}' for j in range(n)]
            tile.append(panels(avatar_stream(r,s,fit,camera),label,sub=subtitle))
        assemble(out/"Video4_feedback_integration.mp4",[tile[:3],tile[3:]],n)
    finally:
        r.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","asset","checkpoint","output","feedback"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--only-feedback",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    model,p=load_predictor(args.checkpoint)
    mean=hand_mean(args.asset)
    args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.asset) as a,TemporaryDirectory() as t:
        temp=Path(t)/"fit.npz"
        if not args.only_feedback:
            show_forecast(args,args.output,mean,model,p["context"],a,temp)
            show_hands(args,args.output,mean,model,p["context"],a,temp)
        if (args.feedback/"integration.csv").exists():
            show_feedback(args,args.output,a,temp)


if __name__=="__main__":
    main()
