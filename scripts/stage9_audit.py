"""Audit Stage 8 stored poses, temporal inheritance, and single-digit mesh controls."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from smplx.joint_names import JOINT_NAMES

from scripts.stage7_optimize import read,write
from src.evaluate_feedback import write_csv
from src.smplx_to_pose import forward
from src.stage9_hand import Controller,DIGITS,SIDES,EDIT_2D,SparseFK
from src.stage9_timing import angular,p95,rotation_matrices,stabilize,track_stats,wrist_local
from src.virtual_pose import apply_affine,load_fit,pinhole

OLDER=("A_geometry","C_sequential","C2_local_hands","C3_local_wide")


def sample_fit(fit,frames):
    n=len(fit["body_pose"])
    return {k:v[frames] if v.ndim and len(v)==n else v for k,v in fit.items()}


def calculate(fit,asset,cam,temp,mesh=False):
    np.savez(temp,**fit)
    xyz,_,*render=forward(temp,asset,"neutral","model",32,"cpu",False,return_mesh=mesh)
    xy=apply_affine(pinhole(xyz,fit["cam_trans"]),cam)
    return xyz,xy,render


def finger_rows(name,label,fit,asset,pts,mean):
    records=[]
    local=wrist_local(pts,fit,asset)
    delta=local-np.median(local,axis=0)
    for h,s in enumerate(SIDES):
        a=fit[f"{s}_hand_pose"].reshape(-1,15,3)+mean[h]
        mats=rotation_matrices(a)
        quaternions=Rotation.from_matrix(mats.reshape(-1,3,3)).as_quat().reshape(len(a),15,4)
        dots=np.einsum("tjd,ujd->jtu",quaternions,quaternions)
        maximum_rotation=2*np.rad2deg(np.arccos(np.clip(np.abs(dots).min(axis=(1,2)),0,1)))
        velocity,accel,jerk=angular(a)
        mid=mats[len(a)//2]
        relative=np.swapaxes(mid,-1,-2)@mats
        turn=Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().reshape(len(a),15)*180/np.pi
        for f,part in enumerate(DIGITS):
            tip=4+4*f
            gap=p95(np.linalg.norm(delta[:,h,tip],axis=-1))*1000
            for j in range(3):
                namej=s+"_"+part+str(j+1)
                at=JOINT_NAMES.index(namej)-(25 if h==0 else 40)
                records.append(dict(source=name,variant=label,side=s,finger=part,joint=namej,
                                    geodesic_range_deg=float(maximum_rotation[at]),
                                    axis_angle_from_first_max_deg=float(np.linalg.norm(a[:,at]-a[0,at],axis=-1).max()*180/np.pi),
                                    geodesic_from_middle_max_deg=float(turn[:,at].max()),
                                    angular_speed_p95_deg_s=p95(np.linalg.norm(velocity[:,at],axis=-1))*24,
                                    angular_accel_p95_deg_s2=p95(np.linalg.norm(accel[:,at],axis=-1))*24**2,
                                    angular_jerk_p95_deg_s3=p95(np.linalg.norm(jerk[:,at],axis=-1))*24**3,
                                    wrist_local_tip_p95_mm=gap))
    return records


def pixel_stats(xy):
    z=xy[:,[91,112,95,99,103,107,111,116,120,124,128,132]]*[1280,720]
    return {f"pixel_{tag}_p95":p95(np.linalg.norm(np.diff(z,n=i,axis=0),axis=-1))
            for i,tag in enumerate(("speed","accel","jerk"),1)}


def diagnostic(source,fit,cam,asset,temp,out):
    one=sample_fit(fit,np.repeat(len(fit["body_pose"])//2,2))
    pts,xy,r=calculate(one,asset,cam,temp,True)
    fk=SparseFK(one,asset,cam,xy)
    control=Controller(fk,4,4)
    with torch.no_grad():
        test=fk.points(fk.base_rot,np.arange(2)).numpy()
    write(out/"sparse_fk_parity.json",dict(max_abs_3d_mm=float(np.abs(test-pts[:,EDIT_2D]).max()*1000)))
    modes=[]
    for side in SIDES:
        for finger in DIGITS:
            mode=side+"."+finger+(".opposition" if finger=="thumb" else ".pip_dip")
            modes.append((mode,[mode]))
        modes.append((side+".wrist",[side+"_wrist.z"]))
    modes.append(("both_open_close",[n for n in control.names if n.endswith(".pip_dip") or n.endswith("thumb.flex")]))
    runs=[]; shots={}
    for label,names in modes:
        x=control.zero()
        ids=[control.names.index(n) for n in names]
        x[:,ids]=.85 if label!="both_open_close" else -.85
        proposal=control.export(x)
        q,_,mesh=calculate(proposal,asset,cam,temp,True)
        left=np.linalg.norm(q[0,91:112]-pts[0,91:112],axis=-1)*1000
        right=np.linalg.norm(q[0,112:133]-pts[0,112:133],axis=-1)*1000
        runs.append(dict(mode=label,left_wrist_mm=float(left[0]),right_wrist_mm=float(right[0]),
                         left_finger_max_mm=float(left[1:].max()),right_finger_max_mm=float(right[1:].max()),
                         mesh_max_mm=float(np.linalg.norm(mesh[0][0]-r[0][0],axis=-1).max()*1000),
                         **{s+"_"+f+"_tip_mm":float(v[(j+1)*4]) for s,v in zip(SIDES,(left,right)) for j,f in enumerate(DIGITS)}))
        for k in ("body_pose","left_hand_pose","right_hand_pose"):
            shots[label+"__"+k]=proposal[k][0]
    write_csv(out/"isolated_controls.csv",runs)
    np.savez_compressed(out/"controls.npz",**one,**shots)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    cfg=read(Path("configs/stage9.json")); root=Path(cfg["upstream"]); d=Path(cfg["observations"])/"data"
    a.output.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(3)
    with np.load(a.asset) as m:
        mean=[m[k].reshape(15,3) for k in ("hands_meanl","hands_meanr")]
    rows=[]; per_finger=[]; temporal=[]
    with TemporaryDirectory() as td:
        temp=Path(td)/"fit.npz"
        for source in cfg["sources"]:
            with np.load(d/"paired"/f"{source}.npz") as x: cam=x["camera"]
            coarse=load_fit(root/source/"B/initial_fit.npz")
            original=load_fit(d/"inputs"/f"{source}.npz")
            cases={"NSA":original,"coarse":coarse}
            cases.update({m:load_fit(root/source/"B"/m/"step_18_fit.npz") for m in OLDER})
            for label,fit in cases.items():
                pts,xy,_=calculate(fit,a.asset,cam,temp)
                rows.append(dict(source=source,variant=label,**track_stats(pts,fit,a.asset),**pixel_stats(xy)))
                per_finger.extend(finger_rows(source,label,fit,a.asset,pts,mean))
            for label,fit in (("raw",coarse),("camera_only",stabilize(coarse,upstream=False)),
                              ("upstream_only",stabilize(coarse,cam=False)),("both",stabilize(coarse))):
                pts,xy,_=calculate(fit,a.asset,cam,temp)
                temporal.append(dict(source=source,variant=label,**track_stats(pts,fit,a.asset),**pixel_stats(xy),
                                    root_accel_p95_deg=p95(np.linalg.norm(angular(fit["global_orient"])[1],axis=-1)),
                                    camera_jerk_p95=p95(np.linalg.norm(np.diff(fit["cam_trans"],n=3,axis=0),axis=-1))))
                if source==cfg["focus"]:
                    np.savez_compressed(a.output/f"jitter_{label}.npz",xy=xy,points=pts)
                    if label=="raw":
                        diagnostic(source,fit,cam,a.asset,temp,a.output)
            print(source,"audited",flush=True)
    write_csv(a.output/"hand_summary.csv",rows)
    write_csv(a.output/"per_joint.csv",per_finger)
    write_csv(a.output/"temporal_ablation.csv",temporal)


if __name__=="__main__":
    main()
