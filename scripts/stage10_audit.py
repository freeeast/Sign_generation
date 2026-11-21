"""Replay focal Stage 9 parameters and localize body and handshape regression."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.stage7_optimize import read,write
from scripts.stage9_audit import calculate
from scripts.stage10_inputs import clip_inputs
from src.evaluate_feedback import write_csv
from src.stage9_hand import Controller,SparseFK,tensor
from src.stage9_timing import wrist_local,stabilize
from src.stage10_world import body_joints,body_metrics,fixed_world,stabilized_body
from src.virtual_pose import load_fit

HANDS=np.array([np.arange(91,112),np.arange(112,133)])
TIPS=[4,8,12,16,20]
W=np.array([1280,720])


def hand_shape(p):
    local=p-p[:,:1]
    scale=np.maximum(np.linalg.norm(local[:,9],axis=-1),1e-6)
    angles=[]
    for j in range(5):
        segment=p[:,1+j*4:5+j*4]
        dirs=np.diff(segment,axis=1)
        dirs=dirs/np.maximum(np.linalg.norm(dirs,axis=-1,keepdims=True),1e-6)
        angles.append(np.rad2deg(np.arccos(np.clip((dirs[:,:-1]*dirs[:,1:]).sum(-1),-1,1))).mean(-1))
    return scale,np.stack(angles,axis=1),np.linalg.norm(local[:,4]-local[:,20],axis=-1)/scale


def snapshot(variant, world, xy, asset, obs, conf, frames, fit=None):
    rows=[]
    for i,side in enumerate(("left","right")):
        pick=HANDS[i]
        p=xy[frames][:,pick]*W
        o=obs[frames][:,pick]*W
        scale,bend,opening=hand_shape(p)
        err=np.linalg.norm((p[:,TIPS]-p[:,:1])-(o[:,TIPS]-o[:,:1]),axis=-1)
        world_h=world[frames][:,pick] if world is not None else None
        if world_h is not None:
            x=world_h[:,5]-world_h[:,0]; y=world_h[:,17]-world_h[:,0]
            normal=np.cross(x,y); normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-6)
            _,world_bend,world_opening=hand_shape(world_h)
        for k,j in enumerate(frames):
            row=dict(variant=variant,frame=int(j),side=side,palm_px=scale[k],opening_2d=opening[k],
                     mean_segment_bend_2d_deg=bend[k,1:].mean(),wrist_x=p[k,0,0],wrist_y=p[k,0,1],
                     wrist_error_px=np.linalg.norm(p[k,0]-o[k,0]),mean_tip_relative_error_px=err[k].mean(),
                     min_score=np.min(conf[j,pick]),vis_count=int((conf[j,pick]>=.6).sum()))
            row.update({f"digit_{f}_tip_error_px":err[k,f] for f in range(5)})
            if world_h is not None:
                row.update(palm_normal_camera_x=normal[k,0],palm_normal_camera_y=normal[k,1],
                           palm_normal_camera_z=normal[k,2],opening_3d=world_opening[k],
                           mean_segment_bend_3d_deg=world_bend[k,1:].mean())
            rows.append(row)
    return rows


def audit(args,cfg):
    output=args.output/"audit"
    output.mkdir(parents=True,exist_ok=True)
    coarse,nsa,idx,obs,conf,aff,mp,s9,_=clip_inputs(cfg,cfg["focus"],args.asset,output)
    baseline=s9/"B/art_joint_factor"
    joint=load_fit(baseline/"final_fit.npz")
    geo=load_fit(s9/"B/art_joint_geo_factor/final_fit.npz")
    fixed=fixed_world(joint)
    stable=stabilized_body(fixed,idx,conf[idx],cfg)
    common=fixed["cam_trans"][0]
    np.savez_compressed(output/"focus_world_variants.npz",extrinsic_t=common,extrinsic_R=np.eye(3),affine=aff,
                        legacy_cam_trans=joint["cam_trans"],body_translation=fixed["transl"],
                        stabilized_translation=stable["transl"])
    rows=[]; hand=[]; trace=[]; timings={}
    with TemporaryDirectory() as td:
        p=Path(td)/"motion.npz"
        old_xy=None
        for m,fit in (("legacy",joint),("fixed_gauge",fixed),("stabilized",stable),
                      ("stable_legs_only",dict(fixed,body_pose=stable["body_pose"])),
                      ("stable_root_only",dict(fixed,transl=stable["transl"],global_orient=stable["global_orient"]))):
            xyz,xy,_=calculate(fit,args.asset,aff,p)
            if old_xy is None:
                old_xy=xy
            j=body_joints(fit,args.asset,aff,xy)
            rows.append(dict(method=m,projection_max_diff_from_legacy_px=float(np.max(np.abs((xy-old_xy)*W))),
                             **body_metrics(fit,j,xyz,xy,aff)))
            timings[m+"_pelvis"]=(j+fit["cam_trans"][:,None])[:,0]
            timings[m+"_knee"]=(j+fit["cam_trans"][:,None])[:,[4,5]]
            np.savez_compressed(output/f"{m}_fit.npz",**fit)
        body=joint["body_pose"].reshape(-1,21,3)
        from src.stage9_timing import angular
        roles={"hips":[0,1],"knees":[3,4],"ankles":[6,7]}
        write(output/"chain.json",dict(n=len(joint["body_pose"]),fps=24,input_has_transl="transl" in joint,
              legacy_translation_range=joint["cam_trans"].ptp(0).tolist() if hasattr(joint["cam_trans"],"ptp") and np.__version__.startswith("1.") else np.ptp(joint["cam_trans"],axis=0).tolist(),
              affine=aff.tolist(),constant_offset=common.tolist(),focal_crop=[5000/192,5000/256],
              root_orientation_jerk_p95_deg_frame3=float(np.percentile(np.linalg.norm(angular(joint["global_orient"])[2],axis=-1),95)),
              local_leg_angular_jerk_p95_deg_frame3={name:float(np.percentile(np.linalg.norm(angular(body[:,ids])[2],axis=-1),95)) for name,ids in roles.items()},
              frame90_in_optimization=bool(90 in idx),nearest_optimization_frames=idx[np.abs(idx-90).argsort()[:4]].tolist()))
        frames=np.array([86,88,89,90,91,92,94,96])
        for m,fit in (("coarse",coarse),("S9_geometry",geo),("S9_joint",joint),("NSA",nsa)):
            xyz,xy,_=calculate(fit,args.asset,aff,p)
            hand.extend(snapshot(m,xyz,xy,args.asset,obs,conf,frames,fit))
        hand+=snapshot("real_RTMW",None,obs,args.asset,obs,conf,frames)
        oldbase=stabilize(coarse)
        _,xy,_=calculate(oldbase,args.asset,aff,p)
        fk=SparseFK(oldbase,args.asset,aff,xy)
        s9cfg=read(Path("configs/stage9.json"))
        ctrl=Controller(fk,s9cfg["knot_spacing"],s9cfg["max_knots"])
        indices=np.r_[86:97]
        with np.load(baseline/"control_trajectory.npz") as a:
            cs,ks=a["coefs"],a["kind"]
        from scripts.stage9_audit import sample_fit
        for i,(coef,kind) in enumerate(zip(cs,ks)):
            fit=coarse if kind=="raw" else ctrl.export(tensor(coef))
            xyz,xy,_=calculate(sample_fit(fit,indices),args.asset,aff,p)
            local=90-indices[0]
            pleft=xy[local,HANDS[0]]*W
            _,bend,aperture=hand_shape(pleft[None])
            q=xyz[local,HANDS[0]]
            _,b3,a3=hand_shape(q[None])
            trace.append(dict(proposal=i,frame=90,opening_px_ratio=float(aperture[0]),opening_3d=float(a3[0]),
                 bend_deg_3d=float(b3[0,1:].mean()),wrist_error_px=float(np.linalg.norm((xy[local,91]-obs[90,91])*W))))
    write_csv(output/"body_decomposition.csv",rows)
    write_csv(output/"handshape_frame90.csv",hand)
    write_csv(output/"joint_frame90_replay.csv",trace)
    np.savez_compressed(output/"world_trajectories.npz",**timings)
    print(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    audit(a,read(Path("configs/stage10.json")))


if __name__=="__main__":
    main()
