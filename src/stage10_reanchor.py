"""Use a neutral SMPL-X hand as a fallback only where dense 2D evidence supports it."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from .stage10_guard import FINGERS, norm
from .stage9_timing import rotation_matrices, rotvec_safe, angular


def neutral_reanchor(search, fit, coarse, obs, confidence, cfg):
    fk=search.fk
    n=len(obs)
    ids=np.arange(0,n,2)
    scores=confidence[ids][:,FINGERS]
    real=obs[ids][:,FINGERS]*[1280,720]
    valid=(scores>=cfg["minimum_score"])&(scores[:,:,:1]>=cfg["minimum_score"])
    valid&=np.all(np.isfinite(real)&(real>=0)&(real<=[1280,720]),axis=-1)
    def measure(rot):
        with torch.no_grad():
            q=fk.project(fk.points(rot[ids],ids),ids).numpy()[:,FINGERS]*[1280,720]
        err=q-real
        penalty=.72*norm(err-err[:,:,:1])+.28*norm(err)
        return (penalty*valid).sum(-1)/valid.sum(-1).clip(1)
    start=fk.base_rot.clone()
    neutral=start.clone()
    hand_mean=fk.model.pose_mean.reshape(55,3)[25:].numpy()
    neutral[:,25:]=torch.from_numpy(rotation_matrices(hand_mean)).to(start.dtype)
    # Try also the wrist orientation from the earlier 2D-only IK.
    # Transfer the world rotation through the current elbow parent, preserving the wrist position.
    def global_body(axis):
        local=rotation_matrices(axis)
        world=[]
        for j,parent in enumerate(fk.model.parents[:22].tolist()):
            world.append(local[:,j] if parent<0 else world[parent]@local[:,j])
        return world
    origin=global_body(np.concatenate([fit["global_orient"],fit["body_pose"]],axis=1).reshape(n,22,3))
    early=global_body(np.concatenate([coarse["global_orient"],coarse["body_pose"]],axis=1).reshape(n,22,3))
    orientation=neutral.clone()
    for j in (20,21):
        parent=int(fk.model.parents[j])
        r=origin[parent].transpose(0,2,1)@early[j]
        orientation[:,j]=torch.from_numpy(r).to(start.dtype)
    current=measure(start)
    reset=measure(neutral)
    rotated=measure(orientation)
    pick=gaussian_filter1d(np.clip((reset-rotated)/8,0,1),1.0,axis=0,mode="nearest")
    pick=np.stack([np.interp(np.arange(n),ids,pick[:,h]) for h in range(2)],axis=1)
    best=torch.clone(neutral)
    for i,j in enumerate((20,21)):
        a=neutral[:,j].numpy()
        d=rotvec_safe(a.transpose(0,2,1)@orientation[:,j].numpy())
        best[:,j]=torch.from_numpy(a@rotation_matrices(d*pick[:,i,None])).to(start.dtype)
    reset=measure(best)
    reliable=valid.sum(-1)>=12
    support=np.clip((current-reset-2)/10,0,1)*reliable
    support=gaussian_filter1d(support,1.0,axis=0,mode="nearest")
    alpha=np.stack([np.interp(np.arange(n),ids,support[:,h]) for h in range(2)],axis=1)
    alpha=gaussian_filter1d(alpha,1.20,axis=0,mode="nearest")
    result=dict(fit)
    ledger={"wrist_hypothesis_weight":pick,"neutral_ik_endpoint_errors_px":rotated}
    joints=fit["body_pose"].copy().reshape(n,21,3)
    for i,side in enumerate(("left","right")):
        key=side+"_hand_pose"
        raw=fit[key].reshape(n,15,3)
        mean=hand_mean[i*15:(i+1)*15]
        wrist=20+i
        raw_hand=raw+mean
        source=np.concatenate([fit["body_pose"].reshape(n,21,3)[:,wrist-1:wrist],raw_hand],axis=1)
        base=rotation_matrices(source)
        target=np.concatenate([best[:,wrist:wrist+1].numpy(),best[:,25+i*15:40+i*15].numpy()],axis=1)
        tangent=rotvec_safe(base.transpose(0,1,3,2)@target)
        base_accel=np.linalg.norm(angular(source)[1],axis=-1)
        base_speed=np.linalg.norm(angular(source)[0],axis=-1)
        scale=1.
        for _ in range(12):
            after=rotvec_safe(base@rotation_matrices(tangent*alpha[:,i,None,None]*scale))
            v,a,_=angular(after)
            if (norm(v)<=np.maximum(base_speed,16)+.06).all() and (norm(a)<=np.maximum(base_accel,2.5)+1.0).all():
                break
            scale*=.75
        else:
            after=source.copy()
            scale=0.
        ledger[side+"_weight"]=alpha[:,i]*scale
        result[key]=(after[:,1:]-mean).astype(raw.dtype).reshape(n,45)
        joints[:,wrist-1]=after[:,0]
        start[:,wrist]=torch.from_numpy(rotation_matrices(after[:,0])).to(start.dtype)
        start[:,25+i*15:40+i*15]=torch.from_numpy(rotation_matrices(after[:,1:])).to(start.dtype)
    result["body_pose"]=joints.reshape(n,63)
    final=measure(start)
    ledger.update(even_frames=ids,anchor_errors_px=current,neutral_errors_px=reset,reanchored_errors_px=final,visible=valid)
    if (final*reliable).sum()>=(current*reliable).sum():
        return fit,ledger,False
    return result,ledger,True
