"""SO(3) motion diagnostics and localized low-amplitude stabilization."""
import numpy as np
from scipy.ndimage import convolve1d
from scipy.spatial.transform import Rotation

from .stage9_hand import DIGITS, SIDES, model_for

KERNEL=np.array([1,2,3,2,1],float)/9


def rotvec_safe(m):
    return Rotation.from_matrix(m.reshape(-1,3,3)).as_rotvec().reshape(*m.shape[:-2],3)


def rotation_matrices(v):
    return Rotation.from_rotvec(v.reshape(-1,3)).as_matrix().reshape(*v.shape[:-1],3,3)


def smooth_rot(v):
    rots=rotation_matrices(v)
    accum=np.zeros_like(v,dtype=float)
    for i,w in enumerate(KERNEL):
        if i==2:
            continue
        take=np.clip(np.arange(len(v))+i-2,0,len(v)-1)
        accum+=w*rotvec_safe(np.swapaxes(rots,-1,-2)@rots[take])
    return rotvec_safe(rots@rotation_matrices(accum)).astype(v.dtype)


def stabilize(fit,cam=True,upstream=True):
    out=dict(fit)
    if cam:
        out["cam_trans"]=convolve1d(fit["cam_trans"],KERNEL,axis=0,mode="nearest")
    if upstream:
        out["global_orient"]=smooth_rot(fit["global_orient"])
        axis=fit["body_pose"].copy().reshape(-1,21,3)
        inherited=[2,5,8,11,12,13,14]
        axis[:,inherited]=smooth_rot(axis[:,inherited])
        out["body_pose"]=axis.reshape(-1,63)
    return out


def angular(v):
    mat=rotation_matrices(v)
    speed=rotvec_safe(np.swapaxes(mat[:-1],-1,-2)@mat[1:])*180/np.pi
    return speed, np.diff(speed,axis=0), np.diff(speed,n=2,axis=0)


def p95(a):
    return float(np.percentile(a,95))


def wrist_frames(fit,asset):
    model=model_for(asset,fit)
    base=np.concatenate([fit["global_orient"],fit["body_pose"]],axis=1).reshape(-1,22,3)
    mats=rotation_matrices(base)
    global_r=[]
    for j,parent in enumerate(model.parents[:22].tolist()):
        global_r.append(mats[:,j] if parent<0 else global_r[parent]@mats[:,j])
    return np.stack([global_r[j] for j in (20,21)],axis=1)


def wrist_local(points,fit,asset):
    local=[]
    wr=wrist_frames(fit,asset)
    for h,(start,end) in enumerate(((91,112),(112,133))):
        v=points[:,start:end]-points[:,start,None]
        local.append(v@wr[:,h])
    return np.stack(local,axis=1)


def track_stats(points,fit,asset):
    local=wrist_local(points,fit,asset)
    tips=local[:,:,4::4]
    aperture=np.linalg.norm(tips[:,:,0]-tips[:,:,1],axis=-1)
    cent=tips-np.median(tips,axis=0)
    speed=np.linalg.norm(np.diff(tips,axis=0),axis=-1)
    return dict(finger_motion_p95_mm=p95(np.linalg.norm(cent,axis=-1))*1000,
                finger_speed_p95_mm_s=p95(speed)*24000,
                finger_jerk_p95_mm_frame3=p95(np.linalg.norm(np.diff(tips,n=3,axis=0),axis=-1))*1000,
                aperture_range_mm=float(np.mean(np.ptp(aperture,axis=0))*1000),
                aperture_speed_p95_mm_frame=p95(np.abs(np.diff(aperture,axis=0)))*1000,
                wrist_jerk_p95_mm_frame3=p95(np.linalg.norm(np.diff(points[:,[91,112]],n=3,axis=0),axis=-1))*1000,
                fingertip_jerk_p95_mm_frame3=p95(np.linalg.norm(np.diff(points[:,[95,99,103,107,111,116,120,124,128,132]],n=3,axis=0),axis=-1))*1000)
