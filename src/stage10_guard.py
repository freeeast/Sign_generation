"""Confidence-supported, per-hand safe refinements with an immutable geometry anchor."""
import numpy as np
from scipy.ndimage import convolve1d
from scipy.spatial.transform import Rotation
import torch

from .stage9_hand import tensor
from .stage9_timing import rotvec_safe

WRISTS = np.array([91,112])
FINGERS = np.array([np.arange(91,112),np.arange(112,133)])
TIPS = np.arange(4,21,4)
SIZE = np.array([1280,720])
ARM_HAND = np.r_[16:22,25:55]


def norm(x):
    return np.linalg.norm(x,axis=-1)


def angle_dist(m):
    return np.rad2deg(np.linalg.norm(rotvec_safe(m),axis=-1))


def relative(a,b):
    return np.swapaxes(a,-1,-2)@b


def measure_hands(x):
    local=x-x[:,:,:1]
    scale=np.maximum(norm(local[:,:,9]),4.)
    aperture=norm(local[:,:,4]-local[:,:,20])/scale
    segments=np.diff(x[:,:,1:].reshape(*x.shape[:2],5,4,2),axis=3)
    directions=segments/np.maximum(norm(segments)[...,None],1e-6)
    dot=np.sum(directions[:,:,:,:-1]*directions[:,:,:,1:],axis=-1)
    bend=np.rad2deg(np.arccos(np.clip(dot,-1,1))).mean(-1)
    return local, aperture, bend


class Guard:
    def __init__(self, cfg, reference, confidence, indices, anchor_xy, anchor_xyz, base_rot):
        self.cfg=cfg
        self.anchor=anchor_xy[:,FINGERS]*SIZE
        self.anchor_xyz=anchor_xyz[:,6:].reshape(-1,2,21,3)
        self.ref=reference[:,FINGERS]*SIZE
        xy=reference[:,FINGERS]
        scores=confidence[:,FINGERS]
        valid=(scores>=cfg["minimum_score"])&np.all(np.isfinite(xy)&(xy>=0)&(xy<=1),axis=-1)
        valid=valid&(convolve1d(valid.astype(int),np.ones(3,int),axis=0,mode="constant")>=cfg["confidence_neighbors"])
        self.valid=valid
        self.base_rot=base_rot
        self.angle_limit=np.maximum(angle_dist(base_rot[:,ARM_HAND]),[158]*4+[120]*2+[130]*30)
        self.speed_limit=np.maximum(angle_dist(relative(base_rot[:-1,ARM_HAND],base_rot[1:,ARM_HAND])),[15]*6+[16]*30)
        fingers=base_rot[:,25:]
        self.acceleration_limit=np.maximum(norm(np.diff(rotvec_safe(relative(fingers[:-1],fingers[1:])),axis=0))*180/np.pi,2.5)
        self.step=np.diff(indices)[:,None,None]
        self.ref_stats=self.observation(self.ref)
        self.anchor_error=self.errors(self.anchor)
        self.anchor_local=self.local3d(self.anchor_xyz,base_rot[indices])
        self.base_clearance=self.clearance(self.anchor_local)

    @staticmethod
    def local3d(p,rot):
        # Wrist-local SO(3) view is unnecessary for scalar inter-digit clearances,
        # and this transform is used only for relative shape after the update.
        return p-p[:,:,:1]

    @staticmethod
    def clearance(hand):
        # Geometry-only proxy between adjacent non-thumb distal phalanges.
        middle=(hand[:,:,7:21:4]+hand[:,:,8:21:4])/2
        return norm(np.diff(middle,axis=2))*1000

    def observation(self,x):
        loc,ap,bend=measure_hands(x)
        v=loc[1:]-loc[:-1]
        return loc,ap,bend,v/self.step[...,None]

    def errors(self,x):
        l,ap,bend,v=self.observation(x)
        re,ra,rb,rv=self.ref_stats
        return dict(wrist=norm(x[:,:,0]-self.ref[:,:,0]),finger=norm(x-self.ref),
                    shape=norm(l-re),motion=norm(v-rv),aperture=np.abs(ap-ra),bend=np.abs(bend-rb))

    def screen(self,rot,xy,xyz,strict):
        r=rot[:,ARM_HAND]
        if (angle_dist(r)>self.angle_limit+.06).any():
            return "absolute_angle"
        if (angle_dist(relative(r[:-1],r[1:]))>self.speed_limit+.06).any():
            return "angular_speed"
        change=angle_dist(relative(self.base_rot,rot))
        if np.max(change[:,25:])>self.cfg["local_rotation_deg"] or np.max(change[:,20:22])>self.cfg["wrist_change_deg"]:
            return "angular_trust"
        f=rot[:,25:]
        acc=norm(np.diff(rotvec_safe(relative(f[:-1],f[1:])),axis=0))*180/np.pi
        if (acc>self.acceleration_limit+self.cfg["extra_finger_acceleration_deg_frame2"]).any():
            return "finger_acceleration"
        if not strict:
            return ""
        p=xy[:,FINGERS]*SIZE
        l=self.local3d(xyz[:,6:].reshape(-1,2,21,3),None)
        if (self.clearance(l)<np.minimum(self.base_clearance-0.25,5.0)).any():
            return "distal_clearance_proxy"
        # Global wrist location can change independently of calibrated local shape.
        if (norm(p[:,:,0]-self.anchor[:,:,0])>self.cfg["wrist_change_px"]).any():
            return "wrist_trust"
        base=self.anchor_local[:,:,TIPS]
        if (norm(l[:,:,TIPS]-base)*1000>self.cfg["local_tip_change_mm"]).any():
            return "local_tip_trust"
        excess={k:v-self.anchor_error[k] for k,v in self.errors(p).items()}
        v=self.valid
        wrist=v[:,:,0]
        finger=v&v[:,:,:1]
        ap=v[:,:,4]&v[:,:,20]&v[:,:,9]&wrist
        digits=v[:,:,1:].reshape(*v.shape[:2],5,4).all(3)&wrist[...,None]
        masks={"wrist":wrist,"finger":v,"shape":finger,"motion":finger[1:]&finger[:-1],
               "aperture":ap,"bend":digits}
        limits={"wrist":self.cfg["wrist_error_px"],"finger":self.cfg["finger_error_px"],
                "shape":self.cfg["shape_error_px"],"motion":self.cfg["motion_error_px_frame"],
                "aperture":self.cfg["aperture_error"],"bend":self.cfg["flex_error_deg"]}
        for key in limits:
            # Window support stays local to each hand and identity of each landmark.
            n=self.cfg["window_samples"]
            m=masks[key]
            samples=convolve1d(m.astype(float),np.ones(n),axis=0,mode="constant")
            smooth=convolve1d(np.where(m,excess[key],0),np.ones(n),axis=0,mode="constant")
            event=(samples>=2)&(smooth/np.maximum(samples,1)>limits[key])
            if event.any():
                return key+"_window"
        worst=(excess["shape"][:,:,TIPS]>self.cfg["tip_window_px"])&finger[:,:,TIPS]
        return "visible_tip_spike" if worst.any() else ""


def compact_basis(n, spacing):
    knots=np.unique(np.r_[np.arange(0,n,spacing),n-1])
    d=np.abs(np.arange(n)[:,None]-knots[None])/spacing
    values=np.where(d<1,(4-6*d*d+3*d**3)/6,np.maximum(2-d,0)**3/6)
    return knots,tensor(values/values.sum(1,keepdims=True))


def blocks(names):
    result={}
    for j,name in enumerate(names):
        key=name.rsplit(".",1)[0]
        if key.endswith("_wrist") or "." in key:
            result.setdefault(key,[]).append(j)
    return result
