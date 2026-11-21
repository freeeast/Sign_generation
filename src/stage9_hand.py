"""Structured per-digit controls and differentiable sparse SMPL-X forward skinning."""
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from smplx.joint_names import JOINT_NAMES
from smplx.lbs import batch_rodrigues, batch_rigid_transform
from smplx.vertex_ids import vertex_ids
import torch

from .smplx_to_pose import COCO_TO_SMPLX, load_model

DIGITS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("left", "right")
ARM = (16,17,18,19,20,21)
EDIT_2D = np.r_[5:11,91:133]
TIPS = np.array([[95,99,103,107,111],[116,120,124,128,132]])
WRIST = np.array([91,112])
BODY_GRAPH = np.r_[0,3:11]


def tensor(x):
    return torch.as_tensor(np.asarray(x),dtype=torch.float32)


def model_for(asset,fit):
    return load_model(Path(asset).resolve(),"neutral","model","cpu",
                      (("num_betas",fit["betas"].shape[-1]),("num_expression_coeffs",fit["expression"].shape[-1])))


def pack(fit):
    n=len(fit["body_pose"])
    extras=[fit.get(k,np.zeros((n,3),np.float32)) for k in ("jaw_pose","leye_pose","reye_pose")]
    return tensor(np.concatenate([fit["global_orient"],fit["body_pose"],*extras,
                                 fit["left_hand_pose"],fit["right_hand_pose"]],axis=1)).reshape(n,55,3)


def unit(v):
    return v/np.maximum(np.linalg.norm(v),1e-8)


def articulated_modes(model):
    with torch.no_grad():
        rest=(model.J_regressor @ model.v_template).numpy()
        mean=model.pose_mean.reshape(55,3).numpy()
    modes=[]; names=[]; bounds=[]
    def add(name, ids, directions, upper):
        axis=np.zeros((55,3),np.float32)
        axis[list(ids)]=directions
        modes.append(axis); names.append(name); bounds.append(upper)
    for j in ARM:
        for i,label in enumerate("xyz"):
            add(f"{JOINT_NAMES[j]}.{label}",[j],np.eye(3)[i],28 if j<20 else 35)
    for side in SIDES:
        idx=lambda name: JOINT_NAMES.index(side+"_"+name)
        middle=rest[idx("middle1")]-rest[idx("wrist")]
        across=rest[idx("index1")]-rest[idx("pinky1")]
        normal=unit(np.cross(middle,across))
        for f in DIGITS:
            ids=np.array([idx(f+str(i)) for i in (1,2,3)])
            direction=unit(rest[ids[1]]-rest[ids[0]])
            flex=unit(np.cross(direction,normal))
            if mean[ids[1:]].sum(axis=0)@flex<0:
                flex=-flex
            if f=="thumb":
                toward=unit(rest[idx("middle1")]-rest[ids[0]])
                opp=unit(np.cross(direction,toward)+.55*flex+.20*direction)
                add(side+".thumb.opposition",ids,[opp,.20*flex,.10*flex],44)
                add(side+".thumb.flex",ids,[.45*flex,flex,.65*flex],58)
                add(side+".thumb.spread",ids[:1],[normal],28)
            else:
                add(f"{side}.{f}.mcp_flex",ids[:1],[flex],58)
                add(f"{side}.{f}.pip_dip",ids[1:],[flex,.65*flex],68)
                add(f"{side}.{f}.spread",ids[:1],[normal],26)
    return names,tensor(modes),tensor(np.deg2rad(bounds))


class SparseFK:
    """Exact skinning for the 10 fingertips on the official mesh, plus mapped joints."""
    def __init__(self,fit,asset,camera,base_xy):
        self.fit=fit
        self.model=model_for(asset,fit)
        m=self.model
        self.base_axis=pack(fit)+m.pose_mean.reshape(1,55,3)
        self.base_rot=batch_rodrigues(self.base_axis.reshape(-1,3)).reshape(-1,55,3,3)
        self.camera=tensor(camera)
        self.translation=tensor(fit["cam_trans"])
        self.base_xy=tensor(base_xy).clone()
        self.identity=torch.eye(3)
        tip_ids=[vertex_ids["smplx"][side[0]+f] for side in SIDES for f in DIGITS]
        self.pose_dirs=m.posedirs.reshape(486,-1,3)[:,tip_ids].reshape(486,-1).contiguous()
        self.skin_weights=m.lbs_weights[tip_ids]
        tip_positions={(s+"_"+f):k for k,(s,f) in enumerate((s,f) for s in SIDES for f in DIGITS)}
        self.out_indices=[JOINT_NAMES.index(COCO_TO_SMPLX[i]) if COCO_TO_SMPLX[i] not in tip_positions
                          else 55+tip_positions[COCO_TO_SMPLX[i]] for i in EDIT_2D]
        with torch.no_grad():
            b=tensor(fit["betas"]); e=tensor(fit["expression"])
            shape_basis=torch.einsum("jv,vck->jck",m.J_regressor,m.shapedirs)
            expr_basis=torch.einsum("jv,vck->jck",m.J_regressor,m.expr_dirs)
            self.rest=m.J_regressor@m.v_template+torch.einsum("tk,jck->tjc",b,shape_basis)+torch.einsum("tk,jck->tjc",e,expr_basis)
            self.v_shaped=(m.v_template[tip_ids]+torch.einsum("tk,vck->tvc",b,m.shapedirs[tip_ids])
                           +torch.einsum("tk,vck->tvc",e,m.expr_dirs[tip_ids]))
        self.transl=tensor(fit.get("transl",np.zeros((len(self.base_rot),3),np.float32)))

    def points(self,rot,frames):
        joints,trans=batch_rigid_transform(rot,self.rest[frames],self.model.parents)
        offsets=((rot[:,1:]-self.identity).reshape(len(rot),-1)@self.pose_dirs).view(len(rot),10,3)
        v=offsets+self.v_shaped[frames]
        skin=(self.skin_weights@trans.reshape(len(rot),55,16)).reshape(len(rot),10,4,4)
        tips=(skin[...,:3,:3]@v[...,None])[...,0]+skin[...,:3,3]
        return torch.cat([joints,tips],dim=1)[:,self.out_indices]+self.transl[frames,None]

    def project(self,points,frames):
        camera=points+self.translation[frames,None]
        virtual=camera[...,:2]/camera[...,2:].clamp_min(.01)*tensor([5000/192,5000/256])+.5
        xy=self.base_xy[frames].clone()
        xy[:,EDIT_2D]=virtual@self.camera[:,:2].T+self.camera[:,2]
        return xy


class Controller:
    def __init__(self,fk,knot_spacing,max_knots):
        self.fk=fk
        self.mobility=None
        n=len(fk.base_rot)
        self.names,self.modes,self.bounds=articulated_modes(fk.model)
        k=min(max_knots,max(4,int(np.ceil((n-1)/knot_spacing))+1))
        self.knots=k
        self.interp=tensor(CubicSpline(np.linspace(0,n-1,k),np.eye(k),bc_type="natural")(np.arange(n)))

    def angles(self,x):
        return torch.tanh(self.interp@x)*self.bounds

    def rotations(self,x):
        delta=torch.einsum("td,dji->tji",self.angles(x),self.modes)
        if self.mobility is not None:
            delta=delta*self.mobility[...,None]
        rot=batch_rodrigues(delta.reshape(-1,3)).reshape(-1,55,3,3)
        return self.fk.base_rot @ rot

    def zero(self):
        return torch.zeros(self.knots,len(self.names))

    def export(self,x):
        from scipy.spatial.transform import Rotation
        rot=self.rotations(x).detach().numpy()
        aa=Rotation.from_matrix(rot.reshape(-1,3,3)).as_rotvec().reshape(-1,55,3)
        # Subtract model mean because official SMPL-X adds it back before Rodrigues.
        raw=aa-self.fk.model.pose_mean.reshape(55,3).numpy()
        out=dict(self.fk.fit)
        out["body_pose"]=raw[:,1:22].reshape(-1,63).astype(np.float32)
        out["left_hand_pose"]=raw[:,25:40].reshape(-1,45).astype(np.float32)
        out["right_hand_pose"]=raw[:,40:55].reshape(-1,45).astype(np.float32)
        return out
