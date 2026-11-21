"""Compact history-conditioned prediction of local bilateral SMPL-X rotations."""
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from scripts.shared_motion_data import ACTIVE,canonical,restore


def hand_mean(asset):
    with np.load(asset,allow_pickle=False) as a:
        m=np.zeros((55,3),np.float32)
        m[25:40]=a["hands_meanl"].reshape(15,3)
        m[40:55]=a["hands_meanr"].reshape(15,3)
    return m


def rotations(fit,means,frames=None):
    q=canonical(fit)
    if frames is not None:q=q[frames]
    q=q[:,ACTIVE]+means[ACTIVE]
    return Rotation.from_rotvec(q.reshape(-1,3)).as_matrix().reshape(*q.shape[:-1],3,3).astype(np.float32)


def decode(fit,r,means):
    q=canonical(fit)
    axis=Rotation.from_matrix(r.reshape(-1,3,3)).as_rotvec().reshape(len(r),len(ACTIVE),3)
    q[:,ACTIVE]=axis-means[ACTIVE]
    return restore(fit,q)


def increments(r):
    rr=r[:-1].swapaxes(-1,-2)@r[1:]
    d=Rotation.from_matrix(rr.reshape(-1,3,3)).as_rotvec().reshape(len(r)-1,len(ACTIVE),3).astype(np.float32)
    return np.concatenate([np.zeros_like(d[:1]),d])


def exp(d):
    x,y,z=d.unbind(-1)
    zero=torch.zeros_like(x)
    k=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*x.shape,3,3)
    n=torch.sqrt((d*d).sum(-1).clamp_min(1e-12))
    s=torch.sinc(n/torch.pi)
    c=.5*torch.sinc(n/(2*torch.pi)).square()
    return torch.eye(3,device=d.device,dtype=d.dtype)+s[...,None,None]*k+c[...,None,None]*(k@k)


def log(r):
    w=.5*torch.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),-1)
    sin=torch.sqrt((w*w).sum(-1).clamp_min(1e-12))
    cos=(r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
    return w*(torch.atan2(sin,cos)/sin)[...,None]


def six(r):
    return r[...,:2].transpose(-1,-2).flatten(start_dim=-2)


def six_to_matrix(s):
    a=torch.nn.functional.normalize(s[...,:3],dim=-1)
    b=s[...,3:6]
    b=torch.nn.functional.normalize(b-(a*b).sum(-1,keepdim=True)*a,dim=-1)
    return torch.stack((a,b,torch.linalg.cross(a,b,dim=-1)),dim=-1)


class Predictor(nn.Module):
    def __init__(self,scale,width=128):
        super().__init__()
        j=len(ACTIVE)
        self.register_buffer("scale",torch.as_tensor(scale,dtype=torch.float32))
        self.input=nn.Linear(j*9,width)
        self.gru=nn.GRU(width,width,batch_first=True)
        self.out=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,j*3))

    def forward(self,r,d,h=None):
        f=torch.cat((six(r),(d/self.scale).clamp(-8,8)),-1).flatten(start_dim=-2)
        f=torch.nn.functional.gelu(self.input(f))
        o,h=self.gru(f,h)
        raw=self.out(o[:,-1]).reshape(-1,len(ACTIVE),3)
        update=4*torch.tanh(raw/4)*self.scale
        return update,h

    def rollout(self,r,d,n):
        pred,h=self(r,d)
        state=r[:,-1]
        rs=[];ds=[]
        for i in range(n):
            state=state@exp(pred)
            rs.append(state);ds.append(pred)
            if i+1<n:pred,h=self(state[:,None],pred[:,None],h)
        return torch.stack(rs,1),torch.stack(ds,1)


def load_predictor(checkpoint):
    c=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model=Predictor(c["scale"],c["width"])
    model.load_state_dict(c["model"])
    return model.eval().requires_grad_(False),c


def geodesic(x,y):
    m=x.transpose(-1,-2)@y
    a=(m.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    return torch.rad2deg(torch.acos(a.clamp(-1,1)))


def baseline(r,d,h,damp):
    pose=r[:,-1]
    pred=d[:,-1]
    ans=[];ds=[]
    for i in range(h):
        vel=pred*(damp**i)
        pose=pose@exp(vel)
        ans.append(pose);ds.append(vel)
    return torch.stack(ans,1),torch.stack(ds,1)
