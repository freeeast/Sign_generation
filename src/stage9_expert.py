"""Differentiable forward through the same frozen Stage 7 bilateral graph."""
import numpy as np
import torch

from .stage7_feedback import HANDS, hand_quality
from .stage9_hand import BODY_GRAPH, tensor


def cosine(a,b):
    x=a.flatten(start_dim=-2)
    y=b.flatten(start_dim=-2)
    return 1-torch.nn.functional.cosine_similarity(x,y,dim=-1,eps=1e-8)


class DifferentiableResponse:
    def __init__(self, response, cfg):
        self.m=response.graphs.adapter.model
        self.m.requires_grad_(False)
        self.real=response.valid
        self.supported=torch.as_tensor(response.supported)
        self.ref=tuple(tensor(x.copy()) for x in response.reference)
        self.weight=tensor(cfg["factor_weights"])

    def terms(self,xy):
        detached=xy.detach().numpy()
        in_bounds=np.isfinite(detached).all(-1)&((detached>=0)&(detached<=1)).all(-1)
        vis=self.real&in_bounds
        present=torch.as_tensor(hand_quality(in_bounds))
        mask=torch.as_tensor(vis[:,BODY_GRAPH])
        body_xy=xy[:,BODY_GRAPH]
        valid=body_xy[mask]
        if len(valid)<4:
            raise ValueError("Insufficient visible body graph points")
        lo,hi=valid.amin(dim=0),valid.amax(dim=0)
        scale=(hi-lo).max().clamp_min(1e-6)
        norm_body=(body_xy-(hi+lo-scale)/2)/scale*2-1
        body=torch.cat([norm_body,mask[...,None].float()],dim=-1).clamp(-1,1)
        body=body*mask[...,None]
        m=self.m
        body=m.gcn_modules["body"](m.proj_linear["body"](body[None]).permute(0,3,1,2))
        spaces=[]; times=[]
        for name,ids,j in zip(("left","right"),HANDS,(-2,-1)):
            x=xy[:,ids]
            seen=torch.as_tensor(vis[:,ids]&vis[:,ids[0],None])
            h=torch.cat([(x-x[:,0,None])/scale,seen[...,None].float()],dim=-1).clamp(-1,1)
            h=h*seen[...,None]
            s=m.gcn_modules[name](m.proj_linear[name](h[None]).permute(0,3,1,2))
            t=m.fusion_gcn_modules[name](s+body[...,j][...,None])
            spaces.append(s[0].permute(1,2,0)); times.append(t[0].permute(1,2,0))
        s,t=torch.stack(spaces,dim=1),torch.stack(times,dim=1)
        present=present & self.supported
        def dist(a,b,p,v,w):
            value=torch.where(v,cosine(a,b).clamp(max=1),1.)
            return (value*w*p).sum()/(w*p).sum().clamp_min(1)
        ones=torch.ones_like(self.supported,dtype=torch.float32)
        spatial=dist(s,self.ref[0],self.supported,present,ones)
        temporal=dist(t,self.ref[1],self.supported,present,ones)
        means=[]
        for width in (4,12):
            a=[]; b=[]; ok=[]; has=[]; w=[]
            for j in range(0,len(t),width):
                sl=slice(j,j+width)
                rs=self.supported[sl]; ps=present[sl]
                a.append((t[sl]*ps[...,None,None]).sum(dim=0))
                b.append((self.ref[1][sl]*rs[...,None,None]).sum(dim=0))
                n=rs.sum(dim=0)
                has.append(n>0); ok.append(ps.sum(dim=0)/n.clamp_min(1)>=.8); w.append(n)
            means.append(dist(torch.stack(a),torch.stack(b),torch.stack(has),torch.stack(ok),torch.stack(w)))
        return torch.stack((spatial,temporal,torch.stack(means).mean()))

    def value(self,xy,which):
        terms=self.terms(xy)
        if which=="factor":
            return (terms*self.weight).sum()
        return terms[("spatial","temporal","multiscale").index(which)]
