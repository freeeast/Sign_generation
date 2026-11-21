"""Frame validity, fixed observation support, and bilateral Uni-Sign graph features."""
import numpy as np

from .pose import LEFT, RIGHT

HANDS=(LEFT,RIGHT)
WRISTS=np.array([91,112])
TIPS=np.array([[95,99,103,107,111],[116,120,124,128,132]])


def in_frame(xy):
    return np.isfinite(xy).all(axis=-1) & ((xy>=0)&(xy<=1)).all(axis=-1)


def valid_pose(xy, scores, confidence):
    return in_frame(xy) & np.isfinite(scores) & (scores>confidence)


def hand_quality(valid, minimum=11):
    return np.column_stack([valid[:, ids[0]] & (valid[:, ids].sum(axis=1)>=minimum) for ids in HANDS])


def expert_input(xy, scores, valid):
    # Hand and face centering in the original adapter precedes masking.
    # An invalid anchor would therefore shift all surviving nodes.
    mask=valid.copy()
    for ids,anchor in [(LEFT,91),(RIGHT,112),(np.arange(23,91),53)]:
        mask[:,ids] &= mask[:,anchor,None]
    return np.where(mask[...,None],xy,0),np.where(mask,scores,0)


class HandGraphs:
    """Use official weights with independently named left/right calls.

    Uni-Sign aliases left and right modules to the same objects. Two hooks on
    either object fire for both calls. Direct named calls preserve the outputs.
    """
    def __init__(self, adapter):
        self.adapter=adapter

    def extract(self, xy, score):
        a=self.adapter
        m=a.model
        with a.torch.inference_mode():
            parts=a.load_part_kp(xy[:,None],score[:,None])
            body=m.proj_linear["body"](parts["body"][None].to(a.device,dtype=a.torch.float32))
            body=m.gcn_modules["body"](body.permute(0,3,1,2))
            spatial,temporal=[],[]
            for part,joint in (("left",-2),("right",-1)):
                f=m.proj_linear[part](parts[part][None].to(a.device,dtype=a.torch.float32))
                f=m.gcn_modules[part](f.permute(0,3,1,2))
                spatial.append(f[0].permute(1,2,0).cpu().numpy())
                t=m.fusion_gcn_modules[part](f+body[...,joint][...,None])
                temporal.append(t[0].permute(1,2,0).cpu().numpy())
        return np.stack(spatial,axis=1),np.stack(temporal,axis=1)


def cosine_error(a,b):
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return 1-np.divide((a*b).sum(axis=-1),den,out=np.zeros_like(den),where=den>1e-8)


def feature_scores(candidate, reference, supported, present, reliability, penalty):
    """Score against fixed real support. Missing virtual regions carry a cost."""
    spatial,temporal=candidate
    ref_spatial,ref_temporal=reference
    denom=supported.sum()
    if not denom:
        return 0.,0.,0.,0.
    inputs=(spatial.reshape(*spatial.shape[:2],-1),temporal.reshape(*temporal.shape[:2],-1))
    refs=(ref_spatial.reshape(*ref_spatial.shape[:2],-1),ref_temporal.reshape(*ref_temporal.shape[:2],-1))
    def mean(error, given, visible, weight):
        total=weight[given].sum()
        if total<=1e-9:
            return 0.
        measured=np.where(visible,np.minimum(error,penalty),penalty)
        return float(np.sum(measured[given]*weight[given])/total)
    spatial_loss=mean(cosine_error(inputs[0],refs[0]),supported,present,reliability)
    temporal_loss=mean(cosine_error(inputs[1],refs[1]),supported,present,reliability)
    scales=[]
    for width in (4,12):
        a,b,s,p,w=[],[],[],[],[]
        for start in range(0,len(supported),width):
            sl=slice(start,start+width)
            real=supported[sl]
            same=real&present[sl]
            # Keep the reference support fixed; missing contributes zero feature, then gets an explicit penalty.
            a.append((inputs[1][sl]*same[...,None]).sum(axis=0))
            b.append((refs[1][sl]*real[...,None]).sum(axis=0))
            n=real.sum(axis=0)
            s.append(n>0)
            p.append(same.sum(axis=0)/np.maximum(n,1)>=.8)
            w.append((reliability[sl]*real).sum(axis=0))
        scales.append(mean(cosine_error(np.array(a),np.array(b)),np.array(s),np.array(p),np.array(w)))
    return spatial_loss,temporal_loss,float(np.mean(scales)),float((supported&present).sum()/denom)


def confidence_diagnostics(xy, scores, threshold):
    sel=np.r_[LEFT,RIGHT]
    observed=(scores[:,sel]>threshold)
    ins=in_frame(xy)[:,sel]
    return {"high_confidence_outside":float(np.mean(observed&~ins)),
            "joint_coverage":float(np.mean(observed&ins)),
            "median_confidence":float(np.median(scores[:,sel])),
            "both_hands":float(hand_quality(valid_pose(xy,scores,threshold)).all(axis=1).mean())}


def landmark_losses(xy, ref, weights, frames, size):
    """Huber pixel distances and wrist velocities on fixed reference support."""
    delta=(xy-ref)*size
    wrist=delta[:,WRISTS]
    tip=delta[:,TIPS]
    ww=weights[:,WRISTS]
    tw=weights[:,TIPS]
    pair_weights=np.minimum(ww[1:],ww[:-1])
    def average(v,w,h=12.):
        norm=np.linalg.norm(v,axis=-1)
        measured=np.where(norm<h,norm*norm/(2*h),norm-h/2)
        return float((measured*w).sum()/max(w.sum(),1e-8))
    return dict(wrist=average(wrist,ww),tips=average(tip,tw),
                shape=average(tip-wrist[:,:,None],np.minimum(tw,ww[:,:,None])),
                velocity=average(np.diff(wrist,axis=0)/np.diff(frames)[:,None,None],pair_weights,4.))


def reliability_weights(xy, scores, valid, paired_mp, raw_feat, masked_feat, size):
    agree=np.linalg.norm((xy[:,WRISTS]-paired_mp[:,WRISTS])*size,axis=-1)
    agree=np.where(np.isfinite(agree),np.exp(-agree/35.),.5)
    raw=raw_feat[1].reshape(len(xy),2,-1)
    masked=masked_feat[1].reshape(len(xy),2,-1)
    stable=np.clip(1-cosine_error(raw,masked),0,1)
    fraction=np.column_stack([np.clip(scores[:,i],0,1).mean(axis=1)*valid[:,i].mean(axis=1) for i in HANDS])
    return np.clip(agree*stable*fraction,0,1)


def longest_visible_region(xy,scores,cfg):
    good=np.flatnonzero(hand_quality(valid_pose(xy,scores,cfg["joint_confidence"]),
                                     cfg["min_hand_points"]).all(axis=1))
    segments=np.split(good,np.flatnonzero(np.diff(good)!=1)+1)
    best=max(segments,key=len)
    if len(best)<cfg["visibility_min_contiguous_frames"]:
        return None
    return (int(best[0]),int(best[-1])+1)


class HandGraphAdapter:
    """Load exactly the official graph parameters without allocating the inactive decoder."""
    def __init__(self, repo, checkpoint, device, threads):
        import importlib
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        import torch

        torch.set_num_threads(threads)
        torch.manual_seed(0)
        self.torch=torch
        self.device=torch.device(device)
        sys.path.insert(0,str(repo.resolve(strict=True)))
        datasets=importlib.import_module("datasets")
        models=importlib.import_module("models")
        self.load_part_kp=datasets.load_part_kp
        args=SimpleNamespace(dataset="How2Sign",hidden_dim=256,rgb_support=False,label_smoothing=0.)
        with patch.object(models.MT5ForConditionalGeneration,"from_pretrained",return_value=torch.nn.Identity()):
            with patch.object(models.T5Tokenizer,"from_pretrained",return_value=None):
                self.model=models.Uni_Sign(args)
        state=torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True)["model"]
        self.model.load_state_dict({key:state[key] for key in self.model.state_dict()},strict=True)
        self.model.eval().to(self.device)
