"""Train and test compact shared SMPL-X next-rotation models without split leakage."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from src.shared_motion import ACTIVE,Predictor,baseline,geodesic,hand_mean,increments,rotations,six,six_to_matrix

GROUPS={"upper":np.arange(13),"shoulder_elbow":np.arange(7,11),"wrist":np.arange(11,13),
        "left":np.arange(13,28),"right":np.arange(28,43),"finger":np.arange(13,43)}
GROUPS.update({f"{side}_{digit}":np.arange(lo+i*3,lo+i*3+3)
               for side,lo in (("left",13),("right",28))
               for i,digit in enumerate(("index","middle","pinky","ring","thumb"))})
HORIZON=(1,3,6,12,24,48)


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def populate(dataset,asset):
    mean=hand_mean(asset)
    with (dataset/"sample.csv").open() as f:
        rows=list(csv.DictReader(f))
    bank=defaultdict(list)
    for row in rows:
        with np.load(dataset/row["path"],allow_pickle=False) as a:
            r=rotations(a,mean,slice(None,None,2))
        d=increments(r)
        assert float(np.max(np.abs(np.linalg.det(r)-1)))<2e-6
        bank[row["split"]].append((row["clip"],row["source"],r,d))
    assert all(not ({x[1] for x in bank[k]} & {x[1] for x in bank[v]}) for k,v in (("train","val"),("train","test"),("val","test")))
    return bank


class Windows(Dataset):
    def __init__(self,bank,context,horizon,stride,include_id=False):
        self.bank=bank
        self.rows=[(i,t) for i,(_,_,r,_) in enumerate(bank) for t in range(1,len(r)-context-horizon+1,stride)]
        self.context=context
        self.horizon=horizon
        self.include_id=include_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        sid,t=self.rows[i]
        clip,group,r,d=self.bank[sid]
        e=t+self.context
        v=(r[t:e],d[t:e],r[e:e+self.horizon],d[e:e+self.horizon])
        return (*v,clip,group,e) if self.include_id else v


def loss(model,p,dr,truth,dt):
    # Balance body and each hand. Normalization prevents high-motion joints dominating.
    w=torch.ones(len(ACTIVE))
    w[:13]=1.35
    scale=model.scale.mean(-1)
    rot=torch.nn.functional.smooth_l1_loss((p-truth)/scale[None,None,:,None,None],
                                          torch.zeros_like(p),reduction="none").mean((-1,-2))
    vel=torch.nn.functional.smooth_l1_loss(dr/model.scale,dt/model.scale,reduction="none").mean(-1)
    cost=(.45*rot+.55*vel)*w
    return cost.mean()


def validate(model,dataset):
    out=[]
    with torch.no_grad():
        for r,d,gt,delta in DataLoader(dataset,batch_size=48):
            p,q=model.rollout(r,d,gt.shape[1])
            e=geodesic(p,gt)
            out.append(e.mean((0,2)).numpy())
    return np.mean(out,0)


def train(bank,args,context,future,scale):
    torch.manual_seed(args.seed)
    model=Predictor(scale,args.width)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015)
    train=Windows(bank["train"],context,future,6)
    val=Windows(bank["val"],context,6,9)
    g=torch.Generator().manual_seed(args.seed)
    logs=[]
    best=float("inf")
    file=args.output/f"gru_context{context}_h{future}.pt"
    for epoch in range(1,args.epochs+1):
        tick=perf_counter(); model.train(); errors=[]
        for n,(r,d,gt,delta) in enumerate(DataLoader(train,batch_size=48,shuffle=True,generator=g)):
            optimizer.zero_grad()
            pred,dp=model.rollout(r,d,future)
            value=loss(model,pred,dp,gt,delta)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            errors.append(float(value.detach()))
            if n+1==args.steps:break
        model.eval(); v=validate(model,val)
        row=dict(context=context,training_horizon=future,epoch=epoch,loss=float(np.mean(errors)),
                 steps=len(errors),val_1_deg=float(v[0]),val_6_deg=float(v[-1]),seconds=perf_counter()-tick)
        logs.append(row)
        print(row,flush=True)
        if v[-1]<best:
            best=float(v[-1])
            torch.save(dict(model=model.state_dict(),context=context,training_horizon=future,width=args.width,
                            scale=model.scale,epoch=epoch,val_6_deg=best,seed=args.seed),file)
    return logs


def eval_methods(bank,model,context,split,method_cycles,output):
    """Matched windows within each horizon; long horizon uses an eligibility subset."""
    rows=[];clip_rows=[]
    for h in method_cycles:
        dataset=Windows(bank[split],context,h,8,True)
        if not len(dataset):continue
        store=defaultdict(list)
        for r,d,gt,delta,clip,groups,t in DataLoader(dataset,batch_size=48):
            with torch.no_grad():
                methods={"gru":model.rollout(r,d,h),"hold":baseline(r,d,h,0.),
                         "velocity":baseline(r,d,h,1.),"damped":baseline(r,d,h,.72)}
                methods["hold"]=(r[:,-1:].expand_as(gt),torch.zeros_like(delta))
            ref=geodesic(r[:,-1:].expand_as(gt),gt).mean(1).numpy()
            for m,(p,dp) in methods.items():
                with torch.no_grad():
                    e=geodesic(p[:,-1],gt[:,-1]).numpy()
                    v=torch.rad2deg(torch.linalg.norm(dp,dim=-1)).numpy()
                    real_v=torch.rad2deg(torch.linalg.norm(delta,dim=-1)).numpy()
                    ang=geodesic(r[:,-1:].expand_as(p),p).mean(1).numpy()
                    acc=np.linalg.norm(np.diff(dp.numpy(),axis=1),axis=-1)
                    demand={g:e[:,j].mean(1) for g,j in GROUPS.items()}
                    hands=GROUPS["finger"]
                    demand["hand_speed_deg_per_step"]=v[:,:,hands].mean((1,2))
                    demand["reference_hand_speed_deg_per_step"]=real_v[:,:,hands].mean((1,2))
                    demand["hand_excursion_deg"]=ang[:,hands].mean(1)
                    demand["reference_hand_excursion_deg"]=ref[:,hands].mean(1)
                    demand["hand_accel_deg_per_step2"]=np.rad2deg(acc[:,:,hands].mean((1,2))) if h>1 else np.zeros(len(p))
                    demand["reference_hand_accel_deg_per_step2"]=np.rad2deg(np.linalg.norm(np.diff(delta.numpy(),axis=1),axis=-1)[:,:,hands].mean((1,2))) if h>1 else np.zeros(len(p))
                    for i,key in enumerate(clip):
                        store[(m,key,groups[i])].append({g:float(v[i]) for g,v in demand.items()})
        for (m,clip,g),vals in store.items():
            clip_rows.append(dict(split=split,horizon=h,method=m,clip=clip,source=g,windows=len(vals),
                                  **{k:np.mean([v[k] for v in vals]) for k in vals[0]}))
        for m in ("hold","velocity","damped","gru"):
            subset=[x for x in clip_rows if x["horizon"]==h and x["method"]==m]
            metrics={k:float(np.mean([r[k] for r in subset])) for k in subset[0] if k not in
                     ("split","horizon","method","clip","source","windows")}
            rows.append(dict(split=split,horizon=h,seconds=h/12,method=m,clips=len(subset),
                             sources=len({x["source"] for x in subset}),windows=sum(x["windows"] for x in subset),**metrics))
        print(split,h,[(r["method"],round(r["finger"],3)) for r in rows if r["horizon"]==h],flush=True)
    write_csv(output/f"{split}_forecast.csv",rows)
    write_csv(output/f"{split}_per_clip.csv",clip_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("dataset","asset","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--epochs",type=int,default=6)
    p.add_argument("--steps",type=int,default=105)
    p.add_argument("--seed",type=int,default=455)
    p.add_argument("--width",type=int,default=128)
    p.add_argument("--threads",type=int,default=3)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    a.output.mkdir(parents=True,exist_ok=True)
    bank=populate(a.dataset,a.asset)
    # The same per-axis scale is shared across x/y/z of a local joint.
    velocities=np.concatenate([d[1:] for _,_,r,d in bank["train"]])
    scale=np.maximum(np.minimum(np.quantile(np.linalg.norm(velocities,axis=-1),.85,axis=0)/1.75,.5),.018)[:,None]
    model_counts={}
    all_logs=[]
    for context,k in ((8,1),(8,3),(16,3)):
        logs=train(bank,a,context,k,scale)
        all_logs.extend(logs)
        write_csv(a.output/"learning.csv",all_logs)
        art=a.output/f"gru_context{context}_h{k}.pt"
        model_counts[art.name]=min(r["val_6_deg"] for r in logs)
    best=min(model_counts,key=model_counts.get)
    from src.shared_motion import load_predictor
    model,c=load_predictor(a.output/best)
    params=sum(p.numel() for p in model.parameters())
    q=torch.from_numpy(bank["train"][0][2][:10])
    check=float((six_to_matrix(six(q))-q).abs().max())
    (a.output/"selection.json").write_text(json.dumps(dict(checkpoint=best,val_6_deg=model_counts,parameters=params,
      clips={s:len(v) for s,v in bank.items()},groups={s:len(set(x[1] for x in v)) for s,v in bank.items()},
      rotation_6d_roundtrip_max=check,training_sample_stride=2,nominal_fps=12),indent=2)+"\n")
    for split in ("val","test"):
        eval_methods(bank,model,c["context"],split,HORIZON,a.output)


if __name__=="__main__":
    main()
