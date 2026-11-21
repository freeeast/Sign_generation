"""Recover an exact accepted or rejected motion and its raw virtual expert activations."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.stage7_optimize import read,write
from scripts.stage8_imitate import Imitator, Response, REFINEMENT, WIDE, zero_prior
from src.stage7_feedback import HandGraphAdapter, HandGraphs, expert_input, valid_pose
from src.stage8_motion import FingerWindowSpace
from src.virtual_pose import load_fit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("asset","repo","checkpoint","root","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--level",required=True,choices=("A","B","C"))
    p.add_argument("--method",required=True)
    p.add_argument("--proposal",type=int,required=True,help="Zero-based row of proposals.csv")
    a=p.parse_args()
    cfg=read(Path("configs/stage8.json"))
    prior=read(Path("configs/stage5.json"))
    data=Path(cfg["stage7_root"])/"data"
    root=a.root/a.source
    with (root/a.level/a.method/"proposals.csv").open() as f:
        rows=list(csv.DictReader(f))
    if not 0<=a.proposal<len(rows):
        p.error("proposal row outside the saved trajectory")
    record=rows[a.proposal]
    x=np.array(json.loads(record["proposed_coefficients"]))
    fit=load_fit(root/a.level/"initial_fit.npz")
    frames=np.array(read(root/"frames.json")["optimization_frames"])
    with np.load(data/"paired"/f"{a.source}.npz") as d:
        camera,mp=d["camera"],d["mp"]
    with np.load(data/"real_pose"/f"{a.source}.npz") as d:
        xy,scores=d["keypoints"],d["scores"]
        size=(int(d["width"]),int(d["height"]))
    g=HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",2))
    response=Response(g,xy[frames],scores[frames],cfg)
    a.output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory() as tmp:
        loop=Imitator(a.asset,cfg,prior,response,fit,camera,xy,scores,mp,size,frames,Path(tmp)/"candidate.npz")
        if a.method in (REFINEMENT,WIDE):
            loop.space=FingerWindowSpace(fit,a.asset,zero_prior(prior))
        metrics=loop.at(x)
        predicted=float(record["response"])
        difference=abs(metrics["response"]-predicted)
        if difference>1e-5:
            raise ValueError(f"Candidate replay mismatch: {difference:g}")
        motion,_=loop.candidate(x)
        points,_,_=loop.project(motion,frames)
        conf=np.ones(points.shape[:2],np.float32)
        layers=g.extract(*expert_input(points,conf,response.valid & valid_pose(points,conf,.3)))
        np.savez_compressed(a.output/"proposal_fit.npz",**motion)
        np.savez_compressed(a.output/"raw_virtual_response.npz",spatial=layers[0],temporal=layers[1],xy=points,
                            frames=frames,confidence=conf)
        write(a.output/"verification.json",dict(source=a.source,level=a.level,method=a.method,
                                               proposal=a.proposal,saved_response=predicted,
                                               reproduced=metrics["response"],difference=difference))


if __name__=="__main__":
    main()
