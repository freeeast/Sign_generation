"""Compare optimizers using the smaller Stage 8 elapsed search budget on the focus clip."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from scripts.stage7_optimize import read,write
from scripts.stage8_imitate import Imitator,WIDE
from scripts.stage8_imitate import zero_prior
from src.stage8_motion import FingerWindowSpace
from scripts.stage9_run import Loop
from src.evaluate_feedback import write_csv
from src.stage7_feedback import HandGraphAdapter,HandGraphs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("repo","checkpoint","asset","output"):
        p.add_argument("--"+k,required=True,type=Path)
    a=p.parse_args()
    cfg=read(Path("configs/stage9.json"))
    old=read(Path("configs/stage8.json")); prior=read(Path("configs/stage5.json"))
    data=Path(cfg["observations"])/"data"; source=cfg["focus"]
    scales=read(Path(cfg["observations"])/"calibration/scales.json")
    graphs=HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",2))
    with TemporaryDirectory() as td,np.load(data/"real_pose"/f"{source}.npz") as real:
        loop=Loop(a,cfg,source,"B",graphs,Path(td)/"new.npz")
        rows=[]
        for name in ("C_sequential",WIDE):
            model=Imitator(a.asset,old,prior,loop.response,loop.raw,loop.camera,real["keypoints"],real["scores"],
                           loop.mp,(1280,720),loop.indices,Path(td)/"old.npz")
            if name==WIDE:
                model.space=FingerWindowSpace(loop.raw,a.asset,zero_prior(prior))
            tick=perf_counter()
            trace,proposals,all_states=model.solve(name,scales)
            elapsed=perf_counter()-tick
            _,_,_,held=model.full_tracks(all_states[-1])
            rows.append(dict(method=name,seconds=elapsed,proposals=len(proposals),response=trace[-1]["response"],
                             held_geometry=held["held_geometry"]))
            write_csv(a.output/f"{name}_trace.csv",trace)
            print(rows[-1],flush=True)
        ceiling=min(r["seconds"] for r in rows)
        for name in ("art_hier","art_joint"):
            out=loop.solve(name,a.output/name,"factor",cfg["budget"],wall_limit=ceiling)
            rows.append(dict(method=name,seconds=out["optim_time_s"],proposals=out["proposal_count"],
                             response=out["feedback_final"],held_geometry=out["held_geometry"]))
        write_csv(a.output/"wall_time_comparison.csv",rows)
        write(a.output/"budget.json",dict(ceiling_seconds=ceiling,threads=2,accelerator="cpu",
                                        source=source,level="B",clock="search excl. final check-point export",
                                        max_proposals=cfg["budget"]))


if __name__=="__main__":
    main()
