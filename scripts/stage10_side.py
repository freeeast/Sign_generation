"""Re-estimate a single fixed side-view evaluation camera in the Stage 10 world gauge."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.stage7_evaluate import project_side,side_camera
from scripts.stage7_optimize import read,write
from scripts.stage8_imitate import geometry_holdout
from scripts.stage9_audit import calculate
from src.evaluate_feedback import write_csv
from src.stage7_feedback import valid_pose
from src.virtual_pose import load_fit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    cfg=read(Path("configs/stage10.json")); source=cfg["focus"]
    root=a.output/source
    side=Path("experiments/Stage7/multiview/pose")
    protocol=read(root/"protocol.json")
    camera=np.asarray(protocol["affine"])
    torch.set_num_threads(2)
    with np.load(side/f"{source}.npz") as real,np.load(side/f"{source}_mp.npz") as held,TemporaryDirectory() as td:
        xy=real["keypoints"]
        valid=valid_pose(xy,real["scores"],.3)
        comparison=read(root/"results.json")
        xyz={}
        for name in comparison:
            f=load_fit(root/name/"final_fit.npz")
            xyz[name]=calculate(f,a.asset,camera,Path(td)/"fit.npz")[0]
        inferred=side_camera(xyz["A_geometry"],xy,valid)
        if inferred is None:
            raise ValueError("Insufficient side camera anchors")
        mp=held["mp"]
        rows=[dict(source=source,method=k,fit_upperbody_px=inferred[0],yaw_deg=inferred[1],
                   **geometry_holdout(project_side(v,inferred)[1::2],mp[1::2],(1280,720))) for k,v in xyz.items()]
        dest=a.output/"summary";dest.mkdir(exist_ok=True)
        write(dest/"side_camera_recalibrated.json",dict(calibration_anchor="A_geometry; real side RTMW even frames",
                                                        opaque_parameters=list(inferred),source=source))
        write_csv(dest/"side_comparison_recalibrated.csv",rows)
        print([(r["method"],r["held_geometry"]) for r in rows])


if __name__=="__main__":
    main()
