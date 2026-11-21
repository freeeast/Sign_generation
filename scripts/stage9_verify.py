"""Replay a saved articulated update and check gradients and mesh-linked keypoints."""
import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.stage7_optimize import read,write
from scripts.stage9_audit import calculate
from scripts.stage9_run import Loop
from src.stage7_feedback import HandGraphAdapter,HandGraphs
from src.stage9_hand import EDIT_2D
from src.virtual_pose import load_fit


def file_digest(path):
    h=sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("repo","checkpoint","asset","output"):
        p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--detector",type=Path)
    p.add_argument("--pose-model",type=Path)
    a=p.parse_args()
    cfg=read(Path("configs/stage9.json"))
    out=a.output/"summary"; out.mkdir(exist_ok=True)
    graphs=HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",2))
    with TemporaryDirectory() as td:
        temp=Path(td)/"fit.npz"
        for source in cfg["sources"]:
            loop=Loop(a,cfg,source,"B",graphs,temp)
            loop.response.save(a.output/source/"real_reference.npz")
            if source!=cfg["focus"]:
                continue
            directory=a.output/source/"B/art_joint_factor"
            with np.load(directory/"control_trajectory.npz") as f:
                x=torch.tensor(f["coefs"][-1],requires_grad=True)
            xy,_=loop.forward(x)
            value=loop.differentiable.value(xy,"factor")
            value.backward()
            idx=np.unravel_index(x.grad.abs().argmax().item(),x.shape)
            checks={}
            with torch.no_grad():
                tangent=torch.zeros_like(x)
                tangent[idx]=1
                for h in (.001,.0003):
                    plus=loop.differentiable.value(loop.forward(x+tangent*h)[0],"factor")
                    minus=loop.differentiable.value(loop.forward(x-tangent*h)[0],"factor")
                    checks[str(h)]=float((plus-minus)/(2*h))
                exported=loop.ctrl.export(x)
                archived=load_fit(directory/"final_fit.npz")
                pts,projected,_=calculate(archived,a.asset,loop.camera,temp)
                sparse_pts=loop.fk.points(loop.ctrl.rotations(x),loop.all).numpy()
                view_max=float(np.abs((xy.numpy()-projected[loop.indices])*[1280,720]).max())
                stored=read(directory/"result.json")["feedback_final"]
                score=loop.measure_xy(xy)["response"]
                full=loop.measure_xy(torch.from_numpy(projected[loop.indices]))["response"]
            record=dict(source=source,frame_count=len(pts),checked_coefficient_name=loop.ctrl.names[idx[1]],
                        checked_knot_index=int(idx[0]),autograd=float(x.grad[idx]),finite_differences=checks,
                        absolute_saved_rotation_diff_rad=float(max(np.abs(archived[k]-exported[k]).max()
                            for k in ("body_pose","left_hand_pose","right_hand_pose"))),
                        sparse_final_fk_max_abs_mm=float(np.abs(sparse_pts-pts[:,EDIT_2D]).max()*1000),
                        sparse_final_projection_max_abs_px=view_max,archived_expert_distance=stored,
                        replayed_expert_distance=score,full_mesh_expert_distance=full)
            write(out/"replay_verification.json",record); print(record,flush=True)
    targets={k:getattr(a,k) for k in ("asset","checkpoint","detector","pose_model") if getattr(a,k)}
    write(out/"assets_sha256.json",{k:dict(basename=v.name,sha256=file_digest(v)) for k,v in targets.items()})


if __name__=="__main__":
    main()
