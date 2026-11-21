"""Validate Stage 8 feedback trajectories through rendering and the same RTMW detector."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from scripts.stage6_rgb import pose_observations, real_frames
from scripts.stage7_optimize import read, write
from scripts.stage8_imitate import Response
from src.evaluate_feedback import write_csv
from src.extract_real_pose import load_detector
from src.smplx_to_pose import forward
from src.stage6_render import AvatarRenderer
from src.stage7_feedback import HandGraphAdapter, HandGraphs
from src.virtual_pose import apply_affine, load_fit, pinhole


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sampled_video(path, indices):
    found = 0
    wanted = set(indices)
    for i,frame in enumerate(real_frames(path)):
        if i in wanted:
            found += 1
            yield frame
        if i >= indices[-1]:
            break
    if found != len(indices):
        raise ValueError(f"Decoded {found} of {len(indices)} requested real frames")


def sample_fit(path, indices):
    fit = load_fit(path)
    n = len(fit["body_pose"])
    return {k:(a[indices] if a.ndim and len(a)==n else a) for k,a in fit.items()}


def validate_source(args, graphs, renderer, detector, cfg, source, temp):
    upstream = Path(cfg["stage7_root"])/"data"
    root = args.output/source
    dest = args.output/"rgb_validation"/source
    dest.mkdir(parents=True,exist_ok=True)
    wide = (root/"B/C3_local_wide/step_18_fit.npz").exists()
    if (dest/"summary.csv").exists() and (not wide or (dest/"C3_local_wide_18.json").exists()):
        return read_csv(dest/"summary.csv")
    if not (root/"B/C_sequential/result.json").exists():
        return []
    idx = np.asarray(read(root/"frames.json")["optimization_frames"])
    idx = idx[np.linspace(0,len(idx)-1,min(cfg["rgb_max_frames"],len(idx))).astype(int)]
    with np.load(upstream/"paired"/f"{source}.npz") as d:
        camera = d["camera"]
    real_xy,real_conf,n = pose_observations(sampled_video(upstream/"inputs"/f"{source}.mp4",idx),detector)
    with np.load(upstream/"real_pose"/f"{source}.npz") as cached:
        cache_drift = float(np.abs(cached["keypoints"][idx]-real_xy).max())
    response = Response(graphs,real_xy,real_conf,cfg)
    response.save(dest/"real_reference.npz")
    write(dest/"metadata.json",dict(sample_frames=idx.tolist(),person_present_frames=int((n>0).sum()),
                                   nearest_cached_keypoint_max_abs=cache_drift,
                                   detector_configuration="YOLOX 416x416 and RTMW 192x256"))
    states = {"initial":root/"B/initial_fit.npz",
              "A_geometry_18":root/"B/A_geometry/step_18_fit.npz",
              "B_expert_18":root/"B/B_expert/step_18_fit.npz",
              "C_sequential_06":root/"B/C_sequential/step_06_fit.npz",
              "C_sequential_12":root/"B/C_sequential/step_12_fit.npz",
              "C_sequential_18":root/"B/C_sequential/step_18_fit.npz",
              "D_joint_18":root/"B/D_joint/step_18_fit.npz"}
    if (root/"B/C2_local_hands/step_18_fit.npz").exists():
        states["C2_local_hands_18"] = root/"B/C2_local_hands/step_18_fit.npz"
    if wide:
        states["C3_local_wide_18"] = root/"B/C3_local_wide/step_18_fit.npz"
    results=[]
    for label,path in states.items():
        row_path = dest/f"{label}.json"
        if row_path.exists():
            results.append(read(row_path))
            continue
        start=perf_counter()
        f=sample_fit(path,idx)
        np.savez(temp,**f)
        points,_,verts,_ = forward(temp,args.asset,"neutral","model",32,"cpu",False,return_mesh=True)
        virtual = apply_affine(pinhole(points,f["cam_trans"]),camera)
        direct,_ = response.score(virtual,np.ones(virtual.shape[:2],np.float32))
        frames=renderer.frames(verts,f["cam_trans"],camera)
        xy,conf,persons=pose_observations(frames,detector)
        rgb,prof=response.score(xy,conf)
        np.savez_compressed(dest/f"{label}_rtmw.npz",xy=xy,confidence=conf,frames=idx,persons=persons,
                            spatial_temporal_cosine=prof)
        status = dict(source=source,label=label,n_frames=len(idx),seconds=perf_counter()-start,
                      person_present=int((persons>0).sum()),**{f"direct_{k}":v for k,v in direct.items()},
                      **{f"rgb_{k}":v for k,v in rgb.items()})
        write(row_path,status)
        results.append(status)
        print(source,label,round(direct["response"],5),round(rgb["response"],5),rgb["support"],flush=True)
    write_csv(dest/"summary.csv",results)
    # Use only periodic C checkpoints satisfying the anchoring gate and RGB support requirement.
    by_label={x["label"]:x for x in results}
    trace=read_csv(root/"B/C_sequential/trace.csv")
    anchor=trace[6]
    selected=[x for x in results if x["label"] in {"C_sequential_06","C_sequential_12","C_sequential_18"}]
    valid=[]
    for x in selected:
        step=int(x["label"].split("_")[-1])
        obs=trace[step]
        keep=all(float(obs[k])<=float(anchor[k])+slack+1e-7 for k,slack in cfg["geometry_slack_px"].items())
        if keep and x["rgb_support"]>=by_label["initial"]["rgb_support"]-cfg["maximum_support_loss"]:
            valid.append(x)
    best=min(valid,key=lambda x:x["rgb_response"]) if valid else None
    write(dest/"checkpoint_reranking.json",dict(
        eligible=[x["label"] for x in valid],selected=best["label"] if best else None,
        selected_response=best["rgb_response"] if best else None,final_precommitted="C_sequential_18",
        geometry_only_comparator="A_geometry_18"))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("repo","checkpoint","asset","detector","pose-model","output"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--sources",nargs="+")
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args()
    cfg=read(Path("configs/stage8.json"))
    graphs=HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",a.threads))
    detector=load_detector(a.repo,a.detector,a.pose_model,a.threads)
    with np.load(a.asset) as f:
        renderer=AvatarRenderer((1280,720),f["f"],f["v_template"])
    try:
        with TemporaryDirectory() as temp:
            rows=[]
            for source in a.sources or cfg["rgb_sources"]:
                rows.extend(validate_source(a,graphs,renderer,detector,cfg,source,Path(temp)/"sampled.npz"))
            if rows:
                write_csv(a.output/"rgb_validation/results.csv",rows)
    finally:
        renderer.close()


if __name__=="__main__":
    main()
