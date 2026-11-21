"""Cross-score fixed RGB and direct endpoints, adding an equal-budget geometry control."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from scripts.stage6_rgb import RGBPilot, coverage, error, pose_observations, save_avatar
from src.evaluate_feedback import write_csv
from src.extract_real_pose import load_detector
from src.smplx_to_pose import forward
from src.stage4_features import FrozenFeedback, distances
from src.stage6_render import AvatarRenderer
from src.unisign import UniSign
from src.pose import LEFT, RIGHT
from src.virtual_pose import apply_affine, load_fit, pinhole


def audit_rgb_endpoints(args, cfg, rgb, expert, detector):
    row = json.loads((args.output/"data/manifest.json").read_text())[0]
    clip = row["clip_id"]
    d = args.output/"rgb"/clip
    with np.load(d/"real_detector.npz") as cached:
        real, scores = cached["keypoints"], cached["scores"]
    with np.load(args.output/"data/paired"/f"{clip}.npz") as pair:
        camera, original3d, mp = pair["camera"], pair["original3d"], pair["mp"]
    wh = (1280,720)
    fit = load_fit(args.motions/f"{clip}.npz")
    with np.load(args.asset) as model:
        render = AvatarRenderer(wh,model["f"],model["v_template"])
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            pilot = RGBPilot(args,cfg,rgb,expert,detector,render,row,fit,camera,real,scores,original3d,
                             mp,Path(temp_dir)/"candidate.npz",False)
            scales = json.loads(Path("experiments/Stage5/calibration_scales.json").read_text())
            coef, trajectory, trace, proposals = pilot.search("B_geometry",scales,axes=rgb["proposal_axes"])
            out_path = d/"loop_geometry_final_parameters.npz"
            final,_ = pilot.space.candidate(coef)
            np.savez_compressed(out_path,**final)
            np.save(d/"loop_geometry_coefficients.npy",trajectory)
            write_csv(d/"loop_geometry_proposals.csv",proposals)
            write_csv(d/"loop_geometry_trajectory.csv",trace)
            save_avatar(args,render,out_path,d/"loop_geometry_final.mp4",fit["cam_trans"],camera)
            frames = pilot.indices
            refs = expert.extract(real[frames],scores[frames])
            descriptors = {
                "initial": d/"loop_direct_initial_parameters.npz",
                "geometry": out_path,
                "direct_feedback": d/"loop_direct_final_parameters.npz",
                "rgb_feedback": d/"loop_rgb_final_parameters.npz",
            }
            results=[]
            for name,path in descriptors.items():
                if not path.exists():
                    continue
                points,_,vertices,_ = forward(path,args.asset,"neutral","model",64,args.device,False,return_mesh=True)
                xy = apply_affine(pinhole(points,fit["cam_trans"]),camera)
                images = render.frames(vertices[frames],fit["cam_trans"][frames],camera)
                vxy,vscore,count = pose_observations(images,detector)
                feat = distances(expert.extract(vxy,vscore),refs)
                force_one = distances(expert.extract(vxy,np.ones_like(vscore)),refs)
                geo = pilot.measure_endpoint_geometry(xy)
                common = error(vxy,vscore,xy[frames],np.ones_like(vscore),wh)
                results.append(dict(method=name,
                    **{f"rgb_{key}":value for key,value in feat.items()},
                    **{f"rgb_forced_confidence_{key}":value for key,value in force_one.items()},
                    **{f"projected_{key}":value for key,value in distances(
                        expert.extract(xy[frames],np.ones_like(vscore)),refs).items()},
                    **{f"detector_vs_same_mesh_{key}_px":value for key,value in common.items()},
                    **coverage(vscore,count,rgb),**geo))
            write_csv(d/"loop_cross_scored_endpoints.csv",results)
            print("Cross-scored",clip,[r["method"] for r in results],flush=True)
    finally:
        render.close()



def visible_window(args, cfg, rgb, expert, detector):
    row = json.loads((args.output/"data/manifest.json").read_text())[0]
    clip = row["clip_id"]
    d = args.output/"rgb"/clip
    with np.load(d/"real_detector.npz") as cached:
        real, scores = cached["keypoints"], cached["scores"]
    in_frame = ((real>=0)&(real<=1)).all(axis=-1)
    good = (scores>.3)&in_frame
    observable = np.flatnonzero(np.all([good[:,ids].sum(axis=1)>=rgb["min_hand_landmarks_over_threshold"]
                                       for ids in (LEFT,RIGHT)],axis=0))
    windows = np.split(observable,np.flatnonzero(np.diff(observable)!=1)+1)
    window = max(windows,key=len)
    if len(window)<24:
        print("No 24-frame window with both real hands observed")
        return
    extent = int(window[0]), int(window[-1]+1)
    with np.load(args.output/"data/paired"/f"{clip}.npz") as pair:
        camera, original3d, mp = pair["camera"], pair["original3d"], pair["mp"]
    fit = load_fit(args.motions/f"{clip}.npz")
    with np.load(args.asset) as model:
        render = AvatarRenderer((1280,720),model["f"],model["v_template"])
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Holdout coverage and the window are defined using observed pixels, never expert scores.
            scales = json.loads(Path("experiments/Stage5/calibration_scales.json").read_text())
            rows = []
            for name, virtual_rgb, arm in (("geometry",False,"B_geometry"),
                                           ("direct_feedback",False,"C_temporal"),
                                           ("rgb_feedback",True,"C_temporal")):
                pilot = RGBPilot(args,cfg,rgb,expert,detector,render,row,fit,camera,real,scores,original3d,
                                 mp,Path(temp_dir)/"candidate.npz",virtual_rgb,frame_range=extent)
                start = pilot.at(pilot.zero)
                coef, traj, trace, probes = pilot.search(arm,scales,axes=rgb["proposal_axes"])
                for phase, m in (("initial",start),("final",trace[-1])):
                    rows.append(dict(method=name,phase=phase,**{k:m[k] for k in start}))
                write_csv(d/f"visible_{name}_trajectory.csv",trace)
                write_csv(d/f"visible_{name}_proposals.csv",probes)
                np.save(d/f"visible_{name}_coefficients.npy",traj)
                params,_ = pilot.space.candidate(coef)
                np.savez_compressed(d/f"visible_{name}_final_parameters.npz",**params)
            write_csv(d/"visible_window_comparison.csv",rows)
            info = dict(window_zero_based_inclusive=[extent[0],extent[1]-1],
                        optimization_frames=pilot.indices.tolist(), independent_frames=pilot.holdout.tolist(),
                        mp_associated_hands=int(np.isfinite(mp[pilot.holdout][:,[91,112],0]).sum()))
            (d/"visible_window.json").write_text(json.dumps(info,indent=2)+"\n")
            print("Visibility audit",clip,info,flush=True)
    finally:
        render.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","checkpoint","mt5","asset","motions","detector","pose-model","output"):
        parser.add_argument(f"--{name}",required=True,type=Path)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--analysis", choices=("endpoints","visible","both"), default="both")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    rgb=json.loads(Path("configs/stage6.json").read_text())["rgb"]
    cfg=json.loads(Path("configs/stage5.json").read_text())
    cfg["sweeps_deg"]=[rgb["step_deg"]]
    detector=load_detector(args.repo,args.detector,args.pose_model,args.threads)
    expert=FrozenFeedback(UniSign(args.repo,args.checkpoint,args.mt5,args.device,args.threads))
    if args.analysis in {"endpoints","both"}:
        audit_rgb_endpoints(args,cfg,rgb,expert,detector)
    if args.analysis in {"visible","both"}:
        visible_window(args,cfg,rgb,expert,detector)


if __name__=="__main__":
    main()
