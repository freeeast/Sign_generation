"""Run the Stage 8 natural-initialization imitation loop with resumable per-arm results."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from scripts.stage4_geometry import evaluate
from scripts.stage7_optimize import read, subset, write
from src.evaluate_feedback import write_csv
from src.pose import FINGERS
from src.smplx_to_pose import forward
from src.stage5_motion import MotionSpace
from src.stage7_feedback import (HandGraphAdapter, HandGraphs, cosine_error, expert_input,
                                 feature_scores, hand_quality, in_frame, landmark_losses, valid_pose)
from src.stage8_coarse import coarse_fit
from src.virtual_pose import apply_affine, load_fit, pinhole

METHODS = ("A_geometry", "B_expert", "C_sequential", "D_joint")
REFINEMENT = "C2_local_hands"
WIDE = "C3_local_wide"
EDITED = ("body_pose", "left_hand_pose", "right_hand_pose")


def fixed_shape(fit):
    out = dict(fit)
    betas = np.broadcast_to(fit["betas"], (len(fit["body_pose"]), fit["betas"].shape[-1]))
    out["betas"] = np.broadcast_to(np.median(betas, axis=0), betas.shape).copy()
    return out


def zero_prior(prior):
    return dict(prior, corruption_coeff_deg=[0]*6, off_basis_deg=0)


def initialize(base, asset, cfg, prior, xy, score, camera, size, output, level):
    path = output/"initial_fit.npz"
    if path.exists():
        return load_fit(path)
    output.mkdir(parents=True, exist_ok=True)
    canonical = fixed_shape(base)
    if level == "B":
        initial, trace, meta = coarse_fit(canonical, asset, camera, xy, score, size, cfg["coarse"])
        write_csv(output/"coarse_fit_trace.csv", trace)
        write(output/"initialization.json", dict(level=level, type="RTMW-even arm inverse kinematics",
                                                 static_betas=True, **meta))
    elif level == "C":
        corruption = dict(prior, corruption_coeff_deg=cfg["challenge_deg"],
                          off_basis_deg=cfg["challenge_off_axis_deg"])
        space = MotionSpace(canonical, asset, corruption)
        initial = space.export(space.initial)
        write(output/"initialization.json", dict(level=level, type="synthetic local rotations",
                                                 static_betas=True, degrees=cfg["challenge_deg"],
                                                 off_axis_degrees=cfg["challenge_off_axis_deg"]))
    else:
        initial = canonical
        write(output/"initialization.json", dict(level=level, type="NSA pose tracks, clip-median betas", static_betas=True))
    np.savez_compressed(path, **initial)
    return initial


def wrapped_norm(z):
    radius = np.linalg.norm(z, axis=-1)
    return np.rad2deg(np.abs((radius + np.pi) % (2*np.pi) - np.pi))


class Response:
    def __init__(self, graphs, xy, confidence, cfg):
        self.graphs, self.cfg = graphs, cfg
        self.xy, self.conf = xy, confidence
        self.valid = valid_pose(xy, confidence, .3)
        self.supported = hand_quality(self.valid)
        self.weights = np.ones(self.supported.shape)
        self.reference = graphs.extract(*expert_input(xy, confidence, self.valid))
        if self.supported.sum() < 6:
            raise ValueError("Insufficient real response support")

    def score(self, xy, confidence):
        valid = valid_pose(xy, confidence, .3)
        together = valid & self.valid
        virtual = self.graphs.extract(*expert_input(xy, confidence, together))
        present = hand_quality(valid)
        terms = feature_scores(virtual, self.reference, self.supported, present, self.weights, 1.)
        profiles = np.stack([cosine_error(a.reshape(*a.shape[:2],-1), b.reshape(*b.shape[:2],-1))
                             for a,b in zip(virtual, self.reference)])
        profiles[:, ~(self.supported & present)] = np.nan
        result = dict(response=float(np.dot(terms[:3], self.cfg["stage7_factor_weights"])),
                      spatial=terms[0], temporal=terms[1], multiscale=terms[2], support=terms[3],
                      visible_fraction=float(valid[:,91:133].mean()), supported_real_hand_frames=int(self.supported.sum()))
        return result, profiles

    def save(self, path):
        if not path.exists():
            np.savez_compressed(path, xy=self.xy, confidence=self.conf, support=self.supported,
                                spatial=self.reference[0], temporal=self.reference[1])


def geometry_holdout(xy, mp, size):
    h = evaluate(xy, mp, size, slice(None))
    keys = ("wrists", "tips", "handshape")
    result = {f"held_{k}": h[k] for k in keys}
    result["held_geometry"] = float(np.mean([h[k] for k in keys])) if all(h[k] is not None for k in keys) else None
    return result


class Imitator:
    def __init__(self, asset, cfg, prior, response, fit, camera, xy, confidence, mp, size, indices, temp):
        self.asset, self.cfg, self.prior, self.feedback = asset, cfg, prior, response
        self.fit, self.camera, self.temp, self.size = fit, camera, temp, size
        self.indices = indices
        self.eval = subset(2, len(xy), 48, start=1)
        self.sampled = np.union1d(indices, self.eval)
        self.train_index = np.searchsorted(self.sampled, indices)
        self.audit_index = np.searchsorted(self.sampled, self.eval)
        self.real = xy[indices]
        self.weights = response.valid * np.maximum(np.clip(confidence[indices],0,1),.25)
        self.mp = np.where(in_frame(mp)[...,None],mp,np.nan)
        self.space = MotionSpace(fit, asset, zero_prior(prior))
        self.cache = {}
        self.bounds = self.angle_bounds(fit)
        self.profile_cache = {}

    def arms_degrees(self, fit):
        body = fit["body_pose"].reshape(-1,21,3)
        return dict(shoulder=wrapped_norm(body[:,15:17]),
                    elbow=wrapped_norm(body[:,17:19]), wrist=wrapped_norm(body[:,19:21]),
                    finger=np.concatenate([wrapped_norm(fit[key].reshape(-1,15,3)+ self.space.means[key])
                                           for key in EDITED[1:]], axis=1))

    def angle_bounds(self, fit):
        return {k:np.maximum(x + 1e-3, self.cfg["kinematic_absolute_deg"][k])
                for k,x in self.arms_degrees(fit).items()}

    def candidate(self, x):
        motion, limits = self.space.candidate(x)
        limits["constraint_violations"] += int(np.max(np.abs(x))>self.prior["max_coeff_deg"])
        absolute = self.arms_degrees(motion)
        limits.update({f"peak_{k}_deg":float(a.max()) for k,a in absolute.items()})
        limits["new_angle_excess"] = int(sum((v>self.bounds[k]+1e-4).sum() for k,v in absolute.items()))
        return motion, limits

    def project(self, motion, selected=None, mesh=False):
        # All local-angle/temporal checks operate on the original 24 Hz parameter trajectory.
        # Only detection/observation frames need full mesh forward kinematics per proposal.
        n = len(motion["body_pose"])
        sampled = motion if selected is None else {k:(x[selected] if x.ndim and len(x)==n else x)
                                                    for k,x in motion.items()}
        np.savez(self.temp, **sampled)
        vals = forward(self.temp, self.asset, "neutral", "model", 32, "cpu", False, return_mesh=mesh)
        points = vals[0]
        xy = apply_affine(pinhole(points,sampled["cam_trans"]), self.camera)
        return xy, points, vals[2] if mesh else None

    def at(self, x):
        key = tuple(x)
        if key in self.cache:
            return self.cache[key]
        start = perf_counter()
        motion, limits = self.candidate(x)
        obs, points, _ = self.project(motion, self.sampled)
        virtual = obs[self.train_index]
        terms, profiles = self.feedback.score(virtual,np.ones(virtual.shape[:2], np.float32))
        self.profile_cache[key] = profiles
        geometry = landmark_losses(virtual, self.real, self.weights, self.indices, self.size)
        geometry["geometry"] = float(sum(geometry[k]*v for k,v in self.cfg["geometry_weights"].items()))
        held = geometry_holdout(obs[self.audit_index], self.mp[self.eval], self.size)
        result = {**limits, **terms, **geometry, **held,
                  "evaluation_time_s":perf_counter()-start}
        self.cache[key] = result
        return result

    def full_tracks(self, x):
        motion, limits = self.candidate(x)
        xy, points, _ = self.project(motion)
        eval_frames = np.arange(1,len(points),2)
        held = geometry_holdout(xy[eval_frames], self.mp[eval_frames], self.size)
        held["held_hand_frames"] = int(np.isfinite(self.mp[eval_frames][:,[91,112],0]).sum())
        speed = np.linalg.norm(np.diff(points[:,FINGERS],axis=0),axis=-1)*24*1000
        accel = np.linalg.norm(np.diff(points[:,FINGERS],n=2,axis=0),axis=-1)*24**2*1000
        held["tip_speed_p95_mm_s"] = float(np.percentile(speed,95))
        held["tip_accel_p95_mm_s2"] = float(np.percentile(accel,95))
        return motion, xy, points, held

    def solve(self, method, scales):
        x = np.zeros(12 if method in (REFINEMENT,WIDE) else 6)
        start = self.at(x)
        current, anchor = start, start
        trace = [dict(step=0, phase="initial", accepted=False, coefficients=x.tolist(), **start)]
        proposals = []
        path = [x.copy()]
        def feasible(m, phase):
            if m["constraint_violations"] or m["new_angle_excess"]:
                return "motion_guard"
            if m["support"] < start["support"] - self.cfg["maximum_support_loss"]:
                return "support_guard"
            if phase == "expert_after_geometry":
                if any(m[k]>anchor[k]+v for k,v in self.cfg["geometry_slack_px"].items()):
                    return "anchored_geometry_guard"
            return ""

        def objective(m, phase):
            g = (m["geometry"]-start["geometry"])/scales["robust_geometry"]
            e = (m["response"]-start["response"])/scales["factor"]
            if phase == "geometry":
                return g
            if phase == "joint":
                return g + self.cfg["joint_feedback_weight"]*e
            return e

        for sweep, radius in enumerate(self.cfg["sweeps_deg"]):
            if sweep == 1:
                anchor = current
            if method == WIDE and sweep:
                radius *= 2
            axes = (6,9,7,10,8,11) if method in (REFINEMENT,WIDE) and sweep else self.cfg["axis_order"]
            for axis in axes:
                phase = {"A_geometry":"geometry", "B_expert":"expert", "D_joint":"joint"}.get(method)
                if method in ("C_sequential", REFINEMENT,WIDE):
                    phase = "geometry" if sweep==0 else "expert_after_geometry"
                step = len(trace)
                tested = []
                for sign in (-1,1):
                    y = x.copy()
                    y[axis] += sign*radius
                    m = self.at(y)
                    rejection = feasible(m,phase)
                    trial = dict(step=step, phase=phase, axis=axis, signed_delta_deg=sign*radius,
                                 current_coefficients=x.tolist(), reference_key="real_reference.npz",
                                 current_response=current["response"], candidate_gain=current["response"]-m["response"],
                                 proposed_coefficients=y.tolist(), rejection=rejection, accepted=False, **m)
                    proposals.append(trial)
                    if not rejection:
                        tested.append((objective(m,phase),y,m,trial))
                moved = False
                if tested:
                    value,y,m,trial = min(tested,key=lambda t:t[0])
                    eps = (self.cfg["minimum_response_gain"] / scales["factor"]
                           if phase.startswith("expert") else 1e-7)
                    if value < objective(current,phase)-eps:
                        x,current = y,m
                        trial["accepted"] = True
                        moved = True
                for trial in proposals[-2:]:
                    if not trial["rejection"] and not trial["accepted"]:
                        trial["rejection"] = "non_selected_or_no_gain"
                    trial["subsequent_response"] = current["response"]
                    trial["subsequent_coefficients"] = x.tolist()
                trace.append(dict(step=step,phase=phase,accepted=moved,coefficients=x.tolist(),**current))
                path.append(x.copy())
        return trace, proposals, path


def arm(run, path, label, scales, source, level):
    start_time = perf_counter()
    trace, proposals, states = run.solve(label,scales)
    path.mkdir(parents=True,exist_ok=True)
    write_csv(path/"trace.csv",trace)
    write_csv(path/"proposals.csv",proposals)
    np.savez_compressed(path/"proposal_response_profiles.npz",
                       frame_order=run.indices,
                       spatial_temporal_cosine=np.stack([run.profile_cache[tuple(p["proposed_coefficients"])] for p in proposals]),
                       reference_mask=run.feedback.supported)
    state_pose = {k:[] for k in EDITED}
    for x in states:
        p,_ = run.candidate(x)
        for k in EDITED:
            state_pose[k].append(p[k])
    np.savez_compressed(path/"motion_trajectory.npz", coefficients=states, **state_pose)
    checkpoints = [s for s in run.cfg["checkpoint_steps"] if s<len(states)]
    frames = []
    for step in checkpoints:
        motion,xy,points,full = run.full_tracks(states[step])
        np.savez_compressed(path/f"step_{step:02}_fit.npz",**motion)
        np.savez_compressed(path/f"step_{step:02}_tracks.npz",xy=xy,points=points)
        frames.append(dict(step=step,**full))
    write_csv(path/"full_held_checkpoints.csv",frames)
    accepted = sum(r["accepted"] for r in trace)
    initial, final = trace[0], trace[-1]
    result = dict(source=source,level=level,method=label,proposals=len(proposals),accepted=accepted,
                  initial_response=initial["response"],final_response=final["response"],
                  response_drop=initial["response"]-final["response"],
                  relative_response_drop=(initial["response"]-final["response"])/max(initial["response"],1e-8),
                  expert_phase_updates=sum(r["accepted"] and r["phase"]=="expert_after_geometry" for r in trace),
                  geometry_anchor_response=trace[6]["response"],
                  initial_full_held=frames[0]["held_geometry"],final_full_held=frames[-1]["held_geometry"],
                  **{f"final_{k}":v for k,v in frames[-1].items() if k!="step"},
                  correction_joint_deg=final["correction_joint_deg"],
                  correction_velocity_deg_per_frame=final["correction_velocity_deg_per_frame"],
                  correction_accel_deg_per_frame2=final["correction_accel_deg_per_frame2"],
                  constraint_violations=final["constraint_violations"],new_angle_excess=final["new_angle_excess"],
                  initial_support=initial["support"], final_support=final["support"],
                  elapsed_s=perf_counter()-start_time)
    write(path/"result.json",result)
    print(source,level,label,round(initial["response"],5),round(final["response"],5),accepted,
          round(frames[-1]["held_geometry"],2),round(result["elapsed_s"],1),flush=True)


def args_for_cli():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("asset","repo","checkpoint","output"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--sources",nargs="+")
    p.add_argument("--levels",nargs="+",default=["B","A","C"],choices=("A","B","C"))
    p.add_argument("--methods",nargs="+",choices=(*METHODS,REFINEMENT,WIDE),default=METHODS)
    p.add_argument("--threads",type=int,default=3)
    p.add_argument("--initialization-only",action="store_true")
    return p.parse_args()


def main():
    a = args_for_cli()
    cfg = read(Path("configs/stage8.json"))
    prior = read(Path("configs/stage5.json"))
    stage7 = Path(cfg["stage7_root"])
    scales = read(stage7/"calibration/scales.json")
    rows = {r["clip_id"]:r for r in read(stage7/"data/manifest.json")}
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads)
    graphs = None if a.initialization_only else HandGraphs(HandGraphAdapter(a.repo,a.checkpoint,"cpu",a.threads))
    for source in a.sources or cfg["sources"]:
        row = rows[source]
        folder = a.output/source
        folder.mkdir(parents=True,exist_ok=True)
        fit = load_fit(stage7/"data/inputs"/f"{source}.npz")
        with np.load(stage7/"data/real_pose"/f"{source}.npz") as d:
            xy,confidence = d["keypoints"],d["scores"]
            size = int(d["width"]),int(d["height"])
        if len(xy)!=row["rgb_frames"]:
            raise ValueError(f"{source}: inconsistent pose and video frame counts")
        with np.load(stage7/"data/paired"/f"{source}.npz") as d:
            camera,mp = d["camera"],d["mp"]
        indices = subset(cfg["reference_stride"],len(xy),cfg["max_expert_frames"])
        write(folder/"frames.json",dict(optimization_frames=indices.tolist(),
                                        held_frames=np.arange(1,len(xy),2).tolist(),size=list(size)))
        expert = None if graphs is None else Response(graphs,xy[indices],confidence[indices],cfg)
        if expert is not None:
            expert.save(folder/"real_reference.npz")
        with TemporaryDirectory() as temp:
            for level in a.levels:
                d = folder/level
                initial = initialize(fit,a.asset,cfg,prior,xy,confidence,camera,size,d,level)
                if a.initialization_only:
                    continue
                run = Imitator(a.asset,cfg,prior,expert,initial,camera,xy,confidence,mp,size,indices,
                               Path(temp)/"candidate.npz")
                for label in a.methods:
                    if (d/label/"result.json").exists():
                        continue
                    if label in (REFINEMENT,WIDE):
                        from src.stage8_motion import FingerWindowSpace
                        local = Imitator(a.asset,cfg,prior,expert,initial,camera,xy,confidence,mp,size,indices,
                                         Path(temp)/"candidate.npz")
                        local.space = FingerWindowSpace(initial,a.asset,zero_prior(prior))
                        arm(local,d/label,label,scales,source,level)
                    else:
                        arm(run,d/label,label,scales,source,level)


if __name__=="__main__":
    main()
