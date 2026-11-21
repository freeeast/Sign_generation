"""Compare direct projection and identical RTMW processing of real and rendered pixels."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from scripts.stage4_geometry import evaluate
from scripts.stage4_review_media import label, pose_on
from scripts.stage5_closed_loop import Pilot, assay, geometrical_average
from src.evaluate_feedback import write_csv
from src.extract_real_pose import detect_frame, load_detector
from src.pose import LEFT, RIGHT, validate
from src.smplx_to_pose import forward
from src.stage4_features import FrozenFeedback, distances
from src.stage5_motion import MotionSpace
from src.stage6_render import AvatarRenderer, video_writer
from src.unisign import UniSign
from src.virtual_pose import apply_affine, load_fit, pinhole

HANDS = (LEFT, RIGHT)


def coverage(scores, count, cfg):
    values = dict(person_fraction=float((count>0).mean()))
    for name, ids in zip(("left","right"), HANDS):
        hand = scores[:, ids]
        values[f"{name}_visible_joint_fraction"] = float((hand>.3).mean())
        values[f"{name}_qualified_frame_fraction"] = float(
            ((hand>.3).sum(axis=1)>=cfg["min_hand_landmarks_over_threshold"]).mean())
        values[f"{name}_median_confidence"] = float(np.median(hand))
    return values


def pose_observations(stream, detector, output=None, wh=None):
    xy, scores, count = [], [], []
    writer = video_writer(output, wh, 24) if output else nullcontext()
    with writer as handle:
        for image in stream:
            if handle:
                handle.write(image.tobytes())
            a, b, n = detect_frame(detector, image)
            xy.append(a)
            scores.append(b)
            count.append(n)
    xy, scores = validate(xy,scores)
    return xy, scores, np.asarray(count)


def real_frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, rgb = cap.read()
            if not ok:
                break
            yield rgb
    finally:
        cap.release()


def save_pose(path, xy, scores, count):
    np.savez_compressed(path, keypoints=xy, scores=scores, person_count=count)


def error(xy, scores, real, real_scores, wh):
    # Compare only shared visible joints, including both wrist and tips in handshape differences.
    candidate = np.where(scores[...,None]>.3, xy, np.nan)
    target = np.where(real_scores[...,None]>.3, real, np.nan)
    return evaluate(candidate, target, wh, slice(None))


def diagnostic_frame(directory, virtual, virtual_scores, real, real_scores, cfg):
    visible = np.column_stack([
        ((virtual_scores[:, ids]>.3).sum(axis=1)>=cfg["min_hand_landmarks_over_threshold"])
        for ids in HANDS])
    real_visible = np.column_stack([
        ((real_scores[:, ids]>.3).sum(axis=1)>=cfg["min_hand_landmarks_over_threshold"])
        for ids in HANDS])
    idx = int(np.argmax((~visible & real_visible).sum(axis=1)))
    panes = []
    for title, video, xy, score in (("How2Sign / RTMW", "real.mp4", real, real_scores),
                                   ("SMPL-X RGB / RTMW", "avatar.mp4", virtual, virtual_scores)):
        cap = cv2.VideoCapture(str(directory/video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, im = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"{video}: cannot read diagnostic frame {idx}")
        xy_px = np.where(score[idx,:,None]>.3, xy[idx], np.nan)*[im.shape[1], im.shape[0]]
        pose_on(im, xy_px, (30,230,252))
        im = cv2.resize(im,(800,450))
        cv2.rectangle(im,(0,0),(800,30),(28,28,28),-1)
        label(im, f"{title}  |  frame {idx}", (10,20))
        panes.append(im)
    cv2.imwrite(str(directory/"same_detector_frame.png"), np.concatenate(panes,axis=1))
    return idx


class RGBPilot(Pilot):
    def __init__(self, args, cfg, rgb, expert, detector, render, row, fit, camera, real, scores, original3d,
                 mp, temp, rendered, *, frame_range=None):
        self.args, self.cfg, self.rgb, self.expert = args, cfg, rgb, expert
        self.detector, self.render, self.row = detector, render, row
        self.fit, self.camera, self.temp, self.rendered = fit, camera, temp, rendered
        self.original3d, self.mp, self.wh = original3d, mp, render.wh
        self.space = MotionSpace(fit, args.asset, cfg)
        self.zero = np.zeros(len(cfg["corruption_coeff_deg"]))
        self.cache = {}
        start, stop = frame_range if frame_range is not None else (0, len(real))
        self.indices = np.arange(start, stop, rgb["frame_stride"])
        self.ref = expert.extract(real[self.indices],scores[self.indices])
        self.real = np.where(scores[...,None] > cfg["rtmw_threshold"], real, np.nan)
        self.ones = np.ones_like(scores[self.indices])
        odd = np.arange(1, len(real), 2)
        self.holdout = np.setdiff1d(odd[(odd >= start) & (odd < stop)], self.indices)

    def at(self, coef):
        key = tuple(coef)
        if key in self.cache:
            return self.cache[key]
        fit, kinematic = self.space.candidate(coef)
        kinematic["constraint_violations"] += int(np.max(np.abs(coef))>self.cfg["max_coeff_deg"])
        if kinematic["constraint_violations"]:
            self.cache[key] = kinematic
            return kinematic
        if self.rendered:
            np.savez(self.temp, **fit)
            points, _, verts, _ = forward(self.temp, self.args.asset, "neutral", "model", 64,
                                         self.args.device, False, return_mesh=True)
            frames = self.render.frames(verts[self.indices],fit["cam_trans"][self.indices],self.camera)
            pose, score, count = pose_observations(frames,self.detector)
            xy = apply_affine(pinhole(points,fit["cam_trans"]),self.camera)
        else:
            xy, _ = self.project(fit)
            pose, score = xy[self.indices], self.ones
            count = np.ones(len(self.indices))
        metrics = coverage(score,count,self.rgb)
        metrics.update(self.measure_endpoint_geometry(xy))
        metrics.update(distances(self.expert.extract(pose,score),self.ref))
        self.cache[key] = dict(**metrics,**kinematic)
        return self.cache[key]

    def measure_endpoint_geometry(self, xy):
        metrics = {}
        metrics.update(assay(xy,self.real,self.wh,self.indices,"rtmw_even"))
        metrics.update(assay(xy,self.mp,self.wh,self.holdout,"mp_held"))
        metrics["rtmw_even_combined"] = geometrical_average(metrics,"rtmw_even")
        metrics["mp_held_combined"] = geometrical_average(metrics,"mp_held")
        return metrics

    def loss(self, metrics, base, coef, arm, scales, **kwargs):
        if not metrics["constraint_violations"] and self.rendered:
            if metrics["person_fraction"] < self.rgb["min_person_fraction"]:
                return float("inf")
            for name in ("left","right"):
                key = f"{name}_qualified_frame_fraction"
                threshold = max(self.rgb["min_hand_qualified_fraction_each"],
                                base[key]-self.rgb["max_candidate_hand_loss_fraction"])
                if metrics[key]<threshold:
                    return float("inf")
        return super().loss(metrics,base,coef,arm,scales,**kwargs)


def passes(quality, rgb):
    return (quality["person_fraction"] >= rgb["min_person_fraction"] and
            all(quality[f"{s}_qualified_frame_fraction"] >= rgb["min_hand_qualified_fraction_each"]
                for s in ("left","right")))


def benchmark(args, cfg, rgb, row, expert, detector, primary, rest):
    clip = row["clip_id"]
    source_dir = args.videos if primary else args.prior_videos
    motion_dir = args.motions if primary else args.prior_motions
    path = source_dir/f"{clip}.mp4"
    fit_file = motion_dir/f"{clip}.npz"
    fit = load_fit(fit_file)
    directory = args.output/"rgb"/clip
    directory.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path, directory/"real.mp4")
    if primary:
        paired = np.load(args.output/"data/paired"/f"{clip}.npz")
        camera, mp = paired["camera"], paired["mp"]
        paired.close()
    else:
        paired = np.load(Path("experiments/E1_real")/clip/"fk_and_projection.npz")
        camera = paired["camera"]
        paired.close()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if abs(fps - 24) > .01:
        raise ValueError(f"Unexpected {clip} metadata")
    points, _, verts, faces = forward(fit_file,args.asset,"neutral","model",64,args.device,False,return_mesh=True)
    render = AvatarRenderer(wh,faces,rest)
    try:
        real, score, count = pose_observations(real_frames(path),detector)
        if len(real)!=len(points):
            raise ValueError("RGB timing and SMPL-X length differ")
        save_pose(directory/"real_detector.npz",real,score,count)
        ref = expert.extract(real[::2], score[::2])
        canonical = apply_affine(pinhole(points,fit["cam_trans"]),camera)
        scale = np.asarray(wh)
        direct_hand_to_body = np.median(np.linalg.norm(
            (canonical[:,[100,121]]-canonical[:,[91,112]])*scale, axis=-1),axis=0)
        observed_hand_to_body = np.median(np.linalg.norm(
            (real[:,[100,121]]-real[:,[91,112]])*scale, axis=-1),axis=0)
        meta = dict(clip=clip, width=wh[0], height=wh[1], fps=fps, frames=len(real),
                    triangles=len(faces), vertices=len(rest), renderer=render.renderer,
                    direct_middle_mcp_to_wrist_px=direct_hand_to_body.tolist(),
                    rtmw_middle_mcp_to_wrist_px=observed_hand_to_body.tolist(),
                    rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows = []
        variants = [("real",real,score,count),("direct",canonical,np.ones_like(score),np.ones_like(count))]
        settings = [("avatar","shirt",0),("clay","clay",0),("camera_shift","shirt",8)] if primary else [("avatar","shirt",0)]
        for name,palette,shift in settings:
            images = render.frames(verts,fit["cam_trans"],camera,palette,shift)
            xy,scores,counts = pose_observations(images,detector,directory/f"{name}.mp4",wh)
            save_pose(directory/f"{name}_detector.npz",xy,scores,counts)
            variants.append((name,xy,scores,counts))
        for name,xy,scores,counts in variants:
            qual = coverage(scores,counts,rgb) if name!="direct" else {}
            rows.append(dict(clip=clip, method=name, **qual,
                             **{k+"_px":v for k,v in error(xy,scores,real,score,wh).items()},
                             **distances(expert.extract(xy[::2],scores[::2]),ref)))
        avatar = variants[2]
        meta["failure_frame_zero_based"] = diagnostic_frame(directory, avatar[1], avatar[2], real,score,rgb)
        if primary:
            with tempfile.TemporaryDirectory() as scratch:
                temp = Path(scratch)/"candidate.npz"
                cfg["sweeps_deg"] = [rgb["step_deg"]]
                scales = json.loads(Path("experiments/Stage5/calibration_scales.json").read_text())
                paths = {}
                loop = []
                for use_render in (True,False):
                    name = "rgb" if use_render else "direct"
                    pilot = RGBPilot(args,cfg,rgb,expert,detector,render,row,fit,camera,real,score,points,mp,temp,use_render)
                    initial = pilot.at(pilot.zero)
                    paths[name] = {"initial": initial, "passes_gate": passes(initial,rgb)}
                    if use_render and not paths[name]["passes_gate"]:
                        continue
                    coef, trajectory, trace, proposals = pilot.search("C_temporal",scales,axes=rgb["proposal_axes"])
                    start_fit,_ = pilot.space.candidate(pilot.zero)
                    best,_ = pilot.space.candidate(coef)
                    for label_, f in (("initial",start_fit),("final",best)):
                        np.savez_compressed(directory/f"loop_{name}_{label_}_parameters.npz",**f)
                        save_avatar(args,render,directory/f"loop_{name}_{label_}_parameters.npz",
                                    directory/f"loop_{name}_{label_}.mp4",f["cam_trans"],camera)
                    write_csv(directory/f"loop_{name}_trajectory.csv",trace)
                    write_csv(directory/f"loop_{name}_proposals.csv",proposals)
                    np.save(directory/f"loop_{name}_coefficients.npy",trajectory)
                    loop.append(dict(method=name,accepted=sum(t["accepted"] for t in trace),
                                     **{k:trace[-1][k] for k in initial}))
                write_csv(directory/"loop_comparison.csv",loop)
                meta["loop_gate"] = paths
        (directory/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
        write_csv(directory/"comparison.csv",rows)
        print(json.dumps(meta),flush=True)
        return rows
    finally:
        render.close()


def save_avatar(args, render, parameters, output, trans, camera):
    _,_,verts,_ = forward(parameters,args.asset,"neutral","model",64,args.device,False,return_mesh=True)
    with video_writer(output,render.wh,24) as writer:
        for image in render.frames(verts,trans,camera):
            writer.write(image.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo","checkpoint","mt5","asset","motions","videos","prior-motions","prior-videos",
                 "detector","pose-model","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args()
    stage = json.loads(Path("configs/stage6.json").read_text())
    cfg = json.loads(Path("configs/stage5.json").read_text())
    rgb = stage["rgb"]
    manifest = json.loads((args.output/"data/manifest.json").read_text())
    secondary = next(row for row in json.loads(Path("configs/validation_pairs.json").read_text())
                     if row["clip_id"]==rgb["secondary_video"])
    with np.load(args.asset,allow_pickle=False) as model:
        rest = model["v_template"]
    detector = load_detector(args.repo,args.detector,args.pose_model,args.threads)
    expert = FrozenFeedback(UniSign(args.repo,args.checkpoint,args.mt5,args.device,args.threads))
    rows = []
    for row,primary in ((manifest[0],True),(secondary,False)):
        rows.extend(benchmark(args,cfg,rgb,row,expert,detector,primary,rest))
    write_csv(args.output/"rgb/comparison.csv",rows)


if __name__=="__main__":
    main()
