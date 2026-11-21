"""Optimize smooth SMPL-X corrections against the same frozen expert on real and virtual poses."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from scripts.stage4_geometry import evaluate
from src.evaluate_feedback import write_csv
from src.pose import FINGERS
from src.probe_3d import diagnostics
from src.run_real_validation import load_manifest, observations
from src.smplx_to_pose import forward
from src.stage4_features import FrozenFeedback, distances
from src.stage5_motion import MotionSpace
from src.unisign import UniSign
from src.virtual_pose import apply_affine, load_fit, model_sha, pinhole

ARMS = {
    "B_geometry": (True, None),
    "C_temporal": (True, "hand_temporal"),
    "D_temporal_only": (False, "hand_temporal"),
    "C_spatial": (True, "hand_spatial"),
    "C_global": (True, "global_encoder"),
    "C_nll": (True, "nll"),
}
SIGNALS = ("rtmw_even_combined", "hand_temporal", "hand_spatial", "global_encoder", "nll")


def assay(virtual, target, wh, frames, prefix):
    return {prefix+"_"+key: value for key, value in evaluate(virtual, target, wh, frames).items()
            if key in {"wrists", "tips", "handshape"}}


def geometrical_average(metrics, prefix):
    return np.mean([metrics[prefix+"_"+k] for k in ("wrists", "tips", "handshape")]).item()


class Pilot:
    def __init__(self, args, cfg, row, expert, temp):
        self.args, self.cfg, self.row, self.expert, self.temp = args, cfg, row, expert, temp
        self.clip = row["clip_id"]
        original_root = Path("experiments")
        real, confidence, self.wh = observations(original_root/"E3/real_pose", row)
        self.real = np.where(confidence[..., None] > cfg["rtmw_threshold"], real, np.nan)
        self.ref = expert.extract(real[::2], confidence[::2])
        self.ones = np.ones(confidence[::2].shape, np.float32)
        with np.load(original_root/"Stage4/geometry"/f"{self.clip}.npz") as data:
            self.mp = data["mp"]
        with np.load(original_root/"E1_real"/self.clip/"fk_and_projection.npz") as data:
            self.original3d, self.camera, self.original2d = data["landmarks3d"], data["camera"], data["keypoints"]
        self.fit = load_fit(args.motions/f"{self.clip}.npz")
        self.space = MotionSpace(self.fit, args.asset, cfg)
        self.zero = np.zeros(len(cfg["corruption_coeff_deg"]))
        self.cache = {}
        self.metric_original = self.measure(self.original2d, self.original3d)

    def measure(self, xy, points):
        expert = self.expert.extract(xy[::2], self.ones, self.row["reference"])
        g = assay(xy, self.real, self.wh, slice(None,None,2), "rtmw_even")
        g.update(assay(xy, self.real, self.wh, slice(1,None,2), "rtmw_odd"),
                 **assay(xy, self.mp, self.wh, slice(1,None,2), "mp_odd"))
        g["rtmw_even_combined"] = geometrical_average(g, "rtmw_even")
        g["rtmw_odd_combined"] = geometrical_average(g, "rtmw_odd")
        g["mp_odd_combined"] = geometrical_average(g, "mp_odd")
        g["mp_odd_hand_instances"] = int(np.isfinite(self.mp[1::2][:,[91,112],0]).sum())
        g.update(diagnostics(points, self.original3d))
        g["fit_distance_3d_mm"] = float(np.median(np.linalg.norm(
            (points-self.original3d)[:,np.r_[9,10,FINGERS]],axis=-1))*1000)
        g["hand_accel_p95_mm_frame2"] = float(np.percentile(np.linalg.norm(
            np.diff(points[:,FINGERS],n=2,axis=0),axis=-1),95)*1000)
        return dict(nll=expert["nll"], **distances(expert, self.ref), **g)

    def project(self, fit):
        np.savez(self.temp, **fit)
        points, _ = forward(self.temp, self.args.asset, "neutral", "model", 64, self.args.device, False)
        xy = apply_affine(pinhole(points, self.fit["cam_trans"]), self.camera)
        return xy, points

    def at(self, coef):
        key = tuple(coef)
        if key in self.cache:
            return self.cache[key]
        fit, kin = self.space.candidate(coef)
        kin["constraint_violations"] += int(np.max(np.abs(coef)) > self.cfg["max_coeff_deg"])
        if kin["constraint_violations"]:
            self.cache[key] = kin
            return kin
        xy, points = self.project(fit)
        g = self.measure(xy, points)
        self.cache[key] = dict(**g, **kin)
        return self.cache[key]

    def scale(self):
        base = self.at(self.zero)
        step = self.cfg["sweeps_deg"][0]
        probes = [self.at(d*step*unit) for unit in np.eye(len(self.zero)) for d in (-1,1)]
        scales = {key: float(max(1e-6, np.median([abs(m[key]-base[key]) for m in probes
                   if m["constraint_violations"]==0]))) for key in SIGNALS}
        return scales

    def loss(self, metrics, base, coef, arm, scales):
        geometry, signal = ARMS[arm]
        if metrics["constraint_violations"]:
            return float("inf")
        g = (metrics["rtmw_even_combined"] - base["rtmw_even_combined"]) / scales["rtmw_even_combined"]
        x = metrics["correction_accel_deg_per_frame2"] / self.cfg["max_correction_accel_deg_per_frame2"]
        reg = self.cfg["coefficient_penalty"] * np.mean((coef/self.cfg["max_coeff_deg"])**2)
        reg += self.cfg["smoothness_penalty"] * x*x
        weight = self.cfg["feedback_weight"] if geometry else 1.
        expert = (metrics[signal] - base[signal]) / scales[signal] if signal else 0.
        return float(geometry*g + weight*expert + reg)

    def search(self, arm, scales):
        coef = self.zero.copy()
        start = self.at(coef)
        current = start
        params = [coef.copy()]
        trace = [dict(step=0, accepted=False, objective=0., **start)]
        probes = []
        step = 0
        for radius in self.cfg["sweeps_deg"]:
            for axis in range(len(coef)):
                step += 1
                old_value = self.loss(current, start, coef, arm, scales)
                proposal = []
                for sign in (-1,1):
                    x = coef.copy()
                    x[axis] += sign*radius
                    measurement = self.at(x)
                    value = self.loss(measurement,start,x,arm,scales)
                    proposal.append((value,x,measurement))
                    probes.append(dict(step=step, axis=axis, sign=sign, radius_deg=radius,
                                       objective=value if np.isfinite(value) else None, coefficients=x.tolist(),
                                       **{k:measurement.get(k) for k in start}))
                value, candidate, measurement = min(proposal, key=lambda r:r[0])
                accept = value < old_value-1e-7
                if accept:
                    coef, current = candidate, measurement
                params.append(coef.copy())
                trace.append(dict(step=step, accepted=accept, objective=self.loss(current,start,coef,arm,scales), **current))
        return coef, params, trace, probes

    def run(self, out, scales, split):
        directory = out/self.clip
        directory.mkdir(parents=True, exist_ok=True)
        bounds = {k:0. for k in self.at(self.zero) if k.startswith("correction_") or k=="constraint_violations"}
        rows = [dict(clip=self.clip, split=split, method="A_original", steps=0, accepted=0, objective=0.,
                     **self.metric_original, **bounds),
                dict(clip=self.clip, split=split, method="I_perturbed", steps=0, accepted=0, objective=0.,
                     **self.at(self.zero))]
        motions = {"A_original__xy":self.original2d, "A_original__points":self.original3d}
        start_fit, _ = self.space.candidate(self.zero)
        np.savez_compressed(directory/"I_perturbed_parameters.npz", **start_fit)
        motions["I_perturbed__xy"], motions["I_perturbed__points"] = self.project(start_fit)
        for arm in ARMS:
            coef, params, trace, probes = self.search(arm, scales)
            write_csv(directory/f"{arm}_trajectory.csv", trace)
            write_csv(directory/f"{arm}_proposals.csv", probes)
            np.save(directory/f"{arm}_coefficients.npy", np.asarray(params))
            fit, _ = self.space.candidate(coef)
            np.savez_compressed(directory/f"{arm}_final_parameters.npz", **fit)
            motions[arm+"__xy"], motions[arm+"__points"] = self.project(fit)
            final = trace[-1]
            rows.append(dict(clip=self.clip, split=split, method=arm, steps=len(trace)-1,
                             accepted=sum(t["accepted"] for t in trace), objective=final["objective"],
                             **{k:final[k] for k in self.at(self.zero)}))
            print(self.clip, arm, rows[-1]["accepted"], rows[-1]["hand_temporal"],
                  rows[-1]["mp_odd_combined"], "uq",len(self.cache), flush=True)
        np.savez_compressed(directory/"final_motion_tracks.npz",**motions)
        write_csv(directory/"comparison.csv",rows)
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "checkpoint", "mt5", "asset", "motions", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/stage5.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    geo = json.loads(Path("experiments/E1_real/geometry_qc.json").read_text())
    if model_sha(args.asset) != geo["asset_sha256"]:
        raise ValueError("Use the asset associated with the fixed camera")
    manifest = {r["clip_id"]:r for r in load_manifest(Path("configs/validation_pairs.json"))}
    expert = FrozenFeedback(UniSign(args.repo,args.checkpoint,args.mt5,args.device,args.threads))
    args.output.mkdir(parents=True,exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        temp = Path(scratch)/"motion.npz"
        ids = [cfg["calibration_clip"],*cfg["evaluation_clips"]]
        for i, clip in enumerate(ids):
            test = Pilot(args,cfg,manifest[clip],expert,temp)
            if i==0:
                scales = test.scale()
                (args.output/"calibration_scales.json").write_text(json.dumps(scales,indent=2)+"\n")
            rows.extend(test.run(args.output,scales, "cal" if i==0 else "heldout"))
    write_csv(args.output/"comparison.csv",rows)


if __name__=="__main__":
    main()
