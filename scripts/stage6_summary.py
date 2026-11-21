"""Consolidate source-level comparisons, uncertainty, curves and a blind-review packet."""
import argparse
import csv
from itertools import combinations, product
import json
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.stage4_geometry import per_frame_metrics
from scripts.stage4_review_media import pose_on, label
from scripts.stage6_rgb import save_avatar
from src.evaluate_feedback import write_csv
from src.stage6_render import AvatarRenderer
from src.virtual_pose import load_fit


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


METHODS = {
    "B_geometry": "Geometry",
    "C_temporal": "+ Hand temporal",
    "C_spatial": "+ Hand spatial",
    "C_global": "+ Global",
    "D_temporal_only": "Hand temporal only",
    "C_temporal_guard": "+ Temporal, wrist guard",
    "C_temporal_staged": "Staged, guard & trust",
}
COLORS = {
    "B_geometry": "#26374a", "C_temporal": "#2869a0", "C_spatial": "#cf7b18",
    "C_global": "#9d5598", "D_temporal_only": "#b0413e",
    "C_temporal_guard": "#338762", "C_temporal_staged": "#646353",
}


def confidence_interval(values):
    # Exhaust all equal-probability resamples of the four source IDs.
    resamples = np.asarray(list(product(range(len(values)), repeat=len(values))))
    return np.quantile(values[resamples].mean(axis=1),[.025,.975])


def report(root, manifest):
    clips = [row["clip_id"] for row in manifest]
    rows, deltas, at_six = [], [], []
    out = root/"direct"
    quantities = ["mp_odd_combined","mp_odd_wrists","mp_odd_tips","mp_odd_handshape","rtmw_odd_combined",
                  "hand_temporal","hand_spatial","global_encoder","hand_accel_p95_mm_frame2"]
    contrasts = ["A_original","I_perturbed","B_geometry"]
    fig, axs = plt.subplots(len(clips), 3, figsize=(12.8, 11.5), sharex=True, layout="constrained")
    for k, clip in enumerate(clips):
        d = out/clip
        by_method = {r["method"]:r for r in read(d/"comparison.csv")}
        rows.extend(by_method.values())
        for method in METHODS:
            r = by_method[method]
            deltas.extend(dict(clip=clip,method=method,baseline=b,**{
                q: float(r[q])-float(by_method[b][q]) for q in quantities}) for b in contrasts)
            trace = read(d/f"{method}_trajectory.csv")
            metrics = ["mp_odd_combined","mp_odd_wrists","hand_temporal"]
            for a, metric in zip(axs[k], metrics):
                difference = [float(t[metric])-float(trace[0][metric]) for t in trace]
                a.plot([int(t["step"]) for t in trace], difference, color=COLORS[method],
                       label=METHODS[method],ls="--" if method.endswith("guard") else "-")
                a.axhline(0,color="#cccccc",lw=.6)
            cut = 6
            at_six.append(dict(clip=clip,method=method,step=cut,**{
                q:float(trace[cut][q])-float(trace[-1][q]) for q in quantities}))
        axs[k,0].set_ylabel(clip.split("_")[0]+"\nChange (px)")
    for a, title in zip(axs[0], ("Held-out MediaPipe composite","Held-out MediaPipe wrists","Hand-temporal cosine distance")):
        a.set_title(title)
    for a in axs[-1]:
        a.set_xlabel("Coordinate decisions (2 proposals each)")
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3,frameon=False)
    fig.savefig(root/"optimization_curves.png",dpi=160)
    plt.close(fig)
    write_csv(out/"comparison.csv",rows)
    write_csv(out/"contrasts.csv",deltas)
    write_csv(out/"checkpoint_6_minus_12.csv",at_six)
    statistics = []
    for method in METHODS:
        if method=="B_geometry":
            continue
        d = [r for r in deltas if r["method"]==method and r["baseline"]=="B_geometry"]
        for q in quantities:
            vals = np.array([r[q] for r in d])
            lo, hi = confidence_interval(vals)
            statistics.append(dict(method=method,metric=q,n_sources=len(vals),mean=vals.mean(),
                                   min=vals.min(),max=vals.max(),lower95=lo,upper95=hi,
                                   lower_count=int((vals<-1e-7).sum()),higher_count=int((vals>1e-7).sum())))
    write_csv(out/"source_summary.csv",statistics)
    vectors, similarities = [], []
    for method in ("B_geometry","C_temporal","D_temporal_only"):
        coefficients = {clip:np.load(out/clip/f"{method}_coefficients.npy")[-1] for clip in clips}
        for clip, coef in coefficients.items():
            vectors.append(dict(clip=clip,method=method,**{f"c{i}_deg":v for i,v in enumerate(coef)}))
        for a,b in combinations(clips,2):
            x,y = coefficients[a],coefficients[b]
            norm = np.linalg.norm(x)*np.linalg.norm(y)
            similarities.append(dict(method=method,source_a=a,source_b=b,
                                     cosine=float(x@y/norm) if norm else None,
                                     coefficient_l2_deg=np.linalg.norm(x-y)))
    write_csv(out/"correction_directions.csv",vectors)
    write_csv(out/"direction_pairwise.csv",similarities)
    rgb_loop_plot(root)


def rgb_loop_plot(root):
    directories = list((root/"rgb").glob("*/loop_comparison.csv"))
    if not directories:
        return
    d = directories[0].parent
    fig, axes = plt.subplots(1,3,figsize=(10,3.7),layout="constrained")
    for name in ("geometry","direct","rgb"):
        path = d/f"loop_{name}_trajectory.csv"
        if not path.is_file():
            continue
        trace = read(path)
        for a, q, title in zip(axes,("hand_temporal","mp_held_wrists","mp_held_combined"),
                               ("Expert temporal distance","Held-out wrists (px)","Held-out composite (px)")):
            a.plot([int(t["step"]) for t in trace],[float(t[q]) for t in trace],marker="o",label=name)
            a.set_title(title)
            a.set_xlabel("Coordinate decision")
            a.legend(frameon=False)
    fig.savefig(root/"rgb_loop_curves.png",dpi=155)
    plt.close(fig)


def failure(root, manifest, videos):
    rows = read(root/"direct"/"contrasts.csv")
    candidates = [r for r in rows if r["method"]=="D_temporal_only" and r["baseline"]=="B_geometry"]
    clip = max(candidates,key=lambda r:float(r["mp_odd_wrists"]))["clip"]
    xy = np.load(root/"direct"/clip/"final_motion_tracks.npz")
    target = np.load(root/"data/paired"/f"{clip}.npz")["mp"]
    wh = (1280,720)
    conditions = ("I_perturbed","B_geometry","D_temporal_only")
    errors = [per_frame_metrics(xy[c+"__xy"],target,wh)["wrists"] for c in conditions]
    held = errors[2][1::2]-errors[1][1::2]
    ix,_ = np.unravel_index(np.nanargmax(held),held.shape)
    idx = int(2*ix + 1)
    cap = cv2.VideoCapture(str(videos/f"{clip}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
    ok,image = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Cannot reopen matched failure frame")
    panes = []
    for name,track in [("Held-out MediaPipe",target[idx]),
                       *((c,xy[c+"__xy"][idx]) for c in conditions)]:
        out = image.copy()
        pose_on(out, track*wh, (35,224,252))
        out = cv2.resize(out,(600,338))
        cv2.rectangle(out,(0,0),(600,29),(28,28,28),-1)
        label(out, f"{clip.split('_')[0]} | frame {idx} | {name}", (9,20))
        panes.append(out)
    cv2.imwrite(str(root/"wrist_failure.png"),
                np.concatenate([np.concatenate(panes[:2],axis=1),np.concatenate(panes[2:],axis=1)]))
    (root/"wrist_failure.json").write_text(json.dumps({"clip":clip,"frame_zero_based":idx,
        "wrist_error_px":{c:[float(v) if np.isfinite(v) else None for v in e[idx]]
                         for c,e in zip(conditions,errors)}},indent=2)+"\n")


def blind(args, manifest):
    reviewer = args.output/"blind_review"
    reviewer.mkdir(exist_ok=True,parents=True)
    seed = 6081
    rng = np.random.default_rng(seed)
    key, tables = [], []
    for clip_number,index in enumerate(rng.permutation(len(manifest)),1):
        row = manifest[index]
        clip = row["clip_id"]
        code = f"sample_{clip_number:02}"
        directory = reviewer/code
        directory.mkdir(exist_ok=True)
        shutil.copyfile(args.videos/f"{clip}.mp4", directory/"reference.mp4")
        source = args.output/"direct"/clip
        camera = np.load(args.output/"data/paired"/f"{clip}.npz")["camera"]
        candidates = ("I_perturbed_parameters.npz","B_geometry_final_parameters.npz","C_temporal_final_parameters.npz")
        for variant,i in zip("ABC",rng.permutation(3)):
            file = source/candidates[i]
            fit = load_fit(file)
            save_avatar(args,args.renderer,file,directory/f"{variant}.mp4",fit["cam_trans"],camera)
            key.append(dict(sample=code,file=variant+".mp4",clip=clip,condition=candidates[i]))
        tables.append(dict(sample=code,relative_directory=code,fps=24,frames=row["rgb_frames"]))
        print(code,"blind videos complete",flush=True)
    write_csv(args.output/"blind_review_key.csv",key)
    write_csv(reviewer/"samples.csv",tables)
    write_csv(reviewer/"review_form.csv",[dict(sample=t["sample"],reviewer_code="",video_a_ASL_understandability="",
        video_b_ASL_understandability="",video_c_ASL_understandability="",best_candidate_or_tie="",
        sign_or_handshape_comments="",nonmanual_comments="",low_visibility_or_cannot_judge="") for t in tables])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("output","asset","videos"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--skip-blind",action="store_true")
    p.add_argument("--blind-only",action="store_true")
    args=p.parse_args()
    args.device="cpu"
    manifest=json.loads((args.output/"data/manifest.json").read_text())
    if not args.blind_only:
        report(args.output,manifest)
        failure(args.output, manifest,args.videos)
    if not args.skip_blind:
        import torch
        torch.set_num_threads(args.threads)
        with np.load(args.asset) as model:
            args.renderer=AvatarRenderer((1280,720),model["f"],model["v_template"])
        try:
            blind(args,manifest)
        finally:
            args.renderer.close()


if __name__=="__main__":
    main()
