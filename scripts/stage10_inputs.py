"""Load the fixed Stage 9 inputs or reconstruct coarse IK for locked older recordings."""
from pathlib import Path
import numpy as np

from scripts.stage7_optimize import read,subset,write
from scripts.stage8_imitate import initialize
from src.stage9_hand import tensor
from src.virtual_pose import load_fit


def clip_inputs(cfg, source, asset, cache):
    archived = Path("experiments/Stage7")
    old = Path("experiments/Stage8")/source
    s9 = Path("experiments/Stage9")/source
    if source.startswith("DKO"):
        old = Path("experiments/Stage9/new_source/stage8_controls")/source
        s9 = Path("experiments/Stage9/new_source")/source
    locked = source in cfg["locked"]
    if locked:
        archived = Path("experiments/Stage6")
    data = archived/"data"
    with np.load(data/"real_pose"/f"{source}.npz") as tracks:
        obs, confidence = tracks["keypoints"],tracks["scores"]
    with np.load(data/"paired"/f"{source}.npz") as tracks:
        affine, holdout = tracks["camera"],tracks["mp"]
    if locked:
        upstream = read(Path("configs/stage8.json"))
        prior = read(Path("configs/stage5.json"))
        initial = data/"source_fits"/f"{source}.npz"
        nsafit = load_fit(initial)
        coarse = initialize(nsafit,asset,upstream,prior,obs,confidence,affine,(1280,720),cache/"coarse","B")
        indices = subset(upstream["reference_stride"],len(obs),upstream["max_expert_frames"])
        videos = {"bcNtUYh3GNs": "sample_01","33YRZ5UZTA4": "sample_02"}
        video = archived/"blind_review"/videos[source.split("_")[0]]/"reference.mp4"
    else:
        nsafit = load_fit(data/"inputs"/f"{source}.npz")
        coarse = load_fit(old/"B/initial_fit.npz")
        indices = np.array(read(old/"frames.json")["optimization_frames"])
        video = data/"inputs"/f"{source}.mp4"
    if len(coarse["body_pose"]) != len(obs):
        raise ValueError(f"Frame alignment is inconsistent for {source}")
    return coarse,nsafit,np.asarray(indices),obs,confidence,affine,holdout,s9,video
