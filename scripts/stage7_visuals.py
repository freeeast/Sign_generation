"""Render front/side skeletal before-after animations for the completed RGB runs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.stage4_review_media import render_pair
from src.smplx_to_pose import forward
from src.virtual_pose import apply_affine, load_fit, pinhole


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--asset",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(3)
    for out in sorted((a.root/"rgb").glob("*/full")):
        before=out/"initial_fit.npz"
        after=out/"robust_rgb_tracks.npz"
        if not before.exists() or not after.exists():
            continue
        clip=out.parent.name
        with np.load(a.root/"data/paired"/f"{clip}.npz") as pdat:
            camera=pdat["camera"]
        start,_=forward(before,a.asset,"neutral","model",64,"cpu",False)
        init=apply_affine(pinhole(start,load_fit(before)["cam_trans"]),camera)
        with np.load(after) as positions:
            render_pair(out/"initial_vs_robust_3d.mp4",
                        [init,positions["xy"]],[start,positions["points"]])


if __name__=="__main__":
    main()
