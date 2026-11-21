"""Select source-disjoint recordings and build RTMW/MediaPipe checkpoints."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from scripts.stage6_prepare import prepare, select
from src.virtual_pose import model_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inventory","annotations","asset","motions","videos","mp-model","output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--clips", nargs="+")
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    old=json.loads(Path("configs/validation_pairs.json").read_text())
    old+=json.loads(Path("experiments/Stage6/data/manifest.json").read_text())
    cfg=json.loads(Path("configs/stage7.json").read_text())
    rows=select(a.inventory,a.annotations,cfg["selection"],old)
    if len(rows)!=cfg["selection"]["source_count"] or model_sha(a.asset)!=json.loads(
        Path("experiments/E1_real/geometry_qc.json").read_text())["asset_sha256"]:
        raise ValueError("Expected the frozen validation inventory and exact SMPL-X asset")
    d=a.output/"data"
    for name in ("paired","inputs"):
        (d/name).mkdir(parents=True,exist_ok=True)
    (d/"manifest.json").write_text(json.dumps(rows,indent=2)+"\n")
    shutil.copyfile(a.inventory,d/"nsa_inventory.txt")
    for r in rows:
        clip=r["clip_id"]
        if a.clips and clip not in a.clips:
            continue
        if not (d/"real_pose"/f"{clip}.npz").exists():
            print("Unavailable exact RGB/pose",clip,flush=True)
            continue
        prepare(r,a)
        for src in (a.motions/f"{clip}.npz",a.motions/f"{clip}.json",a.videos/f"{clip}.mp4"):
            shutil.copyfile(src,d/"inputs"/src.name)


if __name__=="__main__":
    main()
