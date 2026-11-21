"""Render geometry diagnostics and randomized A/B stimuli without model scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from src.pose import FACE
from src.run_real_validation import load_manifest, observations
from src.evaluate_feedback import write_csv


LINKS = [(0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
for wrist in (91, 112):
    for finger in range(5):
        chain = [wrist, *range(wrist + 1 + 4*finger, wrist + 5 + 4*finger)]
        LINKS.extend(zip(chain[:-1], chain[1:]))
POINTS = sorted({i for pair in LINKS for i in pair} | set(FACE))
SIZE = (320, 252)


def label(canvas, text, xy):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1, cv2.LINE_AA)


def pose_on(image, xy, color, shift=(0, 0)):
    finite = np.isfinite(xy).all(axis=1)
    pixels = np.rint(np.nan_to_num(xy) + shift).astype(int)
    for start, end in LINKS:
        if finite[start] and finite[end]:
            cv2.line(image, tuple(pixels[start]), tuple(pixels[end]), color, 1, cv2.LINE_AA)
    for index in POINTS:
        if finite[index]:
            cv2.circle(image, tuple(pixels[index]), 2, color, -1, cv2.LINE_AA)


def diagnostics(output, videos, config, manifest):
    panels = []
    for clip, frames in config["priority_frames"].items():
        frame = frames[-1] if clip.startswith("3Tr") else frames[0]
        real, score, wh = observations(Path("experiments/E3/real_pose"), manifest[clip])
        with np.load(Path("experiments/E1_real") / clip / "fk_and_projection.npz") as data:
            virtual = data["keypoints"][frame]
        with np.load(output / "geometry" / f"{clip}.npz") as data:
            mp, shift = data["mp"][frame], data["shift"][frame]
        cap = cv2.VideoCapture(str(videos / f"{clip}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Missing {clip} frame {frame}")
        focus = real[frame, POINTS]*wh
        x, y = np.clip(np.min(focus, axis=0).astype(int)-82, 0, wh)
        x1, y1 = np.clip(np.max(focus, axis=0).astype(int)+82, 0, wh)
        tiles = []
        name = clip.split("_")[0]
        for title, tracks in (
            ("RTMW yellow / MediaPipe cyan", [(np.where(score[frame, :, None]>.3, real[frame], np.nan), (40,225,255)), (mp,(245,230,40))]),
            ("Baseline fit", [(virtual,(230,90,245))]),
            ("Even-anchor translation", [(virtual+shift,(230,90,245))]),
        ):
            img = image.copy()
            for track, color in tracks:
                pose_on(img, track*wh, color)
            crop = img[y:y1, x:x1]
            factor = min(320/crop.shape[1], 252/crop.shape[0])
            crop = cv2.resize(crop, None, fx=factor, fy=factor)
            tile = np.full((280, 324, 3), 32, np.uint8)
            xx = (324-crop.shape[1])//2
            tile[23:23+crop.shape[0], xx:xx+crop.shape[1]] = crop
            label(tile, f"{name} f{frame}  {title}", (7,15))
            tiles.append(tile)
        panels.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(output/"geometry/failure_frames.png"), np.concatenate(panels, axis=0))


def render_pair(path, front, points):
    width, height = SIZE
    frames = len(front[0])
    front = [xy * [1280,720] for xy in front]
    side = [xyz[..., [2, 1]] * [-1, 1] for xyz in points]
    def convert(tracks):
        all_xy = np.concatenate([x[:, POINTS].reshape(-1, 2) for x in tracks])
        low, high = np.quantile(all_xy, [.002,.998], axis=0)
        scale = min((width - 22) / (high[0]-low[0]), (height - 28) / (high[1]-low[1]))
        center = (low+high)/2
        return [(x-center)*scale+[width/2,height/2+10] for x in tracks]
    front, side = convert(front), convert(side)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width*2}x{height*2}", "-r", "24", "-i", "-", "-an", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "faststart", str(path)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
        for frame in range(frames):
            out = []
            for name, tracks in (("Front", front), ("Side", side)):
                tiles = []
                for k, condition in enumerate(("A", "B")):
                    img = np.full((height, width, 3), 27, np.uint8)
                    pose_on(img, tracks[k][frame], (135,220,246))
                    label(img, f"{condition}  {name}", (12,22))
                    tiles.append(img)
                out.append(np.concatenate(tiles, axis=1))
            process.stdin.write(np.concatenate(out, axis=0).tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Unable to encode blind review video")


def review_set(root, videos, cfg, manifest):
    with (root/"candidate_pairs.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"]=="eval" and
                r["condition"]=="baseline" and abs(float(r["geometry_A"]))<=cfg["geometry_tie_px"]]
    output = root / "blind_review"
    for d in ("pairs","reference"):
        (output/d).mkdir(exist_ok=True,parents=True)
    public, key = [], []
    for clip_n, clip in enumerate(cfg["evaluation"], 1):
        alias = f"source_{clip_n}"
        shutil.copyfile(videos/f"{clip}.mp4", output/"reference"/f"{alias}.mp4")
        with np.load(root/"projection_cache"/f"{clip}.npz") as projected:
            for family in cfg["families"]:
                matched = sorted([r for r in rows if r["clip"]==clip and r["family"]==family],
                                 key=lambda r: hashlib.sha256(("blind-v1"+r["pair_id"]).encode()).digest())[:2]
                for row in matched:
                    pair = row["pair_id"]
                    reverse = int(pair[-1], 16) % 2
                    a,b = (row["case_b"],row["case_a"]) if reverse else (row["case_a"],row["case_b"])
                    ids = [f"{family}__{case}" for case in (a,b)]
                    render_pair(output/"pairs"/f"{pair}.mp4",
                                [projected[x] for x in ids], [projected[x+"__3d"] for x in ids])
                    public.append(dict(pair_id=pair, video=f"pairs/{pair}.mp4",
                        reference=f"reference/{alias}.mp4", approximate_english=manifest[clip]["reference"],
                        handshape_A_B_tie_unclear="", meaning_A_B_tie_unclear="", confidence_1_to_5="", notes=""))
                    key.append(dict(pair_id=pair, clip=clip, family=family, presentation_a=a, presentation_b=b))
    if public:
        write_csv(output/"reviewer_sheet.csv",public)
        write_csv(root/"review_key_KEEP_SEPARATE.csv",key)
    print("Blind pairs",len(public))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--videos",required=True,type=Path)
    p.add_argument("--geometry-only",action="store_true")
    args=p.parse_args()
    cv2.setNumThreads(1)
    config=json.loads(Path("configs/stage4.json").read_text())
    manifest={r["clip_id"]:r for r in load_manifest(Path("configs/validation_pairs.json"))}
    diagnostics(args.root,args.videos,config,manifest)
    if not args.geometry_only:
        review_set(args.root,args.videos,config,manifest)


if __name__=="__main__":
    main()
