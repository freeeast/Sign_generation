"""Audit saved fitted projections against held-out RTMW points and RGB frames."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.pose import BODY, FACE, LEFT, RIGHT
from src.run_real_validation import load_manifest, observations
from src.virtual_pose import TIPS, model_sha

VISIBLE = np.r_[BODY, FACE, LEFT, RIGHT]
WRISTS_TIPS = np.r_[[9, 10], TIPS]
BONES = [(0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
for wrist in (91, 112):
    for first in range(wrist + 1, wrist + 21, 4):
        chain = (wrist, *range(first, first + 4))
        BONES.extend(zip(chain, chain[1:]))


def error(xy, observed, score, wh, obs_index, virtual_index, points):
    delta = np.linalg.norm((xy[virtual_index][:, points] - observed[obs_index][:, points]) * wh, axis=-1)
    return float(np.median(delta[score[obs_index][:, points] > .3]))


def overlay(clip, frames, xy, observed, score, root, output):
    cap = cv2.VideoCapture(str(root / (clip + ".mp4")))
    cards = []
    for i in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode {clip} frame {i}")
        wh = np.array(frame.shape[:2][::-1])
        limits = np.vstack([xy[i, VISIBLE], observed[i, VISIBLE]]) * wh
        x0, y0 = np.maximum(limits.min(axis=0).astype(int) - 40, 0)
        x1, y1 = np.minimum(limits.max(axis=0).astype(int) + 40, wh)
        for points, color in ((observed[i], (55, 231, 255)), (xy[i], (240, 92, 255))):
            points = np.rint(points * wh).astype(int)
            for a, b in BONES:
                if score[i, a] > .3 and score[i, b] > .3:
                    cv2.line(frame, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
            for point in VISIBLE[score[i, VISIBLE] > .3]:
                cv2.circle(frame, tuple(points[point]), 2, color, -1, cv2.LINE_AA)
        crop = frame[y0:y1, x0:x1]
        factor = min(390 / crop.shape[1], 350 / crop.shape[0])
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        card = np.full((408, 408, 3), 25, dtype=np.uint8)
        x = (408 - crop.shape[1]) // 2
        card[35:35+crop.shape[0], x:x+crop.shape[1]] = crop
        cv2.putText(card, f"Frame {i}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .56, (250, 250, 250), 1, cv2.LINE_AA)
        cv2.putText(card, "RTMW", (18, 397), cv2.FONT_HERSHEY_SIMPLEX, .52, (55, 231, 255), 1, cv2.LINE_AA)
        cv2.putText(card, "SMPL-X", (102, 397), cv2.FONT_HERSHEY_SIMPLEX, .52, (240, 92, 255), 1, cv2.LINE_AA)
        cards.append(card)
    cap.release()
    cv2.imwrite(str(output), np.concatenate(cards, axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "poses", "videos", "geometry", "asset"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    names = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "hands_meanl",
             "hands_meanr", "dynamic_lmk_faces_idx", "dynamic_lmk_bary_coords")
    with np.load(args.asset, allow_pickle=False) as file:
        shapes = {name: list(file[name].shape) for name in names}
        if not all(np.isfinite(file[name]).all() for name in names):
            raise ValueError("SMPL-X asset contains nonfinite model arrays")
    (args.geometry / "model_asset_audit.json").write_text(json.dumps({
        "sha256": model_sha(args.asset), "array_shapes": shapes, "finite_arrays": list(shapes),
        "source": "user-supplied SMPLX_NEUTRAL.npz",
    }, indent=2) + "\n")
    output = []
    for r in load_manifest(args.manifest):
        if not r["fit_frames"]:
            continue
        clip = r["clip_id"]
        target = args.geometry / clip
        observed, score, wh = observations(args.poses, r)
        with np.load(target / "fk_and_projection.npz", allow_pickle=False) as data:
            xy = data["keypoints"]
        idx = np.arange(5, len(xy) - 4, 2)
        zeros = np.arange(1, len(xy), 2)
        shoulder_px = float(np.median(np.linalg.norm((observed[:, 5] - observed[:, 6]) * wh, axis=-1)))
        lags = {lag: error(xy, observed, score, wh, idx, idx + lag, WRISTS_TIPS)
                for lag in range(-4, 5)}
        forearm = error(xy, observed, score, wh, zeros, zeros, np.array([7, 8, 9, 10]))
        wrist = error(xy, observed, score, wh, zeros, zeros, np.array([9, 10]))
        tip = error(xy, observed, score, wh, zeros, zeros, TIPS)
        distorted = np.linalg.norm((xy - observed) * wh, axis=-1)
        tip_p95 = float(np.percentile(distorted[zeros][:, TIPS][score[zeros][:, TIPS] > .3], 95))
        worst = int(zeros[np.argmax(np.median(distorted[zeros][:, TIPS], axis=1))])
        selected = ((xy < 0) | (xy > 1)).any(-1)
        best_lag = min(lags, key=lags.get)
        swapped_wrists_px = float(np.median(np.linalg.norm(
            (xy[zeros][:, [10, 9]] - observed[zeros][:, [9, 10]]) * wh, axis=-1)))
        output.append(dict(clip=clip, frames=len(xy), shoulder_px=shoulder_px, wrist_px=wrist, tip_px=tip,
            wrist_over_shoulder=wrist / shoulder_px, tip_over_shoulder=tip / shoulder_px, tip_p95_px=tip_p95,
            elbow_wrist_px=forearm, swapped_wrists_px=swapped_wrists_px, best_lag_frames=best_lag,
            lag0_minus_best_px=lags[0] - lags[best_lag], selected_out_of_image_fraction=float(selected[:, VISIBLE].mean()),
            worst_tip_frame=worst, **{f"lag_{lag}_px": value for lag, value in lags.items()}))
        overlay(clip, sorted({len(xy) // 4, len(xy) // 2, worst}), xy, observed, score, args.videos, target / "rgb_overlay.png")
    with (args.geometry / "alignment_audit.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output[0])
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
