"""Diagnose cached G1 projections using even-frame upper anchors and a second detector."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.run_real_validation import load_manifest, observations
from src.virtual_pose import CALIBRATION_POINTS, TIPS
from src.pose import LEFT, RIGHT

HANDS = np.stack([LEFT, RIGHT])
WRISTS = HANDS[:, 0]
TIP_WRISTS = np.repeat(WRISTS, 5)


def translation(virtual, real, scores):
    even = np.arange(0, len(real), 2)
    residuals = real[even][:, CALIBRATION_POINTS] - virtual[even][:, CALIBRATION_POINTS]
    scores = scores[even][:, CALIBRATION_POINTS]
    good = (scores > .3).sum(1) >= 3
    residuals = np.where(scores[..., None] > .3, residuals, np.nan)[good]
    shift = np.nanmedian(residuals, axis=1)
    if len(shift) < 2:
        raise ValueError("Too few anchor frames")
    return np.stack([np.interp(np.arange(len(real)), even[good], shift[:, axis]) for axis in range(2)], axis=1)


def reference_hands(video, model, real, score, wh):
    vision = mp.tasks.vision
    settings = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2)
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    shape = np.full_like(real, np.nan)
    n_detected = 0
    shoulder = np.median(np.linalg.norm((real[:, 5] - real[:, 6]) * wh, axis=-1))
    with vision.HandLandmarker.create_from_options(settings) as detector:
        for i in range(len(real)):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"{video}: frame {i} is missing")
            pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels))
            hands = np.array([[(p.x, p.y) for p in h] for h in result.hand_landmarks])
            n_detected += len(hands)
            if not len(hands):
                continue
            distances = np.linalg.norm((hands[:, None, 0] - real[i, WRISTS]) * wh, axis=-1)
            from_ids, to_ids = linear_sum_assignment(distances)
            for a, b in zip(from_ids, to_ids):
                other = 1 - b
                if (distances[a, b] < .4 * shoulder and score[i, WRISTS[b]] > .3 and
                        distances[a, other] - distances[a, b] > .075 * shoulder):
                    shape[i, HANDS[b]] = hands[a]
        if cap.read()[0]:
            raise ValueError(f"{video}: additional frames detected after the pose sequence")
    cap.release()
    return shape, n_detected


def per_frame_metrics(xy, target, wh):
    error = (xy - target) * wh
    return {
        "anchors": np.linalg.norm(error[:, CALIBRATION_POINTS], axis=-1),
        "wrists": np.linalg.norm(error[:, WRISTS], axis=-1),
        "tips": np.linalg.norm(error[:, TIPS], axis=-1),
        "handshape": np.linalg.norm(error[:, TIPS] - error[:, TIP_WRISTS], axis=-1),
    }


def evaluate(xy, target, wh, frames):
    def mean_valid(x):
        x = x[frames]
        return None if not np.isfinite(x).any() else float(np.median(x[np.isfinite(x)]))
    return {name: mean_valid(values) for name, values in per_frame_metrics(xy, target, wh).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("videos", "mp-model", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    path = args.output
    path.mkdir(exist_ok=True, parents=True)
    config = json.loads(Path("configs/stage4.json").read_text())
    rows = []
    for row in load_manifest(Path("configs/validation_pairs.json")):
        if not row["fit_frames"]:
            continue
        clip = row["clip_id"]
        real, score, wh = observations(Path("experiments/E3/real_pose"), row)
        with np.load(Path("experiments/E1_real") / clip / "fk_and_projection.npz") as data:
            virtual = data["keypoints"]
        video = args.videos / f"{clip}.mp4"
        meta = json.loads((Path("experiments/E3/real_pose") / f"{clip}.json").read_text())
        if hashlib.sha256(video.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"{clip}: RGB differs from the previous stage")
        independent, count = reference_hands(video, args.mp_model, real, score, wh)
        shift = translation(virtual, real, score)
        np.savez_compressed(path / f"{clip}.npz", mp=independent, shift=shift)
        visible = np.sum(np.isfinite(independent[:, WRISTS, 0]))
        print(clip, count, visible, flush=True)
        sets = {"odd_holdout": np.arange(1, len(real), 2)}
        sets.update({f"frame_{frame}": np.array([frame]) for frame in config["priority_frames"].get(clip, [])})
        masked_real = np.where(score[..., None] > .3, real, np.nan)
        for name, frames in sets.items():
            for key, pose in (("baseline", virtual), ("anchor_shift", virtual + shift[:, None]), ("detector_gap", masked_real)):
                for target, positions in (("RTMW", masked_real), ("MediaPipe", independent)):
                    if key == "detector_gap" and target == "RTMW":
                        continue
                    metrics = evaluate(pose, positions, wh, frames)
                    rows.append(dict(clip=clip, frames=name, method=key, target=target,
                                     n_hands=int(np.isfinite(positions[frames][:, WRISTS, 0]).sum()),
                                     shift_px=float(np.median(np.linalg.norm(shift[frames] * wh, axis=-1))),
                                     **metrics))
    with (path / "geometry.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
