"""Run the Uni-Sign vendored RTMW-192 detector on exact, single-signer How2Sign clips."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from .pose import BODY, FACE, LEFT, RIGHT, validate


def load_detector(repo, detector_path, pose_model, threads):
    import cv2
    import onnxruntime as ort

    sys.path.insert(0, str((repo / "demo/rtmlib-main").resolve(strict=True)))
    from rtmlib import Wholebody

    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    create_session = ort.InferenceSession
    with patch.object(ort, "InferenceSession", side_effect=lambda **kwargs: create_session(
            sess_options=options, **kwargs)):
        return Wholebody(det=str(detector_path.resolve(strict=True)),
                             pose=str(pose_model.resolve(strict=True)), det_input_size=(416, 416), pose_input_size=(192, 256),
                             backend="onnxruntime", device="cpu")


def detect_frame(detector, frame):
    height, width = frame.shape[:2]
    xy, confidence = detector(frame)
    if not len(xy):
        return np.zeros((133, 2), np.float32), np.zeros(133, np.float32), 0
    use = int(np.argmax(confidence[:, BODY].mean(axis=1)))
    return xy[use] / [width, height], confidence[use], len(xy)


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "detector", "pose-model", "video", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    detector = load_detector(args.repo, args.detector, args.pose_model, args.threads)
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    positions, scores, counts, timestamps = [], [], [], []
    try:
        while True:
            readable, frame = capture.read()
            if not readable:
                break
            height, width = frame.shape[:2]
            xy, confidence, n_people = detect_frame(detector, frame)
            positions.append(xy)
            scores.append(confidence)
            counts.append(n_people)
            timestamps.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.)
    finally:
        capture.release()
    positions, scores = validate(positions, scores)
    if declared_frames and declared_frames != len(positions):
        raise ValueError(f"Decoded {len(positions)} of {declared_frames} declared frames")
    if not (scores[:, BODY] > .3).any():
        raise ValueError("The entire video has no visible body landmarks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, keypoints=positions, scores=scores, timestamps_s=timestamps, fps=fps,
                        width=width, height=height)
    evidence = {
        "file": args.video.name, "sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "frames": len(positions), "fps": fps, "width": width, "height": height,
        "missing_detection_frames": counts.count(0), "multiple_detection_frames": sum(x > 1 for x in counts),
        "score_above_one_fraction": float((scores > 1).mean()),
        "visible_fractions": {name: float((scores[:, idx] > .3).mean())
                              for name, idx in (("body", BODY), ("left", LEFT), ("right", RIGHT), ("face", FACE))},
    }
    args.output.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
