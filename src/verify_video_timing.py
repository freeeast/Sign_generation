"""Verify a sentence clip against the host video at both How2Sign timestamp choices."""
import argparse
import json
from pathlib import Path

import numpy as np


def verify(raw, clip, original_start, realigned_start):
    import cv2

    def read(capture, frame):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {frame}")
        height, width = image.shape[:2]
        center = image[height // 5:height * 4 // 5, width // 3:width * 2 // 3]
        return cv2.resize(center, (128, 96), interpolation=cv2.INTER_AREA).astype(np.float32)

    source = cv2.VideoCapture(str(raw))
    sentence = cv2.VideoCapture(str(clip))
    try:
        fps = source.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != sentence.get(cv2.CAP_PROP_FPS):
            raise ValueError("Both MP4 files must decode with the same native FPS")
        count = int(sentence.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = np.linspace(0, count - 1, 7, dtype=int)
        rows = []
        for frame in frames:
            image = read(sentence, frame)
            result = {"sentence_frame": int(frame)}
            for label, start in (("original", original_start), ("realigned", realigned_start)):
                center = round(start * fps) + frame
                best_error, best_frame = min((float(np.abs(read(source, i) - image).mean()), i)
                                             for i in range(max(0, center - 4), center + 5))
                result[label] = {"rgb_mean_absolute_difference": best_error,
                                 "source_frame": int(best_frame)}
            rows.append(result)
    finally:
        sentence.release()
        source.release()
    return dict(raw=raw.name, clip=clip.name, fps=fps, reference_frames=count, evidence=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--original-start", type=float, required=True)
    parser.add_argument("--realigned-start", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.raw, args.clip, args.original_start, args.realigned_start)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
