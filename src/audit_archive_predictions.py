"""Compare the archived Uni-Sign text with its vendored single-clip decoding path."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import random

import numpy as np

from .pose import load_pose
from .unisign import UniSign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repo", "checkpoint", "mt5", "pose-dir", "manifest", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    model = UniSign(args.repo, args.checkpoint, args.mt5, "cpu", 4)
    rows = json.loads(args.manifest.read_text())
    with gzip.open(args.repo / "data/How2Sign/labels.test", "rb") as stream:
        labels = pickle.load(stream)  # pinned, official Uni-Sign repository
    archived = args.repo / "out/eval_how2sign_pose_only"
    def read_records(path, marker):
        records = {}
        for line in path.read_text().splitlines():
            key, value = line.removeprefix("sample: ").split(marker, 1)
            if key in records:
                raise ValueError(f"Duplicate archive key: {key}")
            records[key] = value.strip()
        return records

    refs = read_records(archived / "test_tmp_refs.txt", ", ground-truth: ")
    predictions = read_records(archived / "test_tmp_pres.txt", ", prediction: ")
    keys = list(labels)
    if set(keys) != refs.keys() or set(keys) != predictions.keys():
        raise ValueError("Archive clip IDs and test labels have inconsistent coverage")
    alignment = sum(labels[k]["text"].strip() == refs[k] for k in keys)
    out = []
    for row in rows:
        clip = row["clip_id"]
        key = clip + ".mp4"
        index = keys.index(key)
        xy, scores = load_pose(args.pose_dir / (clip + ".pkl"), trusted_pickle=True)
        if len(xy) > 256:
            choices = [(f"random_seed_{seed}", np.array(sorted(random.Random(seed).sample(range(len(xy)), 256))))
                       for seed in (0, 17, 41)]
            choices.insert(0, ("uniform", np.linspace(0, len(xy) - 1, 256, dtype=int)))
        else:
            choices = [("all", np.arange(len(xy)))]
        for sampling, indices in choices:
            parts = model.load_part_kp(xy[indices, None], scores[indices, None], force_ok=True)
            src, tgt = model.collate([(clip, parts, labels[key]["text"], "", {})])
            for name, value in src.items():
                if model.torch.is_tensor(value):
                    src[name] = value.to(dtype=model.torch.float32)
            with model.torch.inference_mode():
                state = model.model(src, tgt)
                result = model.model.generate(state, max_new_tokens=100, num_beams=4)
            prediction = model.model.mt5_tokenizer.batch_decode(result, skip_special_tokens=True)[0]
            out.append(dict(clip_id=clip, sampling=sampling, raw_frames=len(xy),
                            frames_used=len(indices), archive_index=index, archive_reference=refs[key],
                            reference=labels[key]["text"], archived_prediction=predictions[key],
                            prediction=prediction, exact=prediction == predictions[key].strip(),
                            nll=float(state["loss"])))
            print(clip, sampling, out[-1]["exact"], out[-1]["nll"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(archive_rows=len(keys), aligned_reference_rows=alignment,
                                          cases=out), indent=2) + "\n")


if __name__ == "__main__":
    main()
