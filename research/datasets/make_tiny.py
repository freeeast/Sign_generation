"""Copy two 48-frame NSA examples per train/validation split for a fit check."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = args.dataset.resolve()
    with (dataset / "sample.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for split in ("train", "val"):
        eligible = [row for row in rows if row["split"] == split and int(row["frames"]) >= 48]
        if len(eligible) < 2:
            raise ValueError(f"Expected at least two clips in {split}")
        for row in eligible[:2]:
            source = (dataset / row["path"]).resolve()
            if not source.is_relative_to(dataset):
                raise ValueError("Manifest path escapes dataset")
            target = args.output / source.name
            with np.load(source, allow_pickle=False) as motion:
                frames = int(row["frames"])
                data = {name: value[:48] if value.ndim and len(value) == frames else value
                        for name, value in motion.items()}
            np.savez_compressed(target, **data)
            selected.append(dict(row, path=target.name, frames=48))
    with (args.output / "sample.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=selected[0])
        writer.writeheader()
        writer.writerows(selected)
    cfg = json.loads(Path("experiments/generative_motion/flow_baseline.json").read_text())
    cfg.update(frames=48, width=48, depth=4, steps=300, validate_every=50,
               batch_size=8, validation_batches=4)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    main()
